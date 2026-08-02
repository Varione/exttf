"""Tests for the incremental NAV refresh safety and persistence guarantees.

Covers (audit 2026-08-02):
- provider_revision_report.csv and refresh_report.json are persisted per run
- a fund with CORE NAV revisions is NOT appended in that run and the run
  exits with code 2 (FORWARD_OBSERVATION_PAUSED)
- funds without core revisions still append missing dates
- _round_match float tolerance treats provider decimal artifacts as equal
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "migrations"))
sys.path.insert(0, str(ROOT / "src"))

import refresh_nav_incremental as refresh  # noqa: E402


def _provider_js_text(
    dates_ms: list[int], navs: list[float], returns: list[float | None],
    cum_pairs: list[tuple[int, float]],
) -> str:
    net = ",".join(
        '{"x":%d,"y":%s,"equityReturn":%s}'
        % (x, y, "null" if r is None else r)
        for x, y, r in zip(dates_ms, navs, returns)
    )
    cum = ",".join("[%d,%s]" % (x, y) for x, y in cum_pairs)
    return (
        f"var Data_netWorthTrend = [{net}];\n"
        f"var Data_ACWorthTrend = [{cum}];\n"
    )


def _date_ms(date_str: str) -> int:
    import datetime as dt

    dt_naive = dt.date.fromisoformat(date_str)
    dt_local = dt.datetime(dt_naive.year, dt_naive.month, dt_naive.day, tzinfo=dt.timezone.utc)
    return int(dt_local.timestamp() * 1000)


def _make_db(path: Path, codes: list[str]) -> None:
    with sqlite3.connect(str(path)) as conn:
        conn.execute(
            "CREATE TABLE otf_fund_catalog (fund_code TEXT PRIMARY KEY, "
            "fund_name TEXT, fund_type TEXT, asset_class TEXT, "
            "underlying_name TEXT, inception_date TEXT, termination_date TEXT, "
            "source TEXT)"
        )
        conn.execute(
            "CREATE TABLE otf_fund_nav (fund_code TEXT, nav_date TEXT, "
            "unit_nav REAL, cumulative_nav REAL, daily_growth_pct REAL, "
            "cumulative_nav_imputed INTEGER, distribution_per_share REAL, "
            "share_adjustment_factor REAL, total_return_factor REAL, "
            "PRIMARY KEY (fund_code, nav_date))"
        )
        for code in codes:
            conn.execute(
                "INSERT INTO otf_fund_catalog (fund_code) VALUES (?)", (code,)
            )


def _insert_row(
    conn: sqlite3.Connection, code: str, date: str, unit: float,
    cum: float, growth: float | None, imputed: int,
) -> None:
    conn.execute(
        "INSERT INTO otf_fund_nav VALUES (?,?,?,?,?,?,?,?,?)",
        (code, date, unit, cum, growth, imputed, 0.0, 1.0, 1.0),
    )


def _run_main(monkeypatch: pytest.MonkeyPatch, db_path: Path, js_text: str) -> int:
    class _FakeResponse:
        text = js_text

    def _fake_get(url: str, timeout: int = 30):  # noqa: ARG001
        return _FakeResponse()

    monkeypatch.setattr(refresh.requests, "get", _fake_get)
    monkeypatch.setattr(sys, "argv", ["refresh_nav_incremental.py", "--db", str(db_path)])
    return refresh.main()


class TestRoundMatch:
    def test_float_artifact_tolerance(self):
        values = (0.9067000000000001, 1.0)
        old = (0.9067, 1.0)
        assert refresh._round_match(values, old) is True

    def test_real_diff_detected(self):
        values = (0.9067, 1.0)
        old = (0.9100, 1.0)
        assert refresh._round_match(values, old) is False

    def test_len_mismatch(self):
        assert refresh._round_match((1.0,), (1.0, 2.0)) is False


class TestRefreshPersistence:
    def test_report_files_persisted(self, tmp_path, monkeypatch):
        db = tmp_path / "nav.db"
        _make_db(db, ["000001"])
        dates = [_date_ms(d) for d in ("2026-07-29", "2026-07-30", "2026-07-31")]
        js = _provider_js_text(
            dates, [1.0, 1.01, 1.02], [None, 1.0, 0.9900990099009901],
            [(dates[0], 1.0), (dates[1], 1.01), (dates[2], 1.02)],
        )
        with sqlite3.connect(str(db)) as conn:
            _insert_row(conn, "000001", "2026-07-29", 1.0, 1.0, None, 0)

        run_dir = tmp_path / "report_dir"
        monkeypatch.setattr(refresh, "REFRESH_REPORT_DIR", run_dir)
        exit_code = _run_main(monkeypatch, db, js)
        assert exit_code == 0

        report_files = list(run_dir.glob("nav_refresh_*/refresh_report.json"))
        assert report_files, "refresh_report.json must be persisted"
        report = json.loads(report_files[0].read_text(encoding="utf-8"))
        assert report["rows_added"] == 2
        assert report["blocked_funds"] == []
        csv_files = list(run_dir.glob("nav_refresh_*/provider_revision_report.csv"))
        assert csv_files
        assert "fund_code" in csv_files[0].read_text(encoding="utf-8-sig")

    def test_core_revision_blocks_append(self, tmp_path, monkeypatch):
        db = tmp_path / "nav.db"
        _make_db(db, ["000001"])
        dates = [_date_ms(d) for d in ("2026-07-29", "2026-07-30")]
        js = _provider_js_text(
            dates, [1.05, 1.06], [None, 0.9523809523809523],
            [(dates[0], 1.05), (dates[1], 1.06)],
        )
        with sqlite3.connect(str(db)) as conn:
            _insert_row(conn, "000001", "2026-07-29", 1.00, 1.00, 0.0, 0)

        run_dir = tmp_path / "report_dir"
        monkeypatch.setattr(refresh, "REFRESH_REPORT_DIR", run_dir)
        exit_code = _run_main(monkeypatch, db, js)
        assert exit_code == 2, "core revisions must pause forward observation"

        report_files = list(run_dir.glob("nav_refresh_*/refresh_report.json"))
        report = json.loads(report_files[0].read_text(encoding="utf-8"))
        assert report["core_revision_rows"] >= 1
        assert report["blocked_funds"], "fund with core revisions must be blocked"
        assert report["blocked_funds"][0]["fund_code"] == "000001"
        assert report["rows_added"] == 0

        with sqlite3.connect(str(db)) as conn:
            stored_dates = [
                r[0] for r in conn.execute(
                    "SELECT nav_date FROM otf_fund_nav WHERE fund_code='000001'"
                ).fetchall()
            ]
        assert stored_dates == ["2026-07-29"], "no new row may be appended for blocked fund"
        csv_files = list(run_dir.glob("nav_refresh_*/provider_revision_report.csv"))
        assert csv_files, "provider revision report must be persisted"
        csv_text = csv_files[0].read_text(encoding="utf-8-sig")
        assert "000001" in csv_text and "2026-07-29" in csv_text

    def test_derived_only_diff_does_not_block(self, tmp_path, monkeypatch):
        db = tmp_path / "nav.db"
        _make_db(db, ["000001"])
        dates = [_date_ms(d) for d in ("2026-07-29", "2026-07-30")]
        js = _provider_js_text(
            dates, [1.0, 1.01], [None, 1.0],
            [(dates[0], 1.0), (dates[1], 1.01)],
        )
        with sqlite3.connect(str(db)) as conn:
            conn.execute(
                "INSERT INTO otf_fund_nav VALUES (?,?,?,?,?,?,?,?,?)",
                ("000001", "2026-07-29", 1.0, 1.0, None, 0, 0.0, 1.0, 1.0),
            )

        run_dir = tmp_path / "report_dir"
        monkeypatch.setattr(refresh, "REFRESH_REPORT_DIR", run_dir)
        exit_code = _run_main(monkeypatch, db, js)
        assert exit_code == 0
        report_files = list(run_dir.glob("nav_refresh_*/refresh_report.json"))
        report = json.loads(report_files[0].read_text(encoding="utf-8"))
        assert report["blocked_funds"] == []
        assert report["rows_added"] == 1
