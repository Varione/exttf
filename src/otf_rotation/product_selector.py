"""ProductSelector: time-point-aware dynamic fund selection.

Selects top-N funds per sleeve using only data available as of the
selection date.  Every select() call requires a date parameter; passing
date=None raises for formal research use.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from pathlib import Path
import sqlite3

import pandas as pd
import numpy as np

from otf_trading_rules import ProductRuleBook

# ---------------------------------------------------------------------------
# Shared data / mappings
# ---------------------------------------------------------------------------

SLEEVE_PATTERNS: dict[str, list[str]] = {
    "DIVIDEND": [r"红利低波", r"红利指数", r"中证红利", r"沪深300红利"],
    "GOLD":     [r"黄金"],
    "NASDAQ":   [r"纳斯达克", r"纳指", r"nasdaq", r"NASDAQ"],
    "SP500":    [r"标普500", r"标普 500", r"S&P500", r"sp500", r"标普指数"],
    "HANGSENG": [r"恒生", r"港股", r"H股", r"HANG SENG", r"HANGSENG"],
    "CSI300":   [r"沪深300", r"沪深 300", r"300etf", r"300ETF"],
    "CSI500":   [r"中证500", r"中证 500", r"500etf", r"500ETF"],
    "CSI1000":  [r"中证1000", r"中证 1000", r"1000etf", r"1000ETF"],
    "CD_NCD":   [r"同业存单"],
}

TYPE_TO_SLEEVE: dict[str, str | None] = {
    "指数型-股票":      None,
    "指数型-海外股票":    None,
    "指数型-固收":       "GOV_BOND_1_3Y",
    "指数型-其他":       None,
    "债券型-长债":       "GOV_BOND_3_5Y",
    "债券型-信用债":      "CREDIT_BOND",
    "债券型-利率债":      "GOV_BOND_1_3Y",
    "债券型-中短债":      "ULTRA_SHORT_BOND",
    "债券型-混合一级":    "CREDIT_BOND",
    "债券型-混合二级":    "CREDIT_BOND",
    "混合型-偏债":       "CREDIT_BOND",
    "混合型-灵活":       "CREDIT_BOND",
    "货币型-普通货币":    "MONEY_MARKET",
    "货币型-浮动净值":    "MONEY_MARKET",
    "混合型-偏股":       None,
    "混合型-平衡":       None,
    "股票型":           None,
    "QDII-混合偏股":     None,
    "QDII-普通股票":     None,
    "QDII-纯债":        "GOV_BOND_3_5Y",
    "QDII-混合灵活":     None,
    "QDII-混合债":       "CREDIT_BOND",
    "QDII-商品":        "GOLD",
    "QDII-FOF":        None,
    "QDII-混合平衡":     None,
    "QDII-REITs":      None,
    "FOF-稳健型":        "CREDIT_BOND",
    "FOF-均衡型":        None,
    "FOF-进取型":        None,
    "Reits":           None,
    "商品":             "GOLD",
}

SLEEVE_CANDIDATES: dict[str, list[str]] = {
    "CSI300": ["000051", "160706"],
    "CSI500": ["000008", "000962"],
    "CSI1000": ["006486"],
    "CHINEXT": ["050021", "001592"],
    "DIVIDEND": ["007466", "007605"],
    "GOLD": ["000216", "000218", "002610"],
    "NASDAQ": ["021778", "040046"],
    "SP500": ["050025", "013425"],
    "HANGSENG": ["000071", "000948"],
    "MONEY_MARKET": ["260102", "040003", "217004", "050003", "202301"],
    "CD_NCD": ["014430", "018613", "018809"],
    "ULTRA_SHORT_BOND": ["006663"],
    "GOV_BOND_3_5Y": ["001512"],
    "GOV_BOND_1_3Y": ["005839"],
    "CREDIT_BOND": ["000148"],
}

ALL_SLEEVES = list(SLEEVE_CANDIDATES.keys())


# Share-class suffix patterns for dedup.
_SHARE_SUFFIX = re.compile(r"[_\- .]?([ACDEFHIYZ]|B\d+)$", re.IGNORECASE)


def _base_fund_name(name: str) -> str:
    """Strip trailing share-class suffix for same-family dedup."""
    return _SHARE_SUFFIX.sub("", name.strip()).strip()


def _match_sleeve_by_name(fund_name: str) -> str | None:
    name_lower = fund_name.lower()
    for sleeve, patterns in SLEEVE_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, name_lower, re.IGNORECASE):
                return sleeve
    return None


def _match_sleeve_by_type(fund_type: str) -> str | None:
    return TYPE_TO_SLEEVE.get(fund_type)


# ---------------------------------------------------------------------------
# Audit record
# ---------------------------------------------------------------------------

@dataclass
class SelectionCandidate:
    fund_code: str
    sleeve: str
    sub_fee: float
    nav_observations: int          # actual COUNT(nav_date <= selection_date) per P0-3
    nav_days_available: int        # calendar days from first_nav to selection_date (kept for backward compat)
    last_nav_date: pd.Timestamp | None  # most recent NAV date as of selection_date
    max_gap_days: int              # largest gap between consecutive NAV dates in history window
    rule_source: str               # "verified" | "auto"
    ineligible: bool = False
    ineligible_reason: str = ""


@dataclass
class SelectionAudit:
    sleeve: str
    date: str
    top_n: int
    candidates: list[SelectionCandidate] = field(default_factory=list)
    selected: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "sleeve": self.sleeve,
            "date": self.date,
            "top_n": self.top_n,
            "candidates": [
                {"fund_code": c.fund_code, "sleeve": c.sleeve,
                 "sub_fee": c.sub_fee,
                 "nav_observations": c.nav_observations,
                 "nav_days": c.nav_days_available,
                 "last_nav_date": str(c.last_nav_date.date()) if c.last_nav_date is not None else None,
                 "max_gap_days": c.max_gap_days,
                 "rule_source": c.rule_source,
                 "eligible": not c.ineligible,
                 "reason": c.ineligible_reason}
                for c in self.candidates
            ],
            "selected": self.selected,
        }


# ---------------------------------------------------------------------------
# ProductSelector
# ---------------------------------------------------------------------------

class ProductSelector:
    """Time-point-aware fund selector per sleeve.

    Parameters
    ----------
    db_path : str
        Path to SQLite DB with otf_fund_catalog, otf_fund_nav, otf_default_rules.
    rules_path : str
        Path to CSV with verified product rules.
    min_history_days : int
        Minimum NAV history (calendar days) required for eligibility.
    min_nav_observations : int
        Minimum actual NAV observations (COUNT) as of selection date per P0-3.
    max_last_nav_gap_days : int
        Maximum allowed gap between last NAV and selection date per P0-3.
    max_consecutive_gap_days : int
        Maximum allowed consecutive gap between NAV dates in history window per P0-3.
    exposure_mapping_path : str
        Path to exposure mapping CSV per P0-4. Products must have
        mapping_confidence=HIGH and review_status=APPROVED to be eligible.
    allow_auto_rules : bool
        If True, allow funds with DB-generated rules (products not in CSV).
        Formal research should keep this False.
    """  # noqa: E501

    def __init__(
        self,
        db_path: str = "data/processed/otf_expanded.sqlite",
        rules_path: str = "config/otf_product_rules.csv",
        exposure_mapping_path: str = "config/otf_exposure_mapping.csv",
        min_history_days: int = 120,
        min_nav_observations: int = 60,
        max_last_nav_gap_days: int = 10,
        max_consecutive_gap_days: int = 30,
        allow_auto_rules: bool = False,
        rule_book: ProductRuleBook | None = None,
    ):
        self._db_path = db_path
        self._rules_path = rules_path
        self._exposure_mapping_path = exposure_mapping_path
        self._min_history_days = min_history_days
        self._min_nav_observations = min_nav_observations
        self._max_last_nav_gap_days = max_last_nav_gap_days
        self._max_consecutive_gap_days = max_consecutive_gap_days
        self._allow_auto_rules = allow_auto_rules
        self._rule_book = rule_book

        # Populated in _build()
        self._catalog: dict[str, dict] = {}
        self._first_nav_date: dict[str, pd.Timestamp] = {}
        self._rule_map: dict[str, dict] = {}
        self._exposure_map: dict[str, dict] = {}
        self._candidates: dict[str, list[str]] = {}

        self._build()

    # ----------------------------------------------------------------
    # Build
    # ----------------------------------------------------------------

    def _build(self):
        conn = sqlite3.connect(self._db_path)

        # 1. Catalogue
        cats = pd.read_sql(
            "SELECT fund_code, fund_name, fund_type, inception_date, "
            "termination_date, underlying_name FROM otf_fund_catalog",
            conn,
        )
        for _, r in cats.iterrows():
            fc = str(r["fund_code"]).strip().zfill(6)
            self._catalog[fc] = {
                "name": str(r.get("fund_name", "")),
                "ftype": str(r.get("fund_type", "")),
                "inception": pd.Timestamp(r["inception_date"]) if pd.notna(r.get("inception_date")) and r["inception_date"] else pd.NaT,
                "termination": pd.Timestamp(r["termination_date"]) if pd.notna(r.get("termination_date")) and r["termination_date"] else pd.NaT,
                "underlying": str(r.get("underlying_name", "")),
            }

        # 2. First NAV date per fund
        first_nav = pd.read_sql(
            "SELECT fund_code, MIN(nav_date) as first_date "
            "FROM otf_fund_nav GROUP BY fund_code",
            conn,
        )
        for _, r in first_nav.iterrows():
            fc = str(r["fund_code"]).strip().zfill(6)
            self._first_nav_date[fc] = pd.Timestamp(r["first_date"])

        # 3. Rules: shared ProductRuleBook (if provided) or CSV
        if self._rule_book is not None:
            for fc, rule in self._rule_book.rules.items():
                zc = fc.zfill(6)
                self._rule_map[zc] = {
                    "sub_fee": rule.subscription_fee_rate,
                    "source": "verified",
                    "rule_status": rule.rule_status,
                }
        else:
            try:
                rules_csv = pd.read_csv(self._rules_path, dtype={"fund_code": str})
                for _, r in rules_csv.iterrows():
                    fc = r["fund_code"].strip().zfill(6)
                    self._rule_map[fc] = {
                        "sub_fee": float(r["subscription_fee_rate"]),
                        "source": "verified",
                        "rule_status": str(r.get("rule_status", "ASSUMPTION")),
                    }
            except (FileNotFoundError, pd.errors.EmptyDataError):
                pass

        # 4. DB default rules (auto-generated, lower priority)
        if self._allow_auto_rules:
            try:
                rules_db = pd.read_sql(
                    "SELECT fund_code, subscription_fee_rate FROM otf_default_rules",
                    conn,
                )
                for _, r in rules_db.iterrows():
                    fc = str(r["fund_code"]).strip().zfill(6)
                    if fc not in self._rule_map:
                        self._rule_map[fc] = {
                            "sub_fee": float(r["subscription_fee_rate"]) if pd.notna(r["subscription_fee_rate"]) else 0.0,
                            "source": "auto",
                        }
            except Exception:
                pass

        conn.close()

        # 4.5 Load exposure mapping per P0-4
        try:
            exp_df = pd.read_csv(self._exposure_mapping_path, dtype={"fund_code": str})
            for _, r in exp_df.iterrows():
                fc = str(r["fund_code"]).strip().zfill(6)
                self._exposure_map[fc] = {
                    "asset_sleeve": str(r.get("asset_sleeve", "")).strip(),
                    "asset_class": str(r.get("asset_class", "")).strip(),
                    "underlying_id": str(r.get("underlying_id", "")).strip(),
                    "mapping_confidence": str(r.get("mapping_confidence", "LOW")).strip().upper(),
                    "review_status": str(r.get("review_status", "PENDING")).strip().upper(),
                }
        except (FileNotFoundError, pd.errors.EmptyDataError):
            pass

        # 5. Build base candidate list per sleeve, filtered by exposure mapping
        self._candidates = discover_candidates(
            self._db_path, self._min_history_days,
            self._catalog, self._first_nav_date,
            self._exposure_map,
        )

    def _get_nav_stats(self, fund_codes: list[str], date: pd.Timestamp) -> dict[str, dict]:
        """Batch query NAV statistics as of a given date per P0-3.

        Returns dict[fund_code] -> {
            'nav_observations': int,  # COUNT(nav_date <= date)
            'last_nav_date': Timestamp | None,  # MAX(nav_date WHERE nav_date <= date)
            'max_gap_days': int,  # largest gap between consecutive NAV dates
        }
        """
        if not fund_codes:
            return {}

        result: dict[str, dict] = {fc: {"nav_observations": 0, "last_nav_date": None, "max_gap_days": 0} for fc in fund_codes}

        conn = sqlite3.connect(self._db_path)
        try:
            # Batch query: COUNT and MAX(nav_date) per fund up to selection date
            placeholders = ",".join("?" for _ in fund_codes)
            stats_sql = f"""
                SELECT fund_code,
                       COUNT(*) as nav_count,
                       MAX(nav_date) as last_nav
                FROM otf_fund_nav
                WHERE fund_code IN ({placeholders}) AND nav_date <= ?
                GROUP BY fund_code
            """
            date_str = date.strftime("%Y-%m-%d") if isinstance(date, pd.Timestamp) else str(date)
            stats_df = pd.read_sql(stats_sql, conn, params=fund_codes + [date_str])

            for _, row in stats_df.iterrows():
                fc = str(row["fund_code"]).strip().zfill(6)
                result[fc]["nav_observations"] = int(row["nav_count"])
                if pd.notna(row["last_nav"]):
                    result[fc]["last_nav_date"] = pd.Timestamp(row["last_nav"])

            # For max gap, query recent NAV dates for funds with sufficient observations
            # This is expensive, so only do it for funds that passed the COUNT check
            for fc in fund_codes:
                if result[fc]["nav_observations"] < self._min_nav_observations:
                    continue

                nav_dates_sql = f"""
                    SELECT nav_date FROM otf_fund_nav
                    WHERE fund_code = ? AND nav_date <= ?
                    ORDER BY nav_date DESC LIMIT 200
                """
                nav_dates_df = pd.read_sql(nav_dates_sql, conn, params=[fc.zfill(6), date_str])
                if len(nav_dates_df) >= 2:
                    dates_sorted = sorted(nav_dates_df["nav_date"].apply(pd.Timestamp))
                    gaps = [(dates_sorted[i] - dates_sorted[i-1]).days for i in range(1, len(dates_sorted))]
                    result[fc]["max_gap_days"] = max(gaps) if gaps else 0

        finally:
            conn.close()

        return result

    # ----------------------------------------------------------------
    # Selection
    # ----------------------------------------------------------------

    def select(
        self, sleeve: str, top_n: int = 2, date: str | None = None
    ) -> list[str]:
        """Return top-N eligible fund codes as of *date*.

        Raises ValueError if date is None (formal research mode).
        """
        if date is None:
            raise ValueError(
                "ProductSelector.select() requires a date parameter in "
                "formal research mode."
            )
        audit = self.select_with_audit(sleeve, top_n, date)
        return audit.selected

    def select_with_audit(
        self, sleeve: str, top_n: int, date: str | pd.Timestamp
    ) -> SelectionAudit:
        """Full audit version: returns SelectionAudit with candidate details."""
        dt = pd.Timestamp(date)
        audit = SelectionAudit(sleeve=sleeve, date=str(dt.date()), top_n=top_n)

        base_candidates = self._candidates.get(sleeve, [])
        if not base_candidates:
            return audit

        # Batch NAV statistics for all candidates (P0-3 optimization)
        zipped_codes = [c.zfill(6) for c in base_candidates]
        nav_stats = self._get_nav_stats(zipped_codes, dt)

        eligible: list[SelectionCandidate] = []

        for code in base_candidates:
            zc = code.zfill(6)
            cat = self._catalog.get(zc, {})
            rule = self._rule_map.get(zc, {})
            stats = nav_stats.get(zc, {"nav_observations": 0, "last_nav_date": None, "max_gap_days": 0})

            cand = SelectionCandidate(
                fund_code=code,
                sleeve=sleeve,
                sub_fee=rule.get("sub_fee", 1.0),
                nav_observations=stats["nav_observations"],
                nav_days_available=0,
                last_nav_date=stats["last_nav_date"],
                max_gap_days=stats["max_gap_days"],
                rule_source=rule.get("source", "missing"),
            )

            # --- checks ---

            # (a) has a rule (verified or auto)
            if not rule:
                cand.ineligible = True
                cand.ineligible_reason = "no_rule"
                audit.candidates.append(cand)
                continue

            # (a1) rule status must be allowed for this experiment mode
            rule_status = rule.get("rule_status", "ASSUMPTION")
            allowed_statuses = self._rule_book.allowed_rule_statuses if self._rule_book else {
                ProductRuleBook.OFFICIAL_VERIFIED,
                ProductRuleBook.DISTRIBUTOR_VERIFIED,
            }
            if rule_status not in allowed_statuses:
                cand.ineligible = True
                cand.ineligible_reason = f"rule_status_not_allowed:{rule_status}"
                audit.candidates.append(cand)
                continue

            # (a2) exposure mapping must be approved per P0-4
            exp_entry = self._exposure_map.get(zc)
            if exp_entry is None:
                cand.ineligible = True
                cand.ineligible_reason = "no_exposure_mapping"
                audit.candidates.append(cand)
                continue
            if (
                exp_entry.get("mapping_confidence", "").upper() != "HIGH"
                or exp_entry.get("review_status", "").upper() != "APPROVED"
            ):
                cand.ineligible = True
                cand.ineligible_reason = (
                    f"exposure_mapping_not_approved: "
                    f"confidence={exp_entry.get('mapping_confidence')}, "
                    f"status={exp_entry.get('review_status')}"
                )
                audit.candidates.append(cand)
                continue

            # (b) inception before date
            inception = cat.get("inception")
            if pd.notna(inception) and inception > dt:
                cand.ineligible = True
                cand.ineligible_reason = (
                    f"not_yet_incepted: inception={inception.date()}"
                )
                audit.candidates.append(cand)
                continue

            # (c) not terminated before date
            termination = cat.get("termination")
            if pd.notna(termination) and termination <= dt:
                cand.ineligible = True
                cand.ineligible_reason = (
                    f"already_terminated: termination={termination.date()}"
                )
                audit.candidates.append(cand)
                continue

            # (d) first NAV date exists and is before date
            first_nav = self._first_nav_date.get(zc)
            if first_nav is None or pd.isna(first_nav):
                cand.ineligible = True
                cand.ineligible_reason = "no_nav_data"
                audit.candidates.append(cand)
                continue
            if first_nav > dt:
                cand.ineligible = True
                cand.ineligible_reason = (
                    f"nav_not_yet_available: first_nav={first_nav.date()}"
                )
                audit.candidates.append(cand)
                continue

            # (e) minimum history length (calendar days)
            nav_days = (dt - first_nav).days
            cand.nav_days_available = nav_days
            if nav_days < self._min_history_days:
                cand.ineligible = True
                cand.ineligible_reason = (
                    f"insufficient_history: "
                    f"first_nav={first_nav.date()}, "
                    f"available={nav_days}d < min={self._min_history_days}d"
                )
                audit.candidates.append(cand)
                continue

            # (e1) minimum NAV observations COUNT per P0-3
            nav_obs = stats["nav_observations"]
            if nav_obs < self._min_nav_observations:
                cand.ineligible = True
                cand.ineligible_reason = (
                    f"insufficient_nav_observations: "
                    f"count={nav_obs} < min={self._min_nav_observations}"
                )
                audit.candidates.append(cand)
                continue

            # (e2) last NAV cannot be too stale per P0-3
            last_nav = stats["last_nav_date"]
            if last_nav is None:
                cand.ineligible = True
                cand.ineligible_reason = "no_nav_before_date"
                audit.candidates.append(cand)
                continue
            last_gap = (dt - last_nav).days
            if last_gap > self._max_last_nav_gap_days:
                cand.ineligible = True
                cand.ineligible_reason = (
                    f"stale_nav: last_nav={last_nav.date()}, "
                    f"gap={last_gap}d > max={self._max_last_nav_gap_days}d"
                )
                audit.candidates.append(cand)
                continue

            # (e3) consecutive NAV gap check per P0-3
            if cand.max_gap_days > self._max_consecutive_gap_days:
                cand.ineligible = True
                cand.ineligible_reason = (
                    f"excessive_nav_gap: max_gap={cand.max_gap_days}d "
                    f"> max={self._max_consecutive_gap_days}d"
                )
                audit.candidates.append(cand)
                continue

            # Passed all checks
            eligible.append(cand)
            audit.candidates.append(cand)

        # --- Dedup: same-family same-index ---
        # Group by stripped base name, keep lowest fee per group
        dedup_groups: dict[str, list[SelectionCandidate]] = {}
        for c in eligible:
            cat = self._catalog.get(c.fund_code.zfill(6), {})
            base = _base_fund_name(cat.get("name", "")) or c.fund_code
            dedup_groups.setdefault(base, []).append(c)

        final: list[SelectionCandidate] = []
        for base, group in dedup_groups.items():
            # Sort by fee asc, then nav_observations desc (P0-3), then nav_days desc
            group.sort(key=lambda x: (x.sub_fee, -x.nav_observations, -x.nav_days_available))
            final.append(group[0])

        # --- Rank: sub_fee asc, nav_observations desc (P0-3), nav_days desc ---
        final.sort(key=lambda x: (x.sub_fee, -x.nav_observations, -x.nav_days_available))
        selected = [c.fund_code for c in final[:top_n]]
        audit.selected = selected

        return audit

    def get_sleeve_candidates(self, sleeve: str) -> list[str]:
        return list(self._candidates.get(sleeve, []))


# ---------------------------------------------------------------------------
# Discover candidates from DB (name / type matching)
# ---------------------------------------------------------------------------

def discover_candidates(
    db_path: str,
    min_history_days: int,
    catalog: dict[str, dict],
    first_nav_date: dict[str, pd.Timestamp],
    exposure_mapping: dict[str, dict] | None = None,
) -> dict[str, list[str]]:
    """Build sleeve-to-candidate mapping from catalog + first NAV dates.

    If exposure_mapping is provided, only funds with mapping_confidence=HIGH
    and review_status=APPROVED are included per P0-4. Auto-matched funds
    without approved mapping are excluded from formal experiments.
    """
    conn = sqlite3.connect(db_path)

    # Funds with at least min_history days of NAV
    nav = pd.read_sql(
        "SELECT fund_code, COUNT(*) as cnt FROM otf_fund_nav GROUP BY fund_code",
        conn,
    )
    nav_codes = set(
        nav["fund_code"].astype(str).str.strip().str.zfill(6).values
    )
    conn.close()

    def _is_approved_exposure(fc: str) -> bool:
        """Check if fund has approved exposure mapping per P0-4."""
        if exposure_mapping is None:
            return True  # No mapping enforced if not loaded
        entry = exposure_mapping.get(fc.zfill(6))
        if entry is None:
            return False
        return (
            entry.get("mapping_confidence", "").upper() == "HIGH"
            and entry.get("review_status", "").upper() == "APPROVED"
        )

    result: dict[str, list[str]] = {}
    for sleeve in ALL_SLEEVES:
        result[sleeve] = []
        for c in SLEEVE_CANDIDATES[sleeve]:
            zc = c.zfill(6)
            if zc in first_nav_date and _is_approved_exposure(zc):
                result[sleeve].append(c)

    # Auto-match remaining catalog funds (only if they have approved mapping)
    for fc, info in catalog.items():
        if fc not in first_nav_date:
            continue
        if not _is_approved_exposure(fc):
            continue
        fname = info.get("name", "")
        ftype = info.get("ftype", "")
        sleeve = _match_sleeve_by_name(fname)
        if sleeve is None:
            sleeve = _match_sleeve_by_type(ftype)
        if sleeve and sleeve in result:
            if fc not in result[sleeve]:
                result[sleeve].append(fc)

    return result
