"""Build the audited CN execution calendar snapshot."""

from __future__ import annotations

import json

from otf_rotation.execution_calendar import build_cn_execution_calendar


if __name__ == "__main__":
    calendar = build_cn_execution_calendar()
    print(json.dumps(calendar.facts(), ensure_ascii=False, indent=2))

