"""Product-level trading rules for OTC fund execution simulation."""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd


DEFAULT_RULES_PATH = Path("config/otf_product_rules.csv")
DEFAULT_FEE_TIERS_PATH = Path("config/otf_redemption_fee_tiers.csv")
DEFAULT_SUBSCRIPTION_FEE_TIERS_PATH = Path("config/otf_subscription_fee_tiers.csv")
DEFAULT_EVENTS_PATH = Path("config/otf_trading_events.csv")


def _optional_float(value: object) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    return float(value)


def _bool(value: object, default: bool = True) -> bool:
    if value is None or str(value).strip() == "":
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def normalize_fund_code(value: object) -> str:
    code = str(value).strip()
    return code.zfill(6) if code.isdigit() else code


@dataclass(frozen=True)
class RedemptionFeeTier:
    minimum_holding_days: int
    maximum_holding_days_exclusive: int | None
    fee_rate: float

    def matches(self, holding_days: int) -> bool:
        return holding_days >= self.minimum_holding_days and (
            self.maximum_holding_days_exclusive is None
            or holding_days < self.maximum_holding_days_exclusive
        )


@dataclass(frozen=True)
class SubscriptionFeeTier:
    minimum_amount: float
    maximum_amount_exclusive: float | None
    fee_rate: float | None = None
    fixed_fee: float | None = None

    def matches(self, amount: float) -> bool:
        return amount >= self.minimum_amount and (
            self.maximum_amount_exclusive is None
            or amount < self.maximum_amount_exclusive
        )

    def fee_amount(self, amount: float) -> float:
        if self.fixed_fee is not None:
            return self.fixed_fee
        return amount * float(self.fee_rate or 0.0)


@dataclass(frozen=True)
class FundTradingRule:
    fund_code: str
    subscription_fee_rate: float
    subscription_confirmation_days: int
    redemption_confirmation_days: int
    redemption_settlement_days: int
    minimum_holding_calendar_days: int = 0
    maximum_subscription_amount_per_order: float | None = None
    subscription_open: bool = True
    redemption_open: bool = True
    rule_status: str = "ASSUMPTION"
    official_source: str = ""


@dataclass(frozen=True)
class TradingRestrictionEvent:
    fund_code: str
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    subscription_open: bool
    redemption_open: bool
    maximum_subscription_amount_per_order: float | None
    reason: str
    source: str

    def active_on(self, date: pd.Timestamp) -> bool:
        return self.start_date <= date <= self.end_date


class ProductRuleBook:
    def __init__(
        self,
        rules: dict[str, FundTradingRule] | None = None,
        fee_tiers: dict[str, list[RedemptionFeeTier]] | None = None,
        subscription_fee_tiers: dict[str, list[SubscriptionFeeTier]] | None = None,
        events: dict[str, list[TradingRestrictionEvent]] | None = None,
    ):
        self.rules = rules or {}
        self.fee_tiers = fee_tiers or {}
        self.subscription_fee_tiers = subscription_fee_tiers or {}
        self.events = events or {}

    @classmethod
    def from_csv(
        cls,
        rules_path: str | Path = DEFAULT_RULES_PATH,
        fee_tiers_path: str | Path = DEFAULT_FEE_TIERS_PATH,
        subscription_fee_tiers_path: str | Path = DEFAULT_SUBSCRIPTION_FEE_TIERS_PATH,
        events_path: str | Path = DEFAULT_EVENTS_PATH,
    ) -> "ProductRuleBook":
        rules: dict[str, FundTradingRule] = {}
        with Path(rules_path).open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                code = normalize_fund_code(row["fund_code"])
                rules[code] = FundTradingRule(
                    fund_code=code,
                    subscription_fee_rate=float(row["subscription_fee_rate"]),
                    subscription_confirmation_days=int(row["subscription_confirmation_days"]),
                    redemption_confirmation_days=int(row["redemption_confirmation_days"]),
                    redemption_settlement_days=int(row["redemption_settlement_days"]),
                    minimum_holding_calendar_days=int(row.get("minimum_holding_calendar_days") or 0),
                    maximum_subscription_amount_per_order=_optional_float(
                        row.get("maximum_subscription_amount_per_order")
                    ),
                    subscription_open=_bool(row.get("subscription_open")),
                    redemption_open=_bool(row.get("redemption_open")),
                    rule_status=str(row.get("rule_status", "ASSUMPTION")),
                    official_source=str(row.get("official_source", "")),
                )
        tiers: dict[str, list[RedemptionFeeTier]] = {}
        with Path(fee_tiers_path).open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                code = normalize_fund_code(row["fund_code"])
                tiers.setdefault(code, []).append(
                    RedemptionFeeTier(
                        minimum_holding_days=int(row["minimum_holding_days"]),
                        maximum_holding_days_exclusive=(
                            int(row["maximum_holding_days_exclusive"])
                            if str(row.get("maximum_holding_days_exclusive", "")).strip()
                            else None
                        ),
                        fee_rate=float(row["fee_rate"]),
                    )
                )
        for code, code_tiers in tiers.items():
            code_tiers.sort(key=lambda tier: tier.minimum_holding_days)
            if code_tiers[0].minimum_holding_days != 0:
                raise ValueError(f"FEE_TIERS_MUST_START_AT_ZERO:{code}")
            for previous, current in zip(code_tiers, code_tiers[1:]):
                if previous.maximum_holding_days_exclusive != current.minimum_holding_days:
                    raise ValueError(f"FEE_TIER_GAP_OR_OVERLAP:{code}")
        subscription_tiers: dict[str, list[SubscriptionFeeTier]] = {}
        subscription_path = Path(subscription_fee_tiers_path)
        if subscription_path.exists():
            with subscription_path.open(
                "r", encoding="utf-8-sig", newline=""
            ) as handle:
                for row in csv.DictReader(handle):
                    code = normalize_fund_code(row["fund_code"])
                    subscription_tiers.setdefault(code, []).append(
                        SubscriptionFeeTier(
                            minimum_amount=float(row["minimum_amount"]),
                            maximum_amount_exclusive=_optional_float(
                                row.get("maximum_amount_exclusive")
                            ),
                            fee_rate=_optional_float(row.get("fee_rate")),
                            fixed_fee=_optional_float(row.get("fixed_fee")),
                        )
                    )
        for code, code_tiers in subscription_tiers.items():
            code_tiers.sort(key=lambda tier: tier.minimum_amount)
            if code_tiers[0].minimum_amount != 0:
                raise ValueError(f"SUBSCRIPTION_TIERS_MUST_START_AT_ZERO:{code}")
            for previous, current in zip(code_tiers, code_tiers[1:]):
                if previous.maximum_amount_exclusive != current.minimum_amount:
                    raise ValueError(f"SUBSCRIPTION_TIER_GAP_OR_OVERLAP:{code}")
        events: dict[str, list[TradingRestrictionEvent]] = {}
        event_path = Path(events_path)
        if event_path.exists():
            with event_path.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    if not str(row.get("fund_code", "")).strip():
                        continue
                    code = normalize_fund_code(row["fund_code"])
                    events.setdefault(code, []).append(
                        TradingRestrictionEvent(
                            fund_code=code,
                            start_date=pd.Timestamp(row["start_date"]),
                            end_date=pd.Timestamp(row["end_date"]),
                            subscription_open=_bool(row.get("subscription_open")),
                            redemption_open=_bool(row.get("redemption_open")),
                            maximum_subscription_amount_per_order=_optional_float(
                                row.get("maximum_subscription_amount_per_order")
                            ),
                            reason=str(row.get("reason", "")),
                            source=str(row.get("source", "")),
                        )
                    )
        return cls(rules, tiers, subscription_tiers, events)

    def rule_for(self, fund_code: str) -> FundTradingRule | None:
        return self.rules.get(normalize_fund_code(fund_code))

    def fee_rate(self, fund_code: str, holding_days: int, fallback: float) -> float:
        tiers = self.fee_tiers.get(normalize_fund_code(fund_code), [])
        for tier in tiers:
            if tier.matches(max(0, int(holding_days))):
                return tier.fee_rate
        return fallback

    def subscription_fee_amount(
        self, fund_code: str, amount: float, fallback_rate: float
    ) -> float:
        for tier in self.subscription_fee_tiers.get(
            normalize_fund_code(fund_code), []
        ):
            if tier.matches(amount):
                return tier.fee_amount(amount)
        rule = self.rule_for(fund_code)
        rate = rule.subscription_fee_rate if rule else fallback_rate
        return amount * rate

    def order_constraints(
        self, fund_code: str, date: pd.Timestamp
    ) -> tuple[bool, bool, float | None, str]:
        rule = self.rule_for(fund_code)
        subscription_open = rule.subscription_open if rule else True
        redemption_open = rule.redemption_open if rule else True
        limit = rule.maximum_subscription_amount_per_order if rule else None
        reasons: list[str] = []
        for event in self.events.get(normalize_fund_code(fund_code), []):
            if event.active_on(pd.Timestamp(date)):
                subscription_open = event.subscription_open
                redemption_open = event.redemption_open
                if event.maximum_subscription_amount_per_order is not None:
                    limit = event.maximum_subscription_amount_per_order
                reasons.append(event.reason)
        return subscription_open, redemption_open, limit, ";".join(reasons)

    def coverage(self, fund_codes: list[str]) -> dict[str, object]:
        codes = {normalize_fund_code(code) for code in fund_codes}
        covered = codes & set(self.rules)
        verified = {
            code for code in covered
            if self.rules[code].rule_status == "OFFICIAL_VERIFIED"
        }
        externally_verified = {
            code for code in covered
            if self.rules[code].rule_status in {
                "OFFICIAL_VERIFIED", "DISTRIBUTOR_VERIFIED"
            }
        }
        return {
            "fund_count": len(codes),
            "rule_covered_count": len(covered),
            "official_verified_count": len(verified),
            "externally_verified_count": len(externally_verified),
            "rule_coverage_ratio": len(covered) / len(codes) if codes else 1.0,
            "official_verified_ratio": len(verified) / len(codes) if codes else 1.0,
            "externally_verified_ratio": (
                len(externally_verified) / len(codes) if codes else 1.0
            ),
            "missing_codes": sorted(codes - covered),
        }

    def scaled_fees(self, multiplier: float) -> "ProductRuleBook":
        if multiplier < 0:
            raise ValueError("fee multiplier cannot be negative")
        return ProductRuleBook(
            rules={
                code: replace(
                    rule,
                    subscription_fee_rate=rule.subscription_fee_rate * multiplier,
                )
                for code, rule in self.rules.items()
            },
            fee_tiers={
                code: [replace(tier, fee_rate=tier.fee_rate * multiplier) for tier in tiers]
                for code, tiers in self.fee_tiers.items()
            },
            subscription_fee_tiers={
                code: [
                    replace(
                        tier,
                        fee_rate=(
                            tier.fee_rate * multiplier
                            if tier.fee_rate is not None else None
                        ),
                        fixed_fee=(
                            tier.fixed_fee * multiplier
                            if tier.fixed_fee is not None else None
                        ),
                    )
                    for tier in tiers
                ]
                for code, tiers in self.subscription_fee_tiers.items()
            },
            events=self.events,
        )

    def with_subscription_discount(self, multiplier: float) -> "ProductRuleBook":
        """Discount percentage subscription fees; fixed per-order fees remain."""
        if multiplier < 0:
            raise ValueError("subscription discount multiplier cannot be negative")
        return ProductRuleBook(
            rules={
                code: replace(
                    rule,
                    subscription_fee_rate=rule.subscription_fee_rate * multiplier,
                )
                for code, rule in self.rules.items()
            },
            fee_tiers=self.fee_tiers,
            subscription_fee_tiers={
                code: [
                    replace(
                        tier,
                        fee_rate=(
                            tier.fee_rate * multiplier
                            if tier.fee_rate is not None else None
                        ),
                    )
                    for tier in tiers
                ]
                for code, tiers in self.subscription_fee_tiers.items()
            },
            events=self.events,
        )
