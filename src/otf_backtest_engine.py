"""NAV-based backtesting engine for over-the-counter (OTC) funds.

Unlike the ETF backtest engine which uses daily close prices with a simple
signal-to-return lag, this engine models the full order lifecycle of OTC fund
transactions:

    signal_date -> order_submit_date -> NAV confirmation -> share confirmation
    -> cash settlement

Key differences from ETF mode:

1. Orders execute at the NAV of the confirmation date, not the signal date.
2. Subscription orders freeze cash until shares are confirmed.
3. Redemption proceeds arrive with a settlement delay after confirmation.
4. Fees are asymmetric: subscription fee vs redemption fee.
5. Confirmation rules differ by fund type (T+1 domestic, T+3 QDII).
"""

from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from otf_trading_rules import ProductRuleBook


class OrderSide(str, Enum):
    SUBSCRIBE = "subscribe"
    REDEEM = "redeem"


class OrderStatus(str, Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    SETTLED = "settled"
    CANCELLED = "cancelled"


@dataclass
class PositionLot:
    lot_id: str
    acquired_date: pd.Timestamp
    shares: float
    reserved_shares: float = 0.0


@dataclass
class LotAllocation:
    lot_id: str
    acquired_date: pd.Timestamp
    shares_at_submit: float
    shares_confirmed: float = 0.0
    fee_rate: float = 0.0


@dataclass
class OTFOrder:
    """Single OTC fund order with full lifecycle tracking."""

    order_id: str
    side: OrderSide
    fund_code: str
    signal_date: pd.Timestamp
    submit_date: pd.Timestamp
    requested_amount: float
    nav_at_submit: float | None = None
    confirmation_date: pd.Timestamp | None = None
    confirmed_nav: float | None = None
    shares_confirmed: float | None = None
    fee_paid: float = 0.0
    cash_frozen: float = 0.0
    redemption_arrival_date: pd.Timestamp | None = None
    cash_released: float = 0.0
    status: OrderStatus = OrderStatus.PENDING
    lot_allocations: list[LotAllocation] = field(default_factory=list)
    effective_fee_rate: float = 0.0


class OTFBacktestEngine:
    """Daily accounting engine for OTC fund NAV-based backtesting.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database with ``otf_fund_nav`` and
        ``otf_fund_catalog`` tables.
    confirmation_days_subscribe : int
        Trading days between order submission and NAV confirmation for
        subscription orders (default 1 for domestic funds).
    confirmation_days_redeem : int
        Trading days between order submission and NAV confirmation for
        redemption orders (default 1 for domestic funds).
    settlement_days_redeem : int
        Additional trading days after redemption confirmation before cash
        arrives in the account (default 1).
    subscription_fee_rate : float
        One-time fee charged on the subscription amount (default 0.1%).
    redemption_fee_rate : float
        One-time fee charged on the redemption amount (default 0.15%).
    initial_cash : float
        Starting cash balance (default 1_000_000).
    cash_daily_return : float
        Daily return earned on idle cash (default 0).
    minimum_trade_ratio : float
        Skip target adjustments smaller than this fraction of current account
        equity.  The default 0 preserves legacy behavior.
    """

    def __init__(
        self,
        db_path: str = "data/processed/otf.sqlite",
        *,
        confirmation_days_subscribe: int = 1,
        confirmation_days_redeem: int = 1,
        settlement_days_redeem: int = 1,
        subscription_fee_rate: float = 0.001,
        redemption_fee_rate: float = 0.0015,
        initial_cash: float = 1_000_000.0,
        cash_daily_return: float = 0.0,
        reinvest_distributions: bool = True,
        minimum_trade_ratio: float = 0.0,
        fund_subscription_fee_rates: dict[str, float] | None = None,
        fund_redemption_fee_rates: dict[str, float] | None = None,
        product_rule_book: ProductRuleBook | None = None,
        strict_product_rules: bool = False,
    ):
        self.db_path = db_path
        self.confirmation_days_subscribe = confirmation_days_subscribe
        self.confirmation_days_redeem = confirmation_days_redeem
        self.settlement_days_redeem = settlement_days_redeem
        self.subscription_fee_rate = subscription_fee_rate
        self.redemption_fee_rate = redemption_fee_rate
        self.initial_cash = initial_cash
        self.cash_daily_return = cash_daily_return
        self.minimum_trade_ratio = float(minimum_trade_ratio)
        if not 0.0 <= self.minimum_trade_ratio < 1.0:
            raise ValueError("minimum_trade_ratio must be in [0, 1)")
        self.fund_subscription_fee_rates = {
            str(code): float(rate)
            for code, rate in (fund_subscription_fee_rates or {}).items()
        }
        self.fund_redemption_fee_rates = {
            str(code): float(rate)
            for code, rate in (fund_redemption_fee_rates or {}).items()
        }
        if any(
            rate < 0
            for rate in (
                *self.fund_subscription_fee_rates.values(),
                *self.fund_redemption_fee_rates.values(),
            )
        ):
            raise ValueError("fund fee rates cannot be negative")
        self.reinvest_distributions = bool(reinvest_distributions)
        self.product_rule_book = product_rule_book
        self.strict_product_rules = bool(strict_product_rules)
        self._rejection_log: list[dict[str, Any]] = []
        if not self.reinvest_distributions:
            raise ValueError(
                "Only total-return reinvestment mode is currently supported"
            )

        self._nav_df: pd.DataFrame | None = None
        self._catalog_df: pd.DataFrame | None = None
        self._trading_dates: pd.DatetimeIndex | None = None
        self._date_to_index: dict[pd.Timestamp, int] | None = None
        self._nav_lookup: dict[tuple[str, int], float] | None = None
        self._valuation_nav_lookup: dict[tuple[str, int], float] | None = None
        self._distribution_lookup: dict[tuple[str, int], float] | None = None
        self._share_adjustment_lookup: dict[tuple[str, int], float] | None = None
        self._fund_qdii_map: dict[str, bool] = {}
        self.available_fund_codes: list[str] = []
        self.data_quality: dict[str, Any] = {}

        self._load_data()

    def _load_data(self) -> None:
        """Load NAV and catalog data from SQLite."""
        with sqlite3.connect(self.db_path) as conn:
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }

        required = {"otf_fund_nav", "otf_fund_catalog"}
        missing = required - tables
        if missing:
            raise RuntimeError(f"OTF_DB_MISSING_TABLES: {sorted(missing)}")

        with sqlite3.connect(self.db_path) as conn:
            catalog_columns = {
                row[1]
                for row in conn.execute(
                    "PRAGMA table_info(otf_fund_catalog)"
                ).fetchall()
            }
            self._nav_df = pd.read_sql_query(
                "SELECT fund_code, nav_date, unit_nav, cumulative_nav, "
                "daily_growth_pct, distribution_per_share, "
                "share_adjustment_factor, total_return_factor "
                "FROM otf_fund_nav",
                conn,
            )
            benchmark_expr = (
                "benchmark"
                if "benchmark" in catalog_columns
                else "underlying_name AS benchmark"
                if "underlying_name" in catalog_columns
                else "'' AS benchmark"
            )
            self._catalog_df = pd.read_sql_query(
                "SELECT fund_code, share_class, fund_name, asset_class, "
                f"{benchmark_expr}, inception_date, termination_date "
                "FROM otf_fund_catalog",
                conn,
            )

        self._nav_df["nav_date"] = pd.to_datetime(self._nav_df["nav_date"])
        self._nav_df["unit_nav"] = pd.to_numeric(
            self._nav_df["unit_nav"], errors="coerce"
        )
        self._nav_df["cumulative_nav"] = pd.to_numeric(
            self._nav_df["cumulative_nav"], errors="coerce"
        )
        for column in (
            "daily_growth_pct", "distribution_per_share",
            "share_adjustment_factor", "total_return_factor"
        ):
            self._nav_df[column] = pd.to_numeric(
                self._nav_df[column], errors="coerce"
            )
        duplicate_keys = int(
            self._nav_df.duplicated(["fund_code", "nav_date"]).sum()
        )
        if duplicate_keys:
            raise RuntimeError(f"OTF_DUPLICATE_NAV_KEYS: {duplicate_keys}")
        required_numeric = [
            "unit_nav", "cumulative_nav", "daily_growth_pct",
            "share_adjustment_factor", "total_return_factor",
        ]
        null_counts = self._nav_df[required_numeric].isna().sum()
        if int(null_counts.sum()) > 0:
            raise RuntimeError(
                f"OTF_REQUIRED_NAV_NULLS: {null_counts[null_counts > 0].to_dict()}"
            )
        if (self._nav_df["unit_nav"] <= 0).any():
            raise RuntimeError("OTF_NON_POSITIVE_UNIT_NAV")
        if (self._nav_df["share_adjustment_factor"] <= 0).any():
            raise RuntimeError("OTF_NON_POSITIVE_SHARE_ADJUSTMENT")
        if (self._nav_df["total_return_factor"] <= 0).any():
            raise RuntimeError("OTF_NON_POSITIVE_TOTAL_RETURN_FACTOR")
        self._nav_df = self._nav_df.loc[self._nav_df["unit_nav"] > 0].sort_values(
            ["fund_code", "nav_date"]
        ).drop_duplicates(["fund_code", "nav_date"], keep="last")

        previous_nav = self._nav_df.groupby("fund_code")["unit_nav"].shift(1)
        reconstructed_return = (
            self._nav_df["unit_nav"]
            / previous_nav
            * self._nav_df["share_adjustment_factor"]
            - 1.0
        )
        published_return = self._nav_df["daily_growth_pct"] / 100.0
        reconciliation = (reconstructed_return - published_return).abs().dropna()
        max_reconciliation_error = (
            float(reconciliation.max()) if not reconciliation.empty else 0.0
        )
        if max_reconciliation_error > 1e-8:
            raise RuntimeError(
                "OTF_TOTAL_RETURN_RECONCILIATION_FAILED: "
                f"max_error={max_reconciliation_error:.12g}"
            )
        self.data_quality = {
            "row_count": int(len(self._nav_df)),
            "duplicate_keys": duplicate_keys,
            "max_total_return_reconciliation_error": max_reconciliation_error,
            "price_mode": "published_daily_growth_total_return_reinvested",
        }

        self._catalog_df["inception_date"] = pd.to_datetime(
            self._catalog_df["inception_date"], errors="coerce"
        )
        self._catalog_df["termination_date"] = pd.to_datetime(
            self._catalog_df["termination_date"], errors="coerce"
        )

        all_dates = self._nav_df["nav_date"].drop_duplicates().sort_values()
        self._trading_dates = pd.DatetimeIndex(all_dates)
        self._date_to_index = {
            date: idx for idx, date in enumerate(self._trading_dates)
        }

        nav_pivot = self._nav_df.pivot_table(
            index="fund_code", columns="nav_date", values="unit_nav"
        )
        self._nav_lookup = {}
        self._valuation_nav_lookup = {}
        self._distribution_lookup = {}
        self._share_adjustment_lookup = {}
        for fund_code in nav_pivot.index:
            series = nav_pivot.loc[fund_code]
            valuation_series = series.reindex(self._trading_dates).ffill()
            for idx, date in enumerate(self._trading_dates):
                if date in series.index and not np.isnan(series[date]):
                    self._nav_lookup[(fund_code, idx)] = float(series[date])
                valuation_nav = valuation_series.iloc[idx]
                if not np.isnan(valuation_nav):
                    self._valuation_nav_lookup[(fund_code, idx)] = float(
                        valuation_nav
                    )

        for row in self._nav_df.itertuples(index=False):
            date_idx = self._date_to_index.get(row.nav_date)
            if date_idx is not None and pd.notna(row.distribution_per_share):
                self._distribution_lookup[(row.fund_code, date_idx)] = max(
                    0.0, float(row.distribution_per_share)
                )
            if date_idx is not None and pd.notna(row.share_adjustment_factor):
                self._share_adjustment_lookup[(row.fund_code, date_idx)] = float(
                    row.share_adjustment_factor
                )

        qdii_keywords = ["QDII", "qdii", "纳斯达克", "标普", "海外"]
        for _, row in self._catalog_df.iterrows():
            name = str(row.get("fund_name", ""))
            bench = str(row.get("benchmark", ""))
            asset = str(row.get("asset_class", ""))
            combined = f"{name}{bench}{asset}"
            self._fund_qdii_map[row["fund_code"]] = any(
                kw in combined for kw in qdii_keywords
            )

        available_funds = set(self._nav_df["fund_code"].unique())
        catalog_funds = set(self._catalog_df["fund_code"].unique())
        self.available_fund_codes = sorted(available_funds & catalog_funds)
        self.data_quality["product_rule_coverage"] = (
            self.product_rule_book.coverage(self.available_fund_codes)
            if self.product_rule_book is not None
            else None
        )

        print(
            f"Loaded OTF data: {len(self.available_fund_codes)} funds, "
            f"{len(self._trading_dates)} trading days, "
            f"period={self._trading_dates[0].date()}~{self._trading_dates[-1].date()}"
        )

    def get_nav(self, fund_code: str, date_idx: int) -> float | None:
        """Get NAV for a fund at a given trading day index."""
        return self._nav_lookup.get((fund_code, date_idx))

    def get_valuation_nav(self, fund_code: str, date_idx: int) -> float | None:
        """Get the last published NAV at or before the valuation date."""
        return self._valuation_nav_lookup.get((fund_code, date_idx))

    def get_distribution(self, fund_code: str, date_idx: int) -> float:
        """Return cash distribution per share published on an exact NAV date."""
        return self._distribution_lookup.get((fund_code, date_idx), 0.0)

    def get_share_adjustment(self, fund_code: str, date_idx: int) -> float:
        """Synthetic share multiplier that reproduces published total return."""
        return self._share_adjustment_lookup.get((fund_code, date_idx), 1.0)

    def is_qdii(self, fund_code: str) -> bool:
        """Check if a fund is QDII (requires longer confirmation)."""
        return self._fund_qdii_map.get(fund_code, False)

    def get_subscription_fee_rate(self, fund_code: str) -> float:
        if fund_code in self.fund_subscription_fee_rates:
            return self.fund_subscription_fee_rates[fund_code]
        rule = (
            self.product_rule_book.rule_for(fund_code)
            if self.product_rule_book is not None else None
        )
        return rule.subscription_fee_rate if rule else self.subscription_fee_rate

    def get_subscription_fee_amount(self, fund_code: str, amount: float) -> float:
        if fund_code in self.fund_subscription_fee_rates:
            return amount * self.fund_subscription_fee_rates[fund_code]
        if self.product_rule_book is not None:
            return self.product_rule_book.subscription_fee_amount(
                fund_code, amount, self.subscription_fee_rate
            )
        return amount * self.subscription_fee_rate

    def maximum_affordable_subscription(
        self, fund_code: str, desired_amount: float, available_cash: float
    ) -> float:
        desired = max(0.0, min(float(desired_amount), float(available_cash)))
        if desired + self.get_subscription_fee_amount(fund_code, desired) <= available_cash:
            return desired
        low, high = 0.0, desired
        for _ in range(60):
            midpoint = (low + high) / 2.0
            if (
                midpoint + self.get_subscription_fee_amount(fund_code, midpoint)
                <= available_cash
            ):
                low = midpoint
            else:
                high = midpoint
        return low

    def get_redemption_fee_rate(
        self, fund_code: str, holding_days: int | None = None
    ) -> float:
        if fund_code in self.fund_redemption_fee_rates:
            return self.fund_redemption_fee_rates[fund_code]
        if self.product_rule_book is not None and holding_days is not None:
            return self.product_rule_book.fee_rate(
                fund_code, holding_days, self.redemption_fee_rate
            )
        return self.redemption_fee_rate

    def get_confirmation_days(
        self, fund_code: str, side: OrderSide
    ) -> int:
        """Get confirmation days for a specific fund and order side.

        QDII funds use longer confirmation periods by default.
        """
        is_qdii = self.is_qdii(fund_code)
        rule = (
            self.product_rule_book.rule_for(fund_code)
            if self.product_rule_book is not None else None
        )
        if rule is not None:
            return (
                rule.subscription_confirmation_days
                if side == OrderSide.SUBSCRIBE
                else rule.redemption_confirmation_days
            )
        if side == OrderSide.SUBSCRIBE:
            base = self.confirmation_days_subscribe
            return base + 2 if is_qdii else base
        base = self.confirmation_days_redeem
        return base + 2 if is_qdii else base

    def get_redemption_settlement_days(self, fund_code: str) -> int:
        rule = (
            self.product_rule_book.rule_for(fund_code)
            if self.product_rule_book is not None else None
        )
        return rule.redemption_settlement_days if rule else self.settlement_days_redeem

    def get_minimum_holding_days(self, fund_code: str) -> int:
        rule = (
            self.product_rule_book.rule_for(fund_code)
            if self.product_rule_book is not None else None
        )
        return rule.minimum_holding_calendar_days if rule else 0

    def get_order_constraints(
        self, fund_code: str, submit_date: pd.Timestamp
    ) -> tuple[bool, bool, float | None, str]:
        if self.product_rule_book is None:
            return True, True, None, ""
        return self.product_rule_book.order_constraints(fund_code, submit_date)

    def _record_rejection(
        self, fund_code: str, side: OrderSide, date: pd.Timestamp, reason: str
    ) -> None:
        self._rejection_log.append(
            {
                "fund_code": fund_code,
                "side": side.value,
                "date": pd.Timestamp(date),
                "reason": reason,
            }
        )

    def _allocate_redemption_lots(
        self,
        fund_code: str,
        shares_required: float,
        submit_date: pd.Timestamp,
        position_lots: dict[str, list[PositionLot]],
    ) -> list[LotAllocation] | None:
        minimum_days = self.get_minimum_holding_days(fund_code)
        remaining = shares_required
        allocations: list[LotAllocation] = []
        for lot in sorted(
            position_lots.get(fund_code, []), key=lambda item: item.acquired_date
        ):
            holding_days = (pd.Timestamp(submit_date) - lot.acquired_date).days
            if holding_days < minimum_days:
                continue
            available = max(0.0, lot.shares - lot.reserved_shares)
            take = min(available, remaining)
            if take > 1e-12:
                allocations.append(
                    LotAllocation(lot.lot_id, lot.acquired_date, take)
                )
                remaining -= take
            if remaining <= 1e-10:
                break
        if remaining > 1e-8:
            return None
        lots_by_id = {
            lot.lot_id: lot for lot in position_lots.get(fund_code, [])
        }
        for allocation in allocations:
            lots_by_id[allocation.lot_id].reserved_shares += allocation.shares_at_submit
        return allocations

    def submit_order(
        self,
        order_id: str,
        side: OrderSide,
        fund_code: str,
        signal_date: pd.Timestamp,
        submit_date: pd.Timestamp,
        requested_amount: float,
        available_cash: float,
        current_positions: dict[str, float],
        position_lots: dict[str, list[PositionLot]] | None = None,
    ) -> OTFOrder | None:
        """Submit a new subscription or redemption order.

        Returns None if the order cannot be executed (insufficient cash
        for subscription, insufficient shares for redemption).
        """
        submit_idx = self._date_to_index.get(submit_date)
        if submit_idx is None:
            return None
        if (
            self.strict_product_rules
            and (
                self.product_rule_book is None
                or self.product_rule_book.rule_for(fund_code) is None
            )
        ):
            raise RuntimeError(f"MISSING_PRODUCT_RULE:{fund_code}")

        nav_at_submit = self.get_nav(fund_code, submit_idx)
        if nav_at_submit is None or nav_at_submit <= 0:
            return None
        sub_open, red_open, subscription_limit, restriction_reason = (
            self.get_order_constraints(fund_code, submit_date)
        )
        if side == OrderSide.SUBSCRIBE and not sub_open:
            self._record_rejection(
                fund_code, side, submit_date,
                restriction_reason or "SUBSCRIPTION_CLOSED",
            )
            return None
        if side == OrderSide.REDEEM and not red_open:
            self._record_rejection(
                fund_code, side, submit_date,
                restriction_reason or "REDEMPTION_CLOSED",
            )
            return None
        if (
            side == OrderSide.SUBSCRIBE
            and subscription_limit is not None
            and requested_amount > subscription_limit + 1e-10
        ):
            self._record_rejection(
                fund_code, side, submit_date, "SUBSCRIPTION_LIMIT_EXCEEDED"
            )
            return None

        order = OTFOrder(
            order_id=order_id,
            side=side,
            fund_code=fund_code,
            signal_date=signal_date,
            submit_date=submit_date,
            requested_amount=requested_amount,
            nav_at_submit=nav_at_submit,
        )

        if side == OrderSide.SUBSCRIBE:
            fee = self.get_subscription_fee_amount(fund_code, requested_amount)
            total_cost = requested_amount + fee
            if total_cost > available_cash + 1e-10:
                return None
            order.cash_frozen = total_cost
            order.fee_paid = fee
        else:
            shares_to_redeem = requested_amount / nav_at_submit
            current_shares = current_positions.get(fund_code, 0.0)
            if shares_to_redeem > current_shares + 1e-10:
                return None
            if position_lots is not None:
                allocations = self._allocate_redemption_lots(
                    fund_code, shares_to_redeem, submit_date, position_lots
                )
                if allocations is None:
                    self._record_rejection(
                        fund_code, side, submit_date,
                        "INSUFFICIENT_MATURE_UNRESERVED_LOTS",
                    )
                    return None
                order.lot_allocations = allocations

        return order

    def confirm_order(self, order: OTFOrder, confirm_idx: int) -> bool:
        """Confirm a pending order at the NAV of the confirmation date.

        Returns True if the order was successfully confirmed.
        The key invariant: orders execute at the confirmation-date NAV,
        NOT the submit-date NAV. This prevents same-day NAV execution.
        """
        if order.status != OrderStatus.PENDING:
            return False

        submit_idx = self._date_to_index.get(order.submit_date)
        if submit_idx is None:
            return False
        earliest_confirm_idx = submit_idx + self.get_confirmation_days(
            order.fund_code, order.side
        )
        if confirm_idx < earliest_confirm_idx:
            return False

        confirm_nav = self.get_nav(order.fund_code, confirm_idx)
        if confirm_nav is None or confirm_nav <= 0:
            return False

        order.confirmation_date = self._trading_dates[confirm_idx]
        order.confirmed_nav = confirm_nav

        if order.side == OrderSide.SUBSCRIBE:
            # requested_amount is investment principal; the fee is additional
            # frozen cash.  Deducting it from principal as well would charge
            # the subscription fee twice.
            order.shares_confirmed = order.requested_amount / confirm_nav
        else:
            if order.lot_allocations:
                fee = 0.0
                shares_to_redeem = 0.0
                for allocation in order.lot_allocations:
                    adjusted_shares = allocation.shares_at_submit
                    for idx in range(submit_idx + 1, confirm_idx + 1):
                        adjusted_shares *= self.get_share_adjustment(
                            order.fund_code, idx
                        )
                    holding_days = (
                        self._trading_dates[confirm_idx] - allocation.acquired_date
                    ).days
                    allocation.shares_confirmed = adjusted_shares
                    allocation.fee_rate = self.get_redemption_fee_rate(
                        order.fund_code, holding_days
                    )
                    shares_to_redeem += adjusted_shares
                    fee += adjusted_shares * confirm_nav * allocation.fee_rate
            else:
                shares_to_redeem = order.requested_amount / (
                    order.nav_at_submit or confirm_nav
                )
                for idx in range(submit_idx + 1, confirm_idx + 1):
                    shares_to_redeem *= self.get_share_adjustment(
                        order.fund_code, idx
                    )
                fee = (
                    shares_to_redeem
                    * confirm_nav
                    * self.get_redemption_fee_rate(order.fund_code)
                )
            order.shares_confirmed = shares_to_redeem
            order.fee_paid = fee
            gross_proceeds = shares_to_redeem * confirm_nav
            order.effective_fee_rate = fee / gross_proceeds if gross_proceeds else 0.0
            order.cash_released = shares_to_redeem * confirm_nav - fee

        order.status = OrderStatus.CONFIRMED
        return True

    def settle_order(self, order: OTFOrder, settle_idx: int) -> float:
        """Settle a confirmed order and return cash flow impact.

        For subscription: cash was already frozen at submit time; settlement
        simply converts frozen cash into shares (no additional cash movement).
        For redemption: removes shares, adds net proceeds to available cash.
        Returns the net cash change for the account.
        """
        if order.status != OrderStatus.CONFIRMED:
            return 0.0

        if order.side == OrderSide.SUBSCRIBE:
            order.status = OrderStatus.SETTLED
            return 0.0
        else:
            # Redemption price is fixed at confirmation NAV.  Settlement lag
            # delays cash availability but must not add later NAV exposure.
            net_proceeds = order.cash_released
            order.redemption_arrival_date = self._trading_dates[settle_idx]
            order.status = OrderStatus.SETTLED
            return net_proceeds

    def run_backtest(
        self,
        target_weights: pd.DataFrame,
        start: str | None = None,
        end: str | None = None,
        rebalance_every: int = 5,
        signal_dates: dict[pd.Timestamp, pd.Timestamp] | None = None,
    ) -> pd.DataFrame:
        """Run backtest given daily target weights.

        Parameters
        ----------
        target_weights : pd.DataFrame
            DataFrame indexed by date with fund codes as columns and
            target portfolio weights as values. Weights must sum to <= 1.0.
            NaN means no signal for that day.
        start, end : str
            Date range for backtest (defaults to full available range).
        rebalance_every : int
            Minimum trading days between rebalances.
        signal_dates : dict[pd.Timestamp, pd.Timestamp], optional
            Optional mapping from order submission dates to the earlier dates
            when the source signal became observable.  This preserves the
            listed-ETF close -> next OTC submission audit trail without
            changing the fund confirmation rules.

        Returns
        -------
        pd.DataFrame
            Daily account state with return, exposure, and position info.
        """
        if rebalance_every < 1:
            raise ValueError("rebalance_every must be at least 1")
        target_weights = target_weights.copy()
        target_weights.index = pd.to_datetime(target_weights.index)
        if target_weights.index.has_duplicates:
            raise ValueError("target_weights contains duplicate dates")
        unknown = set(target_weights.columns) - set(self.available_fund_codes)
        if unknown:
            raise ValueError(f"Unknown fund codes in target weights: {sorted(unknown)}")
        numeric_targets = target_weights.apply(pd.to_numeric, errors="coerce")
        if (numeric_targets.fillna(0.0) < -1e-12).any().any():
            raise ValueError("Negative target weights are not allowed")
        if (numeric_targets.fillna(0.0).sum(axis=1) > 1.0 + 1e-8).any():
            raise ValueError("Target weights exceed 100%")
        target_weights = numeric_targets

        if start is None:
            start = self._trading_dates[0].strftime("%Y-%m-%d")
        if end is None:
            end = self._trading_dates[-1].strftime("%Y-%m-%d")
        test_dates = self._trading_dates[
            (self._trading_dates >= pd.Timestamp(start))
            & (self._trading_dates <= pd.Timestamp(end))
        ]
        if test_dates.empty:
            raise ValueError("No trading dates in requested period")

        available_cash = float(self.initial_cash)
        frozen_cash = 0.0
        receivable_cash = 0.0
        positions: dict[str, float] = {}
        position_lots: dict[str, list[PositionLot]] = {}
        reserved_redemptions: dict[str, float] = {}
        pending_orders: list[OTFOrder] = []
        all_orders: list[OTFOrder] = []
        rows: list[dict[str, Any]] = []
        last_rebalance_idx: int | None = None
        prior_end_equity = float(self.initial_cash)
        deferred_subscriptions: dict[str, float] = {}
        deferred_signal_date: pd.Timestamp | None = None
        self._rejection_log = []

        for date in test_dates:
            full_idx = self._date_to_index[date]
            signal_date = pd.NaT
            total_fees = 0.0

            # Apply the source-implied share multiplier.  This handles cash
            # distributions, reinvestment and share conversions while making
            # position value follow the published total daily growth exactly.
            for fund_code in list(positions):
                adjustment = self.get_share_adjustment(fund_code, full_idx)
                positions[fund_code] *= adjustment
                for lot in position_lots.get(fund_code, []):
                    lot.shares *= adjustment
                    lot.reserved_shares *= adjustment
                if fund_code in reserved_redemptions:
                    reserved_redemptions[fund_code] *= adjustment
            available_cash *= 1.0 + self.cash_daily_return

            # Confirm and settle prior orders before accepting a new rebalance.
            for order in list(pending_orders):
                if order.status == OrderStatus.PENDING:
                    if not self.confirm_order(order, full_idx):
                        continue
                    total_fees += order.fee_paid
                    if order.side == OrderSide.SUBSCRIBE:
                        frozen_cash -= order.cash_frozen
                        positions[order.fund_code] = (
                            positions.get(order.fund_code, 0.0)
                            + (order.shares_confirmed or 0.0)
                        )
                        position_lots.setdefault(order.fund_code, []).append(
                            PositionLot(
                                lot_id=f"lot-{uuid.uuid4().hex[:10]}",
                                acquired_date=order.confirmation_date,
                                shares=order.shares_confirmed or 0.0,
                            )
                        )
                        self.settle_order(order, full_idx)
                        pending_orders.remove(order)
                    else:
                        shares = order.shares_confirmed or 0.0
                        if order.lot_allocations:
                            lots_by_id = {
                                lot.lot_id: lot
                                for lot in position_lots.get(order.fund_code, [])
                            }
                            for allocation in order.lot_allocations:
                                lot = lots_by_id[allocation.lot_id]
                                lot.shares -= allocation.shares_confirmed
                                lot.reserved_shares = max(
                                    0.0,
                                    lot.reserved_shares
                                    - allocation.shares_confirmed,
                                )
                            position_lots[order.fund_code] = [
                                lot for lot in position_lots[order.fund_code]
                                if lot.shares > 1e-12
                            ]
                        positions[order.fund_code] = (
                            positions.get(order.fund_code, 0.0) - shares
                        )
                        reserved_redemptions[order.fund_code] = max(
                            0.0,
                            reserved_redemptions.get(order.fund_code, 0.0)
                            - shares,
                        )
                        if positions[order.fund_code] < 1e-12:
                            positions.pop(order.fund_code, None)
                            position_lots.pop(order.fund_code, None)
                        receivable_cash += order.cash_released

                if order.status == OrderStatus.CONFIRMED:
                    confirm_idx = self._date_to_index[order.confirmation_date]
                    if full_idx - confirm_idx >= self.get_redemption_settlement_days(
                        order.fund_code
                    ):
                        cash_flow = self.settle_order(order, full_idx)
                        receivable_cash -= order.cash_released
                        available_cash += cash_flow
                        pending_orders.remove(order)

            current_equity = self._portfolio_value(
                available_cash, frozen_cash, receivable_cash, positions, full_idx
            )
            if current_equity <= 0:
                raise RuntimeError("OTF_ACCOUNT_EQUITY_NON_POSITIVE")
            minimum_trade_amount = max(
                1.0, current_equity * self.minimum_trade_ratio
            )

            # A fund switch cannot spend redemption proceeds before they
            # arrive.  Complete the subscription leg of that same rebalance
            # after settlement instead of leaving the cash idle until the
            # next strategy signal.
            if not pending_orders and deferred_subscriptions:
                for fund_code in sorted(list(deferred_subscriptions)):
                    remaining = deferred_subscriptions[fund_code]
                    if remaining <= minimum_trade_amount:
                        deferred_subscriptions.pop(fund_code, None)
                        continue
                    principal = self.maximum_affordable_subscription(
                        fund_code, remaining, available_cash
                    )
                    _, _, subscription_limit, _ = self.get_order_constraints(
                        fund_code, date
                    )
                    if subscription_limit is not None:
                        principal = min(principal, subscription_limit)
                    if principal <= minimum_trade_amount:
                        continue
                    order = self.submit_order(
                        order_id=f"ord-{uuid.uuid4().hex[:8]}",
                        side=OrderSide.SUBSCRIBE,
                        fund_code=fund_code,
                        signal_date=deferred_signal_date or date,
                        submit_date=date,
                        requested_amount=principal,
                        available_cash=available_cash,
                        current_positions=positions,
                        position_lots=position_lots,
                    )
                    if order is None:
                        continue
                    available_cash -= order.cash_frozen
                    frozen_cash += order.cash_frozen
                    pending_orders.append(order)
                    all_orders.append(order)
                    remaining -= principal
                    if remaining <= minimum_trade_amount:
                        deferred_subscriptions.pop(fund_code, None)
                    else:
                        deferred_subscriptions[fund_code] = remaining
                if not deferred_subscriptions:
                    deferred_signal_date = None

            should_rebalance = (
                not pending_orders
                and (
                    last_rebalance_idx is None
                    or full_idx - last_rebalance_idx >= rebalance_every
                )
                and date in target_weights.index
            )
            if should_rebalance:
                weights_row = target_weights.loc[date]
                target = {
                    str(code): float(weight)
                    for code, weight in weights_row.items()
                    if pd.notna(weight) and float(weight) > 1e-12
                }
                signal_date = (
                    signal_dates.get(date, date)
                    if signal_dates is not None
                    else date
                )
                deferred_subscriptions = {}
                deferred_signal_date = signal_date
                last_rebalance_idx = full_idx
                target_amounts = {
                    code: weight * current_equity for code, weight in target.items()
                }

                # Submit redemptions first.  Funds omitted from the target must
                # be redeemed; iterating only target keys silently kept them.
                all_codes = set(positions) | set(target_amounts)
                for fund_code in sorted(all_codes):
                    exact_nav = self.get_nav(fund_code, full_idx)
                    valuation_nav = self.get_valuation_nav(fund_code, full_idx)
                    if exact_nav is None or valuation_nav is None:
                        continue
                    current_amount = positions.get(fund_code, 0.0) * valuation_nav
                    diff = target_amounts.get(fund_code, 0.0) - current_amount
                    if diff >= -minimum_trade_amount:
                        continue
                    available_shares = max(
                        0.0,
                        positions.get(fund_code, 0.0)
                        - reserved_redemptions.get(fund_code, 0.0),
                    )
                    redemption_amount = min(abs(diff), available_shares * exact_nav)
                    order = self.submit_order(
                        order_id=f"ord-{uuid.uuid4().hex[:8]}",
                        side=OrderSide.REDEEM,
                        fund_code=fund_code,
                        signal_date=signal_date,
                        submit_date=date,
                        requested_amount=redemption_amount,
                        available_cash=available_cash,
                        current_positions={fund_code: available_shares},
                        position_lots=position_lots,
                    )
                    if order is not None:
                        shares_reserved = redemption_amount / exact_nav
                        reserved_redemptions[fund_code] = (
                            reserved_redemptions.get(fund_code, 0.0)
                            + shares_reserved
                        )
                        pending_orders.append(order)
                        all_orders.append(order)

                # Subscriptions are funded from currently available cash only;
                # expected redemption proceeds cannot be spent before arrival.
                for fund_code in sorted(target_amounts):
                    valuation_nav = self.get_valuation_nav(fund_code, full_idx)
                    if valuation_nav is None:
                        continue
                    current_amount = positions.get(fund_code, 0.0) * valuation_nav
                    diff = target_amounts[fund_code] - current_amount
                    if diff <= minimum_trade_amount:
                        continue
                    principal = self.maximum_affordable_subscription(
                        fund_code, diff, available_cash
                    )
                    _, _, subscription_limit, _ = self.get_order_constraints(
                        fund_code, date
                    )
                    if subscription_limit is not None:
                        principal = min(principal, subscription_limit)
                    if principal <= minimum_trade_amount:
                        deferred_subscriptions[fund_code] = diff
                        continue
                    order = self.submit_order(
                        order_id=f"ord-{uuid.uuid4().hex[:8]}",
                        side=OrderSide.SUBSCRIBE,
                        fund_code=fund_code,
                        signal_date=signal_date,
                        submit_date=date,
                        requested_amount=principal,
                        available_cash=available_cash,
                        current_positions=positions,
                        position_lots=position_lots,
                    )
                    if order is not None:
                        available_cash -= order.cash_frozen
                        frozen_cash += order.cash_frozen
                        pending_orders.append(order)
                        all_orders.append(order)
                        remaining = diff - principal
                        if remaining > minimum_trade_amount:
                            deferred_subscriptions[fund_code] = remaining
                    else:
                        deferred_subscriptions[fund_code] = diff

                if not deferred_subscriptions:
                    deferred_signal_date = None

            end_equity = self._portfolio_value(
                available_cash, frozen_cash, receivable_cash, positions, full_idx
            )
            daily_return = (
                end_equity / prior_end_equity - 1.0
                if prior_end_equity > 0 else 0.0
            )
            cost_return = total_fees / prior_end_equity if prior_end_equity > 0 else 0.0
            gross_return = daily_return + cost_return
            exposure_value = sum(
                shares * (self.get_valuation_nav(code, full_idx) or 0.0)
                for code, shares in positions.items()
            )
            exposure = exposure_value / end_equity if end_equity > 0 else 0.0
            rows.append(
                self._make_row(
                    date, signal_date, available_cash, frozen_cash,
                    receivable_cash, positions, daily_return, gross_return,
                    cost_return, end_equity, len(pending_orders), exposure,
                )
            )
            prior_end_equity = end_equity

        result = pd.DataFrame(rows)
        self.last_orders = all_orders
        self.last_rejections = pd.DataFrame(self._rejection_log)
        self.last_position_lots = position_lots
        n_orders = len(all_orders)
        n_sub = sum(1 for o in all_orders if o.side == OrderSide.SUBSCRIBE)
        n_red = sum(1 for o in all_orders if o.side == OrderSide.REDEEM)
        print(
            f"  Orders: {n_orders} total "
            f"({n_sub} subscribe, {n_red} redeem)"
        )
        return result

    def order_audit_frame(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for order in getattr(self, "last_orders", []):
            holding_days = [
                (order.confirmation_date - allocation.acquired_date).days
                for allocation in order.lot_allocations
                if order.confirmation_date is not None
            ]
            rows.append(
                {
                    "order_id": order.order_id,
                    "fund_code": order.fund_code,
                    "side": order.side.value,
                    "status": order.status.value,
                    "signal_date": order.signal_date,
                    "submit_date": order.submit_date,
                    "confirmation_date": order.confirmation_date,
                    "redemption_arrival_date": order.redemption_arrival_date,
                    "requested_amount": order.requested_amount,
                    "cash_frozen": order.cash_frozen,
                    "confirmed_nav": order.confirmed_nav,
                    "shares_confirmed": order.shares_confirmed,
                    "fee_paid": order.fee_paid,
                    "effective_fee_rate": order.effective_fee_rate,
                    "fifo_lot_count": len(order.lot_allocations),
                    "minimum_holding_days": min(holding_days) if holding_days else None,
                    "maximum_holding_days": max(holding_days) if holding_days else None,
                }
            )
        return pd.DataFrame(rows)

    def _portfolio_value(
        self,
        available_cash: float,
        frozen_cash: float,
        receivable_cash: float,
        positions: dict[str, float],
        date_idx: int,
    ) -> float:
        position_value = sum(
            shares * (self.get_valuation_nav(code, date_idx) or 0.0)
            for code, shares in positions.items()
        )
        return available_cash + frozen_cash + receivable_cash + position_value

    @staticmethod
    def _make_row(
        date: pd.Timestamp,
        signal_date: pd.Timestamp | None,
        available_cash: float,
        frozen_cash: float,
        receivable_cash: float,
        positions: dict[str, float],
        daily_return: float,
        gross_return: float,
        transaction_cost: float,
        equity: float,
        pending_order_count: int,
        exposure: float = 0.0,
    ) -> dict[str, Any]:
        return {
            "date": date,
            "signal_date": signal_date if pd.notna(signal_date) else None,
            "available_cash": round(available_cash, 2),
            "frozen_cash": round(frozen_cash, 2),
            "receivable_cash": round(receivable_cash, 2),
            "equity": round(equity, 2),
            "daily_return": daily_return,
            "gross_return": gross_return,
            "transaction_cost": transaction_cost,
            "position_count": len(positions),
            "pending_order_count": pending_order_count,
            "exposure": exposure,
            "positions": str(sorted(positions.keys())),
        }

    def calculate_metrics(
        self,
        daily: pd.DataFrame,
        benchmark_returns: pd.Series | None = None,
    ) -> dict[str, float]:
        """Calculate performance metrics from daily backtest results."""
        returns = pd.Series(daily["daily_return"].values, dtype=float).fillna(0.0)
        if returns.empty:
            return {}

        wealth = (1.0 + returns).cumprod()
        total_return = wealth.iloc[-1] - 1.0
        n_days = len(returns)
        final_wealth = max(wealth.iloc[-1], 1e-10)
        ann_return = final_wealth ** (252.0 / n_days) - 1.0
        ann_std = returns.std(ddof=1) * np.sqrt(252) if n_days > 1 else 0.0
        sharpe = (
            returns.mean() / returns.std(ddof=1) * np.sqrt(252)
            if ann_std > 0 else 0.0
        )

        downside = returns[returns < 0]
        downside_std = (
            downside.std(ddof=1) * np.sqrt(252) if len(downside) > 1 else 0.0
        )
        sortino = (
            returns.mean() / downside.std(ddof=1) * np.sqrt(252)
            if downside_std > 0 else 0.0
        )

        drawdown = wealth / wealth.cummax() - 1.0
        max_dd = float(drawdown.min())

        in_drawdown = (drawdown < -1e-12).astype(int)
        longest_dd_days = 0
        if in_drawdown.any():
            groups = in_drawdown.ne(in_drawdown.shift(1)).cumsum()
            dd_lengths = in_drawdown.groupby(groups).sum()
            longest_dd_days = int(dd_lengths.max())

        metrics: dict[str, float] = {
            "CAGR%": ann_return * 100,
            "Total_Return%": total_return * 100,
            "Annualized_Volatility%": ann_std * 100,
            "Sharpe": float(sharpe),
            "Sortino": float(sortino),
            "Max_Drawdown%": max_dd * 100,
            "Calmar": float(ann_return / abs(max_dd)) if max_dd < 0 else 0.0,
            "Daily_Win_Rate%": float((returns > 0).mean() * 100),
            "Skewness": float(returns.skew()),
            "Kurtosis": float(returns.kurtosis()),
            "Longest_DD_Duration_days": longest_dd_days,
        }

        if n_days > 1:
            var_95 = float(np.percentile(returns, 5)) * 100
            cvar_95 = float(
                returns[returns <= np.percentile(returns, 5)].mean()
            ) * 100
            metrics["VaR_95%"] = var_95
            metrics["CVaR_95%"] = cvar_95

        if "transaction_cost" in daily.columns:
            tc_series = pd.Series(daily["transaction_cost"], dtype=float)
            metrics["Cumulative_Cost_Ratio%"] = float(tc_series.sum() * 100)

        if "gross_return" in daily.columns:
            gross_returns = pd.Series(
                daily["gross_return"].values, dtype=float
            ).fillna(0.0)
            gross_wealth = (1.0 + gross_returns).cumprod()
            gross_ann = gross_wealth.iloc[-1] ** (252.0 / n_days) - 1.0
            metrics["Gross_Total_Return%"] = float(
                (gross_wealth.iloc[-1] - 1.0) * 100
            )
            metrics["Gross_CAGR%"] = float(gross_ann * 100)
            metrics["Annualized_Cost_Drag%"] = float(
                (gross_ann - ann_return) * 100
            )
            metrics["Transaction_Cost_Drag%"] = float(
                (gross_wealth.iloc[-1] - wealth.iloc[-1]) * 100
            )
        elif "transaction_cost" in daily.columns:
            metrics["Transaction_Cost_Drag%"] = metrics[
                "Cumulative_Cost_Ratio%"
            ]

        if "exposure" in daily.columns:
            exp_series = pd.Series(daily["exposure"], dtype=float)
            metrics["Exposure%"] = float(exp_series.mean() * 100)

        return metrics


class OTFStrategySignal:
    """Generate target weights for OTC fund universe.

    This is a simplified strategy interface for the OTF engine.
    Strategies receive NAV-based returns and produce target weights.
    """

    def __init__(
        self,
        engine: OTFBacktestEngine,
        lookback_days: int = 20,
    ):
        self.engine = engine
        self.lookback_days = lookback_days
        self._returns_wide: pd.DataFrame | None = None

    def build_returns_wide(self) -> pd.DataFrame:
        """Build wide return DataFrame from NAV data."""
        if self._returns_wide is not None:
            return self._returns_wide

        nav_pivot = self.engine._nav_df.pivot_table(
            index="nav_date", columns="fund_code", values="total_return_factor"
        ).sort_index()

        self._returns_wide = nav_pivot.ffill().pct_change(fill_method=None).fillna(0.0)
        return self._returns_wide

    def momentum_signal(
        self, date: pd.Timestamp, n_hold: int = 5
    ) -> dict[str, float]:
        """Cross-sectional momentum signal for OTC funds."""
        ret = self.build_returns_wide()
        if date not in ret.index:
            return {}
        idx = ret.index.get_loc(date)
        if idx < self.lookback_days:
            return {}

        window = ret.iloc[idx - self.lookback_days : idx]
        cum_ret = (1.0 + window).prod() - 1.0

        moments = cum_ret.dropna().sort_values(ascending=False)
        if len(moments) < n_hold:
            return {}

        top_n = moments.head(n_hold)
        weight = 1.0 / len(top_n)
        return {sym: weight for sym in top_n.index}

    def equal_weight_signal(self) -> dict[str, float]:
        """Equal weight across all available OTC funds."""
        n = len(self.engine.available_fund_codes)
        if n == 0:
            return {}
        weight = 1.0 / n
        return {f: weight for f in self.engine.available_fund_codes}

    def generate_target_weights(
        self,
        signal_func,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        """Generate a DataFrame of daily target weights.

        Parameters
        ----------
        signal_func : callable
            Function that takes a date and returns a dict of fund_code -> weight.
        start, end : str
            Date range for signal generation.

        Returns
        -------
        pd.DataFrame
            Index=date, columns=fund_codes, values=target weights.
        """
        if start is None:
            start = self.engine._trading_dates[0].strftime("%Y-%m-%d")
        if end is None:
            end = self.engine._trading_dates[-1].strftime("%Y-%m-%d")

        dates = self.engine._trading_dates[
            (self.engine._trading_dates >= pd.Timestamp(start))
            & (self.engine._trading_dates <= pd.Timestamp(end))
        ]

        rows: list[dict[str, Any]] = []
        for date in dates:
            weights = signal_func(date)
            row = {"date": date}
            for fund_code in self.engine.available_fund_codes:
                row[fund_code] = weights.get(fund_code, 0.0)
            rows.append(row)

        df = pd.DataFrame(rows).set_index("date")
        return df
