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


class OrderSide(str, Enum):
    SUBSCRIBE = "subscribe"
    REDEEM = "redeem"


class OrderStatus(str, Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    SETTLED = "settled"
    CANCELLED = "cancelled"


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
    ):
        self.db_path = db_path
        self.confirmation_days_subscribe = confirmation_days_subscribe
        self.confirmation_days_redeem = confirmation_days_redeem
        self.settlement_days_redeem = settlement_days_redeem
        self.subscription_fee_rate = subscription_fee_rate
        self.redemption_fee_rate = redemption_fee_rate
        self.initial_cash = initial_cash
        self.cash_daily_return = cash_daily_return
        self.reinvest_distributions = bool(reinvest_distributions)
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
            self._nav_df = pd.read_sql_query(
                "SELECT fund_code, nav_date, unit_nav, cumulative_nav, "
                "daily_growth_pct, distribution_per_share, "
                "share_adjustment_factor, total_return_factor "
                "FROM otf_fund_nav",
                conn,
            )
            self._catalog_df = pd.read_sql_query(
                "SELECT fund_code, share_class, fund_name, asset_class, "
                "benchmark, inception_date, termination_date "
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

    def get_confirmation_days(
        self, fund_code: str, side: OrderSide
    ) -> int:
        """Get confirmation days for a specific fund and order side.

        QDII funds use longer confirmation periods by default.
        """
        is_qdii = self.is_qdii(fund_code)
        if side == OrderSide.SUBSCRIBE:
            base = self.confirmation_days_subscribe
            return base + 2 if is_qdii else base
        else:
            base = self.confirmation_days_redeem
            return base + 2 if is_qdii else base

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
    ) -> OTFOrder | None:
        """Submit a new subscription or redemption order.

        Returns None if the order cannot be executed (insufficient cash
        for subscription, insufficient shares for redemption).
        """
        submit_idx = self._date_to_index.get(submit_date)
        if submit_idx is None:
            return None

        nav_at_submit = self.get_nav(fund_code, submit_idx)
        if nav_at_submit is None or nav_at_submit <= 0:
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
            fee = requested_amount * self.subscription_fee_rate
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
            shares_to_redeem = order.requested_amount / (
                order.nav_at_submit or confirm_nav
            )
            for idx in range(submit_idx + 1, confirm_idx + 1):
                shares_to_redeem *= self.get_share_adjustment(
                    order.fund_code, idx
                )
            order.shares_confirmed = shares_to_redeem
            fee = shares_to_redeem * confirm_nav * self.redemption_fee_rate
            order.fee_paid = fee
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
        reserved_redemptions: dict[str, float] = {}
        pending_orders: list[OTFOrder] = []
        all_orders: list[OTFOrder] = []
        rows: list[dict[str, Any]] = []
        last_rebalance_idx: int | None = None
        prior_end_equity = float(self.initial_cash)

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
                        self.settle_order(order, full_idx)
                        pending_orders.remove(order)
                    else:
                        shares = order.shares_confirmed or 0.0
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
                        receivable_cash += order.cash_released

                if order.status == OrderStatus.CONFIRMED:
                    confirm_idx = self._date_to_index[order.confirmation_date]
                    if full_idx - confirm_idx >= self.settlement_days_redeem:
                        cash_flow = self.settle_order(order, full_idx)
                        receivable_cash -= order.cash_released
                        available_cash += cash_flow
                        pending_orders.remove(order)

            current_equity = self._portfolio_value(
                available_cash, frozen_cash, receivable_cash, positions, full_idx
            )
            if current_equity <= 0:
                raise RuntimeError("OTF_ACCOUNT_EQUITY_NON_POSITIVE")

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
                signal_date = date
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
                    if diff >= -1.0:
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
                        signal_date=date,
                        submit_date=date,
                        requested_amount=redemption_amount,
                        available_cash=available_cash,
                        current_positions={fund_code: available_shares},
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
                    if diff <= 1.0:
                        continue
                    principal = min(
                        diff,
                        available_cash / (1.0 + self.subscription_fee_rate),
                    )
                    if principal <= 1.0:
                        continue
                    order = self.submit_order(
                        order_id=f"ord-{uuid.uuid4().hex[:8]}",
                        side=OrderSide.SUBSCRIBE,
                        fund_code=fund_code,
                        signal_date=date,
                        submit_date=date,
                        requested_amount=principal,
                        available_cash=available_cash,
                        current_positions=positions,
                    )
                    if order is not None:
                        available_cash -= order.cash_frozen
                        frozen_cash += order.cash_frozen
                        pending_orders.append(order)
                        all_orders.append(order)

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
        n_orders = len(all_orders)
        n_sub = sum(1 for o in all_orders if o.side == OrderSide.SUBSCRIBE)
        n_red = sum(1 for o in all_orders if o.side == OrderSide.REDEEM)
        print(
            f"  Orders: {n_orders} total "
            f"({n_sub} subscribe, {n_red} redeem)"
        )
        return result

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
