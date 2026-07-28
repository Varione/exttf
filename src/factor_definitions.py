"""Extended factor library with 120+ factors across diverse categories."""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Callable


@dataclass
class FactorDef:
    name: str
    category: str
    description: str
    compute: Callable[[pd.DataFrame], pd.Series]


def _safe_div(a, b, default=0):
    return (a / b.replace(0, np.nan)).fillna(default)


# ===== 1. MOMENTUM (12 factors) =====

def _momentum(df, window):
    return df["close"].pct_change(window).fillna(0)


def _weighted_momentum(df, w1, w2, w3):
    m1 = df["close"].pct_change(w1).fillna(0)
    m2 = df["close"].pct_change(w2).fillna(0)
    m3 = df["close"].pct_change(w3).fillna(0)
    return (m1 + m2 + m3) / 3


def _skew_momentum(df, window):
    ret = df["close"].pct_change().fillna(0)
    pos_ret = ret.clip(lower=0).rolling(window, min_periods=5).sum()
    neg_ret = ret.clip(upper=0).rolling(window, min_periods=5).sum()
    return (pos_ret - abs(neg_ret)) / (abs(neg_ret) + 1e-8)


def _win_rate(df, window):
    ret = df["close"].pct_change().fillna(0)
    wins = (ret > 0).astype(int).rolling(window, min_periods=5).sum()
    total = pd.Series(np.ones(len(df)), index=df.index).rolling(window, min_periods=5).sum()
    return _safe_div(wins, total, 0.5) - 0.5


def _max_drawdown_window(df, window):
    peak = df["close"].rolling(window, min_periods=5).max()
    dd = (df["close"] / peak) - 1
    return dd.fillna(0)


def _recovery_speed(df, window):
    low_n = df["close"].rolling(window, min_periods=5).min()
    ret_from_low = (df["close"] / low_n) - 1
    return ret_from_low.fillna(0)


# ===== 2. TREND (12 factors) =====

def _ma_distance(df, window):
    ma = df["close"].rolling(window, min_periods=window).mean()
    return _safe_div(df["close"] - ma, ma, 0)


def _ema_distance(df, span):
    ema = df["close"].ewm(span=span, adjust=False).mean()
    return _safe_div(df["close"] - ema, ema, 0)


def _linear_slope(df, window):
    def _slope(x):
        if len(x) < 5:
            return np.nan
        idx = np.arange(len(x), dtype=float)
        return np.polyfit(idx, x, 1)[0]
    return df["close"].rolling(window, min_periods=5).apply(_slope, raw=True).fillna(0)


def _linear_r_squared(df, window):
    def _r2(x):
        if len(x) < 5:
            return np.nan
        idx = np.arange(len(x), dtype=float)
        z = np.polyfit(idx, x, 1)
        p = np.poly1d(z)
        ss_res = np.sum((x - p(idx)) ** 2)
        ss_tot = np.sum((x - np.mean(x)) ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return df["close"].rolling(window, min_periods=5).apply(_r2, raw=True).fillna(0)


def _ma_cross(df, fast, slow):
    ma_f = df["close"].rolling(fast, min_periods=fast).mean()
    ma_s = df["close"].rolling(slow, min_periods=slow).mean()
    return _safe_div(ma_f - ma_s, ma_s, 0)


def _ma_stack_score(df, windows):
    score = pd.Series(0, index=df.index, dtype=float)
    close = df["close"]
    for w in windows:
        ma = close.rolling(w, min_periods=w).mean()
        score += (close > ma).astype(float)
    return (score / len(windows) - 0.5).fillna(0)


def _hull_ma_distance(df, period):
    half = period // 2
    ema1 = df["close"].ewm(span=half, adjust=False).mean()
    ema2 = df["close"].ewm(span=period, adjust=False).mean()
    hull_input = 2 * ema1 - ema2
    hull_ma = hull_input.ewm(span=int(np.sqrt(period)), adjust=False).mean()
    return _safe_div(df["close"] - hull_ma, hull_ma, 0)


# ===== 3. VOLATILITY (15 factors) =====

def _realized_vol(df, window):
    log_r = np.log(df["close"] / df["close"].shift(1))
    return log_r.rolling(window, min_periods=5).std().fillna(0)


def _true_range_vol(df, window):
    tr = _compute_tr(df)
    return tr.rolling(window, min_periods=5).std().fillna(0)


def _parkinson_vol(df, window):
    park_var = 0.25 * (np.log(df["high"] / df["low"])) ** 2
    return np.sqrt(park_var.rolling(window, min_periods=5).mean()).fillna(0)


def _garman_klass_vol(df, window):
    log_hl = np.log(df["high"] / df["low"])
    log_cc = np.log(df["close"] / df["close"].shift(1))
    gk_var = 0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_cc ** 2
    return np.sqrt(gk_var.clip(lower=0).rolling(window, min_periods=5).mean()).fillna(0)


def _rogers_satchev_vol(df, window):
    log_hl = np.log(df["high"] / df["low"])
    log_hpc = np.log(df["high"] / df["close"].shift(1))
    log_lpc = np.log(df["low"] / df["close"].shift(1))
    rs_var = log_hl ** 2 * (2 * np.log(2) - 1) + (log_hpc ** 2 - 2 * np.log(2) * log_lpc ** 2)
    return np.sqrt(rs_var.clip(lower=0).rolling(window, min_periods=5).mean()).fillna(0)


def _yin_vol(df, window):
    log_ho = np.log(df["high"] / df["open"])
    log_lo = np.log(df["low"] / df["open"])
    yin_var = log_ho ** 2 + log_lo ** 2
    return np.sqrt(yin_var.rolling(window, min_periods=5).mean()).fillna(0)


def _vol_ratio(df, short, long):
    log_r = np.log(df["close"] / df["close"].shift(1))
    vs = log_r.rolling(short, min_periods=3).std()
    vl = log_r.rolling(long, min_periods=5).std()
    return _safe_div(vs, vl, 1)


def _vol_expansion(df, window):
    log_r = np.log(df["close"] / df["close"].shift(1))
    vol = log_r.rolling(window, min_periods=5).std()
    vol_ma = vol.rolling(window * 2, min_periods=window).mean()
    return _safe_div(vol - vol_ma, vol_ma, 0)


def _vol_regime(df, window):
    log_r = np.log(df["close"] / df["close"].shift(1))
    vol = log_r.rolling(window, min_periods=5).std()
    vol_mean = vol.rolling(window * 4, min_periods=window).mean()
    vol_std = vol.rolling(window * 4, min_periods=window).std()
    return _safe_div(vol - vol_mean, vol_std, 0)


def _dual_theta(df, window):
    close = df["close"]
    open_p = df["open"]
    log_r = np.log(close / close.shift(1))
    theta_2 = (log_r ** 2).rolling(window, min_periods=5).mean()
    close_open = np.log(close / open_p)
    theta_1 = (close_open ** 2).rolling(window, min_periods=5).mean()
    return _safe_div(theta_2, theta_1, 1)


def _atr_ratio(df, window):
    return _safe_div(_compute_tr(df).rolling(window, min_periods=window).mean(), df["close"], 0)


# ===== 4. VOLUME / LIQUIDITY (15 factors) =====

def _volume_ma_ratio(df, window):
    vol_ma = df["volume"].rolling(window, min_periods=window).mean()
    return _safe_div(df["volume"], vol_ma, 1)


def _volume_trend(df, window):
    vol_ma = df["volume"].rolling(window, min_periods=window).mean()
    vol_ma2 = df["volume"].rolling(window * 2, min_periods=window).mean()
    return _safe_div(vol_ma - vol_ma2, vol_ma2, 0)


def _volume_acceleration(df, window):
    vol_ret = df["volume"].pct_change().fillna(0)
    return vol_ret.rolling(window, min_periods=5).mean().fillna(0)


def _amount_volume_ratio(df):
    amount_per_vol = _safe_div(df["amount"], df["volume"], 0)
    close = df["close"]
    return _safe_div(amount_per_vol - close, close, 0)


def _volume_price_correlation(df, window):
    ret = df["close"].pct_change().fillna(0)
    return df["volume"].rolling(window, min_periods=10).corr(ret).fillna(0)


def _accumulation_distribution(df):
    ad = _safe_div((df["close"] - df["low"]) - (df["high"] - df["close"]),
                   (df["high"] - df["low"]).replace(0, np.nan), 0) * df["volume"]
    ad_cum = ad.cumsum()
    return ad_cum.pct_change(20).fillna(0)


def _chaikin_money_flow(df, window):
    ad = _safe_div((df["close"] - df["low"]) - (df["high"] - df["close"]),
                   (df["high"] - df["low"]).replace(0, np.nan), 0) * df["volume"]
    ad_sum = ad.rolling(window, min_periods=window).sum()
    vol_sum = df["volume"].rolling(window, min_periods=window).sum()
    return _safe_div(ad_sum, vol_sum, 0)


def _ease_of_movement(df, window):
    price_change = (df["close"] - df["close"].shift(2)) / 2
    vol_amount = _safe_div(df["amount"], df["close"].replace(0, np.nan), 0)
    eom = _safe_div(price_change, vol_amount, 0)
    return eom.rolling(window, min_periods=5).sum().fillna(0)


def _on_balance_volume_mom(df, window):
    obv = (np.sign(df["close"].diff()) * df["volume"]).cumsum()
    return _safe_div(obv - obv.shift(window), obv.shift(window), 0)


def _volume_cluster(df, window):
    vol_ma = df["volume"].rolling(window, min_periods=window).mean()
    above = (df["volume"] > vol_ma).astype(int).rolling(window, min_periods=5).sum()
    total = pd.Series(np.ones(len(df)), index=df.index).rolling(window, min_periods=5).sum()
    return (_safe_div(above, total, 0.5) - 0.5).fillna(0)


def _noble_volume(df):
    ret = df["close"].pct_change().fillna(0)
    vol_ret = df["volume"].pct_change().fillna(0)
    return (ret * vol_ret).fillna(0)


# ===== 5. MEAN REVERSION (10 factors) =====

def _bollinger_position(df, window, std_mult):
    ma = df["close"].rolling(window, min_periods=window).mean()
    std = df["close"].rolling(window, min_periods=window).std()
    return _safe_div(df["close"] - ma, std * std_mult, 0)


def _keltner_position(df, window):
    ma = df["close"].rolling(window, min_periods=window).mean()
    atr = _compute_tr(df).rolling(window, min_periods=window).mean()
    return _safe_div(df["close"] - ma, atr, 0)


def _price_ema_ratio(df, fast, slow):
    ef = df["close"].ewm(span=fast, adjust=False).mean()
    es = df["close"].ewm(span=slow, adjust=False).mean()
    return _safe_div(ef - es, es, 0)


def _range_position(df, window):
    high_n = df["high"].rolling(window, min_periods=window).max()
    low_n = df["low"].rolling(window, min_periods=window).min()
    return _safe_div(df["close"] - low_n, high_n - low_n, 0.5)


def _ibias(df, window):
    ema = df["close"].ewm(span=window, adjust=False).mean()
    sma = df["close"].rolling(window, min_periods=window).mean()
    return _safe_div(ema - sma, sma, 0)


def _donchian_position(df, window):
    high_n = df["high"].rolling(window, min_periods=window).max()
    low_n = df["low"].rolling(window, min_periods=window).min()
    mid = (high_n + low_n) / 2
    half = (high_n - low_n) / 2
    return _safe_div(df["close"] - mid, half, 0.5)


def _mean_rev_speed(df, window):
    ma = df["close"].rolling(window, min_periods=window).mean()
    dist = _safe_div(df["close"] - ma, ma, 0)
    ret = df["close"].pct_change().fillna(0)
    return (-dist.shift(1) * ret).fillna(0)


# ===== 6. PRICE PATTERN (15 factors) =====

def _intraday_range(df):
    return _safe_div(df["high"] - df["low"], df["close"].shift(1), 0)


def _body_ratio(df):
    body = abs(df["close"] - df["open"])
    total = (df["high"] - df["low"]).replace(0, np.nan)
    return _safe_div(body, total, 0.5)


def _upper_shadow(df):
    body_top = np.maximum(df["open"], df["close"])
    total = (df["high"] - df["low"]).replace(0, np.nan)
    return _safe_div(df["high"] - body_top, total, 0)


def _lower_shadow(df):
    body_bottom = np.minimum(df["open"], df["close"])
    total = (df["high"] - df["low"]).replace(0, np.nan)
    return _safe_div(body_bottom - df["low"], total, 0)


def _close_position(df):
    total = (df["high"] - df["low"]).replace(0, np.nan)
    return _safe_div(df["close"] - df["low"], total, 0.5)


def _gap_ratio(df):
    prev = df["close"].shift(1)
    gap = (df["open"] / prev) - 1
    daily_ret = (df["close"] / prev) - 1
    return _safe_div(gap, daily_ret, 0)


def _gap_persistence(df, window):
    gaps = (df["open"] / df["close"].shift(1)) - 1
    filled = abs(gaps).rolling(window, min_periods=5).mean()
    return filled.fillna(0)


def _candle_engulfing(df):
    body_today = df["close"] - df["open"]
    body_yest = (df["close"].shift(1) - df["open"].shift(1))
    ratio = _safe_div(body_today, body_yest.replace(0, np.nan), 0)
    return ratio.fillna(0)


def _inside_bar(df):
    cond = ((df["high"] < df["high"].shift(1)) & (df["low"] > df["low"].shift(1))).astype(int)
    return cond


def _three_day_pattern(df):
    ret3 = df["close"].pct_change(3).fillna(0)
    ret1_sum = df["close"].pct_change().fillna(0).rolling(3, min_periods=3).sum()
    return (ret3 - ret1_sum).fillna(0)


def _open_close_asymmetry(df):
    up_days = (df["close"] > df["open"]).astype(int).rolling(20, min_periods=10).mean()
    down_days = (df["close"] < df["open"]).astype(int).rolling(20, min_periods=10).mean()
    return _safe_div(up_days - down_days, up_days + down_days, 0)


# ===== 7. RISK ADJUSTED (8 factors) =====

def _sharpness(df, window):
    ret = df["close"].pct_change().fillna(0)
    m = ret.rolling(window, min_periods=10).mean()
    s = ret.rolling(window, min_periods=10).std()
    return _safe_div(m, s, 0)


def _sortino_ratio(df, window):
    ret = df["close"].pct_change().fillna(0)
    m = ret.rolling(window, min_periods=10).mean()
    ds = ret.where(ret < 0).rolling(window, min_periods=5).std()
    return _safe_div(m, ds, 0)


def _omega_ratio(df, window, threshold=0):
    ret = df["close"].pct_change().fillna(0)
    gains = ret.clip(lower=threshold).rolling(window, min_periods=10).sum()
    losses = (-ret.clip(upper=threshold)).rolling(window, min_periods=10).sum()
    return _safe_div(gains, losses, 1)


def _max_drawdown_ratio(df, window):
    peak = df["close"].rolling(window, min_periods=5).max()
    dd = (df["close"] / peak) - 1
    ret_mean = df["close"].pct_change().fillna(0).rolling(window, min_periods=10).mean()
    max_dd = dd.rolling(window, min_periods=5).min().abs()
    return _safe_div(ret_mean, max_dd, 0)


def _up_capture(df, window):
    ret = df["close"].pct_change().fillna(0)
    gains = ret.clip(lower=0)
    threshold = gains.rolling(window, min_periods=10).quantile(0.75)
    tail_gains = gains.where(gains.ge(threshold), 0)
    total_gains = gains.rolling(window, min_periods=10).sum()
    captured = tail_gains.rolling(window, min_periods=10).sum()
    return _safe_div(captured, total_gains.replace(0, np.nan), 0)


def _down_capture(df, window):
    ret = df["close"].pct_change().fillna(0)
    losses = (-ret).clip(lower=0)
    threshold = losses.rolling(window, min_periods=10).quantile(0.75)
    tail_losses = losses.where(losses.ge(threshold), 0)
    total_losses = losses.rolling(window, min_periods=10).sum()
    captured = tail_losses.rolling(window, min_periods=10).sum()
    return _safe_div(captured, total_losses.replace(0, np.nan), 0)


def _tail_ratio(df, window):
    ret = df["close"].pct_change().fillna(0)
    skew = ret.rolling(window, min_periods=20).skew().fillna(0)
    kurt = ret.rolling(window, min_periods=20).kurt().fillna(0)
    return _safe_div(skew.abs(), (kurt + 3).abs(), 0)


# ===== 8. DISTRIBUTION (10 factors) =====

def _return_skew(df, window):
    return df["close"].pct_change().fillna(0).rolling(window, min_periods=20).skew().fillna(0)


def _return_kurt(df, window):
    return df["close"].pct_change().fillna(0).rolling(window, min_periods=20).kurt().fillna(0)


def _return_entropy(df, window):
    ret = df["close"].pct_change().fillna(0)
    def _entropy(x):
        if len(x) < 5 or x.std() == 0:
            return np.nan
        bins = np.histogram_bin_edges(x, bins=5)
        hist, _ = np.histogram(x, bins=bins)
        probs = hist / hist.sum()
        probs = probs[probs > 0]
        return -np.sum(probs * np.log(probs))
    return ret.rolling(window, min_periods=10).apply(_entropy, raw=True).fillna(0)


def _vol_skew(df, window):
    log_r = np.log(df["close"] / df["close"].shift(1))
    return (log_r ** 3).rolling(window, min_periods=20).mean().fillna(0)


def _vol_kurt(df, window):
    log_r = np.log(df["close"] / df["close"].shift(1))
    return ((log_r ** 4).rolling(window, min_periods=20).mean() /
            (log_r.rolling(window, min_periods=20).std() ** 4).replace(0, np.nan)).fillna(0)


def _return_autocorr(df, lag):
    ret = df["close"].pct_change().fillna(0)
    return ret.autocorr(lag) if len(ret.dropna()) > lag else 0


def _return_autocorr_series(df, window):
    ret = df["close"].pct_change().fillna(0)
    def _autocorr(x):
        if len(x) < 10:
            return np.nan
        r = (x - x.mean()) / (x.std() * len(x))
        return np.sum(r[:-1] * r[1:])
    return ret.rolling(window, min_periods=10).apply(_autocorr, raw=True).fillna(0)


def _positive_ratio(df, window):
    ret = df["close"].pct_change().fillna(0)
    pos = (ret > 0).astype(int).rolling(window, min_periods=5).sum()
    total = pd.Series(np.ones(len(df)), index=df.index).rolling(window, min_periods=5).sum()
    return _safe_div(pos, total, 0.5) - 0.5


# ===== 9. TREND STRENGTH (8 factors) =====

def _trend_strength(df, window):
    up = df["close"].diff().clip(lower=0)
    down = (-df["close"].diff()).clip(lower=0)
    su = up.rolling(window, min_periods=5).mean()
    sd = down.rolling(window, min_periods=5).mean()
    return _safe_div(abs(su - sd), su + sd, 0)


def _choppiness_index(df, window):
    atr_sum = (df["high"] - df["low"]).rolling(window, min_periods=window).sum()
    price_range = (df["high"].rolling(window, min_periods=window).max() -
                   df["low"].rolling(window, min_periods=window).min())
    return (100 * np.log10(_safe_div(atr_sum, price_range, 1)) / np.log10(window)).fillna(50)


def _adx_proxy(df, window):
    high = df["high"]
    low = df["low"]
    dm_plus = (high - high.shift(1)).clip(lower=0)
    dm_minus = (low.shift(1) - low).clip(lower=0)
    dm_plus = dm_plus.where(dm_plus > dm_minus, 0)
    dm_minus = dm_minus.where(dm_minus > dm_plus, 0)
    tr = pd.Series(_compute_tr(df), index=df.index)
    atr = tr.rolling(window, min_periods=window).mean()
    di_plus = 100 * _safe_div(dm_plus.rolling(window, min_periods=window).mean(), atr, 0)
    di_minus = 100 * _safe_div(dm_minus.rolling(window, min_periods=window).mean(), atr, 0)
    dx = _safe_div(abs(di_plus - di_minus), di_plus + di_minus, 0)
    return dx.rolling(window, min_periods=5).mean().fillna(0)


def _ci_commodity(df, window):
    typical = (df["high"] + df["low"] + df["close"]) / 3
    ma = typical.rolling(window, min_periods=window).mean()
    sum_abs = typical.diff().abs().rolling(window, min_periods=window).sum()
    return _safe_div(100 * (typical - ma), sum_abs, 0)


def _mass_index(df, window):
    hl = df["high"] - df["low"]
    ema1 = hl.ewm(span=window, adjust=False).mean()
    ema2 = ema1.ewm(span=window, adjust=False).mean()
    ratio = _safe_div(ema1, ema2, 1)
    return ratio.rolling(window, min_periods=window).sum().fillna(0)


def _directional_movement(df, window):
    high = df["high"]
    low = df["low"]
    dm_plus = (high - high.shift(1)).clip(lower=0)
    dm_minus = (low.shift(1) - low).clip(lower=0)
    dm_plus = dm_plus.where(dm_plus > dm_minus, 0)
    dm_minus = dm_minus.where(dm_minus > dm_plus, 0)
    tr = pd.Series(_compute_tr(df), index=df.index)
    atr = tr.rolling(window, min_periods=5).mean()
    di_plus = 100 * _safe_div(dm_plus.rolling(window, min_periods=5).mean(), atr, 0)
    di_minus = 100 * _safe_div(dm_minus.rolling(window, min_periods=5).mean(), atr, 0)
    return (di_plus - di_minus).fillna(0)


# ===== 10. DIVERGENCE (6 factors) =====

def _vp_divergence(df, window):
    pc = df["close"].pct_change(window).fillna(0)
    vr = _safe_div(df["volume"], df["volume"].rolling(window, min_periods=window).mean(), 1)
    return (pc - (vr - 1)).fillna(0)


def _price_volume_trend(df):
    pvt = df["close"].pct_change().fillna(0) * df["volume"]
    return pvt.rolling(20, min_periods=5).sum().fillna(0)


def _force_index(df, window):
    fi = df["close"].pct_change().fillna(0) * df["volume"]
    return fi.ewm(span=window, adjust=False).mean()


def _money_flow_index(df, window):
    typical = (df["high"] + df["low"] + df["close"]) / 3
    mf = typical * df["volume"]
    mf_pos = mf.where(typical.diff() > 0, 0).rolling(window, min_periods=window).sum()
    mf_neg = mf.where(typical.diff() < 0, 0).rolling(window, min_periods=window).sum()
    mfr = _safe_div(mf_pos, mf_neg, 1)
    return (100 - 100 / (1 + mfr)).fillna(50)


def _negative_volume_index(df):
    vol_down = df["volume"].where(df["close"].pct_change() < 0, 0)
    nvix = vol_down.cumsum()
    return nvix.pct_change(20).fillna(0)


# ===== 11. OSCILLATOR (10 factors) =====

def _rsi(df, window):
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(window, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).rolling(window, min_periods=window).mean()
    rs = _safe_div(gain, loss, 1)
    return (100 - 100 / (1 + rs)).fillna(50)


def _stochastic_k(df, window):
    low_n = df["low"].rolling(window, min_periods=window).min()
    high_n = df["high"].rolling(window, min_periods=window).max()
    return (100 * _safe_div(df["close"] - low_n, high_n - low_n, 0.5)).fillna(50)


def _stochastic_d(df, window, smooth):
    k = _stochastic_k(df, window)
    return k.rolling(smooth, min_periods=smooth).mean().fillna(50)


def _williams_r(df, window):
    high_n = df["high"].rolling(window, min_periods=window).max()
    low_n = df["low"].rolling(window, min_periods=window).min()
    return (100 * _safe_div(high_n - df["close"], high_n - low_n, 0.5) - 50).fillna(0)


def _cci(df, window):
    typical = (df["high"] + df["low"] + df["close"]) / 3
    ma = typical.rolling(window, min_periods=window).mean()
    mad = (typical - ma).abs().rolling(window, min_periods=window).mean()
    return _safe_div(typical - ma, 0.015 * mad, 0)


def _ultimate_oscillator(df):
    bp = df["close"] - np.minimum(df["low"], df["close"].shift(1))
    tr = np.maximum(df["high"], df["close"].shift(1)) - \
         np.minimum(df["low"], df["close"].shift(1))
    bp_s = pd.Series(bp.values, index=df.index)
    tr_s = pd.Series(tr.values, index=df.index)
    avg7 = bp_s.rolling(7, min_periods=7).sum() / tr_s.rolling(7, min_periods=7).sum().replace(0, np.nan)
    avg14 = bp_s.rolling(14, min_periods=14).sum() / tr_s.rolling(14, min_periods=14).sum().replace(0, np.nan)
    avg28 = bp_s.rolling(28, min_periods=28).sum() / tr_s.rolling(28, min_periods=28).sum().replace(0, np.nan)
    return (100 * (4 * avg7 + 2 * avg14 + avg28) / 7).fillna(50)


def _roc(df, window):
    return df["close"].pct_change(window).fillna(0)


# ===== 12. FRACTAL / NONLINEAR (6 factors) =====

def _hurst_proxy(df, window):
    ret = np.log(df["close"] / df["close"].shift(1)).fillna(0)
    def _hurst(x):
        if len(x) < 20:
            return np.nan
        n = len(x)
        lags = np.arange(2, min(n // 4, 20))
        stds = [np.std(np.diff(x, n=int(lag))) for lag in lags]
        if len(stds) < 3:
            return np.nan
        log_lags = np.log(lags)
        log_stds = np.log(stds)
        slope, _ = np.polyfit(log_lags, log_stds, 1)
        return -slope + 0.5
    return ret.rolling(window, min_periods=20).apply(_hurst, raw=True).fillna(0.5)


def _fractal_dimension(df, window):
    high_n = df["high"].rolling(window, min_periods=window).max()
    low_n = df["low"].rolling(window, min_periods=window).min()
    trend = _safe_div(df["close"] - df["close"].shift(window), high_n - low_n, 0)
    return trend.fillna(0)


def _nonlinearity_test(df, window):
    ret = df["close"].pct_change().fillna(0)
    ret_sq = ret ** 2
    shifted = ret.shift(1) ** 2
    corr = ret_sq.rolling(window, min_periods=10).corr(shifted).fillna(0)
    return corr


# ===== 13. CROSS-SECTIONAL RANK PROXIES (4 factors) =====

def _cross_sectional_momentum_rank(df, window):
    ret = df["close"].pct_change(window).fillna(0)
    ma_ret = ret.rolling(window, min_periods=5).mean()
    std_ret = ret.rolling(window, min_periods=5).std()
    return _safe_div(ret - ma_ret, std_ret.replace(0, np.nan), 0)


def _volatility_rank(df, window):
    vol = np.log(df["close"] / df["close"].shift(1)).rolling(window, min_periods=5).std()
    vol_ma = vol.rolling(window * 2, min_periods=window).mean()
    return _safe_div(vol - vol_ma, vol_ma.replace(0, np.nan), 0)


# ===== 14. INTRA-DAY MICROSTRUCTURE (6 factors) =====

def _amihud_illiquidity(df, window):
    ret = df["close"].pct_change().fillna(0).abs()
    return _safe_div(ret.rolling(window, min_periods=10).sum(),
                     df["amount"].rolling(window, min_periods=10).sum(), 0)


def _vwap_deviation(df):
    typical = (df["high"] + df["low"] + df["close"]) / 3
    return _safe_div(df["close"] - typical, df["close"], 0)


def _price_impact(df):
    ret = df["close"].pct_change().fillna(0).abs()
    vol_std = df["volume"].rolling(20, min_periods=10).std()
    return _safe_div(ret, vol_std.replace(0, np.nan), 0)


def _intraday_momentum(df):
    return _safe_div(df["close"] - df["open"], df["open"], 0)


# ===== Helper =====

def _compute_tr(df):
    close_prev = df["close"].shift(1)
    return np.maximum(
        df["high"] - df["low"],
        np.maximum(abs(df["high"] - close_prev), abs(df["low"] - close_prev))
    )


# ===== FACTORY REGISTRY (120+ factors) =====

FACTORS: list[FactorDef] = []


def _register():
    # 1. Momentum (12)
    for w in [5, 10, 20, 60, 120]:
        FACTORS.append(FactorDef(f"mom_{w}", "momentum", f"{w}-day return", lambda d, w=w: _momentum(d, w)))
    FACTORS.append(FactorDef("mom_w_5_20_60", "momentum", "Weighted momentum 5/20/60",
                              lambda d: _weighted_momentum(d, 5, 20, 60)))
    FACTORS.append(FactorDef("skew_mom_20", "momentum", "Skew-adjusted momentum 20d",
                              lambda d: _skew_momentum(d, 20)))
    FACTORS.append(FactorDef("win_rate_20", "momentum", "Win rate 20d",
                              lambda d: _win_rate(d, 20)))
    FACTORS.append(FactorDef("mdd_60", "momentum", "Max drawdown 60d window",
                              lambda d: _max_drawdown_window(d, 60)))
    FACTORS.append(FactorDef("recovery_20", "momentum", "Recovery speed 20d",
                              lambda d: _recovery_speed(d, 20)))

    # 2. Trend (12)
    for w in [10, 20, 60, 120]:
        FACTORS.append(FactorDef(f"ma_dist_{w}", "trend", f"Distance from MA{w}", lambda d, w=w: _ma_distance(d, w)))
    for s in [12, 26, 50]:
        FACTORS.append(FactorDef(f"ema_dist_{s}", "trend", f"Distance from EMA{s}", lambda d, s=s: _ema_distance(d, s)))
    for w in [20, 60]:
        FACTORS.append(FactorDef(f"slope_{w}", "trend", f"Linear slope {w}d", lambda d, w=w: _linear_slope(d, w)))
        FACTORS.append(FactorDef(f"r2_{w}", "trend", f"R-squared {w}d", lambda d, w=w: _linear_r_squared(d, w)))
    FACTORS.append(FactorDef("ma_cross_5_20", "trend", "MA5/MA20 cross", lambda d: _ma_cross(d, 5, 20)))
    FACTORS.append(FactorDef("ma_cross_20_60", "trend", "MA20/MA60 cross", lambda d: _ma_cross(d, 20, 60)))
    FACTORS.append(FactorDef("ma_stack_5_10_20_60", "trend", "MA stack score [5,10,20,60]",
                              lambda d: _ma_stack_score(d, [5, 10, 20, 60])))
    FACTORS.append(FactorDef("hull_dist_20", "trend", "Hull MA distance 20d",
                              lambda d: _hull_ma_distance(d, 20)))

    # 3. Volatility (15)
    for w in [10, 20, 60]:
        FACTORS.append(FactorDef(f"real_vol_{w}", "volatility", f"Realized vol {w}d", lambda d, w=w: _realized_vol(d, w)))
    for w in [10, 20]:
        FACTORS.append(FactorDef(f"tr_vol_{w}", "volatility", f"True range vol {w}d", lambda d, w=w: _true_range_vol(d, w)))
        FACTORS.append(FactorDef(f"park_vol_{w}", "volatility", f"Parkinson vol {w}d", lambda d, w=w: _parkinson_vol(d, w)))
    FACTORS.append(FactorDef("gk_vol_20", "volatility", "Garman-Klass vol 20d",
                              lambda d: _garman_klass_vol(d, 20)))
    FACTORS.append(FactorDef("rs_vol_20", "volatility", "Rogers-Satchev vol 20d",
                              lambda d: _rogers_satchev_vol(d, 20)))
    FACTORS.append(FactorDef("yin_vol_20", "volatility", "Yin vol 20d",
                              lambda d: _yin_vol(d, 20)))
    FACTORS.append(FactorDef("vol_ratio_5_20", "volatility", "Vol ratio 5/20d",
                              lambda d: _vol_ratio(d, 5, 20)))
    FACTORS.append(FactorDef("vol_expansion_20", "volatility", "Vol expansion 20d",
                              lambda d: _vol_expansion(d, 20)))
    FACTORS.append(FactorDef("vol_regime_20", "volatility", "Vol regime z-score 20d",
                              lambda d: _vol_regime(d, 20)))
    FACTORS.append(FactorDef("dual_theta_20", "volatility", "Dual theta ratio 20d",
                              lambda d: _dual_theta(d, 20)))
    FACTORS.append(FactorDef("atr_ratio_14", "volatility", "ATR/close ratio 14d",
                              lambda d: _atr_ratio(d, 14)))

    # 4. Volume / Liquidity (15)
    for w in [5, 10, 20]:
        FACTORS.append(FactorDef(f"vol_ma_{w}", "volume", f"Volume vs MA{w}", lambda d, w=w: _volume_ma_ratio(d, w)))
    FACTORS.append(FactorDef("vol_trend_20", "volume", "Volume trend 20d",
                              lambda d: _volume_trend(d, 20)))
    FACTORS.append(FactorDef("vol_accel_10", "volume", "Volume acceleration 10d",
                              lambda d: _volume_acceleration(d, 10)))
    FACTORS.append(FactorDef("amount_vol_ratio", "volume", "Amount/volume ratio deviation",
                              _amount_volume_ratio))
    FACTORS.append(FactorDef("vp_corr_20", "volume", "Volume-price correlation 20d",
                              lambda d: _volume_price_correlation(d, 20)))
    FACTORS.append(FactorDef("ad_line_mom", "volume", "Accumulation/distribution momentum",
                              _accumulation_distribution))
    FACTORS.append(FactorDef("cmf_20", "volume", "Chaikin money flow 20d",
                              lambda d: _chaikin_money_flow(d, 20)))
    FACTORS.append(FactorDef("eom_14", "volume", "Ease of movement 14d",
                              lambda d: _ease_of_movement(d, 14)))
    FACTORS.append(FactorDef("obv_mom_20", "volume", "OBV momentum 20d",
                              lambda d: _on_balance_volume_mom(d, 20)))
    FACTORS.append(FactorDef("vol_cluster_20", "volume", "Volume clustering 20d",
                              lambda d: _volume_cluster(d, 20)))
    FACTORS.append(FactorDef("noble_vol", "volume", "Noble volume signal",
                              _noble_volume))

    # 5. Mean Reversion (10)
    for w in [10, 20, 60]:
        FACTORS.append(FactorDef(f"boll_pos_{w}", "mean_reversion", f"Bollinger position {w}d",
                                  lambda d, w=w: _bollinger_position(d, w, 2)))
    FACTORS.append(FactorDef("keltner_pos_20", "mean_reversion", "Keltner position 20d",
                              lambda d: _keltner_position(d, 20)))
    FACTORS.append(FactorDef("ema_ratio_12_26", "mean_reversion", "EMA12/EMA26 ratio",
                              lambda d: _price_ema_ratio(d, 12, 26)))
    for w in [10, 20, 60]:
        FACTORS.append(FactorDef(f"range_pos_{w}", "mean_reversion", f"Range position {w}d",
                                  lambda d, w=w: _range_position(d, w)))
    FACTORS.append(FactorDef("ibias_20", "mean_reversion", "I-Bias 20d",
                              lambda d: _ibias(d, 20)))
    FACTORS.append(FactorDef("donchian_pos_20", "mean_reversion", "Donchian position 20d",
                              lambda d: _donchian_position(d, 20)))
    FACTORS.append(FactorDef("mr_speed_20", "mean_reversion", "Mean reversion speed 20d",
                              lambda d: _mean_rev_speed(d, 20)))

    # 6. Price Pattern (15)
    FACTORS.append(FactorDef("intraday_range", "pattern", "Intraday range ratio",
                              _intraday_range))
    FACTORS.append(FactorDef("body_ratio", "pattern", "Body to total range ratio",
                              _body_ratio))
    FACTORS.append(FactorDef("upper_shadow", "pattern", "Upper shadow ratio",
                              _upper_shadow))
    FACTORS.append(FactorDef("lower_shadow", "pattern", "Lower shadow ratio",
                              _lower_shadow))
    FACTORS.append(FactorDef("close_pos", "pattern", "Close position in range",
                              _close_position))
    FACTORS.append(FactorDef("gap_ratio", "pattern", "Gap to return ratio",
                              _gap_ratio))
    FACTORS.append(FactorDef("gap_persist_20", "pattern", "Gap persistence 20d",
                              lambda d: _gap_persistence(d, 20)))
    FACTORS.append(FactorDef("engulfing", "pattern", "Engulfing candle ratio",
                              _candle_engulfing))
    FACTORS.append(FactorDef("inside_bar", "pattern", "Inside bar indicator",
                              _inside_bar))
    FACTORS.append(FactorDef("three_day_pat", "pattern", "Three-day pattern anomaly",
                              _three_day_pattern))
    FACTORS.append(FactorDef("oc_asymmetry", "pattern", "Open-close asymmetry 20d",
                              _open_close_asymmetry))

    # 7. Risk Adjusted (8)
    FACTORS.append(FactorDef("sharpness_20", "risk_return", "Local sharpness 20d",
                              lambda d: _sharpness(d, 20)))
    FACTORS.append(FactorDef("sortino_20", "risk_return", "Sortino ratio 20d",
                              lambda d: _sortino_ratio(d, 20)))
    FACTORS.append(FactorDef("omega_20", "risk_return", "Omega ratio 20d",
                              lambda d: _omega_ratio(d, 20)))
    FACTORS.append(FactorDef("calmar_20", "risk_return", "Calmar proxy 20d",
                              lambda d: _max_drawdown_ratio(d, 20)))
    FACTORS.append(FactorDef("up_capture_20", "risk_return", "Up capture ratio 20d",
                              lambda d: _up_capture(d, 20)))
    FACTORS.append(FactorDef("down_capture_20", "risk_return", "Down capture ratio 20d",
                              lambda d: _down_capture(d, 20)))
    FACTORS.append(FactorDef("tail_ratio_60", "risk_return", "Tail ratio 60d",
                              lambda d: _tail_ratio(d, 60)))

    # 8. Distribution (10)
    for w in [20, 60]:
        FACTORS.append(FactorDef(f"skew_{w}", "distribution", f"Return skewness {w}d",
                                  lambda d, w=w: _return_skew(d, w)))
        FACTORS.append(FactorDef(f"kurt_{w}", "distribution", f"Return kurtosis {w}d",
                                  lambda d, w=w: _return_kurt(d, w)))
    FACTORS.append(FactorDef("entropy_20", "distribution", "Return entropy 20d",
                              lambda d: _return_entropy(d, 20)))
    FACTORS.append(FactorDef("vol_skew_60", "distribution", "Volatility skewness 60d",
                              lambda d: _vol_skew(d, 60)))
    FACTORS.append(FactorDef("vol_kurt_60", "distribution", "Volatility kurtosis 60d",
                              lambda d: _vol_kurt(d, 60)))
    FACTORS.append(FactorDef("autocorr_20", "distribution", "Return autocorrelation 20d",
                              lambda d: _return_autocorr_series(d, 20)))
    for w in [10, 20]:
        FACTORS.append(FactorDef(f"pos_ratio_{w}", "distribution", f"Positive return ratio {w}d",
                                  lambda d, w=w: _positive_ratio(d, w)))

    # 9. Trend Strength (8)
    for w in [10, 14]:
        FACTORS.append(FactorDef(f"trend_str_{w}", "trend_strength", f"Trend strength {w}d",
                                  lambda d, w=w: _trend_strength(d, w)))
    FACTORS.append(FactorDef("choppiness_14", "trend_strength", "Choppiness index 14d",
                              lambda d: _choppiness_index(d, 14)))
    FACTORS.append(FactorDef("adx_proxy_14", "trend_strength", "ADX proxy 14d",
                              lambda d: _adx_proxy(d, 14)))
    FACTORS.append(FactorDef("ci_commodity_20", "trend_strength", "Commodity channel index 20d",
                              lambda d: _ci_commodity(d, 20)))
    FACTORS.append(FactorDef("mass_index_20", "trend_strength", "Mass index 20d",
                              lambda d: _mass_index(d, 20)))
    FACTORS.append(FactorDef("dir_movement_14", "trend_strength", "Directional movement 14d",
                              lambda d: _directional_movement(d, 14)))

    # 10. Divergence (6)
    FACTORS.append(FactorDef("vp_div_20", "divergence", "Volume-price divergence 20d",
                              lambda d: _vp_divergence(d, 20)))
    FACTORS.append(FactorDef("pvt_20", "divergence", "Price-volume trend 20d",
                              _price_volume_trend))
    FACTORS.append(FactorDef("force_index_13", "divergence", "Force index 13d",
                              lambda d: _force_index(d, 13)))
    FACTORS.append(FactorDef("mfi_20", "divergence", "Money flow index 20d",
                              lambda d: _money_flow_index(d, 20)))
    FACTORS.append(FactorDef("nvix_20", "divergence", "Negative volume index momentum 20d",
                              _negative_volume_index))

    # 11. Oscillator (10)
    for w in [6, 14, 21]:
        FACTORS.append(FactorDef(f"rsi_{w}", "oscillator", f"RSI {w}d", lambda d, w=w: _rsi(d, w)))
    FACTORS.append(FactorDef("stoch_k_14", "oscillator", "Stochastic %K 14d",
                              lambda d: _stochastic_k(d, 14)))
    FACTORS.append(FactorDef("stoch_d_14_3", "oscillator", "Stochastic %D 14/3d",
                              lambda d: _stochastic_d(d, 14, 3)))
    FACTORS.append(FactorDef("williams_r_14", "oscillator", "Williams %R 14d",
                              lambda d: _williams_r(d, 14)))
    FACTORS.append(FactorDef("cci_20", "oscillator", "CCI 20d",
                              lambda d: _cci(d, 20)))
    FACTORS.append(FactorDef("ultimate_osc", "oscillator", "Ultimate oscillator",
                              _ultimate_oscillator))
    for w in [10, 20]:
        FACTORS.append(FactorDef(f"roc_{w}", "oscillator", f"Rate of change {w}d",
                                  lambda d, w=w: _roc(d, w)))

    # 12. Fractal / Nonlinear (6)
    FACTORS.append(FactorDef("hurst_60", "fractal", "Hurst exponent proxy 60d",
                              lambda d: _hurst_proxy(d, 60)))
    FACTORS.append(FactorDef("fractal_dim_20", "fractal", "Fractal dimension proxy 20d",
                              lambda d: _fractal_dimension(d, 20)))
    FACTORS.append(FactorDef("nonlinearity_20", "fractal", "Nonlinearity test 20d",
                              lambda d: _nonlinearity_test(d, 20)))

    # 13. Cross-sectional (4)
    for w in [20, 60]:
        FACTORS.append(FactorDef(f"cs_zscore_{w}", "cross_sectional", f"Cross-sectional z-score {w}d",
                                  lambda d, w=w: _cross_sectional_momentum_rank(d, w)))
    FACTORS.append(FactorDef("vol_rank_20", "cross_sectional", "Volatility rank 20d",
                              lambda d: _volatility_rank(d, 20)))

    # 14. Intraday Microstructure (6)
    FACTORS.append(FactorDef("amihud_20", "microstructure", "Amihud illiquidity 20d",
                              lambda d: _amihud_illiquidity(d, 20)))
    FACTORS.append(FactorDef("vwap_dev", "microstructure", "VWAP deviation",
                              _vwap_deviation))
    FACTORS.append(FactorDef("price_impact", "microstructure", "Price impact proxy",
                              _price_impact))
    FACTORS.append(FactorDef("intraday_mom", "microstructure", "Intraday momentum (open-close)",
                              _intraday_momentum))


_register()


def get_factors_by_category(category: str) -> list[FactorDef]:
    return [f for f in FACTORS if f.category == category]


def get_all_categories() -> list[str]:
    return sorted(set(f.category for f in FACTORS))
