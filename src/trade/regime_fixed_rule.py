from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


# ============================================================
# 1. Basic utilities
# ============================================================

def validate_tau(tau: int):
    if tau % 5 != 0:
        raise ValueError("tau must be a multiple of 5.")


def prepare_trading_data(trading_data: pd.DataFrame) -> pd.DataFrame:
    df = trading_data.copy().sort_values("time").reset_index(drop=True)
    df["time"] = pd.to_datetime(df["time"])

    # Required trade size
    df["target_position"] = df["position"].astype(float)
    df["delta_q"] = df["target_position"].diff().fillna(0.0)

    return df


def sparse_ema(series: pd.Series, step: int, span: int) -> pd.Series:
    """
    Non-overlapping tau-grid EMA.

    If tau = 10 minutes and data is 5-minute frequency, step = 2.

    For timestamp t:
        EMA uses t, t-10, t-20, ...

    For timestamp t+5:
        EMA uses t+5, t-5, t-15, ...

    So state variables are still available every 5 minutes,
    but each EMA path uses non-overlapping tau intervals.
    """
    values = series.to_numpy(dtype=float)
    out = np.full(len(series), np.nan)

    alpha = 2.0 / (span + 1.0)

    for offset in range(step):
        idx = np.arange(offset, len(series), step)
        sub = values[idx]

        prev = np.nan

        for j, x in enumerate(sub):
            loc = idx[j]

            if np.isnan(x):
                out[loc] = prev
            elif np.isnan(prev):
                prev = x
                out[loc] = prev
            else:
                prev = alpha * x + (1.0 - alpha) * prev
                out[loc] = prev

    return pd.Series(out, index=series.index)


# ============================================================
# 2. State variable construction
# ============================================================

def build_tau_state_features(
    data1: pd.DataFrame,
    data5: pd.DataFrame,
    tau: int,
    vol_ema_span: int = 100,
    price_short_span: int = 20,
    price_long_span: int = 100,
    annualize_scalar: float | None = None,
) -> pd.DataFrame:
    """
    Build state variables every 5 minutes.

    State variables:

    1. volatility:
        For each 5-minute timestamp t, calculate realized volatility
        using 1-minute returns over the current tau-minute interval.

        Then apply sparse EMA using non-overlapping tau intervals.

    2. crossover:
        Use 5-minute close price.
        Apply sparse EMA with short and long spans.
        Then compute:

            crossover = EMA_short - EMA_long

        and also:

            crossover_pct = crossover / close * 100
    """
    validate_tau(tau)

    step = tau // 5

    if annualize_scalar is None:
        annualize_scalar = 252 * 24 * 60 / tau

    x1 = data1.copy().sort_values("time")
    x5 = data5.copy().sort_values("time")

    x1["time"] = pd.to_datetime(x1["time"])
    x5["time"] = pd.to_datetime(x5["time"])

    x1 = x1.set_index("time")
    x5 = x5.set_index("time")

    # 1-minute log return
    log_ret_1m = np.log(x1["close"]).diff()

    # tau-minute realized volatility ending at each 1-minute timestamp
    rv_tau = log_ret_1m.pow(2).rolling(tau).sum()

    vol_tau = np.sqrt(rv_tau * annualize_scalar)

    feat = pd.DataFrame(index=x5.index)
    feat["close"] = x5["close"]

    # Reindex tau-window volatility to every 5-minute timestamp
    feat["volatility_raw"] = vol_tau.reindex(feat.index)

    # Sparse EMA volatility
    feat["ema_volatility"] = sparse_ema(
        feat["volatility_raw"],
        step=step,
        span=vol_ema_span,
    )

    # Sparse EMA price crossover
    feat["ema_price_short"] = sparse_ema(
        feat["close"],
        step=step,
        span=price_short_span,
    )

    feat["ema_price_long"] = sparse_ema(
        feat["close"],
        step=step,
        span=price_long_span,
    )

    feat["crossover"] = feat["ema_price_short"] - feat["ema_price_long"]
    feat["crossover_pct"] = feat["crossover"] / feat["close"] * 100.0

    return feat.reset_index()


# ============================================================
# 3. Future filling distribution dataset
# ============================================================

def build_forward_distribution_dataset(data5: pd.DataFrame, tau: int) -> pd.DataFrame:
    """
    For each 5-minute timestamp t, calculate future tau-minute favorable moves.

    Buy order:
        We care whether price goes lower after t.

        future_low_move = close_t - future_low

    Sell order:
        We care whether price goes higher after t.

        future_high_move = future_high - close_t

    The future window is:

        (t, t + tau]

    because the order is placed after observing close_t.
    """
    validate_tau(tau)

    bars = tau // 5

    df = data5.copy().sort_values("time").reset_index(drop=True)
    df["time"] = pd.to_datetime(df["time"])

    highs = df["high"].to_numpy(float)
    lows = df["low"].to_numpy(float)
    closes = df["close"].to_numpy(float)
    times = df["time"].to_numpy(dtype="datetime64[ns]")

    future_high = np.full(len(df), np.nan)
    future_low = np.full(len(df), np.nan)
    future_close = np.full(len(df), np.nan)
    available_time = np.full(len(df), np.datetime64("NaT"), dtype="datetime64[ns]")

    for i in range(len(df)):
        start = i + 1
        end = i + bars

        if end >= len(df):
            continue

        future_high[i] = np.nanmax(highs[start:end + 1])
        future_low[i] = np.nanmin(lows[start:end + 1])
        future_close[i] = closes[end]
        available_time[i] = times[end]

    out = df[["time", "close"]].copy()

    out["future_high_move"] = future_high - out["close"]
    out["future_low_move"] = out["close"] - future_low
    out["future_close"] = future_close
    out["available_time"] = pd.to_datetime(available_time)

    return out


# ============================================================
# 4. Regime distribution state
# ============================================================

def choose_k_from_moves(
    moves: pd.Series,
    fill_prob: float,
    tick_size: float,
    max_k: int | None = None,
) -> tuple[int, float]:
    """
    Choose the largest k such that:

        P(move >= k * tick_size) >= fill_prob

    Larger k means better execution price but lower fill probability.
    """
    valid = moves.dropna().to_numpy(float)

    if len(valid) == 0:
        return 0, np.nan

    if max_k is None:
        max_k = max(0, int(np.floor(np.nanmax(valid) / tick_size)))

    chosen_k = 0
    chosen_prob = float((valid >= 0.0).mean())

    for k in range(max_k + 1):
        prob = float((valid >= k * tick_size).mean())

        if prob >= fill_prob:
            chosen_k = k
            chosen_prob = prob
        else:
            break

    return chosen_k, chosen_prob


@dataclass
class RegimeDistributionState:
    """
    Conditional filling distribution based on:

        volatility regime + crossover regime
    """
    history: pd.DataFrame
    features: pd.DataFrame
    fill_prob: float
    tick_size: float

    max_k: int | None = None
    min_obs_per_regime: int = 25
    n_vol_regimes: int = 3
    n_cross_regimes: int = 3

    vol_cuts: list[float] | None = None
    cross_cuts: list[float] | None = None
    tagged_history: pd.DataFrame | None = None

    @staticmethod
    def _bucket(x: float, cuts: list[float] | None) -> int:
        if cuts is None or pd.isna(x):
            return 0

        for i, c in enumerate(cuts):
            if x <= c:
                return i

        return len(cuts)

    def get_regime(self, ema_volatility: float, crossover_pct: float) -> str:
        vol_bucket = self._bucket(ema_volatility, self.vol_cuts)
        cross_bucket = self._bucket(crossover_pct, self.cross_cuts)

        return f"vol_{vol_bucket}_cross_{cross_bucket}"

    def rebuild(self):
        """
        Rebuild regime buckets and tag history.

        This should NOT be called every 5 minutes.
        In the strategy below, it is called once per day.
        """
        merged = self.history.merge(
            self.features[["time", "ema_volatility", "crossover_pct"]],
            on="time",
            how="inner",
        )

        merged = merged.dropna(
            subset=[
                "future_high_move",
                "future_low_move",
                "ema_volatility",
                "crossover_pct",
            ]
        ).copy()

        if merged.empty:
            self.vol_cuts = None
            self.cross_cuts = None
            self.tagged_history = merged
            return

        vol_q = np.linspace(0, 1, self.n_vol_regimes + 1)[1:-1]
        cross_q = np.linspace(0, 1, self.n_cross_regimes + 1)[1:-1]

        self.vol_cuts = merged["ema_volatility"].quantile(vol_q).to_list()
        self.cross_cuts = merged["crossover_pct"].quantile(cross_q).to_list()

        merged["regime"] = [
            self.get_regime(v, c)
            for v, c in zip(merged["ema_volatility"], merged["crossover_pct"])
        ]

        self.tagged_history = merged.reset_index(drop=True)

    def choose_k(self, feature_row: pd.Series, side: str) -> tuple[int, float, str]:
        """
        Choose k based on current state and order side.
        """
        if self.tagged_history is None or self.tagged_history.empty:
            return 0, np.nan, "fallback_empty"

        ema_vol = feature_row["ema_volatility"]
        cross = feature_row["crossover_pct"]

        regime = self.get_regime(ema_vol, cross)

        subset = self.tagged_history[self.tagged_history["regime"] == regime]

        if side == "buy":
            regime_moves = subset["future_low_move"]
            fallback_moves = self.tagged_history["future_low_move"]
        else:
            regime_moves = subset["future_high_move"]
            fallback_moves = self.tagged_history["future_high_move"]

        if regime_moves.dropna().shape[0] >= self.min_obs_per_regime:
            moves = regime_moves
            state_key = regime
        else:
            moves = fallback_moves
            state_key = "fallback_global"

        k, prob = choose_k_from_moves(
            moves=moves,
            fill_prob=self.fill_prob,
            tick_size=self.tick_size,
            max_k=self.max_k,
        )

        return k, prob, state_key

    def daily_update(self, new_history: pd.DataFrame, new_features: pd.DataFrame):
        """
        Update the distribution once per day.
        """
        if not new_history.empty:
            self.history = (
                pd.concat([self.history, new_history], ignore_index=True)
                .drop_duplicates(subset=["time"], keep="last")
                .sort_values("time")
                .reset_index(drop=True)
            )

        if not new_features.empty:
            self.features = (
                pd.concat([self.features, new_features], ignore_index=True)
                .drop_duplicates(subset=["time"], keep="last")
                .sort_values("time")
                .reset_index(drop=True)
            )

        self.rebuild()


# ============================================================
# 5. Order helpers
# ============================================================

def make_order(
    signal_time,
    signal_idx: int,
    side: str,
    qty: float,
    ref_price: float,
    target_price: float,
    k: int,
    fill_prob_estimate: float,
    state_key: str,
    ema_volatility: float,
    crossover_pct: float,
) -> dict:
    return {
        "signal_time": signal_time,
        "signal_idx": signal_idx,
        "side": side,
        "qty": qty,
        "delta_q": qty if side == "buy" else -qty,
        "ref_price": ref_price,
        "target_price": target_price,
        "k": k,
        "fill_prob_estimate": fill_prob_estimate,
        "state_key": state_key,
        "ema_volatility": ema_volatility,
        "crossover_pct": crossover_pct,
        "status": "active",
    }


def check_fill(order: dict, row: pd.Series) -> bool:
    if order["side"] == "buy":
        return pd.notna(row["low"]) and row["low"] <= order["target_price"]
    else:
        return pd.notna(row["high"]) and row["high"] >= order["target_price"]


def make_execution_row(
    order: dict,
    exec_time,
    exec_idx: int,
    exec_price: float,
    execution_state: str,
    tau: int,
) -> dict:
    bars_per_tau = tau // 5

    delay_5m_bars = exec_idx - order["signal_idx"]

    if delay_5m_bars <= bars_per_tau:
        delay_tau_intervals = 1
    else:
        delay_tau_intervals = int(np.ceil(delay_5m_bars / bars_per_tau))

    return {
        "signal_time": order["signal_time"],
        "exec_time": exec_time,
        "side": order["side"],
        "qty": order["qty"],
        "delta_q": order["delta_q"],

        "ref_price": order["ref_price"],
        "target_price": order["target_price"],
        "exec_price": exec_price,

        "k": order["k"],
        "fill_prob_estimate": order["fill_prob_estimate"],
        "state_key": order["state_key"],

        "ema_volatility": order["ema_volatility"],
        "crossover_pct": order["crossover_pct"],

        "execution_state": execution_state,
        "delay_5m_bars": delay_5m_bars,
        "delay_tau_intervals": delay_tau_intervals,
    }


# ============================================================
# 6. PnL calculation
# ============================================================

def build_pnl_df(
    trading_df: pd.DataFrame,
    exec_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return PnL dataframe with:

        1. agent original PnL
        2. strategy PnL
        3. difference
    """
    df = trading_df.copy().sort_values("time").reset_index(drop=True)

    # Agent benchmark:
    # execute every required delta_q immediately at agent price if available,
    # otherwise use close.
    if "price" in df.columns:
        agent_exec_price = df["price"].fillna(df["close"])
    else:
        agent_exec_price = df["close"]

    df["agent_cash_flow"] = -(df["delta_q"] * agent_exec_price)
    df["agent_cash"] = df["agent_cash_flow"].cumsum()
    df["agent_position"] = df["delta_q"].cumsum()
    df["agent_pnl"] = df["agent_cash"] + df["agent_position"] * df["close"]

    # Strategy PnL
    if exec_df.empty:
        exec_agg = pd.DataFrame(columns=["time", "strategy_cash_flow", "strategy_delta_q"])
    else:
        exec_agg = (
            exec_df.groupby("exec_time", as_index=False)
            .apply(
                lambda x: pd.Series(
                    {
                        "strategy_cash_flow": (-(x["delta_q"] * x["exec_price"])).sum(),
                        "strategy_delta_q": x["delta_q"].sum(),
                    }
                ),
                include_groups=False,
            )
            .rename(columns={"exec_time": "time"})
        )

    out = df.merge(exec_agg, on="time", how="left")

    out["strategy_cash_flow"] = out["strategy_cash_flow"].fillna(0.0)
    out["strategy_delta_q"] = out["strategy_delta_q"].fillna(0.0)

    out["strategy_cash"] = out["strategy_cash_flow"].cumsum()
    out["strategy_position"] = out["strategy_delta_q"].cumsum()
    out["strategy_pnl"] = out["strategy_cash"] + out["strategy_position"] * out["close"]

    out["pnl_diff"] = out["strategy_pnl"] - out["agent_pnl"]

    return out[
        [
            "time",
            "close",

            "agent_cash_flow",
            "agent_cash",
            "agent_position",
            "agent_pnl",

            "strategy_cash_flow",
            "strategy_cash",
            "strategy_position",
            "strategy_pnl",

            "pnl_diff",
        ]
    ].reset_index(drop=True)


# ============================================================
# 7. Main strategy
# ============================================================

def regime_fixed(
    trading_data: pd.DataFrame,
    data1: pd.DataFrame,
    data5: pd.DataFrame,
    prep_data1: pd.DataFrame,
    prep_data5: pd.DataFrame,
    tau: int = 10,
    fill_prob: float = 0.8,
    tick_size: float = 0.25,
    vol_ema_span: int = 100,
    price_short_span: int = 20,
    price_long_span: int = 100,
    max_k: int | None = None,
    min_obs_per_regime: int = 25,
    n_vol_regimes: int = 3,
    n_cross_regimes: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Regime fixed strategy with daily distribution update.

    Logic:

    1. Build volatility and crossover state variables.
    2. Use prep data to initialize conditional filling distributions.
    3. During trading:
        - every 5 minutes, observe required position change
        - choose target price based on current state
        - check whether new order is filled during the next tau interval
        - check whether old unfilled orders are filled
    4. Update filling probability distribution only after each trading day ends.
    5. At the end, force all unexecuted orders at the final close.

    Returns:
        pnl_df, trade_detail_df
    """
    validate_tau(tau)

    bars_per_tau = tau // 5

    # --------------------------------------------------------
    # Prepare trading data
    # --------------------------------------------------------

    trading_df = prepare_trading_data(trading_data)

    # --------------------------------------------------------
    # Build features on prep + trading data together
    # --------------------------------------------------------

    all_data1 = (
        pd.concat([prep_data1, data1], ignore_index=True)
        .drop_duplicates("time")
        .sort_values("time")
        .reset_index(drop=True)
    )

    all_data5 = (
        pd.concat([prep_data5, data5], ignore_index=True)
        .drop_duplicates("time")
        .sort_values("time")
        .reset_index(drop=True)
    )

    all_features = build_tau_state_features(
        data1=all_data1,
        data5=all_data5,
        tau=tau,
        vol_ema_span=vol_ema_span,
        price_short_span=price_short_span,
        price_long_span=price_long_span,
    )

    all_features["time"] = pd.to_datetime(all_features["time"])

    feature_map = (
        all_features
        .sort_values("time")
        .drop_duplicates("time")
        .set_index("time")
    )

    # --------------------------------------------------------
    # Initial prep distribution
    # --------------------------------------------------------

    prep_forward = build_forward_distribution_dataset(prep_data5, tau)

    prep_history = prep_forward.dropna(
        subset=["future_high_move", "future_low_move", "available_time"]
    ).copy()

    prep_features = all_features[
        all_features["time"].isin(pd.to_datetime(prep_history["time"]))
    ].copy()

    state = RegimeDistributionState(
        history=prep_history,
        features=prep_features,
        fill_prob=fill_prob,
        tick_size=tick_size,
        max_k=max_k,
        min_obs_per_regime=min_obs_per_regime,
        n_vol_regimes=n_vol_regimes,
        n_cross_regimes=n_cross_regimes,
    )

    state.rebuild()

    # --------------------------------------------------------
    # Forward outcomes from trading period
    # These are used only for daily distribution updates.
    # --------------------------------------------------------

    trading_forward = build_forward_distribution_dataset(
        trading_df[["time", "open", "high", "low", "close", "volume"]],
        tau=tau,
    )

    trading_forward = (
        trading_forward
        .dropna(subset=["future_high_move", "future_low_move", "available_time"])
        .sort_values("available_time")
        .reset_index(drop=True)
    )

    last_distribution_update_time = pd.Timestamp.min

    # --------------------------------------------------------
    # Trading loop
    # --------------------------------------------------------

    active_orders: list[dict] = []
    exec_rows: list[dict] = []

    trading_df["trading_day"] = trading_df["time"].dt.date

    previous_day = None
    previous_time = None

    for i, row in trading_df.iterrows():
        t = row["time"]
        current_day = row["trading_day"]

        # ----------------------------------------------------
        # Daily distribution update
        # ----------------------------------------------------
        # When a new day starts, update the distribution using
        # all outcomes that became observable before the previous
        # day ended.
        # ----------------------------------------------------

        if previous_day is not None and current_day != previous_day:
            update_cutoff = previous_time

            new_history = trading_forward[
                (trading_forward["available_time"] <= update_cutoff)
                & (trading_forward["available_time"] > last_distribution_update_time)
            ].copy()

            if not new_history.empty:
                new_features = (
                    feature_map
                    .reindex(pd.to_datetime(new_history["time"]))
                    .reset_index()
                    .rename(columns={"index": "time"})
                )

                state.daily_update(new_history, new_features)

                last_distribution_update_time = update_cutoff

        # ----------------------------------------------------
        # 1. Check old unexecuted orders
        # ----------------------------------------------------

        still_active = []

        for order in active_orders:
            if check_fill(order, row):
                exec_rows.append(
                    make_execution_row(
                        order=order,
                        exec_time=t,
                        exec_idx=i,
                        exec_price=order["target_price"],
                        execution_state="delayed_execution",
                        tau=tau,
                    )
                )
            else:
                still_active.append(order)

        active_orders = still_active

        # ----------------------------------------------------
        # 2. If required position changes, create new order
        # ----------------------------------------------------

        dq = float(row["delta_q"])

        if dq != 0:
            side = "buy" if dq > 0 else "sell"
            qty = abs(dq)
            ref_price = float(row["close"])

            if t in feature_map.index:
                frow = feature_map.loc[t]

                k, prob, state_key = state.choose_k(frow, side)

                ema_volatility = frow["ema_volatility"]
                crossover_pct = frow["crossover_pct"]
            else:
                k, prob, state_key = 0, np.nan, "missing_feature"

                ema_volatility = np.nan
                crossover_pct = np.nan

            if side == "buy":
                target_price = ref_price - k * tick_size
            else:
                target_price = ref_price + k * tick_size

            order = make_order(
                signal_time=t,
                signal_idx=i,
                side=side,
                qty=qty,
                ref_price=ref_price,
                target_price=target_price,
                k=k,
                fill_prob_estimate=prob,
                state_key=state_key,
                ema_volatility=ema_volatility,
                crossover_pct=crossover_pct,
            )

            # ------------------------------------------------
            # 3. Check whether new order is filled during
            #    its first tau interval.
            #
            # Since the order is submitted after close_t,
            # the checking window is:
            #
            #     i+1, ..., i+bars_per_tau
            # ------------------------------------------------

            first_start = i + 1
            first_end = min(i + bars_per_tau, len(trading_df) - 1)

            first_window = trading_df.iloc[first_start:first_end + 1]

            filled_in_first_interval = False
            fill_idx = None
            fill_time = None

            for j, future_row in first_window.iterrows():
                if check_fill(order, future_row):
                    filled_in_first_interval = True
                    fill_idx = j
                    fill_time = future_row["time"]
                    break

            if filled_in_first_interval:
                exec_rows.append(
                    make_execution_row(
                        order=order,
                        exec_time=fill_time,
                        exec_idx=fill_idx,
                        exec_price=target_price,
                        execution_state="successful",
                        tau=tau,
                    )
                )
            else:
                # If not filled in the first tau interval,
                # it remains active and may be filled later.
                active_orders.append(order)

        previous_day = current_day
        previous_time = t

    # --------------------------------------------------------
    # Final distribution update is not necessary for trading,
    # because no future decision uses it.
    # --------------------------------------------------------

    # --------------------------------------------------------
    # Force close all remaining active orders
    # --------------------------------------------------------

    if len(trading_df) > 0:
        final_idx = len(trading_df) - 1
        final_time = trading_df["time"].iloc[-1]
        final_close = float(trading_df["close"].iloc[-1])

        for order in active_orders:
            exec_rows.append(
                make_execution_row(
                    order=order,
                    exec_time=final_time,
                    exec_idx=final_idx,
                    exec_price=final_close,
                    execution_state="forced_close",
                    tau=tau,
                )
            )

    trade_detail_df = pd.DataFrame(exec_rows)

    pnl_df = build_pnl_df(
        trading_df=trading_df,
        exec_df=trade_detail_df,
    )

    return pnl_df, trade_detail_df