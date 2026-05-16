from __future__ import annotations

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
    df["delta_q"] = df["position"].diff().fillna(0.0)
    return df


def empty_exec_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "signal_time",
            "exec_time",
            "side",
            "qty",
            "delta_q",
            "exec_price",
            "fill_type",
            "placed_prices",
            "placed_times",
            "ideal_prices",
        ]
    )


def target_price(side: str, ref_price: float, k: int, tick_size: float) -> float:
    if side == "buy":
        return ref_price - k * tick_size
    else:
        return ref_price + k * tick_size


# ============================================================
# 2. PnL and stats
# ============================================================

def finalize_pnl(df: pd.DataFrame, exec_df: pd.DataFrame):
    if exec_df.empty:
        exec_agg = pd.DataFrame(columns=["time", "cash_flow", "delta_q_exec"])
    else:
        exec_agg = (
            exec_df.groupby("exec_time", as_index=False)
            .apply(
                lambda x: pd.Series(
                    {
                        "cash_flow": (-(x["delta_q"] * x["exec_price"])).sum(),
                        "delta_q_exec": x["delta_q"].sum(),
                    }
                ),
                include_groups=False,
            )
            .rename(columns={"exec_time": "time"})
        )

    out = df.merge(exec_agg, on="time", how="left")
    out["cash_flow"] = out["cash_flow"].fillna(0.0)
    out["delta_q_exec"] = out["delta_q_exec"].fillna(0.0)

    out["cash"] = out["cash_flow"].cumsum()
    out["position"] = out["delta_q_exec"].cumsum()
    out["pnl"] = out["cash"] + out["position"] * out["close"]

    pnl_df = out[["time", "cash_flow", "cash", "position", "pnl"]].copy()

    if exec_df.empty:
        exec_df = empty_exec_df()
    else:
        exec_df = exec_df[
            [
                "signal_time",
                "exec_time",
                "side",
                "qty",
                "delta_q",
                "exec_price",
                "fill_type",
                "placed_prices",
                "placed_times",
                "ideal_prices",
            ]
        ].reset_index(drop=True)

    return pnl_df.reset_index(drop=True), exec_df.reset_index(drop=True)


def compute_stats(
    pnl_df: pd.DataFrame,
    bars_per_year: int = 252 * 24 * 12,
) -> dict:
    pnl = pnl_df["pnl"].astype(float).reset_index(drop=True)
    pnl_change = pnl.diff().fillna(0.0)

    running_max = pnl.cummax()
    drawdown = pnl - running_max

    vol = pnl_change.std(ddof=1)

    sharpe = (
        np.sqrt(bars_per_year) * pnl_change.mean() / vol
        if vol > 0
        else np.nan
    )

    return {
        "start_time": pnl_df["time"].iloc[0] if len(pnl_df) > 0 else pd.NaT,
        "end_time": pnl_df["time"].iloc[-1] if len(pnl_df) > 0 else pd.NaT,
        "total_return": pnl.iloc[-1] if len(pnl) > 0 else np.nan,
        "max_drawdown": drawdown.min() if len(drawdown) > 0 else np.nan,
        "sharpe": sharpe,
    }


# ============================================================
# 3. Order helpers
# ============================================================

def make_order(
    signal_time,
    submit_time,
    submit_idx,
    side,
    qty,
    limit_price,
    ideal_price=None,
):
    return {
        "signal_time": signal_time,
        "submit_time": submit_time,
        "submit_idx": submit_idx,
        "side": side,
        "qty": qty,
        "delta_q": qty if side == "buy" else -qty,
        "limit_price": limit_price,
        "placed_prices": [limit_price],
        "placed_times": [submit_time],
        "ideal_prices": [ideal_price] if ideal_price is not None else None,
        "status": "active",
    }


def fill_order(order: dict, exec_time, exec_price, fill_type: str) -> dict:
    return {
        "signal_time": order["signal_time"],
        "exec_time": exec_time,
        "side": order["side"],
        "qty": order["qty"],
        "delta_q": order["delta_q"],
        "exec_price": exec_price,
        "fill_type": fill_type,
        "placed_prices": order["placed_prices"].copy(),
        "placed_times": order["placed_times"].copy(),
        "ideal_prices": (
            order["ideal_prices"].copy()
            if isinstance(order["ideal_prices"], list)
            else order["ideal_prices"]
        ),
    }


def is_limit_filled(row: pd.Series, order: dict) -> bool:
    if order["side"] == "buy":
        return pd.notna(row["low"]) and row["low"] <= order["limit_price"]
    else:
        return pd.notna(row["high"]) and row["high"] >= order["limit_price"]


def window_ideal_price(
    df: pd.DataFrame,
    start_idx: int,
    side: str,
    tau: int,
) -> float:
    bars_per_tau = tau // 5
    window = df.iloc[start_idx + 1 : min(start_idx + 1 + bars_per_tau, len(df))]

    if window.empty:
        return np.nan

    fallback_price = window["close"].iloc[-1]

    if side == "buy":
        valid = window.dropna(subset=["low"])
        return fallback_price if valid.empty else valid["low"].min()
    else:
        valid = window.dropna(subset=["high"])
        return fallback_price if valid.empty else valid["high"].max()


# ============================================================
# 4. Agent benchmark
# ============================================================

def agent_rule(
    trading_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Agent benchmark:
    execute immediately at agent price.
    """
    df = prepare_trading_data(trading_data)

    df["cash_flow"] = -df["delta_q"] * df["price"]
    df["cash"] = df["cash_flow"].cumsum()
    df["position"] = df["delta_q"].cumsum()
    df["pnl"] = df["cash"] + df["position"] * df["price"]

    exec_df = df.loc[df["delta_q"] != 0, ["time", "delta_q", "price"]].copy()

    if exec_df.empty:
        exec_df = empty_exec_df()
    else:
        exec_df["signal_time"] = exec_df["time"]
        exec_df["exec_time"] = exec_df["time"]
        exec_df["side"] = np.where(exec_df["delta_q"] > 0, "buy", "sell")
        exec_df["qty"] = exec_df["delta_q"].abs()
        exec_df["exec_price"] = exec_df["price"]
        exec_df["fill_type"] = "agent"
        exec_df["placed_prices"] = None
        exec_df["placed_times"] = None
        exec_df["ideal_prices"] = None

        exec_df = exec_df[
            [
                "signal_time",
                "exec_time",
                "side",
                "qty",
                "delta_q",
                "exec_price",
                "fill_type",
                "placed_prices",
                "placed_times",
                "ideal_prices",
            ]
        ].reset_index(drop=True)

    pnl_df = df[["time", "cash_flow", "cash", "position", "pnl"]].reset_index(drop=True)

    return pnl_df, exec_df


# ============================================================
# 5. Ideal benchmark
# ============================================================

def ideal_rule(
    trading_data: pd.DataFrame,
    tau: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Unrealistic ideal benchmark:
    - buy at the lowest price in the next tau interval
    - sell at the highest price in the next tau interval
    """
    validate_tau(tau)

    df = prepare_trading_data(trading_data)
    bars_per_tau = tau // 5

    exec_rows = []

    for i, row in df.iterrows():
        dq = row["delta_q"]

        if dq == 0:
            continue

        window = df.iloc[i + 1 : min(i + 1 + bars_per_tau, len(df))].copy()

        if window.empty:
            continue

        side = "buy" if dq > 0 else "sell"
        qty = abs(float(dq))

        fallback_idx = window.index[-1]
        fallback_price = df.loc[fallback_idx, "close"]

        if side == "buy":
            valid = window.dropna(subset=["low"])
            if valid.empty:
                exec_idx = fallback_idx
                exec_price = fallback_price
            else:
                exec_idx = valid["low"].idxmin()
                exec_price = df.loc[exec_idx, "low"]
        else:
            valid = window.dropna(subset=["high"])
            if valid.empty:
                exec_idx = fallback_idx
                exec_price = fallback_price
            else:
                exec_idx = valid["high"].idxmax()
                exec_price = df.loc[exec_idx, "high"]

        exec_rows.append(
            {
                "signal_time": row["time"],
                "exec_time": df.loc[exec_idx, "time"],
                "side": side,
                "qty": qty,
                "delta_q": dq,
                "exec_price": exec_price,
                "fill_type": "ideal",
                "placed_prices": None,
                "placed_times": None,
                "ideal_prices": None,
            }
        )

    exec_df = pd.DataFrame(exec_rows)

    if exec_df.empty:
        exec_df = empty_exec_df()

    return finalize_pnl(df, exec_df)


# ============================================================
# 6. Shared tick-limit engine
# ============================================================

def run_tick_limit_strategy(
    trading_data: pd.DataFrame,
    rule_type: str,
    tau: int = 10,
    k: int = 1,
    tick_size: float = 0.25,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    rule_type:

    reset:
        set target price.
        if not executed after tau, reset target price using latest close.

    fixed:
        set target price.
        if not executed after tau, keep using original target price.

    close:
        set target price.
        if not executed after tau, execute at current close price.

    prev_close:
        unrealistic rule.
        set target price.
        if not executed after tau, assume we know this in advance,
        execute immediately at the original close price.
    """
    validate_tau(tau)

    if rule_type not in {"reset", "fixed", "close", "prev_close"}:
        raise ValueError("rule_type must be one of: reset, fixed, close, prev_close.")

    df = prepare_trading_data(trading_data)

    active_orders = []
    exec_rows = []

    for i, row in df.iterrows():
        t = row["time"]
        ref_price = row["close"]

        # ----------------------------------------------------
        # 1. Check whether existing active orders are filled
        # ----------------------------------------------------
        for order in active_orders:
            if order["status"] != "active":
                continue

            if i <= order["submit_idx"]:
                continue

            if is_limit_filled(row, order):
                order["status"] = "filled"
                exec_rows.append(
                    fill_order(
                        order=order,
                        exec_time=t,
                        exec_price=order["limit_price"],
                        fill_type="limit",
                    )
                )

        # ----------------------------------------------------
        # 2. Handle orders that survive for tau minutes
        # ----------------------------------------------------
        for order in active_orders:
            if order["status"] != "active":
                continue

            if i <= order["submit_idx"]:
                continue

            elapsed = (t - order["submit_time"]).total_seconds() / 60.0

            if elapsed < tau:
                continue

            if rule_type == "reset":
                new_price = target_price(
                    side=order["side"],
                    ref_price=ref_price,
                    k=k,
                    tick_size=tick_size,
                )

                new_ideal = window_ideal_price(
                    df=df,
                    start_idx=i,
                    side=order["side"],
                    tau=tau,
                )

                order["limit_price"] = new_price
                order["submit_time"] = t
                order["submit_idx"] = i
                order["placed_prices"].append(new_price)
                order["placed_times"].append(t)

                if isinstance(order["ideal_prices"], list):
                    order["ideal_prices"].append(new_ideal)

            elif rule_type == "fixed":
                order["submit_time"] = t
                order["submit_idx"] = i
                order["placed_prices"].append(order["limit_price"])
                order["placed_times"].append(t)

            elif rule_type == "close":
                order["status"] = "filled"
                order["placed_prices"].append(ref_price)
                order["placed_times"].append(t)

                exec_rows.append(
                    fill_order(
                        order=order,
                        exec_time=t,
                        exec_price=ref_price,
                        fill_type="tau_close",
                    )
                )

            elif rule_type == "prev_close":
                # Should not happen for prev_close because this rule executes
                # immediately when the first tau-window miss is known in advance.
                pass

        # ----------------------------------------------------
        # 3. Create new order from target position change
        # ----------------------------------------------------
        dq = row["delta_q"]

        if dq == 0:
            continue

        side = "buy" if dq > 0 else "sell"
        qty = abs(float(dq))
        px_target = target_price(side, ref_price, k, tick_size)

        if rule_type == "prev_close":
            bars_per_tau = tau // 5
            window = df.iloc[i + 1 : i + 1 + bars_per_tau].copy()

            if side == "buy":
                hit = window[window["low"].notna() & (window["low"] <= px_target)]
            else:
                hit = window[window["high"].notna() & (window["high"] >= px_target)]

            if not hit.empty:
                exec_idx = hit.index[0]
                exec_time = df.loc[exec_idx, "time"]
                exec_price = px_target
                fill_type = "limit"
            else:
                exec_time = t
                exec_price = ref_price
                fill_type = "prev_close"

            exec_rows.append(
                {
                    "signal_time": t,
                    "exec_time": exec_time,
                    "side": side,
                    "qty": qty,
                    "delta_q": dq,
                    "exec_price": exec_price,
                    "fill_type": fill_type,
                    "placed_prices": [px_target],
                    "placed_times": [t],
                    "ideal_prices": None,
                }
            )

            continue

        ideal_price = None

        if rule_type == "reset":
            ideal_price = window_ideal_price(
                df=df,
                start_idx=i,
                side=side,
                tau=tau,
            )

        active_orders.append(
            make_order(
                signal_time=t,
                submit_time=t,
                submit_idx=i,
                side=side,
                qty=qty,
                limit_price=px_target,
                ideal_price=ideal_price,
            )
        )

    # --------------------------------------------------------
    # 4. Force close unfinished orders at final close
    # --------------------------------------------------------
    if len(df) > 0:
        final_time = df["time"].iloc[-1]
        final_close = df["close"].iloc[-1]

        for order in active_orders:
            if order["status"] != "active":
                continue

            order["status"] = "filled"
            order["placed_prices"].append(final_close)
            order["placed_times"].append(final_time)

            if isinstance(order["ideal_prices"], list):
                order["ideal_prices"].append(final_close)

            exec_rows.append(
                fill_order(
                    order=order,
                    exec_time=final_time,
                    exec_price=final_close,
                    fill_type="forced_close",
                )
            )

    exec_df = pd.DataFrame(exec_rows)

    if exec_df.empty:
        exec_df = empty_exec_df()

    return finalize_pnl(df, exec_df)


# ============================================================
# 7. Public tick rules
# ============================================================

def tick_reset(
    trading_data: pd.DataFrame,
    tau: int = 10,
    k: int = 1,
    tick_size: float = 0.25,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Set target price first.
    If not executed after tau, reset target price using latest close.
    """
    return run_tick_limit_strategy(
        trading_data=trading_data,
        rule_type="reset",
        tau=tau,
        k=k,
        tick_size=tick_size,
    )


def tick_fixed(
    trading_data: pd.DataFrame,
    tau: int = 10,
    k: int = 1,
    tick_size: float = 0.25,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Set target price first.
    If not executed after tau, keep using the original target price.
    """
    return run_tick_limit_strategy(
        trading_data=trading_data,
        rule_type="fixed",
        tau=tau,
        k=k,
        tick_size=tick_size,
    )


def tick_close(
    trading_data: pd.DataFrame,
    tau: int = 10,
    k: int = 1,
    tick_size: float = 0.25,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Set target price first.
    If not executed after tau, execute at current close price.
    """
    return run_tick_limit_strategy(
        trading_data=trading_data,
        rule_type="close",
        tau=tau,
        k=k,
        tick_size=tick_size,
    )


def tick_prev_close(
    trading_data: pd.DataFrame,
    tau: int = 10,
    k: int = 1,
    tick_size: float = 0.25,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Unrealistic benchmark.
    Set target price first.
    If the order will not be executed in the next tau interval,
    assume we know this in advance and execute immediately at the original close.
    """
    return run_tick_limit_strategy(
        trading_data=trading_data,
        rule_type="prev_close",
        tau=tau,
        k=k,
        tick_size=tick_size,
    )