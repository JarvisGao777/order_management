import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
import re

import warnings
warnings.filterwarnings

sys.path.append(os.path.abspath(".."))
from src.utils.paths import RAW_DATA_DIR, DATA_DIR
from src.utils.month_codes import MONTH_CODE



def parse_contract_maturity(contract_name: str) -> pd.Timestamp:
    contract_name = str(contract_name).upper()
    m = re.search(r'([FGHJKMNQUVXZ])(\d{2})$', contract_name)
    if m is None:
        raise ValueError(f"Cannot parse maturity from contract name: {contract_name}")
    
    month = MONTH_CODE[m.group(1)]
    year = 2000 + int(m.group(2))
    return pd.Timestamp(year=year, month=month, day=1)


def build_contract_rank_map(futures_dict: dict) -> dict:
    contract_names = list(futures_dict.keys())
    maturity_pairs = [(name, parse_contract_maturity(name)) for name in contract_names]
    maturity_pairs = sorted(maturity_pairs, key=lambda x: x[1])
    
    rank_map = {name: i + 1 for i, (name, _) in enumerate(maturity_pairs)}
    return rank_map


def resample_one_contract_to_5min(
    df: pd.DataFrame, 
    time_col: str = "time",
) -> pd.DataFrame:
    """
    Resample one contract's 1-minute dataframe to 5-minute left-inclusive bars.
    """
    x = df.copy()
    x[time_col] = pd.to_datetime(x[time_col])
    x = x.sort_values(time_col).set_index(time_col)

    out = x.resample(
        "5min",
        label="right",
        closed="left"
    ).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    })

    out = out.dropna(subset=["close"]).reset_index()
    return out


def resample_all_contracts(futures_dict: dict) -> dict:
    futures_5m_dict = {}
    for contract_name, df in futures_dict.items():
        futures_5m_dict[contract_name] = resample_one_contract_to_5min(df)
    return futures_5m_dict


def identify_agent_contracts(
    agent_df: pd.DataFrame,
    futures_dict: dict,
    price_tol: float = 0.0,
    time_col: str = "time",
    agent_price_col: str = "price"
) -> pd.DataFrame:

    agent = agent_df.copy()
    agent[time_col] = pd.to_datetime(agent[time_col])
    agent = agent.sort_values(time_col).reset_index(drop=True)

    # build rank map
    rank_map = build_contract_rank_map(futures_dict)

    # resample all contracts to 5min
    futures_5m_dict = resample_all_contracts(futures_dict)
    
    # same last timestamp of agent & futures data
    last_agent_timestamp = agent[time_col].max()
    last_futures_timestamp = max(
        df_5m[time_col].max()
        for df_5m in futures_5m_dict.values()
        if len(df_5m) > 0
    )
    last_timestamp = min(last_agent_timestamp, last_futures_timestamp)
    agent = agent[agent[time_col] <= last_timestamp].copy()

    # build quick lookup for each contract:
    lookup_dict = {}
    for contract_name, df_5m in futures_5m_dict.items():
        temp = df_5m.copy()
        temp = temp[temp[time_col] <= last_timestamp]
        temp = temp.set_index(time_col)[["close", "volume"]]
        lookup_dict[contract_name] = temp
        
    # to store result row by row
    matched_contracts_all = []
    matched_ranks_all = []
    matched_volumes_all = []
    match_status_all=[]

    # first iterate: agent
    for _, row in agent.iterrows():
        t = row[time_col]
        p = row[agent_price_col]

        matched_contracts = []
        matched_ranks = []
        matched_volumes = []
        time_found = False

        # second iterate: future contracts
        for contract_name, lookup_df in lookup_dict.items():
            if t not in lookup_df.index:
                continue
            
            time_found = True

            close_price = lookup_df.at[t, "close"]
            vol = lookup_df.at[t, "volume"]

            if price_tol == 0.0:
                is_match = (p == close_price)
            else:
                is_match = abs(p - close_price) <= price_tol

            if is_match:
                matched_contracts.append(contract_name)
                matched_ranks.append(rank_map[contract_name])
                matched_volumes.append(vol)
                
        if len(matched_contracts) > 0:
            match_status_all.append("done")
        elif not time_found:
            match_status_all.append("time_not_found")
        else:
            match_status_all.append("no_price")

        matched_contracts_all.append(matched_contracts)
        matched_ranks_all.append(matched_ranks)
        matched_volumes_all.append(matched_volumes)

    agent["matched_contracts"] = matched_contracts_all
    agent["matched_contract_ranks"] = matched_ranks_all
    agent["matched_contract_volumes"] = matched_volumes_all
    agent["match_status"] = match_status_all

    # choose one by largest volume if multiple contracts match
    chosen_contract = []
    chosen_rank = []
    chosen_volume = []

    for contracts, ranks, vols in zip(
        agent["matched_contracts"],
        agent["matched_contract_ranks"],
        agent["matched_contract_volumes"]
    ):
        if len(contracts) == 0:
            chosen_contract.append(np.nan)
            chosen_rank.append(np.nan)
            chosen_volume.append(np.nan)
        elif len(contracts) == 1:
            chosen_contract.append(contracts[0])
            chosen_rank.append(ranks[0])
            chosen_volume.append(vols[0])
        else:
            idx = int(np.argmax(vols))
            chosen_contract.append(contracts[idx])
            chosen_rank.append(ranks[idx])
            chosen_volume.append(vols[idx])

    agent["chosen_contract"] = chosen_contract
    agent["chosen_contract_rank"] = chosen_rank
    agent["chosen_contract_volume"] = chosen_volume

    return agent


def build_front_5min_from_agent(
    agent: pd.DataFrame,
    futures_dict: dict,
    time_col: str = "time",
    agent_price_col: str = "price"
) -> pd.DataFrame:

    futures_5m_dict = resample_all_contracts(futures_dict)

    lookup_5m = {}
    for contract, df in futures_5m_dict.items():
        temp = df.copy()
        temp[time_col] = pd.to_datetime(temp[time_col])
        temp = temp.set_index(time_col)
        lookup_5m[contract] = temp

    rows = []

    for _, row in agent.iterrows():
        t = row[time_col]
        p_agent = row[agent_price_col]
        contract = row["front_contract"]
        status = row["match_status"]

        out = {
            time_col: t,
            "contract": contract,
            "rank": row["front_rank"],
            "match_status": status,
        }

        if contract in lookup_5m and t in lookup_5m[contract].index:
            bar = lookup_5m[contract].loc[t]

            if status == "done":
                out.update({
                    "open": bar["open"],
                    "high": bar["high"],
                    "low": bar["low"],
                    "close": bar["close"],
                    "volume": bar["volume"],
                })
            else:
                out.update({
                    "open": np.nan,
                    "high": np.nan,
                    "low": np.nan,
                    "close": p_agent,
                    "volume": np.nan,
                })
        else:
            out.update({
                "open": np.nan,
                "high": np.nan,
                "low": np.nan,
                "close": p_agent,
                "volume": np.nan,
            })

        rows.append(out)

    return pd.DataFrame(rows)


def build_front_1min_from_agent(
    agent: pd.DataFrame,
    futures_dict: dict,
    time_col: str = "time",
    agent_price_col: str = "price"
) -> pd.DataFrame:

    lookup_1m = {}

    for contract, df in futures_dict.items():
        temp = df.copy()
        temp[time_col] = pd.to_datetime(temp[time_col])
        temp = temp.sort_values(time_col).set_index(time_col)
        lookup_1m[contract] = temp

    rows = []

    for _, row in agent.iterrows():
        t = row[time_col]
        p_agent = row[agent_price_col]
        contract = row["front_contract"]
        status = row["match_status"]

        interval_times = pd.date_range(
            start=t - pd.Timedelta(minutes=5),
            end=t - pd.Timedelta(minutes=1),
            freq="1min"
        )

        if status == "done" and contract in lookup_1m:
            contract_1m = lookup_1m[contract]
            bars = contract_1m.reindex(interval_times).reset_index()
            bars = bars.rename(columns={"index": time_col})

            bars["contract"] = contract
            bars["rank"] = row["front_rank"]
            bars["agent_time"] = t
            bars["match_status"] = status

        else:
            bars = pd.DataFrame({
                time_col: interval_times,
                "open": np.nan,
                "high": np.nan,
                "low": np.nan,
                "close": np.nan,
                "volume": np.nan,
                "contract": contract,
                "rank": row["front_rank"],
                "agent_time": t,
                "match_status": status,
            })

            # last 1-min bar close equals agent price
            bars.loc[bars[time_col] == t - pd.Timedelta(minutes=1), "close"] = p_agent

        rows.append(bars)

    return pd.concat(rows, ignore_index=True)


def build_prep_5min_from_agent(
    agent: pd.DataFrame,
    futures_dict: dict,
    time_col: str = "time",
) -> pd.DataFrame:
    if agent.empty:
        return pd.DataFrame(
            columns=[time_col, "open", "high", "low", "close", "volume", "contract", "rank"]
        )

    first_agent_timestamp, first_front_contract, first_front_rank, prep_start_date = (
        get_prep_window_info(agent, futures_dict, time_col=time_col)
    )

    contract_5m = resample_one_contract_to_5min(
        futures_dict[first_front_contract], time_col=time_col
    )
    prep = contract_5m[
        (contract_5m[time_col] < first_agent_timestamp)
        & (contract_5m[time_col].dt.normalize() >= prep_start_date)
    ].copy()
    prep["contract"] = first_front_contract
    prep["rank"] = first_front_rank

    return prep[[time_col, "open", "high", "low", "close", "volume", "contract", "rank"]]


def build_prep_1min_from_agent(
    agent: pd.DataFrame,
    futures_dict: dict,
    time_col: str = "time",
) -> pd.DataFrame:
    if agent.empty:
        return pd.DataFrame(
            columns=[time_col, "open", "high", "low", "close", "volume", "contract", "rank"]
        )

    first_agent_timestamp, first_front_contract, first_front_rank, prep_start_date = (
        get_prep_window_info(agent, futures_dict, time_col=time_col)
    )

    prep = futures_dict[first_front_contract].copy()
    prep[time_col] = pd.to_datetime(prep[time_col])
    prep = prep.sort_values(time_col)
    prep = prep[
        (prep[time_col] < first_agent_timestamp)
        & (prep[time_col].dt.normalize() >= prep_start_date)
    ].copy()
    prep["contract"] = first_front_contract
    prep["rank"] = first_front_rank

    return prep[[time_col, "open", "high", "low", "close", "volume", "contract", "rank"]]


def get_prep_window_info(
    agent: pd.DataFrame,
    futures_dict: dict,
    time_col: str = "time",
):
    agent_sorted = agent.sort_values(time_col)
    first_agent_timestamp = pd.to_datetime(agent_sorted[time_col]).min()
    first_front_contract = agent_sorted["front_contract"].iloc[0]
    first_front_rank = int(agent_sorted["front_rank"].iloc[0])

    contract_df = futures_dict[first_front_contract].copy()
    contract_df[time_col] = pd.to_datetime(contract_df[time_col])
    contract_df = contract_df.sort_values(time_col)
    contract_df["trade_date"] = contract_df[time_col].dt.normalize()

    daily_volume = contract_df.groupby("trade_date", as_index=False)["volume"].sum()
    volume_threshold = daily_volume["volume"].max() / 10.0
    liquid_days = daily_volume[daily_volume["volume"] >= volume_threshold]

    if liquid_days.empty:
        prep_start_date = contract_df["trade_date"].min()
    else:
        prep_start_date = liquid_days["trade_date"].iloc[0]

    return first_agent_timestamp, first_front_contract, first_front_rank, prep_start_date