import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
from src.utils.paths import RAW_DATA_DIR, DATA_DIR
from src.trade.regime_fixed_rule import regime_fixed


def main():
    raw_data_path=RAW_DATA_DIR
    futures_path=DATA_DIR/'futures_right'
    agent_path=DATA_DIR/'agent_right'
    result_path=DATA_DIR/'result'
    markets=sorted([p.name for p in raw_data_path.iterdir() if p.is_dir()])

    tick_size = 0.25
    tau_fill_prob={
        'EuroStoxx': [60,0.5],
        'GBP - British Pound': [5,0.9],
        'German Bunds - German Government Bonds': [5,0.9],
        'Gold':[20,0.8],
        'HeatingOil':[60,0.5],
        'JPY - Japanese Yen': [5,0.9],
        'Nasdaq':[10,0.9]
    }

    for market in markets:
        tau, fill_prob=tau_fill_prob[market]
        market_result_path = result_path / market
        market_result_path.mkdir(parents=True, exist_ok=True)
        data1=pd.read_parquet(futures_path/'minute_1'/f'{market}.parquet').reset_index(drop=True)
        data1_prep=pd.read_parquet(futures_path/'minute_1'/f'{market}_prep.parquet').reset_index(drop=True)
        data5=pd.read_parquet(futures_path/'minute_5'/f'{market}.parquet').reset_index(drop=True)
        data5_prep=pd.read_parquet(futures_path/'minute_5'/f'{market}_prep.parquet').reset_index(drop=True)
        agent_data=pd.read_parquet(agent_path/f'{market}.parquet').reset_index(drop=True)
        trading_data=data5[['time','open','high','low','close','volume']].merge(agent_data[['time','price','position']],on='time')


        pnl_df, trade_detail_df = regime_fixed(
            trading_data=trading_data,
            data1=data1,
            data5=data5,
            prep_data1=data1_prep,
            prep_data5=data5_prep,
            tau=tau,
            fill_prob=fill_prob,
            tick_size=tick_size,

            vol_ema_span=100,
            price_short_span=20,
            price_long_span=100,

            max_k=10,
            min_obs_per_regime=25,
            n_vol_regimes=3,
            n_cross_regimes=3,
            
        )
        
        # --------------------------------------------------------
        # Plot PnL
        # --------------------------------------------------------

        pnl_plot = pnl_df.copy()
        pnl_plot["agent_pnl_from_start"] = (pnl_plot["agent_pnl"] - pnl_plot["agent_pnl"].iloc[0])
        pnl_plot["strategy_pnl_from_start"] = (pnl_plot["strategy_pnl"] - pnl_plot["strategy_pnl"].iloc[0])
        pnl_plot["pnl_diff_from_start"] = (pnl_plot["pnl_diff"] - pnl_plot["pnl_diff"].iloc[0])

        plt.figure(figsize=(12, 5))
        plt.plot(pnl_plot["time"],pnl_plot["agent_pnl_from_start"],label="agent original pnl",color='blue')
        plt.plot(pnl_plot["time"],pnl_plot["strategy_pnl_from_start"],label="strategy pnl",color='red')
        plt.plot(pnl_plot["time"],pnl_plot["pnl_diff_from_start"],label="strategy - agent",color='orange')
        plt.title(f"{market} PnL from Start")
        plt.xlabel("Time")
        plt.ylabel("PnL")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(market_result_path / "pnl.png", dpi=300, bbox_inches="tight")
        plt.close()

        # --------------------------------------------------------
        # Plot position
        # --------------------------------------------------------

        plt.figure(figsize=(12, 4))
        plt.plot(pnl_df["time"],pnl_df["agent_position"],label="agent target position",color='blue')
        plt.plot(pnl_df["time"],pnl_df["strategy_position"],label="strategy executed position",color='red')
        plt.title(f"{market} Position")
        plt.xlabel("Time")
        plt.ylabel("Position")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(market_result_path / "position.png", dpi=300, bbox_inches="tight")
        plt.close()

        trade_detail_df.to_pickle(market_result_path / "trading_detail.pkl")

        # --------------------------------------------------------
        # Basic execution summary
        # --------------------------------------------------------

        print(f"\n===== {market} Summary =====")
        print("Final agent PnL:    ", pnl_df["agent_pnl"].iloc[-1])
        print("Final strategy PnL: ", pnl_df["strategy_pnl"].iloc[-1])
        print("Final PnL diff:     ", pnl_df["pnl_diff"].iloc[-1])
        print("Final PnL improve   ", f'{round(pnl_df["pnl_diff"].iloc[-1]/pnl_df["agent_pnl"].iloc[-1]*100,2)}%')

        print("\nExecution state counts:")
        print(trade_detail_df["execution_state"].value_counts(dropna=False))

        print("\nAverage delay by execution state:")
        print(trade_detail_df.groupby("execution_state")[["delay_5m_bars", "delay_tau_intervals"]].mean())
        


   
if __name__=='__main__':
    main()