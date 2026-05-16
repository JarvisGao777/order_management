import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
import re

import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.utils.paths import RAW_DATA_DIR, DATA_DIR
from src.data.data import *

def save_data(
    raw_data_path:Path=RAW_DATA_DIR,
    agent_output_path:Path=DATA_DIR/'agent_right',
    futures_output_path:Path=DATA_DIR/'futures_right',
):
    
    agent_output_path.mkdir(parents=True,exist_ok=True)
    futures_1min=futures_output_path/'minute_1'
    futures_5min=futures_output_path/'minute_5'
    futures_1min.mkdir(parents=True,exist_ok=True)
    futures_5min.mkdir(parents=True,exist_ok=True)


    markets=sorted([p.name for p in raw_data_path.iterdir() if p.is_dir()])
    for market in markets:
        print(f'======== {market} ========')
        market_path=raw_data_path/market
        
        # files
        files=sorted([p.name for p in market_path.glob("*.csv")
                    if not p.name.startswith("AIAgent")])
        ai_file=[p.name for p in market_path.glob("*.csv")
                    if p.name.startswith("AIAgent")][0]

        # futures_data
        futures_dict=dict()
        for f in files:
            data=pd.read_csv(RAW_DATA_DIR/market/f,header=None)
            data=data.dropna() # delete rows with missing value, this doesn't change the results here
            data.columns=['time','open','high','low','close','volume']
            data['time']=pd.to_datetime(data['time'])+ pd.Timedelta(hours=6) # change to trading clock
            data = data[data["time"].dt.dayofweek < 5]                    # delete Saturdays and Sundays
            data['hour']=(data['time'].dt.hour).astype(int)
            data=data[data['hour']!=23]                                   # delete breaking time
            data=data[['time','open','high','low','close','volume']]
            futures_dict[f[:-4]]=data
        
        # agent_data
        agent_df=pd.read_csv(RAW_DATA_DIR/market/ai_file,header=None)
        agent_df.columns=['date','hour','minute','price','position']
        agent_df['date']=pd.to_datetime(agent_df['date'], unit='D', origin='1899-12-30')
        agent_df['time'] = (agent_df['date']
                            + pd.to_timedelta(agent_df['hour'], unit='h') 
                            + pd.to_timedelta(agent_df['minute'], unit='m'))
        agent_df['time']=pd.to_datetime(agent_df['time']) + pd.Timedelta(hours=6) # change to trading clock
        agent_df = agent_df[agent_df["time"].dt.dayofweek < 5]                    # delete Saturdays and Sundays
        agent_df['hour']=(agent_df['time'].dt.hour).astype(int)
        # delete breaking time
        agent_df = agent_df[
            ~(
                ((agent_df["time"].dt.hour == 23) & (agent_df["time"].dt.minute >= 5))
                | ((agent_df["time"].dt.hour == 0) & (agent_df["time"].dt.minute == 0))
            )
        ] 
        
        # result
        agent=identify_agent_contracts(agent_df,futures_dict)
        agent["front_rank"] = agent["chosen_contract_rank"].ffill()
        agent["front_rank"] = agent["front_rank"].fillna(1.0).astype(int)
        rank_to_contract = {
            rank: contract
            for contract, rank in build_contract_rank_map(futures_dict).items()
        }
        agent["front_contract"] = agent["front_rank"].map(rank_to_contract)

        front_5m = build_front_5min_from_agent(agent, futures_dict)
        front_1m = build_front_1min_from_agent(agent, futures_dict)
        prep_5m = build_prep_5min_from_agent(agent, futures_dict)
        prep_1m = build_prep_1min_from_agent(agent, futures_dict)
        
        front_5m=front_5m.reset_index(drop=True)
        front_1m=front_1m.reset_index(drop=True)
        prep_5m=prep_5m.reset_index(drop=True)
        prep_1m=prep_1m.reset_index(drop=True)
        agent=agent.reset_index(drop=True)
        
        front_5m.to_parquet(futures_5min/f'{market}.parquet')
        front_1m.to_parquet(futures_1min/f'{market}.parquet')
        prep_5m.to_parquet(futures_5min / f"{market}_prep.parquet")
        prep_1m.to_parquet(futures_1min / f"{market}_prep.parquet")
        agent.to_parquet(agent_output_path/f'{market}.parquet')


    
# solve the problem of HeatingOil

def fix_heatingoil(
    raw_path:Path,
    vr_path:Path,
    ar_path:Path,
):
    
    market='HeatingOil'
    
    vr_1=pd.read_parquet(vr_path/'minute_1'/f'{market}.parquet')
    vr_5=pd.read_parquet(vr_path/'minute_5'/f'{market}.parquet')
    ar_trading_1=pd.read_parquet(ar_path/'minute_1'/f'{market}.parquet')
    ar_trading_5=pd.read_parquet(ar_path/'minute_5'/f'{market}.parquet')
    ar_prep_1=pd.read_parquet(ar_path/'minute_1'/f'{market}_prep.parquet')
    ar_prep_5=pd.read_parquet(ar_path/'minute_5'/f'{market}_prep.parquet')
    
    # change trading
    ar_trading_1['rank']=ar_trading_1['rank'].replace(1,4)
    ar_trading_5['rank']=ar_trading_5['rank'].replace(1,4)
    # chanege prep
    # 1.
    ar_prep_1=vr_1[vr_1['time']<ar_trading_1['time'].iloc[0]]
    ar_prep_5=vr_5[vr_5['time']<ar_trading_5['time'].iloc[0]]
    # 2.
    ar_prep_1=ar_prep_1.drop('date',axis=1)
    ar_prep_5=ar_prep_5.drop(['date','n_rows'],axis=1)
    # 3.
    ar_prep_1=ar_prep_1.rename(columns={'contract':'rank'})
    ar_prep_5=ar_prep_5.rename(columns={'contract':'rank'})
    # 4.
    market_path=raw_path/market
    files=sorted([p.name for p in market_path.glob("*.csv")
                if not p.name.startswith("AIAgent")])
    files=[f[:-4] for f in files]
    rank_to_contract = dict(enumerate(files))
    ar_prep_1['contract']=ar_prep_1['rank'].map(rank_to_contract)
    ar_prep_1['rank']=ar_prep_1['rank']+1
    ar_prep_5['contract']=ar_prep_5['rank'].map(rank_to_contract)
    ar_prep_5['rank']=ar_prep_5['rank']+1
    # 5.
    column_seq=['time','open','high','low','close','volume','contract','rank']
    ar_prep_1=ar_prep_1[column_seq]
    ar_prep_5=ar_prep_5[column_seq]
    
    ar_prep_1.to_parquet(ar_path/'minute_1'/f'{market}_prep.parquet')
    ar_prep_5.to_parquet(ar_path/'minute_5'/f'{market}_prep.parquet')
    ar_trading_1.to_parquet(ar_path/'minute_1'/f'{market}.parquet')
    ar_trading_5.to_parquet(ar_path/'minute_5'/f'{market}.parquet')
    


def main():
    raw_path=RAW_DATA_DIR
    agent_output_path=DATA_DIR/'agent_right'
    futures_output_path=DATA_DIR/'futures_right'
    save_data(raw_path, agent_output_path, futures_output_path)
    
    vr_path=DATA_DIR/'futures'
    ar_path=DATA_DIR/'futures_right'
    fix_heatingoil(raw_path,vr_path,ar_path)
    
    
if __name__=='__main__':
    main()