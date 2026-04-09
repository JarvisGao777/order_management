from  __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

MONTH_CODES={
    "F": 1, "G": 2, "H": 3, "J": 4, "K": 5, "M": 6,
    "N": 7, "Q": 8, "U": 9, "V": 10, "X": 11, "Z": 12
}


@dataclass
class FuturesCleanConfig:
    data_dir:Path
    session_start_hour:int=18
    session_end_hour:int=17
    maintenance_end_hour:int=18
    
    # activity filters
    min_coverage: float=0.90    # discard day if active bars < 90% of expected
    min_rel_volume:float=0.05   # discard contract-days with volume < 5% of that contract's max daily volume
    min_rel_bars:float=0.50     # discard contract-days with bars < 50% of that contract's typical active-day bar count
    
    # resample output frequency
    out_freq:str='1min'
    
    
def parse_contract_from_filename(path:Path)->dict:
    """
    Example:
        NQH20.csv -> root='NQ', month_code='H', year=2020, expiry_month=3
    """
    stem=path.stem.upper()
    m=re.fullmatch(r"([A-Z]+)([FGHJKMNQUVXZ])(\d{1,2})", stem)
    if m is None:
        raise ValueError(f"Cannot parse futures contract from filename: {path.name}")
    
    root,month_code,yy=m.groups()
    yy=int(yy)
    year=2000+yy if yy<80 else 1990+yy
    expiry_month=MONTH_CODES[month_code]
    
    return {
        'contract':stem,
        'root':root,
        'month_code':month_code,
        'year':year,
        'expiry_month':expiry_month,
    }
    

def assign_trade_date(
    ts:pd.Series,
    sess
)
    


def load_data():
    pass

def clean_data():
    pass