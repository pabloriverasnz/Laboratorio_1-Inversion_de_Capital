
import os
import glob
import pandas as pd
import yfinance as yf
import numpy as np
import itertools
from datetime import datetime

def read_multiple_csv(path):
    
    'Función que lee multiples archivos csv descargados del NAFTRAC'
    
    all_files = glob.glob(os.path.join(path, "*.csv"))     # advisable to use os.path.join as this makes concatenation OS independent

    df_from_each_file = [pd.read_csv(f, skiprows = 2) for f in all_files]

    concatenated_df = pd.concat(df_from_each_file)
    
    concatenated_df.dropna(axis = 0, how = 'all', inplace = True)
    
    concatenated_df = concatenated_df[pd.notnull(concatenated_df['Peso (%)'])] # Quitar nulos en la columna de peso
 
    return concatenated_df


def get_constant_tickers(csv_files_df, r_ticker):
    
    all_tickers = csv_files_df.iloc[:,0].unique()
    
    c_tickers = [csv_files_df[csv_files_df['Ticker'] == i].iloc[0,0] + '.MX' for i in all_tickers if len(csv_files_df[csv_files_df['Ticker'] == i]) == 25]

    c_tickers = [s.replace('*', '') for s in c_tickers]
    c_tickers = [s.replace('LIVEPOLC.1', 'LIVEPOLC-1') for s in c_tickers]
    c_tickers.remove(r_ticker)
    c_tickers.sort()

    return c_tickers