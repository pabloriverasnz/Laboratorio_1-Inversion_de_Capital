
import os
import glob
import pandas as pd
import yfinance as yf
import numpy as np
import itertools
from datetime import datetime
import locale
locale.setlocale(locale.LC_TIME, 'es_ES')

def read_multiple_csv(path):
    
    'Función que lee multiples archivos csv descargados del NAFTRAC'
    
    all_files = glob.glob(os.path.join(path, "*.csv"))     # advisable to use os.path.join as this makes concatenation OS independent

    df_from_each_file = [pd.read_csv(f, skiprows = 2) for f in all_files]

    concatenated_df = pd.concat(df_from_each_file)
    
    concatenated_df.dropna(axis = 0, how = 'all', inplace = True)
    
    concatenated_df = concatenated_df[pd.notnull(concatenated_df['Peso (%)'])] # Quitar nulos en la columna de peso
 
    return all_files, concatenated_df


def get_constant_tickers(csv_files_df, r_ticker):
    
    all_tickers = csv_files_df.iloc[:,0].unique()
    
    c_tickers = [csv_files_df[csv_files_df['Ticker'] == i].iloc[0,0] + '.MX' for i in all_tickers if len(csv_files_df[csv_files_df['Ticker'] == i]) == 25]

    c_tickers = [s.replace('*', '') for s in c_tickers]
    c_tickers = [s.replace('LIVEPOLC.1', 'LIVEPOLC-1') for s in c_tickers]
    c_tickers.remove(r_ticker)
    c_tickers.sort()

    return c_tickers

def get_all_dates(all_files: 'dirección de todos los archivos'):
    
    # Obtenemos los nombres de la primer fila de cada archivo
    dates = [pd.read_csv(f, nrows = 1).columns for f in all_files]
    
    # Tomar solamente las fechas, si la columna tiene "unnamed" o "fecha" no se toma
    dates = [[x for x in dates[j] if not 'Unnamed' in x] for j in range(len(dates))]
    dates = [[x for x in dates[j] if not 'fecha' in x] for j in range(len(dates))]
    
    dates = [item for sublist in dates for item in sublist]

    return dates

def replace_list_value(the_list, to_replace, new_value):
    
    idx = the_list.index(to_replace)
    
    the_list[idx] = new_value
    
    return the_list

def str_to_datetime(date: str, current_format: str, new_format: str):
    
    change_format = datetime.strptime(date, current_format)
    
    given_format = change_format.strftime(new_format)
    
    return given_format

def get_ticker_prices(tickers, dates):
    
    data = [yf.download(tickers, start = i).iloc[0:1,len(tickers):len(tickers) * 2] for i in dates]

    prices = pd.concat(data)
    
    return prices


