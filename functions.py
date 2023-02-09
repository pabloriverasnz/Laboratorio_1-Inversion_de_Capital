
import os
import glob
import pandas as pd
import yfinance as yf
import numpy as np
import itertools
from datetime import datetime
import matplotlib.pyplot as plt
#import locale
#locale.setlocale(locale.LC_TIME, 'es_ES')
from scipy.optimize import minimize

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

def get_all_dates2(all_files: 'dirección de todos los archivos'):
    
    dates = [all_files[i][14:22] for i in range(len(all_files))]

    return dates

def replace_list_value(the_list, to_replace, new_value):
    
    idx = the_list.index(to_replace)
    
    the_list[idx] = new_value
    
    return the_list

def str_to_datetime(date: str, current_format: str, new_format: str):
    
    change_format = datetime.strptime(date, current_format)
    
    given_format = change_format.strftime(new_format)
    
    return given_format

def get_ticker_prices(tickers, start: 'Si date_format = 0, en lista', date_format, end = None, interval = None):
    
    if date_format == 0:
        data = [yf.download(tickers, start = i).iloc[0:1,len(tickers):len(tickers) * 2] for i in start]
        prices = pd.concat(data)
    elif date_format == 1:
        prices = yf.download(tickers, start = start, end = end, interval = interval).iloc[:,len(tickers):len(tickers) * 2]
        #prices = pd.concat(data)
    else:
        prices = 'Error, selecciona 0 o 1 en date_format'
        
    return prices

def get_weights(pond_file, tickers, cash_ticks):
    
    ponderaciones = pd.read_csv(pond_file, skiprows = 2).dropna()

    p2 = ponderaciones.iloc[:,0] + '.MX'
    p2 = [s.replace('*', '') for s in p2]
    p2 = [s.replace('LIVEPOLC.1', 'LIVEPOLC-1') for s in p2]
    ponderaciones['Ticker'] = p2
    
    ponderaciones.set_index('Ticker', inplace = True)
    
    global cash
    cash = ponderaciones.loc[cash_ticks, 'Peso (%)'] / 100
    
    pond = [ponderaciones.loc[i, 'Peso (%)'] for i in tickers]

    return pond

def cash():
    
    return cash

def capital_values(total_cap, cash_pond, ponds):
    
    cash_amount = total_cap * cash_pond.values
    invested_cap = sum(ponds) / 100 * total_cap
    remaining = total_cap - (invested_cap)
    
    df = pd.DataFrame(data = {'Cash': cash_amount,
                             'Invested': invested_cap,
                             'Not Invested': remaining - cash_amount,
                             'Total Not Invested': remaining},
                     index = ['Amount'])
    
    return df

def get_titulos(prices, date, comision, capital, ponderaciones):
    
    prices_w_comision = prices.loc[date].values * (1 + (comision))
    
    titulos = [int((i / 100) * capital) for i in ponderaciones] / prices_w_comision
    titulos = titulos.astype(int)
    
    return titulos

def monthly_perf_pasive(ticks, weights, prices, titulos, date: str, capital, res_df):

    df = pd.DataFrame(data = {'Tickers': ticks,
                             'Ponderación': weights,
                             f"Precio ({date})": prices,
                             'Títulos': titulos
                             })
    cap = capital

    money = [(w / 100) * cap for w in weights]
    
    #df['Títulos'] = (money / df['Precio']).astype(int)
    df['Valor posición'] = round(df[f"Precio ({date})"] * df['Títulos'], 2)
    #df['Restante'] = money - df['Valor posición']
    
    # Diferencia entre precio * títulos y capital x ponderacion
    res = [sum(money - df['Valor posición'])]
    res_df.loc[date] = res
     
    #capital_f = np.sum(df['Valor posición'])
    
    df = df.append(df.sum(numeric_only=True), ignore_index=True)
    
    return df# + capital_f#, sum(df['Valor posición'])

def get_pasive_capital_results(reports):
    
    res = [reports[i].iloc[-1,4] for i in range(len(reports))]

    return res

def display_pasive_results(pasive_results, dates, df_or_graph):
        
    df = pd.DataFrame(data = {'Capital': pasive_results},
                     index = dates)
    
    df['Rendimiento'] = df.iloc[:,0].pct_change()
    df['Rend. Acum'] = df.iloc[:,1].cumsum()
    
    if df_or_graph == 1:
        
        plt.figure(figsize = (12,8))
        plt.plot(df.iloc[:,0])
        plt.title('Capital Estrategia Pasiva')
        plt.xlabel('Fecha')
        plt.ylabel('Capital')
        plt.grid()
    
    return df if df_or_graph == 0 else None

def mkwtz_port(rf, prices):
    
    returns = np.log(1 + prices.pct_change()).dropna()

    summary = pd.DataFrame()
    summary ['Media'] = 12 * returns.mean()
    summary['Vol'] = np.sqrt(12) * returns.std()

    cov = 12 * returns.cov()

    Eind = summary.T.loc['Media']

    # Función obtener varianza portafolio
    def var(w, Sigma):
        return w.T.dot(Sigma).dot(w)
    
    # Número de activos
    N = len(Eind)
    # Dato inicial
    w0 = np.ones(N) / N
    # Cotas de las variables
    bnds = ((0,1), ) * N
    # Restricciones
    cons = {'type': 'eq', 'fun': lambda w:w.sum() - 1}
    
    # Función objetivo
    def menos_RS(w, Eind, rf, Sigma):
        E_port = Eind.T.dot(w)
        s_port = var(w, Sigma)**0.5
        RS = (E_port - rf) / s_port
        return -RS
    
    emv = minimize(fun=menos_RS, x0 = w0, args = (Eind, rf, cov), bounds = bnds, constraints = cons, tol = 1e-8)

    report = pd.DataFrame()
    report['Ticker'] = returns['Close'].columns
    report.set_index('Ticker', inplace = True)
    report['W'] = emv.x.round(6) * 100
    report['Precio'] = prices.iloc[-1,:].values
    #report = report[report['W'] != 0]
    
    return report