
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

def read_multiple_csv(path: 'dirección de los archivos'):
    
    'Función que lee multiples archivos csv descargados del NAFTRAC almacenados en una carpeta'
    
    # Obtener el nombre de todos los archivos
    all_files = glob.glob(os.path.join(path, "*.csv"))
    
    # Leer cada uno de los archivos de acuerdo con los nombres 
    df_from_each_file = [pd.read_csv(f, skiprows = 2) for f in all_files]

    # Concatenar todos los DataFrames individuales ya leídos
    concatenated_df = pd.concat(df_from_each_file)
    
    # Eliminar filas que sean puros "0"
    concatenated_df.dropna(axis = 0, how = 'all', inplace = True)
    
    # Además de eliminar los 0, solamente obtener los tickers que tienen un peso como dato
    concatenated_df = concatenated_df[pd.notnull(concatenated_df['Peso (%)'])] # Quitar nulos en la columna de peso
 
    return all_files, concatenated_df


def get_constant_tickers(csv_files_df: 'DataFrame de todos los csv concatenados', r_ticker: '1 ticker a eliminar (cash)'):
                                                                                  # Faltaría implementar para varios tickers
    
    'Función que obtiene los tickers que se repiten (constantes) en todos los archivos a leer, es decir, los que siempre aparecen o no salen del NAFTRAC'
    
    # Tickers en todo el archivo (valores únicos)
    all_tickers = csv_files_df.iloc[:,0].unique()
    
    # Dar formato a cada ticker para poder descargar de yf y obtener los que se repiten 25 veces (25 archivos concatenados)
    c_tickers = [csv_files_df[csv_files_df['Ticker'] == i].iloc[0,0] + '.MX' for i in all_tickers if len(csv_files_df[csv_files_df['Ticker'] == i]) == 25]

    # Formato
    c_tickers = [s.replace('*', '') for s in c_tickers]
    c_tickers = [s.replace('LIVEPOLC.1', 'LIVEPOLC-1') for s in c_tickers]

    # Eliminar ticker anteriormente especificado
    c_tickers.remove(r_ticker) 
    
    # Ordenar por orden alfabético
    c_tickers.sort()

    return c_tickers

def get_all_dates(all_files: 'dirección de todos los archivos'):
    
    'Función que obtiene las fechas de cada archivo, utilizando locale y obteniéndola de la primer fila de cada archivo'
    
    # Obtenemos los nombres de la primer fila de cada archivo
    dates = [pd.read_csv(f, nrows = 1).columns for f in all_files]
    
    # Tomar solamente las fechas, si la columna tiene "unnamed" o "fecha" no se toma
    dates = [[x for x in dates[j] if not 'Unnamed' in x] for j in range(len(dates))]
    dates = [[x for x in dates[j] if not 'fecha' in x] for j in range(len(dates))]
    
    # Quitar lista de listas
    dates = [item for sublist in dates for item in sublist]

    return dates

def get_all_dates2(all_files: 'dirección de todos los archivos'):
    
    'Función que obtiene las fechas de cada archivo, sin utilizar locale y obteniéndola del nombre del archivo'
    
    # Obtenemos únicamente la fecha del nombre de cada archivo
    dates = [all_files[i][14:22] for i in range(len(all_files))]

    return dates

def replace_list_value(the_list: 'lista con valor a reemplazar', to_replace: 'valor a reemplazar', new_value: 'nuevo valor'):

    'Función que reemplaza un valor de una lista por otro valor deseado'
    
    # Índice del elemento a cambiar
    idx = the_list.index(to_replace)
    
    # Sustitución de valor
    the_list[idx] = new_value
    
    return the_list

def str_to_datetime(date: str, current_format: str, new_format: str):
    
    # Formato actual de las fechas a cambiar
    change_format = datetime.strptime(date, current_format)
    
    # Formato nuevo 
    given_format = change_format.strftime(new_format)
    
    return given_format

def get_ticker_prices(tickers: 'tickers', start: 'Si date_format = 0, en lista', date_format: '0 para 1 fecha deseada, 1 para un rango de fechas', end = None, interval = None):
    
    'Función que obtiene los precios de los tickers deseados en el rango o en el día deseado'
    
    if date_format == 0: # Si se elige únicamente una fecha, toma la primer fila de los precios
        data = [yf.download(tickers, start = i, progress=False).iloc[0:1,len(tickers):len(tickers) * 2] for i in start]
        prices = pd.concat(data)
    elif date_format == 1: # Si se elige un rango de fechas, se incluye end
        prices = yf.download(tickers, start = start, end = end, interval = interval, progress=False).iloc[:,len(tickers):len(tickers) * 2]
        #prices = pd.concat(data)
    else:
        prices = 'Error, selecciona 0 o 1 en date_format'
        
    return prices

def get_weights(pond_file: 'archivo donde se encuentran las ponderaciones iniciales', tickers: 'tickers de activos', cash_ticks: 'ticker de activo designado a cash'):
    
    'Función que obtiene los pesos iniciales del portafolio NAFTRAC y da formato a los tickers'
    
    # Leer las ponderaciones del archivo NAFTRAC
    ponderaciones = pd.read_csv(pond_file, skiprows = 2).dropna()

    # Modificaciones al formato de los tickers
    p2 = ponderaciones.iloc[:,0] + '.MX'
    p2 = [s.replace('*', '') for s in p2]
    p2 = [s.replace('LIVEPOLC.1', 'LIVEPOLC-1') for s in p2]
    ponderaciones['Ticker'] = p2
    
    ponderaciones.set_index('Ticker', inplace = True)
    
    # Ponderación del cash
    global cash
    cash = ponderaciones.loc[cash_ticks, 'Peso (%)'] / 100
    
    # Ponderaciones de los tickers seleccionados
    pond = [ponderaciones.loc[i, 'Peso (%)'] for i in tickers]

    return pond

def cash():
    
    'Variable de cash contenida en la función de get_weights'
    
    return cash

def capital_values(total_cap: 'capital total inicial', cash_pond: 'ponderación de tickers designados a cash', ponds: 'ponderaciones de los tickers'):
    
    'Función que regresa información del capital total invertido'
    
    # Datos del cash actual
    cash_amount = total_cap * cash_pond.values
    invested_cap = sum(ponds) / 100 * total_cap
    remaining = total_cap - (invested_cap)
    
    df = pd.DataFrame(data = {'Cash': cash_amount,
                             'Invested': invested_cap,
                             'Not Invested': remaining - cash_amount,
                             'Total Not Invested': remaining},
                     index = ['Amount'])
    
    return df

def get_titulos(prices: 'df de precios', date: 'Fecha de los precios', comision: 'comisión por compra/venta activos', capital: 'capital a invertir', ponderaciones: 'ponderaciones de tickers'):
    
    'Función que obtiene los títulos a comprar de cada activo de acuerdo con las ponderaciones y tomando en cuenta el cash disponible para comprar a los precios con la comisión incluida'
    
    # Obtenemos los precios de la fecha indicada con comisión
    prices_w_comision = prices.loc[date].values * (1 + (comision))
    
    # Cantidad entera de títulos que se alcanzan a comprar con el capital dado
    titulos = [int((i / 100) * capital) for i in ponderaciones] / prices_w_comision
    titulos = titulos.astype(int)
    
    # Variable global del primer pago de comisión
    global comision_inicial
    comision_inicial = ((prices_w_comision - prices.loc[date].values) * titulos).sum()
    
    return titulos

def monthly_perf_pasive(ticks: 'tickers de los activos', weights: 'ponderaciones de activos', prices: 'df de precios', titulos: 'títulos a comprar de cada activo', date: str, capital: 'capital a invertir', res_df: 'df donde se almacena el movimiento del cash disponible'):
    
    'Función que obtiene el rendimiento del portafolio de administración pasiva en un periodo de tiempo dado. Se puede iterar sobre un rango de fechas'

    # Dataframe inicial con información importante
    df = pd.DataFrame(data = {'Tickers': ticks,
                             'Ponderación': weights,
                             f"Precio ({date})": prices,
                             'Títulos': titulos
                             })
    cap = capital

    # Dinero colocado en cada posición
    money = [(w / 100) * cap for w in weights]
    
    # Cálculo del valor de la posición individual
    df['Valor posición'] = round(df[f"Precio ({date})"] * df['Títulos'], 2)
    
    # Diferencia entre precio * títulos y capital x ponderacion
    res = [sum(money - df['Valor posición'])]
    res_df.loc[date] = res
    
    # Fila con suma de todos los valores numéricos
    df = df.append(df.sum(numeric_only=True), ignore_index=True)
    
    return df# + capital_f#, sum(df['Valor posición'])

def get_pasive_capital_results(reports: 'resultados del rendimiento pasivo'):
    
    'Función que obtiene los valores del capital de la estrategia pasiva a lo largo del tiempo'
    
    # Obtener el capital final de cada reporte (suma final de la posiciones)
    res = [reports[i].iloc[-1,4] for i in range(len(reports))]

    return res

def display_pasive_results(pasive_results: 'resultados (capital) del portafolio pasivo', dates: 'fechas', df_or_graph: '0 para devolver el df o 1 para devolver gráfica'):
    
    'Función que devuelve los resultados finales del portafolio de administración pasivo, mostrando sus rendimientos. Si se especifica, regresa una serie de tiempo de los movimiento del capital invertido'
        
    # Datos de los resultados pasivos
    df = pd.DataFrame(data = {'Capital': pasive_results},
                     index = dates)
    
    # Cálculo de rendimiento
    df['Rendimiento'] = df.iloc[:,0].pct_change()
    df['Rend. Acum'] = df.iloc[:,1].cumsum()
    
    # Gráfica del capital en caso de ser especificado en la función
    if df_or_graph == 1:
        
        plt.figure(figsize = (12,8))
        plt.plot(df.iloc[:,0])
        plt.title('Capital Estrategia Pasiva')
        plt.xlabel('Fecha')
        plt.ylabel('Capital')
        plt.xticks(rotation=90)
        plt.grid()
    
    return df if df_or_graph == 0 else None

def mkwtz_port(rf: 'tasa libre de riesgo', prices: 'precios históricos'):
    
    'Función que devuelve las ponderaciones óptimas de acuerdo con la Teoría Moderna de Portafolio de Markowitz'
    
    # Cáluclo de retornos
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
    cons = {'type': 'eq', 'fun': lambda w:w.sum() - 1} # Ningún w sea 0
    
    # Función objetivo
    def menos_RS(w, Eind, rf, Sigma):
        E_port = Eind.T.dot(w)
        s_port = var(w, Sigma)**0.5
        RS = (E_port - rf) / s_port
        return -RS
    
    # Función minimize
    emv = minimize(fun=menos_RS, x0 = w0, args = (Eind, rf, cov), bounds = bnds, constraints = cons, tol = 1e-8)

    # DataFrame de los resultados del portafolio óptimo
    report = pd.DataFrame()
    report['Ticker'] = returns['Close'].columns
    report.set_index('Ticker', inplace = True)
    report['W'] = emv.x.round(6) * 100
    report['Precio'] = prices.iloc[-1,:].values
    #report = report[report['W'] != 0]
    
    return report

def percentage_change(col1,col2):
    
    'Función que obtiene la variación porcentual de 2 columnas'
    
    return ((col2 - col1) / col1) * 100

def initial_active_port(port_info: 'información de portafolio', prices: 'precios', titulos: 'títulos iniciales', date: 'fecha inicial', capital: 'capital a invertir', res_df: 'df de movimientos del cash', cash: 'cantidad de cash disponible'):
    
    'Función que obtiene el portafolio incial así como su información importante del mismo que se puede usar en otras funciones'
    
    # Información inicial
    info = port_info[port_info['W'] != 0] # obtener los tickers que su ponderación no sea 0
    ticks = info.index
    weights = info.iloc[:,0].values
    pcs = prices.loc[date]['Close'].loc[ticks].values # Precios de la fecha indicada
    titulos = [i for i in titulos if i != 0] # Remover los ticker con 0 títulos
    
    # DataFrame inicial
    df = pd.DataFrame(data = {'Tickers': ticks,
                             'Ponderación': weights,
                             f"Precio ({date})": pcs,
                             'Títulos': titulos
                             })
    
    cap = capital

    # Dinero colocado en cada posición
    money = [(w / 100) * cap for w in weights]
    
    # Cálculo del valor de la posición individual
    df['Valor posición'] = round(df[f"Precio ({date})"] * df['Títulos'], 2)
    
    # Diferencia entre precio * títulos y capital x ponderacion
    res = [sum(money - df['Valor posición'])]
    res_df.loc[date] = cash + res
         
    # Fila con suma de todos los valores numéricos
    df = df.append(df.sum(numeric_only=True), ignore_index=True)
    
    return df

def monthly_perf_active(ticks, weights, prices, titulos, date1: str, date2, capital, res_df, comision, port_inicial):
    
    'Función que obtiene el rendimiento del portafolio de administración activa en un periodo de tiempo dado. Se puede iterar sobre un rango de fechas'

    # DataFrame con información inicial
    df = pd.DataFrame(data = {'Tickers': ticks[:-1],
                             'Ponderación': weights[:-1], # PONDERACION Y TITULOS ALMACENADOS EN OTRO DF
                             f"Precio ({date2})": prices.loc[date2]['Close'][ticks.dropna()].values,
                             'Títulos': titulos[:-1]
                             })
    
    # Cálculo de precios nuevos y su diferencia para obtener la variación
    new_prices = df[f"Precio ({date2})"]
    old_prices = prices.loc[date1]['Close'][ticks.dropna()].values
    price_dif = percentage_change(old_prices, new_prices)
    
    # DataFrame de la diferencia de precios
    price_dif_df = pd.DataFrame(data = {'Ticker': ticks[:-1],
                                       'Diferencia': price_dif}).sort_values(by = 'Diferencia', 
                                                                             ascending=False, key = abs)
    
    # Filtras el DataFrame para obtener únicamente las variaciones mayores al 5% (hacia arriba o hacia abajo)
    price_dif_df = price_dif_df[(price_dif_df['Diferencia'] >= 5) | (price_dif_df['Diferencia'] <= -5)]
    
    # DataFrame para almacenar información de los movimientos de los títulos
    cambio_titulos = df.set_index('Tickers').loc[price_dif_df['Ticker']].reset_index(inplace=False) # En orden
    cambio_titulos['Títulos nuevos'] = 0
    
    # Aumento o disminución de títulos de acuerdo al a condición especificada en las intrucciones
    for i in range(len(cambio_titulos)):
        if price_dif_df.iloc[i,1] >= 5: # Si el cambio es mayor a 5%
            cambio_titulos.iloc[i,4] = cambio_titulos.iloc[i,3] * (1 + 0.025) # Títulos aumentan 2.5%
        elif price_dif_df.iloc[i,1] <= -5: # Si el cambio es menor a 5%
            cambio_titulos.iloc[i,4] = cambio_titulos.iloc[i,3] * (1 - 0.025) # Títulos disminuyen 2.5% 

    # Cálculos de los nuevos títulos de acuerdo a la condición anterior
    dif_titulos = cambio_titulos.iloc[:,4] - cambio_titulos.iloc[:,3]
    cambio_titulos['Títulos nuevos'] = cambio_titulos['Títulos nuevos'].astype(int)
    #print((cambio_titulos.iloc[:,4] - cambio_titulos.iloc[:,3]).values)
    cambio_titulos['Títulos Dif'] = (cambio_titulos.iloc[:,4] - cambio_titulos.iloc[:,3]).values
    cambio_titulos['Prices w/com'] = 0
    
    # Precios con comisión
    for i in range(len(cambio_titulos)):
        if cambio_titulos.iloc[i,5] > 0:
            cambio_titulos.iloc[i,6] = cambio_titulos.iloc[i,2] * (1 + comision)
        elif cambio_titulos.iloc[i,5] < 0:
            cambio_titulos.iloc[i,6] = cambio_titulos.iloc[i,2] * (1 - comision)
    
    # Comisión cobrada
    cambio_titulos['Commission'] = cambio_titulos['Prices w/com'] - cambio_titulos.iloc[:,2]
    
    # Ventas de acciones si existen
    ing = -cambio_titulos[cambio_titulos['Títulos Dif'] < 0].iloc[:,6] * cambio_titulos[cambio_titulos['Títulos Dif'] < 0].iloc[:,5]
    ing = ing.sum()
    
    # Suma del capital más los ingresos por ventas
    cap = (capital + ing)
    #cap = 12300
    
    # Ajuste de acuerdo al capital
    for i in range(len(cambio_titulos[cambio_titulos['Títulos Dif'] > 0])): # Para los de compra
        if (cap / (cambio_titulos.iloc[i,5] * cambio_titulos.iloc[i,6])) >= 1: # Si hay dinero suficiente
            cambio_titulos.iloc[i,5] = cambio_titulos.iloc[i,5] # Los títulos quedan igual
            cap = cap - (cambio_titulos.iloc[i,5] * cambio_titulos.iloc[i,6]) # Se disminuye el capital
        else: # Si no hay dinero suficiente
            cambio_titulos.iloc[i,5] = int((cap / cambio_titulos.iloc[i,6])) # Se compran los títulos que alcancen
            cap = cap - (cambio_titulos.iloc[i,5] * cambio_titulos.iloc[i,6]) # Se le resta la cantidad correspondiente
    
    df.set_index('Tickers', inplace = True)
    tick_change = cambio_titulos['Tickers']
    
    # Ajuste de títulos de acuerdo con el capital calculado anteriormente
    cambio_titulos['Títulos nuevos'] = cambio_titulos['Títulos Dif'] + cambio_titulos['Títulos']
    
    # Comisión por títulos comprados
    cambio_titulos['T * C'] = abs(cambio_titulos['Commission'] * cambio_titulos['Títulos Dif'])
    
    # Actualizar variable df
    for i in range(len(cambio_titulos)):
        df.loc[tick_change[i]]['Títulos'] = cambio_titulos[cambio_titulos['Tickers'] == tick_change[i]]['Títulos nuevos']
        
    df.reset_index(inplace = True)
    
    df['Valor posición'] = round(df[f"Precio ({date2})"] * df['Títulos'], 2)
    
    # Ajuste Cash
    res_df.loc[date2] = cap
    
    port_inicial['Títulos Act'] = df['Títulos']
     
    # Suma en una fila final de los valores numéricos
    df = df.append(df.sum(numeric_only=True), ignore_index=True)
    
    return df, cambio_titulos

def display_active_results(active_results: 'resultados del portafolio activo', 
                           initial_p: 'portafolio inicial',
                           dates: 'fechas', 
                           df_or_graph: '0 para devolver el df o 1 para devolver gráfica'):
    
    'Función que devuelve los resultados finales del portafolio de administración activa, mostrando sus rendimientos. Si se especifica, regresa una serie de tiempo de los movimiento del capital invertido'
        
    # Plantilla del DataFrame
    df1 = pd.DataFrame(columns=['timestamp', 'capital', 'rend', 'rend_acum'])
 
    # Información del DataFrame
    df1['timestamp'] = dates
    df1['capital'] = [initial_p['Valor posición'].iloc[-1]] + [active_results[i][0]['Valor posición'].iloc[-1] for i in range(len(active_results))]
    df1['rend'] = df1['capital'].pct_change()
    df1['rend_acum'] = df1['rend'].cumsum()
    
    # En caso de ser especificado, se devuelve la gráfica de los movimientos del capital
    if df_or_graph == 1:
        
        plt.figure(figsize = (12,8))
        plt.plot(df1.iloc[:,0], df1.iloc[:,1])
        plt.title('Capital Estrategia Activa')
        plt.xlabel('Fecha')
        plt.ylabel('Capital')
        plt.xticks(rotation=90)
        plt.grid()
    
    return df1 if df_or_graph == 0 else None

def get_comission_history(active_results: 'resultados del portafolio activo', 
                           initial_p: 'portafolio inicial',
                           dates: 'fechas'):
    
    'Función que devuelve el historial de comisiones cobradas en una administración de un portafolio activo'
    
    # Plantilla del DataFrame
    df2 = pd.DataFrame(columns=['timestamp', 'titulos_totales', 'titulos_compra', 'comisión', 'comision_acum'])

    # Información del DataFrame
    df2['timestamp'] = dates
    df2['titulos_totales'] = [initial_p['Títulos'].iloc[-1]] + [active_results[i][0]['Títulos'].iloc[-1] for i in range(len(active_results))]
    df2['titulos_compra'] = [initial_p['Títulos'].iloc[-1]] + [active_results[i][1]['Títulos Dif'].sum() for i in range(len(active_results))]
    df2['comisión'] = [comision_inicial] + [active_results[i][1]['T * C'].sum() for i in range(len(active_results))]
    df2['comision_acum'] = df2['comisión'].cumsum()

    return df2
