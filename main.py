import data
import functions # Archivo de funciones
import pandas as pd

### Estrategia Pasiva

# DataFrame de todos los archivos csv descargados del NAFTRAC
csv_files = functions.read_multiple_csv("files")
csv_df = csv_files[1] # 1 para dataframe 0 para archivos (direcciones)

# Tickers repetidos en todos los archivos, eliminando MXN
c_tickers = functions.get_constant_tickers(csv_df, 'MXN.MX')

# Obtener las fecgas de los archivos
all_dates = functions.get_all_dates2(csv_files[0])
dates_fd = [functions.str_to_datetime(i, '%Y%m%d', '%Y-%m-%d') for i in all_dates]
dates_fd.sort() # Importante

# Precios mensuales, de acuerdo con las fechas anteriormente obtenidas
prices = data.prices

# Ponderaciones de los tickers repetidos obtenidas del primer archivo descargado
pond = functions.get_weights('files/NAFTRAC_20210129.csv', c_tickers, ['MXN.MX'])

# Información del capital, comisión y cash (% que queda afuera debido a posiciones de cash como MXN)
cash = functions.cash
total_cap = 1000000
comision = 0.00125

# DataFrame con la información anterior 
caps = functions.capital_values(total_cap, cash, pond)

# Títulos a comprar de acuerdo al capital, la ponderación y además considerando precios con comisión
titulos = functions.get_titulos(prices, '2021-01-29', comision, total_cap, pond)

# Inicialización de DataFrame para almacenar el cash del portafolio pasivo
res_df = pd.DataFrame(columns=['Amount'], index = [dates_fd])

# Función que devuelve el comportamiento/desempeño del portafolio creado, se itera por cada fecha
# Se puede observar la evolución del portafolio en cada fecha
monthly_reports = [functions.monthly_perf_pasive(c_tickers, pond, prices.loc[i].values, titulos, i, total_cap, res_df) for i in dates_fd]
monthly_reports[0]

# Movimientos del capital en todas las fechas
cap_res = functions.get_pasive_capital_results(monthly_reports)

# Resultados de la inversión pasiva
p_results = functions.display_pasive_results(cap_res, dates_fd, 0)

# Resultados de la inversión pasiva (gráfica)
p_resultsg = functions.display_pasive_results(cap_res, dates_fd, 1)

### Estrategia Activa

# Tasa libre de riesgo mensual
rf = 0.0429 / 12

# Precios históricos diarios desde el inicio hasta Febrero
pcs = functions.get_ticker_prices(c_tickers, dates_fd[0], 1, '2022-03-01', '1d') # 1 día adelante

# Ponderaciones óptimas usando Teoría de Portafolio de Markowitz (minimize)
pond_mk = functions.mkwtz_port(rf, pcs)

# Precios mensuales de Febrero en adelante para el análisis
pcs2 = data.prices[13:]

# Títulos comprados al inicio de acuerdo con los resultados del portafolio de Markowitz
# Se considera el capital inicial y la comisión
titulos_ac = functions.get_titulos(pcs, '2022-02-28', comision, 965600.0, pond_mk.iloc[:,0])

# Inicialización de DataFrame para almacenar el cash del portafolio pasivo
# Registra cada movimiento en el mismo en cada fecha si hay compras o ventas
res_df_a = pd.DataFrame(columns=['Amount'], index = [dates_fd[13:]])

# Características del portafolio activo inicial
initial_ap = functions.initial_active_port(pond_mk, pcs2, titulos_ac, dates_fd[13], caps['Invested'][0], res_df_a, caps['Total Not Invested'][0])
initial_ap['Títulos Act'] = initial_ap['Títulos']

# Función que devuelve el comportamiento/desempeño del portafolio activo, se itera por cada fecha
# Se puede observar la evolución del portafolio en cada fecha
r = [functions.monthly_perf_active(initial_ap['Tickers'], initial_ap['Ponderación'], pcs2, initial_ap['Títulos Act'], dates_fd[13:][i], dates_fd[13:][i+1], res_df_a.iloc[i,0], res_df_a, comision, initial_ap) for i in range(len(dates_fd[13:]) - 1)]

# Resultados de la inversión activa
df1 = functions.display_active_results(r, initial_ap, dates_fd[13:], 0)

# Resultados de las comisiones a lo largo del periodo de tiempo
df2 = functions.get_comission_history(r, initial_ap, dates_fd[13:])

# Resultados finales/comparación final
df_final = pd.DataFrame(columns = ['Medida' , 'Descrpición', 'Inv Activa', 'Inv Pasiva'])
df_final['Medida'] = ['rend_m', 'rend_c', 'sharpe']
df_final['Descrpición'] = ['Rendimiento Promedio Mensual', 'Rendimiento Mensual Acumulado', 'Sharpe Ratio']
df_final['Inv Activa'] = [df1['rend'].mean(), df1['rend_acum'].iloc[-1], (df1['rend'].mean() - rf) / df1['rend'].std()]
df_final['Inv Pasiva'] = [p_results['Rendimiento'].mean(), p_results['Rend. Acum'].iloc[-1], (p_results['Rendimiento'].mean() - rf) / p_results['Rendimiento'].std()]



