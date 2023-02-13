import functions

csv_files = functions.read_multiple_csv("files")
csv_df = csv_files[1] # 1 para dataframe 0 para archivos (direcciones)

# Tickers repetidos en todos los archivos, eliminando MXN
c_tickers = functions.get_constant_tickers(csv_df, 'MXN.MX')

all_dates = functions.get_all_dates2(csv_files[0])
dates_fd = [functions.str_to_datetime(i, '%Y%m%d', '%Y-%m-%d') for i in all_dates]
dates_fd.sort() # Importante

prices = functions.get_ticker_prices(c_tickers, dates_fd, 0)