import os
import pandas as pd
import yfinance as yf

class StockDataDownloader:
    def __init__(self, data_dir='../data/stock_data'):
        self.data_dir = data_dir
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def fetch_sp500_tickers(self):
        """Fetches the list of S&P 500 tickers."""
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        sp500_list = pd.read_html(url)[0]
        return sp500_list['Symbol'].values.tolist()

    def fetch_and_save_weekly_stock_data(self, ticker):
        """Fetches historical weekly stock data and appends new data to the existing CSV file."""
        filepath = os.path.join(self.data_dir, f"{ticker}_weekly.csv")
        if os.path.exists(filepath):
           # Load existing data
           existing_data = pd.read_csv(filepath, index_col='Date', parse_dates=True)
           last_date = existing_data.index.max()

           # Format the last_date as a string in the format expected by yfinance
           last_date_str = last_date.strftime('%Y-%m-%d')

           # Fetch new data starting from the day after the last date in the existing data
           # Increment the last_date by one day to ensure we're starting the fetch from the correct point
           new_data = yf.Ticker(ticker).history(start=last_date + pd.Timedelta(days=1), interval="1wk")
           if not new_data.empty:
                 updated_data = pd.concat([existing_data, new_data])  # Combine existing with new data
                 updated_data.to_csv(filepath)  # Save combined data back to CSV
        else:
           # If the CSV doesn't exist, fetch the full historical data and save
           updated_data = yf.Ticker(ticker).history(period="5y", interval="1wk")
           updated_data.to_csv(filepath)


# Usage
data_downloader = StockDataDownloader()
tickers = data_downloader.fetch_sp500_tickers()
for ticker in tickers[:5]:  # Example: Process only the first 5 tickers for demonstration
    data_downloader.fetch_and_save_weekly_stock_data(ticker)
