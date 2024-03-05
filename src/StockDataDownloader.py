import pandas as pd
import yfinance as yf
import sqlite3
import warnings
from Indicators import Indicators  # Ensure this import is correct

# Suppress FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class StockDataDownloader:
    def __init__(self, db_path='stock_data.db'):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self._create_table()

    def _create_table(self):
        """Modifies the table to include predefined columns for indicators."""
        create_table_query = """
        CREATE TABLE IF NOT EXISTS stock_data (
            Date DATE,
            Ticker TEXT,
            Open FLOAT,
            High FLOAT,
            Low FLOAT,
            Close FLOAT,
            Volume INTEGER,
            MA200 FLOAT,
            MA150 FLOAT,
            MA50 FLOAT,
            Volume50 FLOAT,
            RSI14 FLOAT,
            PRIMARY KEY (date, ticker)
        );
        """
        self.conn.execute(create_table_query)
        self.conn.commit()

    def fetch_and_process_stock_data(self, ticker, period='5y', interval='1d'):
        """Fetches stock data, calculates indicators, and saves everything into the database."""
        try:
            data = yf.Ticker(ticker).history(period=period, interval=interval)
            if data.empty:
                print(f"No data found for {ticker}. It might be delisted or the ticker symbol may have changed.")
                return

            data.reset_index(inplace=True)
            data['Ticker'] = ticker  # Add ticker column
            data['Date'] = data['Date'].dt.strftime('%Y-%m-%d')  # Convert Date to string format

            # Exclude unwanted columns
            data = data[['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']]  # Add any other columns you need

            # Calculate indicators
            indicators = Indicators(data)
            data['MA200'] = indicators.moving_average(window=200)
            data['MA150'] = indicators.moving_average(window=150)
            data['MA50'] = indicators.moving_average(window=50)
            data['Volume50'] = indicators.volume_moving_average(window=50)
            data['RSI14'] = indicators.relative_strength_index(window=14)

            # Prepare data for insertion, including indicators
            data_tuples = [tuple(row) for row in data.to_numpy()]

            # Ensure all column names are properly quoted for SQL
            cols = ', '.join([f'"{col}"' for col in data.columns])

            placeholders = ', '.join(['?'] * len(data.columns))
            insert_query = f"INSERT OR IGNORE INTO stock_data ({cols}) VALUES ({placeholders})"

            self.conn.executemany(insert_query, data_tuples)
            self.conn.commit()
        except Exception as e:
            print(f"An error occurred while processing {ticker}: {e}")

    def update_SP500_data(self):
        """Fetches, processes, and updates data for all S&P 500 tickers."""
        tickers = fetch_sp500_tickers()
        for ticker in tickers:
            self.fetch_and_process_stock_data(ticker)

    def view_data(self, ticker=None, limit=10):
        """Fetches and prints the latest 'limit' rows for a given ticker from the database."""
        with self.conn:  # Automatically handles commit/rollback
            cursor = self.conn.cursor()
            query = "SELECT * FROM stock_data"
            if ticker:
                query += " WHERE ticker = ?"
                query += f" ORDER BY date DESC LIMIT {limit}"
                cursor.execute(query, (ticker,))
            else:
                query += f" ORDER BY date DESC LIMIT {limit}"
                cursor.execute(query)

            rows = cursor.fetchall()
            for row in rows:
                print(row)

    def load_db_into_memory(self):
        """Loads the database into memory for faster operations."""
        # Connect to an in-memory database
        mem_conn = sqlite3.connect(':memory:')
        # Connect to the database file
        disk_conn = sqlite3.connect(self.db_path)
        # Copy the database from disk to memory
        query = "".join(line for line in disk_conn.iterdump())
        mem_conn.executescript(query)
        # Close the disk connection and switch to using the in-memory connection
        disk_conn.close()
        self.conn = mem_conn
        print("Database has been loaded into memory.")

    def get_data_as_dataframe(self, ticker):
        """
        Fetches stock data for a specific ticker from the database (loaded in memory) and returns it as a pandas DataFrame.
        :param ticker: The ticker symbol of the stock to fetch data for.
        :return: A pandas DataFrame containing the stock data for the specified ticker.
        """
        query = f"SELECT * FROM stock_data WHERE ticker = '{ticker}'"
        df = pd.read_sql_query(query, self.conn)
        return df

def fetch_sp500_tickers():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    sp500_list = pd.read_html(url)[0]
    return sp500_list['Symbol'].values.tolist()