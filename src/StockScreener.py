from StockAnalyzer import StockAnalyzer
from StockDataDownloader import StockDataDownloader, fetch_sp500_tickers
import sqlite3

class StockScreener:
    def __init__(self, db_path='stock_data.db'):
        self.data_downloader = StockDataDownloader(db_path=db_path)
        self.tickers = fetch_sp500_tickers()
        self.db_path = db_path
        self._prepare_database()

    def _prepare_database(self):
        """Prepares the database for storing Stage 2 stocks."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS stage2_stocks (
                    ticker TEXT PRIMARY KEY
                );
            """)
            conn.commit()

    def find_stage2_stocks(self):
        """Finds and saves Stage 2 stocks to the database."""
        stage2_stocks = []
        for ticker in self.tickers:
            df = self.data_downloader.get_data_as_dataframe(ticker)
            if not df.empty:
                analyzer = StockAnalyzer(df)
                if analyzer.is_in_stage_2() \
                    and analyzer.volume_boost():
                    # and analyzer.has_volume_contraction()
                    # and analyzer.check_recent_contraction():
                    stage2_stocks.append(ticker)
        self._save_stage2_stocks_to_db(stage2_stocks)
        return stage2_stocks

    def find_squeeze_expansion_stocks(self):
        """Finds and saves Stage 2 stocks to the database."""
        squeeze_expansion_stocks = []
        for ticker in self.tickers:
            df = self.data_downloader.get_data_as_dataframe(ticker)
            if not df.empty:
                analyzer = StockAnalyzer(df)
                if analyzer.is_squeeze_expansion():
                    squeeze_expansion_stocks.append(ticker)
        return squeeze_expansion_stocks

    def _save_stage2_stocks_to_db(self, tickers):
        """Saves the list of Stage 2 stocks to the SQLite database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Optional: Clear the table before adding new entries
            cursor.execute("DELETE FROM stage2_stocks")
            for ticker in tickers:
                cursor.execute("INSERT INTO stage2_stocks (ticker) VALUES (?)", (ticker,))
            conn.commit()

