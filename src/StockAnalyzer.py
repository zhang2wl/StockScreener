import pandas as pd
import numpy as np
from StockDataDownloader import fetch_sp500_tickers
from scipy.stats import linregress

class StockAnalyzer:
    def __init__(self, df):
        """
        Initialize with a DataFrame containing stock data.
        The DataFrame should have columns: Date, Open, High, Low, Close, Volume.
        :param df: pandas DataFrame
        """
        self.df = df

    def is_in_stage_2(self):
        """Check if the stock meets the criteria for being in Stage 2."""
        # Check if DataFrame is empty
        if self.df.empty:
            print("DataFrame is empty. Cannot determine if the stock is in Stage 2.")
            return False  # Or any other indicator/value you prefer for this case

        # Use .iloc[-1] to get the latest row and ensure no NaN values in key columns
        latest = self.df.iloc[-1]

        # Check if any required values are NaN, which would make the comparison invalid
        if pd.isna(latest['Close']) or pd.isna(latest['MA150']) or pd.isna(latest['MA200']) or pd.isna(latest['MA50']):
            print("Missing data for the latest entry. Cannot accurately determine if the stock is in Stage 2.")
            return False

        ma_conditions = (
            latest['Close'] > latest['MA150'] > latest['MA200'] and
            latest['MA50'] > latest['MA150']
        )

        # Criteria 5 and 6
        fifty_two_week_low = self.df['Close'].iloc[-260:].min()
        fifty_two_week_high = self.df['Close'].iloc[-260:].max()
        price_conditions = (
            latest['Close'] >= 1.25 * fifty_two_week_low and
            latest['Close'] >= 0.75 * fifty_two_week_high
        )

        # Assuming current price above MA50 as part of Criteria 8
        ma50_condition = latest['Close'] > latest['MA50']

        # New RSI Trend Condition
        rsi_trend_condition = self.is_rsi_trending_up()

        return ma_conditions and price_conditions and ma50_condition and rsi_trend_condition

    def is_rsi_trending_up(self):
        """Check if the RSI is on an increasing trend over the past 13 weeks (65 trading days)."""
        rsi = self.df['RSI14'].tail(65).dropna()  # Consider the last 65 days
        if len(rsi) < 65:
            return False  # Not enough data to assess the trend

        # Use linear regression to determine the trend
        x = np.arange(len(rsi))
        slope, _, _, _, _ = linregress(x, rsi)
        return slope > 0

    def is_vcp_candidate(self):
        """
        Checks if the stock shows a Volatility Contraction Pattern (VCP).
        """
        # Ensure there's enough data to analyze
        if len(self.df) < 3:  # Need at least 3 points to define two contractions
            return False

        # Calculate daily price ranges and volume changes
        self.df['PriceRange'] = self.df['High'] - self.df['Low']
        self.df['VolumeChange'] = self.df['Volume'].pct_change()

        # Identify contractions: successive reductions in price range
        contractions = (self.df['PriceRange'].shift(-1) < self.df['PriceRange']) & (self.df['VolumeChange'] < 0)

        # Count the number of contractions and check if volume is reducing over these contractions
        contraction_count = contractions.sum()
        if contraction_count >= 2:  # Basic check for multiple contractions
            last_contraction_volume = self.df[contractions]['Volume'].iloc[-1]
            first_contraction_volume = self.df[contractions]['Volume'].iloc[0]

            # Ensure volume is reducing
            if last_contraction_volume < first_contraction_volume:
                return True

        return False

    def has_volume_contraction(self, threshold=1):
        """
        Identifies periods of volume contraction based on the Volume50 moving average.

        :param threshold: The fraction of the Volume50 that current volume must be below to indicate a contraction.
        :return: boolean, whether the stock has significant volume contraction
        """
        # Identify days where volume is significantly below the Volume50 average
        return self.df['Volume'].iloc[-1] < (self.df['Volume50'].iloc[-1] * threshold)

    def check_recent_contraction(self):
        """
        Checks the last 63 trading days for a contraction where the minimum close value
        within this period is more than 10% lower than the maximum close value, and
        this minimum occurs after the maximum.
        """
        # Focus on the last 63 trading days
        recent_df = self.df[-63:]

        if recent_df.empty:
            print("Insufficient data.")
            return False

        # Find the maximum close value and its index
        max_close = recent_df['Close'].max()
        max_close_date = recent_df['Close'].idxmax()

        # Find the minimum close value and its index
        min_close = recent_df['Close'].min()
        min_close_date = recent_df['Close'].idxmin()

        # Check if the min is newer than the max and is more than 10% lower
        if min_close_date > max_close_date and min_close < (max_close * 0.9):
            return True

        return False