import pandas as pd
import numpy as np
# from StockDataDownloader import fetch_sp500_tickers
from scipy.stats import linregress
import mplfinance as mpf

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

    def volume_boost(self):
        """
        Checks if there is a week in the past two months where the trading volume
        is more than twice the 50-day moving average volume.
        """
        # Ensure there's at least 50 days of data to calculate the average
        if len(self.df) < 50:
            print("Insufficient data for 50-day average volume calculation.")
            return False

        # Calculate the 50-day average volume if not already present
        if 'Volume50' not in self.df.columns:
            self.df['Volume50'] = self.df['Volume'].rolling(window=50).mean()

        # Focus on the last two months (approximately 40 trading days)
        recent_df = self.df[-40:]

        # Resample to weekly data, summing the volume
        weekly_volumes = recent_df['Volume'].resample('W').sum()

        # Check if any week's volume is more than twice the 50-day average
        for week_volume in weekly_volumes:
            if week_volume > 2 * recent_df['Volume50'].mean():
                return True

        return False

    def mark_local_extrema(self):
        """
        Marks local minima and maxima in the closing price curve.
        Local minima are marked with -1, local maxima with 1, and all other points with 0.
        """
        prices = self.df['Close'].tolist()
        self.df['extrema'] = 0  # Initialize all points as neither (0)

        for i in range(1, len(prices) - 1):
            # Check for local maximum
            if prices[i] > prices[i - 1] and prices[i] > prices[i + 1]:
                self.df.at[self.df.index[i], 'extrema'] = 1
            # Check for local minimum
            elif prices[i] < prices[i - 1] and prices[i] < prices[i + 1]:
                self.df.at[self.df.index[i], 'extrema'] = -1

        return self.df

    def wedge_breakout(self):
        """
        Tests for a wedge breakout pattern over the past two months.
        """
        recent_df = self.df[-40:]  # Focus on the last 40 trading days

        # Extract indices where local maxima (1) and minima (-1) occur
        max_indices = recent_df[recent_df['extrema'] == 1].index
        min_indices = recent_df[recent_df['extrema'] == -1].index

        if len(max_indices) < 2 or len(min_indices) < 2:
            # Not enough extrema to determine pattern
            return False

        # Check local highs for horizontal consistency
        max_values = recent_df.loc[max_indices, 'Close']
        slope, _, _, _, _ = linregress(range(len(max_values)), max_values)

        # Assuming a tolerance for the slope to still be considered 'horizontal'
        if abs(slope) < 0.0:
            # Local highs are decreasing
            return False

        # Check for increasing local lows
        min_values = recent_df.loc[min_indices, 'Close']
        low_slope, _, _, _, _ = linregress(range(len(min_values)), min_values)

        if low_slope <= 0:
            # Local lows are not increasing
            return False

        return True

    def calculate_bollinger_bands(self, window=20, num_std=2):
        """
        Calculates Bollinger Bands for the stock data.

        :param window: The moving average window size. Defaults to 20.
        :param num_std: The number of standard deviations for the upper and lower bands. Defaults to 2.
        """
        self.df['MiddleBB'] = self.df['Close'].rolling(window=window).mean()  # Middle Band
        self.df['STD'] = self.df['Close'].rolling(window=window).std()

        self.df['UpperBB'] = self.df['MiddleBB'] + (self.df['STD'] * num_std)  # Upper Band
        self.df['LowerBB'] = self.df['MiddleBB'] - (self.df['STD'] * num_std)  # Lower Band

        # Calculate Bollinger Bandwidth as a percentage of the Middle Band
        self.df['BBWidth'] = ((self.df['UpperBB'] - self.df['LowerBB']) / self.df['MiddleBB']) * 100

    def is_squeeze_expansion(self):
        """
        Checks for 'squeeze expansion' pattern in Bollinger Bands.

        Returns True if the middle band is trending up and the lower band is trending down.
        """
        # Assuming Bollinger Bands have been calculated
        if 'MiddleBB' not in self.df.columns or 'LowerBB' not in self.df.columns:
            self.calculate_bollinger_bands()

        # Check the trend of the Middle and Lower Bollinger Bands
        middle_trend = self.df['MiddleBB'].iloc[-3:].is_monotonic_increasing
        lower_trend = self.df['LowerBB'].iloc[-3:].is_monotonic_decreasing

        # Check if the minimum Bandwidth of the last 3 days is the lowest in the last 60 days
        last_3_days_min_bbwidth = self.df['BBWidth'].iloc[-3:].min()
        last_60_days_min_bbwidth = self.df['BBWidth'].iloc[-60:].min()

        bbwidth_criterion = last_3_days_min_bbwidth <= last_60_days_min_bbwidth*1.05

        return middle_trend and lower_trend and bbwidth_criterion

    def is_squeeze_expansion(self):
        """
        Checks for 'squeeze expansion' pattern in Bollinger Bands.

        Returns True if the middle band is trending up and the lower band is trending down.
        """
        # Assuming Bollinger Bands have been calculated
        if 'MiddleBB' not in self.df.columns or 'LowerBB' not in self.df.columns:
            self.calculate_bollinger_bands()

        # Check the trend of the Middle and Lower Bollinger Bands
        middle_trend = self.df['MiddleBB'].iloc[-3:].is_monotonic_increasing
        lower_trend = self.df['LowerBB'].iloc[-3:].is_monotonic_decreasing

        # Check if the minimum Bandwidth of the last 3 days is the lowest in the last 60 days
        last_3_days_min_bbwidth = self.df['BBWidth'].iloc[-3:].min()
        last_60_days_min_bbwidth = self.df['BBWidth'].iloc[-60:].min()

        bbwidth_criterion = last_3_days_min_bbwidth <= last_60_days_min_bbwidth*1.05

        return middle_trend and lower_trend and bbwidth_criterion

    def plot_with_bollinger_bands(self, plot_df, occurrences):
        # Create a list of Bollinger Bands for the 'addplot' argument
        ap = [
            mpf.make_addplot(plot_df['UpperBB'], color='g', linestyle='dashdot' , width=0.75),  # Upper Bollinger Band
            mpf.make_addplot(plot_df['MiddleBB'], color='b', linestyle='dashdot', width=0.75),  # Middle Bollinger Band
            mpf.make_addplot(plot_df['LowerBB'], color='r', linestyle='dashdot' , width=0.75),  # Lower Bollinger Band
        ]

        # Calculate min and max for y-axis limits
        price_min = plot_df[['Low', 'LowerBB']].min().min()
        price_max = plot_df[['High', 'UpperBB']].max().max()

        # # Add a vertical line on the squeeze expansion day
        # vline = [(squeeze_day, plot_df['Low'].min(), squeeze_day, plot_df['High'].max())]

        # Plot configuration
        mpf.plot(plot_df,
            type='candle',
            style='charles',
            volume=True,
            title=f"Squeeze Expansion Occurrence {occurrences}",
            addplot=ap,
            # alines=vline,  # Adding the vertical line
            figratio=(12, 8),
            figscale=1.2,
            panel_ratios=(6,3),
            show_nontrading=False,
            ylim=(price_min, price_max))  # Set y-axis limits to match price range)

    def calculate_alligator(self):
        # Williams Alligator parameters
        jaw_length = 13
        teeth_length = 8
        lips_length = 5

        # Smoothed Moving Averages (SMMA)
        self.df['Jaw'] = self.df['Close'].rolling(window=jaw_length, min_periods=1).mean().shift(8)
        self.df['Teeth'] = self.df['Close'].rolling(window=teeth_length, min_periods=1).mean().shift(5)
        self.df['Lips'] = self.df['Close'].rolling(window=lips_length, min_periods=1).mean().shift(3)

    def sliding_window_analysis(self, win_threshold=0.1, window=14):
        successes = 0
        occurrences = 0
        self.calculate_bollinger_bands()
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df.set_index('Date', inplace=True)

        # Start from day 61 to have 60 days of data for calculating the lowest BBWidth
        for i in range(60, len(self.df)-window):  # Leave 14 days at the end for the price increase check
            window_end = i + 1  # +1 because the upper bound is exclusive

            # Check the trend of the Middle and Lower Bollinger Bands
            check_start = -3+window_end
            check_end = window_end

            middle_trend = self.df['MiddleBB'].iloc[check_start:check_end].is_monotonic_increasing
            lower_trend = self.df['LowerBB'].iloc[check_start:check_end].is_monotonic_decreasing

            # Check if the minimum Bandwidth of the last 3 days is the lowest in the last 60 days
            last_2_days_min_bbwidth = self.df['BBWidth'].iloc[check_start:check_end].min()
            last_60_days_min_bbwidth = self.df['BBWidth'].iloc[-60+window_end:window_end+1].min()

            bbwidth_criterion = last_2_days_min_bbwidth <= last_60_days_min_bbwidth*1.05

            squeeze_expansion = middle_trend and lower_trend and bbwidth_criterion

            # Check if the squeeze expansion condition is met
            if squeeze_expansion:
                occurrences += 1

                # Check for a 10% price increase in the following 14 data points
                current_price = self.df.iloc[i]['Close']
                future_prices = self.df.iloc[i+1:i+window+1]['Close']  # Next 14 days
                max_future_price = future_prices.max()

                if max_future_price >= (1+win_threshold) * current_price:
                    successes += 1

                # if occurrences <=5:
                #     # Define the plot window including the squeeze event and the following window days
                #     plot_df = self.df.iloc[i-10:i+window+1]  # Plot 20 days before the event for context and window days after

                #     self.plot_with_bollinger_bands(plot_df, occurrences)

        success_rate = successes / occurrences if occurrences else 0
        # print(f"Occurrences = {occurrences}, successes = {successes}, winrate = {success_rate}")
        return (occurrences, successes, success_rate)


    def sliding_window_alligator_analysis(self, threshold=0.05,window=14):
        self.calculate_alligator()
        successes = 0
        occurrences = 0

        # Start from the point where the Alligator Indicator can be calculated
        for i in range(13, len(self.df)-window):
            # Check for a Lips crossover above Teeth and Jaw (Buy Signal)
            if self.df.iloc[i]['Lips'] > self.df.iloc[i]['Teeth'] and self.df.iloc[i]['Lips'] > self.df.iloc[i]['Jaw']:
                # This is where the signal occurs; we mark it as an occurrence
                occurrences += 1

                # Check for a 10% price increase in the following window days
                current_price = self.df.iloc[i]['Close']
                future_prices = self.df.iloc[i+1:i+window+1]['Close']
                max_future_price = future_prices.max()

                if max_future_price >= (1+threshold) * current_price:
                    successes += 1

        success_rate = successes / occurrences if occurrences else 0
        # print(f"Alligator Strategy - Occurrences: {occurrences}, Successes: {successes}, Success Rate: {success_rate:.2%}")
        return (occurrences, successes, success_rate)