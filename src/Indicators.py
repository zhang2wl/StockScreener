import pandas as pd
import numpy as np

class Indicators:
    def __init__(self, data):
        """
        Initializes the Indicators object with stock data.
        :param data: pandas DataFrame with stock price data, including 'Close' prices.
        """
        self.data = data

    def moving_average(self, window=20):
        """
        Calculates the Moving Average (MA) and adds it as a new column to the DataFrame.
        :param window: The number of periods to calculate the average over.
        """
        column_name = f'MA{window}'  # Dynamic column name based on the window size
        self.data[column_name] = self.data['Close'].rolling(window=window).mean()
        return self.data[column_name]  # Return only the Series of the calculated MA

    def volume_moving_average(self, window=20):
        """
        Calculates the Volume Moving Average (Volume MA).
        :param window: The number of periods to calculate the average over.
        """
        column_name = f'Volume{window}'  # Dynamic column name based on the window size
        self.data[column_name] = self.data['Volume'].rolling(window=window).mean()
        return self.data[column_name]  # Return the Series of the calculated Volume MA

    def relative_strength_index(self, window=14):
        """
        Calculates the Relative Strength Index (RSI).
        :param window: The number of periods to calculate the RSI over.
        """
        delta = self.data['Close'].diff(1)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=window, min_periods=1).mean()
        avg_loss = loss.rolling(window=window, min_periods=1).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        self.data[f'RSI{window}'] = rsi
        return self.data[f'RSI{window}']  # Return the Series of the calculated RSI
