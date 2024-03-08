import pandas as pd
from StockScreener import StockScreener
from StockDataDownloader import StockDataDownloader
from StockAnalyzer import StockAnalyzer
from StockDataDownloader import fetch_sp500_tickers
def main():
    data_downloader = StockDataDownloader()
    # tickers = fetch_sp500_tickers()
    tickers = ['ACN', 'ADBE', 'AMD', 'ABNB', 'ALB', 'ALGN', 'AXP', 'AMGN', 'APA', 'BKR', 'BAC', 'BIO', 'BX', 'BA', 'CCL', 'CRL', 'CMA', 'STZ', 'GLW', 'DXCM', 'ENPH', 'EPAM', 'ETSY', 'EXR', 'FITB', 'FTNT', 'FCX', 'GNRC', 'HAL', 'INCY', 'KHC', 'LYV', 'MHK', 'MOS', 'NFLX', 'NEM', 'NWSA', 'NWS', 'NUE', 'NVDA', 'PCAR', 'PFE', 'RF', 'RVTY', 'RHI', 'ROL', 'SLB', 'SNPS', 'TDG', 'VLTO', 'WBD', 'WRK', 'WYNN', 'ZBH']
    # data_downloader.update_SP500_data()
    aapl_df = data_downloader.get_data_as_dataframe('AAPL')
    total_occurences = 0
    total_successes = 0
    high_winrate_list = []
    for ticker in tickers:
        df = data_downloader.get_data_as_dataframe(ticker)
        analyzer = StockAnalyzer(df)
        # occurences, successes, _ = analyzer.sliding_window_analysis(win_threshold=0.05)
        occurences, successes, _ = analyzer.sliding_window_alligator_analysis(threshold=0.05)

        total_occurences += occurences
        total_successes += successes
        if occurences > 0:
            winrate = successes/occurences
            if winrate > 0.6:
                high_winrate_list.append(ticker)
                print(f"{ticker}: winrate = {winrate}")
    print(f"total_occurences = {total_occurences}, total_successes = {total_successes}, winrate = {total_successes/total_occurences}")
    print(high_winrate_list)
    # screener = StockScreener()
    # squeen_expansion_stocks = screener.find_squeeze_expansion_stocks()
    # stage2_volume_contraction_stocks = screener.find_stage2_stocks()
    # print(squeen_expansion_stocks)
    # print(f'total of stage 2 stocks {len(stage2_volume_contraction_stocks)}')
if __name__ == '__main__':
   main()
