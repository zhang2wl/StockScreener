import pandas as pd
from StockScreener import StockScreener
from StockDataDownloader import StockDataDownloader
def main():
    data_downloader = StockDataDownloader()
    data_downloader.update_SP500_data()
    screener = StockScreener()
    squeen_expansion_stocks = screener.find_squeeze_expansion_stocks()
    # stage2_volume_contraction_stocks = screener.find_stage2_stocks()
    print(squeen_expansion_stocks)
    # print(f'total of stage 2 stocks {len(stage2_volume_contraction_stocks)}')
if __name__ == '__main__':
   main()
