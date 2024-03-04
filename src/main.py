import pandas as pd
from StockScreener import StockScreener
def main():
    screener = StockScreener()
    stage2_volume_contraction_stocks = screener.find_stage2_stocks()
    print(stage2_volume_contraction_stocks)
    print(f'total of stage 2 stocks {len(stage2_volume_contraction_stocks)}')
if __name__ == '__main__':
   main()
