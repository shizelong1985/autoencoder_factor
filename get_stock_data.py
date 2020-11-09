"""
    Yahoo!ファイナンスより株価を収集
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pandas_datareader.data as pdr
import matplotlib.pyplot as plt

FILE_PATH = "data/TOPIX_core_large.xls"
# FILE_PATH = "data/TOPIX_一部上場.xls"


def load_asset_code_list(file_path):
    code_list = pd.read_excel(file_path)
    code_list = code_list.iloc[:, 1:3]
    code_list.columns = ['code', 'name']
    return code_list


def get_asset_return(asset_list, num_of_days):
    # データ取得期間を指定
    # TODO:
    date_start = datetime.today() - timedelta(days=num_of_days)
    date_end = datetime.today()

    # Yahoo!Financeより株価を取得
    asset_data = pd.DataFrame()
    for elem in asset_list.iterrows():
        asset_data[elem[1]["name"]] = pdr.DataReader(
            "{}.T".format(elem[1]["code"]), 'yahoo', date_start, date_end)['Adj Close']
    asset_data = asset_data.fillna(method='ffill')

    # リターンに変換
    asset_return = asset_data.pct_change()
    # asset_return = asset_return.dropna(axis=1)  # 上手く行かない...
    # asset_return.fillna(0, inplace=True)  # 0で補完はしない方が良さそう
    # asset_return = asset_return.dropna()  # NAを含む行は削除

    # リターンを返却
    return asset_return


def main():
    asset_list = load_asset_code_list(FILE_PATH)
    asset_return = get_asset_return(asset_list, 3600)  # 最大2年程度しか取得出来ない???
    asset_return.to_csv("return_data_TOPIX_all.csv", encoding="shift_jis")


if __name__ == '__main__':
    main()
