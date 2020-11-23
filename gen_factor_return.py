"""
    オートエンコーダを使用したファクターの生成
    元論文: https://www.sciencedirect.com/science/article/abs/pii/S0304407620301998
"""
from datetime import datetime
import math
import os

# keras
from keras import backend, regularizers
from keras.callbacks import EarlyStopping, LearningRateScheduler
from keras.datasets import mnist
from keras.layers import BatchNormalization, Dense, dot, Dropout, Flatten, Input, LeakyReLU, Multiply, ReLU, Reshape
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional
from keras.models import Model
from keras.optimizers import Nadam, Adam, SGD
from keras.utils.vis_utils import plot_model  # to visualize model structure
# sklearn
import sklearn
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import numpy as np
import pandas as pd
import pandas_datareader.data as pdr
import matplotlib.pyplot as plt

# パラメタ
FILE_PATH = "data/index_data.csv"
TRAIN_START = '2011/1/1'
TRAIN_END = '2018/12/31'
TEST_START = '2019/1/1'
TEST_END = '2019/12/31'

# 日付演算用に型変換
TRAIN_START = datetime.strptime(TRAIN_START, '%Y/%m/%d')
TRAIN_END = datetime.strptime(TRAIN_END, '%Y/%m/%d')
TEST_START = datetime.strptime(TEST_START, '%Y/%m/%d')
TEST_END = datetime.strptime(TEST_END, '%Y/%m/%d')

# ---------------------------------------------------------
#  訓練データとテストデータを準備
# ---------------------------------------------------------
# データを読み込み
asset_index = pd.read_csv(FILE_PATH, encoding="shift_jis", index_col=0)
asset_index = asset_index.dropna(how="all").dropna(axis=1).dropna()

# 前日と全く同じ指数値の場合には取引なしと見なして削除
dup_list = []
for i in range(asset_index.index.shape[0]-1):
    if all(asset_index.iloc[i+1, :] == asset_index.iloc[i, :]):
        dup_list.append(i+1)
asset_index = asset_index.drop(asset_index.index[dup_list])
asset_return = asset_index.pct_change().dropna()

# ---------------------------------------------------------
#  特徴データ
# ---------------------------------------------------------
# モメンタム
asset_mom = asset_index.pct_change(250).dropna().shift(1).dropna()
# ボラティリティ
asset_vol = asset_return.rolling(250).std(ddof=0).shift(1).dropna()
# 共通部分のみ使用する
asset_return.index = np.array([datetime.strptime(x, '%Y/%m/%d')
                               for x in asset_return.index])
asset_mom.index = np.array([datetime.strptime(x, '%Y/%m/%d')
                            for x in asset_mom.index])
common_index = np.intersect1d(asset_return.index, asset_mom.index)
asset_return = asset_return.loc[common_index]
asset_mom = asset_mom.loc[common_index]

# （モメンタム－リバーサル）ファクター
mom_factor_ret = []
for i in range(asset_mom.shape[0]):
    curr_char = asset_mom.iloc[i, :]
    curr_ret = asset_return.iloc[i, :]
    lower_q = curr_char.quantile(0.2)
    upper_q = curr_char.quantile(0.8)
    lower_index = curr_char[curr_char < lower_q].index
    upper_index = curr_char[curr_char > upper_q].index
    ret = curr_ret[upper_index].mean() - curr_ret[lower_index].mean()
    mom_factor_ret.append(ret)
mom_factor_ret = pd.DataFrame(mom_factor_ret, index=asset_mom.index)
mom_factor_ret.to_csv("stats/mom_factor_ret.csv")

# （高ボラ－低ボラ）ファクター
vol_factor_ret = []
for i in range(asset_vol.shape[0]):
    curr_char = asset_vol.iloc[i, :]
    curr_ret = asset_return.iloc[i, :]
    lower_q = curr_char.quantile(0.2)
    upper_q = curr_char.quantile(0.8)
    lower_index = curr_char[curr_char < lower_q].index
    upper_index = curr_char[curr_char > upper_q].index
    ret = curr_ret[upper_index].mean() - curr_ret[lower_index].mean()
    vol_factor_ret.append(ret)
vol_factor_ret = pd.DataFrame(vol_factor_ret, index=asset_vol.index)
vol_factor_ret.to_csv("stats/vol_factor_ret.csv")
