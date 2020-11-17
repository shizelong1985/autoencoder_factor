"""
    オートエンコーダを使用したファクターの生成
    元論文: https://www.sciencedirect.com/science/article/abs/pii/S0304407620301998
"""
from datetime import datetime
import math
import os

# keras
from keras import regularizers
from keras.callbacks import EarlyStopping, LearningRateScheduler
from keras.datasets import mnist
from keras.layers import Activation, Input, Dense, LeakyReLU, Reshape, dot, BatchNormalization, Flatten, Dropout
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional
from keras.models import Model
from keras.optimizers import Adam, SGD
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
NUM_PLOT = 20
FILE_PATH = "data/return_data.csv"
STATS_PATHS = {
    "VOL5": "data/vol_data_5days.csv",
    "VOL20": "data/vol_data_20days.csv",
    "MOM5": "data/mom_data_pct_5days.csv",
    "MOM20": "data/mom_data_pct_20days.csv",
    "LIQ": "data/ln_volume_data.csv",
}
NUM_FACTOR = 3
NUM_BETA_LAG = 1
NUM_EPOCH = 500
NUM_BATCH = 100
# LR = 0.01
TRAIN_START = '2011/1/1'
TRAIN_END = '2018/12/31'
TEST_START = '2019/1/1'
TEST_END = '2019/12/31'
IS_RECURRENT = False
IS_STATS = True

# 日付演算用に型変換
TRAIN_START = datetime.strptime(TRAIN_START, '%Y/%m/%d')
TRAIN_END = datetime.strptime(TRAIN_END, '%Y/%m/%d')
TEST_START = datetime.strptime(TEST_START, '%Y/%m/%d')
TEST_END = datetime.strptime(TEST_END, '%Y/%m/%d')


# ---------------------------------------------------------
# グラフ作成用関数
# ---------------------------------------------------------


def plot_learn_history(history, save_path="image/learn_history.png"):
    # 損失関数の推移を確認
    plt.plot(range(1, NUM_EPOCH+1), history['loss'], label="Training Loss")
    plt.plot(range(1, NUM_EPOCH+1),
             history['val_loss'], label="Validation Loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def plot_factors(x_data, factors, save_path):
    factors = factors.T

    # 平均0%・標準偏差年率10%にスケーリング（年間260日基準）
    for i in range(NUM_FACTOR):
        m = np.mean(factors[i])
        sd = np.std(factors[i])
        factors[i] = [(f - m) / sd * (0.1 / math.sqrt(260))
                      for f in factors[i]]
    factors = np.array(factors)

    # 100スタートのインデックスに変形
    for i in range(NUM_FACTOR):
        factors[i] = 100 * (1 + factors[i]).cumprod()

    # 比較してプロット
    plt.figure()
    for i in range(NUM_FACTOR):
        plt.plot(x_data.index, factors[i], label="factor_{}".format(i+1))
    plt.legend(loc='upper right')
    plt.savefig(save_path)
    plt.close()

    return factors


def round_data(x_data, quantile_point=0.01):
    down_quantiles = x_data.quantile(quantile_point)
    upper_quantiles = x_data.quantile(1.0-quantile_point)
    outliers_low = (x_data < down_quantiles)
    outliers_up = (x_data > upper_quantiles)
    x_data[outliers_low] = np.nan
    x_data = x_data.fillna(down_quantiles)
    x_data[outliers_up] = np.nan
    x_data = x_data.fillna(upper_quantiles)
    return x_data


# ---------------------------------------------------------
# 決定係数
# 検討したけど上手く行かなかった...
# see: https://statisticsbyjim.com/regression/interpret-adjusted-r-squared-predicted-r-squared-regression/
# ---------------------------------------------------------


def calcu_r_squared(ret, pred_ret, is_sep=False):
    # 決定係数を計算
    if not is_sep:
        ret = np.array(ret)
    sse = np.square((ret - pred_ret)).sum()
    sst = np.square(ret - ret.mean()).sum()
    return 1 - sse/sst


def press_statistic(y_true, y_pred, xs):
    """
    Calculation of the `Press Statistics <https://www.otexts.org/1580>`_
    """
    res = y_pred - y_true
    hat = xs.dot(np.linalg.pinv(xs))
    den = (1 - np.diagonal(hat))
    sqr = np.square(res/den)
    return sqr.sum()


def predicted_r2(y_true, y_pred, xs):
    """
    Calculation of the `Predicted R-squared <https://rpubs.com/RatherBit/102428>`_
    """
    press = press_statistic(y_true=y_true,
                            y_pred=y_pred,
                            xs=xs)

    sst = np.square(y_true - y_true.mean()).sum()
    return 1 - press / sst


# ---------------------------------------------------------
#  訓練データとテストデータを準備
# ---------------------------------------------------------
# データを読み込み
asset_return = pd.read_csv(FILE_PATH, encoding="shift_jis", index_col=0)
asset_return = asset_return.dropna(how="all").dropna(axis=1).dropna()

if IS_STATS:
    # 経済データを使用する場合
    asset_chars = pd.DataFrame()
    # asset_chars = asset_return
    for val in STATS_PATHS.values():
        asset_char = pd.read_csv(val, encoding="shift_jis", index_col=0).dropna(
            how="all").dropna(axis=1).dropna()
        # データを結合
        asset_chars = pd.concat([asset_chars, asset_char], axis=1).dropna()

    # 差分追加版
    asset_chars = pd.concat([asset_chars, asset_chars.diff(5)],
                            axis=1).shift(1).dropna()

    # asset charsは一日ラグを設ける
    asset_chars = asset_chars.shift(1).dropna()

    # 過去NUM_BETA_LAG日前～前日のデータを使用
    base_chars = asset_chars
    for i in range(NUM_BETA_LAG-1):
        base_chars = pd.concat([base_chars, asset_chars.shift(i+1)], axis=1)
    asset_chars = base_chars
    asset_chars = asset_chars.dropna()
else:
    # 経済データは使用せず、過去NUM_BETA_LAG日前～前日のデータを使用
    asset_chars = []
    for i in range(asset_return.shape[0] - NUM_BETA_LAG):
        asset_char = []
        for j in range(NUM_BETA_LAG):
            asset_char.append(np.array(asset_return.iloc[i+j, :]))
        asset_chars.append(asset_char)
    asset_chars = np.array(asset_chars)

# インデックスの型を変更し、ラグ部分のデータを削除
asset_return.index = np.array([datetime.strptime(x, '%Y/%m/%d')
                               for x in asset_return.index])
asset_return = asset_return.iloc[NUM_BETA_LAG:, :]

# 特徴データを使用する場合に使用
if IS_STATS:
    asset_chars.index = np.array([datetime.strptime(x, '%Y/%m/%d')
                                  for x in asset_chars.index])
    common_index = np.intersect1d(asset_return.index, asset_chars.index)
    # 共通部分のみ使用する
    asset_return = asset_return.loc[common_index]
    asset_chars = asset_chars.loc[common_index]

# サンプル分割を実施
is_train = (asset_return.index >= TRAIN_START) * \
    (asset_return.index <= TRAIN_END)
is_test = (asset_return.index >= TEST_START) * (asset_return.index <= TEST_END)

x_train = asset_return[is_train]
x_test = asset_return[is_test]
x_train_chars = asset_chars[is_train]
x_test_chars = asset_chars[is_test]

# 学習データの中で明らかに変な値があるので、丸め処理を実施
x_train = round_data(x_train, quantile_point=0.005)
# x_train_chars = round_data(x_train_chars)  # 何故かこっちだけ機能しない...
print(np.max(x_train.max()))

# 標準化：今は資産の特徴データにのみ適用
# x_scaler = StandardScaler()
# x_scaler.fit(x_train)
# x_train = x_scaler.transform(x_train)
# x_test = x_scaler.transform(x_test)
x_chars_scaler = StandardScaler()
x_chars_scaler.fit(x_train_chars)
x_train_chars = x_chars_scaler.transform(x_train_chars)
x_test_chars = x_chars_scaler.transform(x_test_chars)


# ---------------------------------------------------------
# ネットワークの定義
# - Dropoutをしないと過学習が酷い
# - 元論文を参考にl1正則化を使用
#   see also：https://liaoyuan.hatenablog.jp/entry/2018/02/13/231430
# ---------------------------------------------------------
input_shape = x_train.shape[1]

# ベータを生成する層
if IS_RECURRENT:
    # LSTM
    input_beta = Input(shape=(NUM_BETA_LAG, input_shape))
    x = Bidirectional(LSTM(64, batch_input_shape=(
        None, x_train_chars.shape[1], x_train_chars.shape[2]), recurrent_dropout=0.25))(input_beta)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.25)(x)
else:
    input_beta = Input(shape=(x_train_chars.shape[1],))
    # x = Flatten()(input_beta)
    x = Dense(32, kernel_regularizer=regularizers.l1(1e-4))(input_beta)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.25)(x)
x = Dense(32, kernel_regularizer=regularizers.l1(1e-4))(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.2)(x)
x = Dropout(0.25)(x)
x = Dense(16, kernel_regularizer=regularizers.l1(1e-4))(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.2)(x)
x = Dropout(0.50)(x)
x = Dense(NUM_FACTOR * input_shape, activation='linear')(x)
output_beta = Reshape((input_shape, NUM_FACTOR))(x)
model_beta = Model(inputs=input_beta, outputs=output_beta)  # beta確認用

# ファクターを生成する層
input_factor = Input(shape=(input_shape,))
output_factor = Dense(NUM_FACTOR, activation='linear',
                      kernel_regularizer=regularizers.l1(1e-4))(input_factor)
model_factor = Model(inputs=input_factor, outputs=output_factor)  # factor確認用
output = dot([output_beta, output_factor], axes=(2, 1))

# 上記を結合してオートエンコーダを定義
adam = Adam(lr=0.001)
autoencoder = Model(inputs=[input_beta, input_factor], output=output)
autoencoder.compile(optimizer=adam, loss='mean_squared_error')
autoencoder.summary()
plot_model(autoencoder, to_file="model_image.png",
           show_shapes=True, show_layer_names=False)  # モデルの構造を出力

# 学習率をエポックごとに調整 (see https://keras.io/ja/optimizers/)
# あまり上手く機能しなかった
# def step_decay(epoch):
#     x = LR
#     if epoch >= NUM_EPOCH * 0.25:
#         x *= 0.1
#     if epoch >= NUM_EPOCH * 0.50:
#         x *= 0.1
#     return x
# lr_decay = LearningRateScheduler(step_decay)


# 学習を実施
early_stopping = EarlyStopping(
    monitor='val_loss',
    min_delta=0.0,
    patience=10)

result = autoencoder.fit(x=[x_train_chars, x_train],
                         y=x_train,
                         validation_split=0.1,
                         epochs=NUM_EPOCH,
                         batch_size=NUM_BATCH,
                         callbacks=[early_stopping])
# 損失関数の推移を確認
plot_learn_history(result.history)

# 各ファクターを構成する銘柄の内訳情報を取得
factor_weight = autoencoder.get_layer(index=-2).get_weights()[0].T
norm_weight = pd.DataFrame()
for i, fw in enumerate(factor_weight):
    norm_weight["factor_{}".format(i+1)] = fw/sum(fw)  # ウェイト合計100%になるように
norm_weight.index = x_train.columns
norm_weight.to_csv("factor_weight.csv", encoding="shift_jis")

# ---------------------------------------------------------
# インサンプルの系列比較
# ---------------------------------------------------------
# デコードされたリターン系列を作成
decoded_return = autoencoder.predict([x_train_chars, x_train])
r_2s = np.array(calcu_r_squared(x_train, decoded_return, is_sep=True))

for num_index in range(NUM_PLOT):
    # 累積リターンに変更
    cum_ret = 100 * (1 + x_train.iloc[:, num_index]).cumprod()
    cum_ret_decoded = 100 * (1 + decoded_return[:, num_index]).cumprod()

    # 比較してプロット
    plt.figure()
    plt.title("{} R2={}".format(
        x_train.iloc[:, num_index].name, round(r_2s[num_index]*100, 2)), fontname="MS Gothic")
    plt.plot(x_train.index, cum_ret, label="Original")
    plt.plot(x_train.index, cum_ret_decoded, label="After AE")
    plt.legend(loc='upper right')
    # plt.show()
    plt.savefig("image/res_insample_{}.png".format(num_index))
    plt.close()

# モデルの評価を実施
print(calcu_r_squared(x_train, decoded_return))
# print(calcu_r_squared(x_train, decoded_return, is_sep=True))
factors = model_factor.predict(x_train)

# predicted_r2(np.array(x_train), decoded_return, factors)  # 多次元に未対応...
# pythonで計算するのを諦めてRのライブラリを使用する方向で調整
# Rでpredictive R-squaredを計算するために出力->上手く行かなかった
# np.savetxt('data/pred_R/y_true.csv', np.array(x_train), delimiter=',')
# np.savetxt('data/pred_R/y_pred.csv', decoded_return, delimiter=',')
# np.savetxt('data/pred_R/factor.csv', factors, delimiter=',')


# ファクター系列を確認・解釈
factors = model_factor.predict(x_train)
plot_factors(x_train, factors, "image/factor_insample.png")

# ベータ系列を確認・解釈
betas = model_beta.predict(x_train_chars)

for i_firm in range(NUM_PLOT):
    firm_name = x_train.iloc[:, i_firm].name
    for i_factor in range(NUM_FACTOR):
        beta = betas.T[i_factor][i_firm]

        plt.figure()
        plt.title(firm_name, fontname="MS Gothic")
        plt.plot(x_train.index, beta, label="beta_insample_factor_{}".format(
            i_factor+1))
        plt.legend(loc='upper right')
        # plt.show()
        plt.savefig(
            "image/beta_insample_{}_factor_{}".format(firm_name, i_factor+1))
        plt.close()

# ---------------------------------------------------------
# アウトオブサンプルの系列比較
# ---------------------------------------------------------
# デコードされたリターン系列を作成
decoded_return = autoencoder.predict([x_test_chars, x_test])
r_2s = np.array(calcu_r_squared(x_test, decoded_return, is_sep=True))

for num_index in range(NUM_PLOT):
    # 累積リターンに変更
    cum_ret = 100 * (1 + x_test.iloc[:, num_index]).cumprod()
    cum_ret_decoded = 100 * (1 + decoded_return[:, num_index]).cumprod()

    # 比較してプロット
    plt.figure()
    plt.title("{} R2={}".format(
        x_test.iloc[:, num_index].name, round(r_2s[num_index]*100, 2)), fontname="MS Gothic")
    plt.plot(x_test.index, cum_ret, label="Original")
    plt.plot(x_test.index, cum_ret_decoded, label="After AE")
    plt.legend(loc='upper right')
    # plt.show()
    plt.savefig("image/res_outsample_{}.png".format(num_index))
    plt.close()
print(calcu_r_squared(x_test, decoded_return))

# ファクター系列を確認・解釈
factors = model_factor.predict(x_test)
plot_factors(x_test, factors, "image/factor_outsample.png")


# ---------------------------------------------------------
# ファクター解釈用に経済データを追加
# - マクロファクターっぽい解釈は難しそう
# - 比較対象は株のスタイルファクターの方が良さげ？
# ---------------------------------------------------------
factors = model_factor.predict(x_train)
factors = factors.T

# 平均0%・標準偏差年率10%にスケーリング（年間260日基準）
for i in range(NUM_FACTOR):
    m = np.mean(factors[i])
    sd = np.std(factors[i])
    factors[i] = [(f - m) / sd * (0.1 / math.sqrt(260))
                  for f in factors[i]]
factor_data = pd.DataFrame(factors).T
factor_data.index = x_train.index
factor_data.columns = ["factor_{}".format(i+1) for i in range(NUM_FACTOR)]

# 経済データを取得
factor_data["日経225"] = pdr.DataReader(
    '^N225', 'yahoo', TRAIN_START, TRAIN_END)['Close'].pct_change()
factor_data["ドル円"] = pdr.DataReader(
    'JPY=X', 'yahoo', TRAIN_START, TRAIN_END)['Close'].pct_change()
# 日経225の特徴データ
nikkei = pdr.DataReader('^N225', 'yahoo', TRAIN_START, TRAIN_END)['Close']
factor_data["日経_MOM_PCT_{}D".format(5)] = nikkei.pct_change(5)
factor_data["日経_MOM_PCT_{}D".format(20)] = nikkei.pct_change(20)
factor_data["日経_VOL_DIFF_{}D".format(
    5)] = nikkei.rolling(5).std(ddof=0).diff(1)
factor_data["日経_VOL_DIFF_{}D".format(
    20)] = nikkei.rolling(20).std(ddof=0).diff(1)

# 結果を整形して出力
factor_data = factor_data.dropna()
factor_corr = factor_data.corr()
factor_corr.to_csv("factor_corr.csv", encoding="shift_jis")
# factor_data.to_csv("factor.csv", encoding="shift_jis")


# ---------------------------------------------------------
# PCAを用いた分解
# ---------------------------------------------------------
pca = PCA(n_components=NUM_FACTOR)
pca.fit(x_train)
pca_factors = pca.transform(x_train)
pca_factors = pca_factors.T

# 平均0%・標準偏差年率10%にスケーリング（年間260日基準）
for i in range(NUM_FACTOR):
    m = np.mean(pca_factors[i])
    sd = np.std(pca_factors[i])
    pca_factors[i] = [(f - m) / sd * (0.1 / math.sqrt(260))
                      for f in pca_factors[i]]
pca_factors = np.array(pca_factors)

# 100スタートのインデックスに
for i in range(NUM_FACTOR):
    pca_factors[i] = 100 * (1 + pca_factors[i]).cumprod()

# 比較してプロット
plt.figure()
for i in range(NUM_FACTOR):
    plt.plot(x_train.index, pca_factors[i],
             label="pca_factor_{}".format(i+1))
plt.legend(loc='upper right')
# plt.show()
plt.savefig("image/pca_factor_insample.png")
plt.close()

# ---------------------------------------------------------
# PCA:インサンプルの比較
# ---------------------------------------------------------
# デコードされたリターン系列を作成
decoded_return = pca.inverse_transform(pca.transform(x_train))

for num_index in range(NUM_PLOT):
    # 累積リターンに変更
    cum_ret = 100 * (1 + x_train.iloc[:, num_index]).cumprod()
    cum_ret_decoded = 100 * (1 + decoded_return[:, num_index]).cumprod()

    # 比較してプロット
    plt.figure()
    plt.title(asset_return.iloc[:, num_index].name, fontname="MS Gothic")
    plt.plot(x_train.index, cum_ret, label="Original")
    plt.plot(x_train.index, cum_ret_decoded, label="After PCA")
    plt.legend(loc='upper right')
    # plt.show()
    plt.savefig("image/PCA_insample_{}.png".format(num_index))

print(calcu_r_squared(x_train, decoded_return))

# ---------------------------------------------------------
# PCA:アウトサンプルの比較
# ---------------------------------------------------------
# デコードされたリターン系列を作成
decoded_return = pca.inverse_transform(pca.transform(x_test))

for num_index in range(NUM_PLOT):
    # 累積リターンに変更
    cum_ret = 100 * (1 + x_test.iloc[:, num_index]).cumprod()
    cum_ret_decoded = 100 * (1 + decoded_return[:, num_index]).cumprod()

    # 比較してプロット
    plt.figure()
    plt.title(asset_return.iloc[:, num_index+1].name, fontname="MS Gothic")
    plt.plot(x_test.index, cum_ret, label="Original")
    plt.plot(x_test.index, cum_ret_decoded, label="After PCA")
    plt.legend(loc='upper right')
    # plt.show()
    plt.savefig("image/PCA_res_insample_{}.png".format(num_index))

plt.close()
print(calcu_r_squared(x_test, decoded_return))
# 0.3338396309481597
