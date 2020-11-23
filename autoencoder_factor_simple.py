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
from keras.layers import BatchNormalization, Dense, dot, Dropout, Flatten, Input, LeakyReLU, ReLU, Reshape
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
NUM_PLOT = 20
FILE_PATH = "data/index_data.csv"
STATS_PATHS = {
    "LIQ": "data/volume_data.csv",
}
NUM_FACTOR = 7
NUM_BETA_LAG = 20
NUM_EPOCH = 500
NUM_BATCH = 100
LR = 0.01
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


def plot_scatter_chart(x_true, x_pred, title):
    plt.scatter(np.array(x_true).flatten(),
                np.array(x_pred).flatten(),
                s=1)
    plt.plot(np.linspace(-0.2, 0.2, 1000),
             np.linspace(-0.2, 0.2, 1000),
             color="black")
    plt.title("R2:{}".format(round(calcu_r_squared(x_true, x_pred)*100, 2)),
              fontname="MS Gothic")
    plt.xlabel("実リターン", fontname="MS Gothic")
    plt.ylabel("推定リターン", fontname="MS Gothic")
    plt.xlim(-0.2, 0.2)
    plt.ylim(-0.2, 0.2)
    plt.grid(True)
    plt.savefig("image/return_scatter_{}".format(title))
    plt.show()


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
# see: https://statisticsbyjim.com/regression/interpret-adjusted-r-squared-predicted-r-squared-regression/
# ---------------------------------------------------------


def calcu_r_squared(ret, pred_ret, is_sep=False):
    # 決定係数を計算
    if not is_sep:
        ret = np.array(ret)
    sse = np.square((ret - pred_ret)).sum()
    sst = np.square(ret - ret.mean()).sum()
    return 1 - sse/sst


def calcu_pred_r_squared(x_data, factors, betas):
    # factorのt-1までの平均
    factors = pd.DataFrame(factors)
    factors.index = x_data.index
    factors.columns = ["factor_{}".format(i+1) for i in range(NUM_FACTOR)]
    factors = factors.expanding().mean().shift(1).dropna()

    # 個別のベータを取得してbeta*上記の平均を計算
    upper_sum = 0
    for i_firm in range(x_data.shape[1]):
        f_times_b = []
        for i_factor in range(NUM_FACTOR):
            beta = betas.T[i_factor][i_firm][1:]
            factor = factors.iloc[:, i_factor]
            f_times_b.append(beta * factor)
        f_times_b = pd.DataFrame(f_times_b).T.sum(axis=1)
        ret = x_data.iloc[1:, i_firm]
        upper_sum += np.square(ret - f_times_b).sum()
    # 合算してR-squaredに
    lower_sum = np.square(x_data.iloc[1:, :]).sum().sum()
    pred_r_squared = 1 - upper_sum / lower_sum
    return pred_r_squared


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

# 過去NUM_BETA_LAG日前～前日のデータを使用
asset_chars = asset_return.shift(1)
for i in range(NUM_BETA_LAG-1):
    asset_chars = pd.concat(
        [asset_chars, asset_return.shift(i+2)], axis=1)
asset_chars = asset_chars.dropna()

# 共通部分のみ使用する
asset_return.index = np.array([datetime.strptime(x, '%Y/%m/%d')
                               for x in asset_return.index])
asset_chars.index = np.array([datetime.strptime(x, '%Y/%m/%d')
                              for x in asset_chars.index])
common_index = np.intersect1d(asset_return.index, asset_chars.index)
asset_return = asset_return.loc[common_index]
asset_chars = asset_chars.loc[common_index]

# サンプル分割を実施
is_train = (asset_return.index >= TRAIN_START) * \
    (asset_return.index <= TRAIN_END)
is_test = (asset_return.index >= TEST_START) * (asset_return.index <= TEST_END)
x_train_org = asset_return[is_train]
x_test_org = asset_return[is_test]
x_train_chars = asset_chars[is_train]
x_test_chars = asset_chars[is_test]

# 学習データの中で明らかに変な値があるので、丸め処理を実施
# TRデータの場合には実施しない
# x_train_org = round_data(x_train_org, quantile_point=0.005)
# x_train_chars = round_data(x_train_chars, quantile_point=0.005)  # 何故かこっちだけ機能しない
print(np.max(x_train_org.max()))

# 学習時は標準化
x_scaler = StandardScaler()
x_scaler.fit(x_train_org)
x_train = pd.DataFrame(x_scaler.transform(x_train_org),
                       index=x_train_org.index, columns=x_train_org.columns)
x_test = pd.DataFrame(x_scaler.transform(x_test_org),
                      index=x_test_org.index, columns=x_test_org.columns)
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
    x = Dropout(0.20)(x)
else:
    input_beta = Input(shape=(x_train_chars.shape[1],))
    x = Dense(256, kernel_regularizer=regularizers.l1(1e-3))(input_beta)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dropout(0.20)(x)
x = Dense(128, kernel_regularizer=regularizers.l1(1e-3))(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.3)(x)
x = Dropout(0.50)(x)
x = Dense(128, kernel_regularizer=regularizers.l1(1e-3))(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.3)(x)
x = Dropout(0.50)(x)
x = Dense(NUM_FACTOR * input_shape, activation='linear')(x)
output_beta = Reshape((input_shape, NUM_FACTOR))(x)
model_beta = Model(inputs=input_beta, outputs=output_beta)  # beta確認用

# ファクターを生成する層
input_factor = Input(shape=(input_shape,))
output_factor = Dense(NUM_FACTOR, activation='linear',
                      kernel_regularizer=regularizers.l1(1e-3))(input_factor)
model_factor = Model(inputs=input_factor, outputs=output_factor)  # factor確認用
output = dot([output_beta, output_factor], axes=(2, 1))

# 上記を結合してオートエンコーダを定義
# adam = Adam(lr=0.001)
nadam = Nadam()
autoencoder = Model(inputs=[input_beta, input_factor], output=output)
autoencoder.compile(optimizer=nadam, loss='mean_squared_error')
autoencoder.summary()
plot_model(autoencoder, to_file="image/model_image.png",
           show_shapes=True, show_layer_names=False)  # モデルの構造を出力


def step_decay(epoch):
    # 学習率をエポックごとに調整
    # see: https://keras.io/ja/optimizers/
    x = LR
    if epoch >= NUM_EPOCH * 0.25:
        x *= 0.1
    if epoch >= NUM_EPOCH * 0.50:
        x *= 0.1
    return x


# 早期終了を実施
# early_stopping = EarlyStopping(
#     monitor='val_loss',
#     min_delta=0.0,
#     patience=10)
lr_decay = LearningRateScheduler(step_decay)
result = autoencoder.fit(x=[x_train_chars, x_train],
                         y=x_train,
                         validation_split=0.1,
                         epochs=NUM_EPOCH,
                         batch_size=NUM_BATCH,
                         callbacks=[lr_decay])
# 損失関数の推移を確認
plot_learn_history(result.history)

# 各ファクターを構成する銘柄の内訳情報を取得
factor_weight = autoencoder.get_layer(index=-2).get_weights()[0].T
norm_weight = pd.DataFrame()
for i, fw in enumerate(factor_weight):
    norm_weight["factor_{}".format(i+1)] = fw/sum(fw)  # ウェイト合計100%になるように
norm_weight.index = x_train.columns
norm_weight.to_csv("stats/factor_weight.csv", encoding="shift_jis")


# ---------------------------------------------------------
# インサンプルの系列比較
# ---------------------------------------------------------
# デコードされたリターン系列を作成
decoded_return = autoencoder.predict([x_train_chars, x_train])
decoded_return = x_scaler.inverse_transform(decoded_return)
r_2s = np.array(calcu_r_squared(x_train_org, decoded_return, is_sep=True))

for num_index in range(NUM_PLOT):
    # 累積リターンに変更
    cum_ret = 100 * (1 + x_train_org.iloc[:, num_index]).cumprod()
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

# 決定係数ベースでモデルの評価を実施
print(calcu_r_squared(x_train_org, decoded_return))
# predictive R-squared
# factors = model_factor.predict(x_train)
# betas = model_beta.predict(x_train_chars)
# print("pred R2:{}".format(calcu_pred_r_squared(x_train, factors, betas)))

# ファクター系列を確認
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

# 散布図（全リターン）
plot_scatter_chart(x_train_org, decoded_return, "insample")


# ---------------------------------------------------------
# アウトオブサンプルの系列比較
# ---------------------------------------------------------
# デコードされたリターン系列を作成
decoded_return = autoencoder.predict([x_test_chars, x_test])
decoded_return = x_scaler.inverse_transform(decoded_return)
r_2s = np.array(calcu_r_squared(x_test, decoded_return, is_sep=True))

for num_index in range(NUM_PLOT):
    # 累積リターンに変更
    cum_ret = 100 * (1 + x_test_org.iloc[:, num_index]).cumprod()
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
print(calcu_r_squared(x_test_org, decoded_return))

# ファクター系列を確認
factors = model_factor.predict(x_test)
plot_factors(x_test, factors, "image/factor_outsample.png")

# 散布図（全リターン）
plot_scatter_chart(x_test_org, decoded_return, "outsample")


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

# MOMとVOL
asset_mom = pd.read_csv("stats/mom_factor_ret.csv", encoding="shift_jis", index_col=0).dropna()
asset_vol = pd.read_csv("stats/vol_factor_ret.csv", encoding="shift_jis", index_col=0).dropna()
asset_mom.index = np.array([datetime.strptime(x, '%Y/%m/%d')
                               for x in asset_mom.index])
asset_vol.index = np.array([datetime.strptime(x, '%Y/%m/%d')
                               for x in asset_vol.index])
# 共通部分のみ使用する
factor_data["MOM"] = asset_mom
factor_data["VOL"] = asset_vol
factor_data = factor_data.dropna()

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
decoded_return = x_scaler.inverse_transform(decoded_return)

for num_index in range(NUM_PLOT):
    # 累積リターンに変更
    cum_ret = 100 * (1 + x_train_org.iloc[:, num_index]).cumprod()
    cum_ret_decoded = 100 * (1 + decoded_return[:, num_index]).cumprod()

    # 比較してプロット
    plt.figure()
    plt.title(asset_return.iloc[:, num_index].name, fontname="MS Gothic")
    plt.plot(x_train.index, cum_ret, label="Original")
    plt.plot(x_train.index, cum_ret_decoded, label="After PCA")
    plt.legend(loc='upper right')
    # plt.show()
    plt.savefig("image/PCA_insample_{}.png".format(num_index))

print(calcu_r_squared(x_train_org, decoded_return))
plot_scatter_chart(x_train_org, decoded_return, "PCAinsample")

# ---------------------------------------------------------
# PCA:アウトサンプルの比較
# ---------------------------------------------------------
# デコードされたリターン系列を作成
decoded_return = pca.inverse_transform(pca.transform(x_test))
decoded_return = x_scaler.inverse_transform(decoded_return)

for num_index in range(NUM_PLOT):
    # 累積リターンに変更
    cum_ret = 100 * (1 + x_test_org.iloc[:, num_index]).cumprod()
    cum_ret_decoded = 100 * (1 + decoded_return[:, num_index]).cumprod()

    # 比較してプロット
    plt.figure()
    plt.title(asset_return.iloc[:, num_index+1].name, fontname="MS Gothic")
    plt.plot(x_test.index, cum_ret, label="Original")
    plt.plot(x_test.index, cum_ret_decoded, label="After PCA")
    plt.legend(loc='upper right')
    # plt.show()
    plt.savefig("image/PCA_res_outsample_{}.png".format(num_index))
    plt.close()
print(calcu_r_squared(x_test_org, decoded_return))
plot_scatter_chart(x_test_org, decoded_return, "PCAoutsample")

# ファクター分析用に系列を生成
pca_factors = pca.transform(x_train)
pca_factors = pca_factors.T

# 平均0%・標準偏差年率10%にスケーリング（年間260日基準）
for i in range(NUM_FACTOR):
    m = np.mean(pca_factors[i])
    sd = np.std(pca_factors[i])
    pca_factors[i] = [(f - m) / sd * (0.1 / math.sqrt(260))
                      for f in pca_factors[i]]
pca_factors = np.array(pca_factors)

for i, factor_val in enumerate(pca_factors):
    factor_data["pca_factor_{}".format(i)] = factor_val

# 結果を出力
factor_corr = factor_data.corr()
factor_corr.to_csv("factor_corr.csv", encoding="shift_jis")
# factor_data.to_csv("factor.csv", encoding="shift_jis")
