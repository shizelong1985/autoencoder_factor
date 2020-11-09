"""
    オートエンコーダを使用したファクターの生成
    元論文: https://www.sciencedirect.com/science/article/abs/pii/S0304407620301998
"""
from datetime import datetime
import math
import os

# keras
from keras import regularizers
from keras.datasets import mnist
from keras.layers import Activation, Input, Dense, LeakyReLU, Reshape, dot, BatchNormalization, Flatten, Dropout
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.utils.vis_utils import plot_model  # to visualize model structure
# sklearn
import sklearn
from sklearn.decomposition import PCA

import numpy as np
import pandas as pd
import pandas_datareader.data as pdr
import matplotlib.pyplot as plt

# パラメタ
FILE_PATH = "return_data.csv"
NUM_FACTOR = 3
NUM_BETA_LAG = 5  # betaにX個
NUM_EPOCH = 1000
TRAIN_START = '2011/1/1'
TRAIN_END = '2018/12/31'
TEST_START = '2019/1/1'
TEST_END = '2019/12/31'
# TRAIN_START = '2011/1/1'
# TRAIN_END = '2019/12/31'
# TEST_START = '2020/1/1'
# TEST_END = '2020/10/31'


# 日付演算用に型変換
TRAIN_START = datetime.strptime(TRAIN_START, '%Y/%m/%d')
TRAIN_END = datetime.strptime(TRAIN_END, '%Y/%m/%d')
TEST_START = datetime.strptime(TEST_START, '%Y/%m/%d')
TEST_END = datetime.strptime(TEST_END, '%Y/%m/%d')


def plot_learn_history(history, save_path="image/learn_history.png"):
    # 損失関数の推移を確認
    plt.plot(range(1, NUM_EPOCH+1), history['loss'], label="Training Loss")
    plt.plot(range(1, NUM_EPOCH+1),
             history['val_loss'], label="Validation Loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    # plt.show()
    plt.savefig(save_path)


def calcu_r_squared(ret, pred_ret):
    # 決定係数を計算
    # 元論文に合わせて平均を引かない定式化
    upper = (ret - pred_ret).flatten()
    upper = np.sum(upper**2)
    lower = ret.flatten()
    lower = np.sum(lower**2)
    res = 1 - upper/lower
    return res


# ---------------------------------------------------------
#  訓練データとテストデータを準備
# ---------------------------------------------------------

# データを読み込み
asset_return = pd.read_csv(FILE_PATH, encoding="shift_jis")
asset_return = asset_return.iloc[1:, :].dropna(axis=1).dropna()
# サンプル分割用に日付をstrからdatetimeに変換
date_label = np.array([datetime.strptime(x, '%Y/%m/%d')
                       for x in asset_return.iloc[:, 0]])
asset_return.iloc[:, 0] = date_label

# サンプル分割を実施
x_train = asset_return[(asset_return.iloc[:, 0] >= TRAIN_START)
                       * (asset_return.iloc[:, 0] <= TRAIN_END)]
x_test = asset_return[(asset_return.iloc[:, 0] >= TEST_START)
                      * (asset_return.iloc[:, 0] <= TEST_END)]
x_train = np.array(x_train.iloc[:, 1:])
x_test = np.array(x_test.iloc[:, 1:])

# beta推計用にラグ付きデータを作成
# 訓練データ
x_train_beta = []
for i in range(x_train.shape[0]-NUM_BETA_LAG):
    elem = [np.array(x_train[i-j+NUM_BETA_LAG-1]) for j in range(NUM_BETA_LAG)]
    elem = np.array(elem)
    x_train_beta.append(elem)
x_train_beta = np.array(x_train_beta)

# テストデータ
x_test_beta = []
for i in range(x_test.shape[0]-NUM_BETA_LAG):
    elem = [np.array(x_test[i-j+NUM_BETA_LAG-1]) for j in range(NUM_BETA_LAG)]
    elem = np.array(elem)
    x_test_beta.append(elem)
x_test_beta = np.array(x_test_beta)

# 作図用に日付ラベルを用意
date_label_train = date_label[NUM_BETA_LAG:x_train.shape[0]]
date_label_test = date_label[(
    x_train.shape[0]+NUM_BETA_LAG):(x_train.shape[0]+x_test.shape[0])]

# ラグ分のデータは削除
x_train = x_train[NUM_BETA_LAG:]
x_test = x_test[NUM_BETA_LAG:]
input_shape = x_train.shape[1]  # インプットの次元数


# ---------------------------------------------------------
# ネットワークの定義
# NOTE:バッチ正規化をしないと過学習が酷い...
# ---------------------------------------------------------

# ベータを生成する層
input_beta = Input(shape=(NUM_BETA_LAG, input_shape))
x = Flatten()(input_beta)  # Flattenで一次元データに変換
x = Dense(64, activity_regularizer=regularizers.l1(1e-4))(x)
x = BatchNormalization()(x)  # 元論文を参考にバッチ正規化
x = LeakyReLU()(x)
x = Dropout(0.25)(x)
x = Dense(64)(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = Dropout(0.25)(x)
x = Dense(64)(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = Dropout(0.50)(x)
x = Dense(NUM_FACTOR * input_shape, activation='linear')(x)
output_beta = Reshape((input_shape, NUM_FACTOR))(x)
model_beta = Model(inputs=input_beta, outputs=output_beta)  # beta確認用

# ファクターを生成する層
input_factor = Input(shape=(input_shape,))
output_factor = Dense(NUM_FACTOR, activation='linear')(input_factor)
model_factor = Model(inputs=input_factor, outputs=output_factor)  # factor確認用
output = dot([output_beta, output_factor], axes=(2, 1))

# 上記を結合してオートエンコーダを定義
autoencoder = Model(inputs=[input_beta, input_factor], output=output)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')
autoencoder.summary()
plot_model(autoencoder, to_file="model_image.png",
           show_shapes=True, show_layer_names=False)  # モデルの構造を出力

# 学習を実施
result = autoencoder.fit(x=[x_train_beta, x_train],
                         y=x_train,
                         validation_split=0.1,
                         epochs=NUM_EPOCH,
                         batch_size=100)
# 損失関数の推移を確認
plot_learn_history(result.history)


# ---------------------------------------------------------
# インサンプルの系列比較
# ---------------------------------------------------------
# デコードされたリターン系列を作成
decoded_return = autoencoder.predict([x_train_beta, x_train])

for num_index in range(5):
    # 累積リターンに変更
    cum_ret = 100 * (1 + x_train[:, num_index]).cumprod()
    cum_ret_decoded = 100 * (1 + decoded_return[:, num_index]).cumprod()

    # 比較してプロット
    plt.figure()
    plt.title(asset_return.iloc[:, num_index+1].name, fontname="MS Gothic")
    plt.plot(date_label_train, cum_ret, label="Original")
    plt.plot(date_label_train, cum_ret_decoded, label="After AE")
    plt.legend(loc='upper right')
    # plt.show()
    plt.savefig("image/res_insample_{}.png".format(num_index))

print(calcu_r_squared(x_train, decoded_return))

# ---------------------------------------------------------
# アウトオブサンプルの系列比較
# ---------------------------------------------------------
# デコードされたリターン系列を作成
decoded_return = autoencoder.predict([x_test_beta, x_test])

for num_index in range(5):
    # 累積リターンに変更
    cum_ret = 100 * (1 + x_test[:, num_index]).cumprod()
    cum_ret_decoded = 100 * (1 + decoded_return[:, num_index]).cumprod()

    # 比較してプロット
    plt.figure()
    plt.title(asset_return.iloc[:, num_index+1].name, fontname="MS Gothic")
    plt.plot(date_label_test, cum_ret, label="Original")
    plt.plot(date_label_test, cum_ret_decoded, label="After AE")
    plt.legend(loc='upper right')
    # plt.show()
    plt.savefig("image/res_outsample_{}.png".format(num_index))

# 決定係数を計算
print(calcu_r_squared(x_test, decoded_return))

# ---------------------------------------------------------
# ファクター系列を確認・解釈
# ---------------------------------------------------------
factors = model_factor.predict([x_train])
factors = factors.T

# 平均0%・標準偏差年率10%にスケーリング（年間260日基準）
for i in range(NUM_FACTOR):
    m = np.mean(factors[i])
    sd = np.std(factors[i])
    factors[i] = [(f - m) / sd * (0.1 / math.sqrt(260)) for f in factors[i]]
factors = np.array(factors)

# 100スタートのインデックスに
for i in range(NUM_FACTOR):
    factors[i] = 100 * (1 + factors[i]).cumprod()

# 比較してプロット
plt.figure()
for i in range(NUM_FACTOR):
    plt.plot(date_label_train, factors[i], label="factor_{}".format(i+1))
plt.legend(loc='upper right')
# plt.show()
plt.savefig("image/factor_insample.png")

# ---------------------------------------------------------
# ファクター解釈用に経済データを追加
# - マクロファクターっぽい解釈は難しそう
# - 比較対象は株のスタイルファクターの方が良さげ？
# ---------------------------------------------------------
factor_data = pd.DataFrame(factors).T
factor_data.index = date_label_train
factor_data.columns = ["factor_{}".format(i+1) for i in range(NUM_FACTOR)]
factor_data["日経225"] = pdr.DataReader(
    '^N225', 'yahoo', TRAIN_START, TRAIN_END)['Close']
factor_data["ドル円"] = pdr.DataReader(
    'JPY=X', 'yahoo', TRAIN_START, TRAIN_END)['Close']
factor_data["米10年金利"] = pdr.DataReader(
    'DGS10', 'fred', TRAIN_START, TRAIN_END)
factor_data = factor_data.dropna()
factor_data.to_csv("factor.csv", encoding="shift_jis")


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
    plt.plot(date_label_train, pca_factors[i],
             label="pca_factor_{}".format(i+1))
plt.legend(loc='upper right')
# plt.show()
plt.savefig("image/pca_factor_insample.png")

# ---------------------------------------------------------
# PCA:インサンプルの比較
# ---------------------------------------------------------
# デコードされたリターン系列を作成
decoded_return = pca.inverse_transform(pca.transform(x_train))

for num_index in range(5):
    # 累積リターンに変更
    cum_ret = 100 * (1 + x_train[:, num_index]).cumprod()
    cum_ret_decoded = 100 * (1 + decoded_return[:, num_index]).cumprod()

    # 比較してプロット
    plt.figure()
    plt.title(asset_return.iloc[:, num_index+1].name, fontname="MS Gothic")
    plt.plot(date_label_train, cum_ret, label="Original")
    plt.plot(date_label_train, cum_ret_decoded, label="After PCA")
    plt.legend(loc='upper right')
    # plt.show()
    plt.savefig("image/PCA_insample_{}.png".format(num_index))

print(calcu_r_squared(x_train, decoded_return))

# ---------------------------------------------------------
# PCA:アウトサンプルの比較
# ---------------------------------------------------------
# デコードされたリターン系列を作成
decoded_return = pca.inverse_transform(pca.transform(x_test))

for num_index in range(5):
    # 累積リターンに変更
    cum_ret = 100 * (1 + x_test[:, num_index]).cumprod()
    cum_ret_decoded = 100 * (1 + decoded_return[:, num_index]).cumprod()

    # 比較してプロット
    plt.figure()
    plt.title(asset_return.iloc[:, num_index+1].name, fontname="MS Gothic")
    plt.plot(date_label_test, cum_ret, label="Original")
    plt.plot(date_label_test, cum_ret_decoded, label="After PCA")
    plt.legend(loc='upper right')
    # plt.show()
    plt.savefig("image/PCA_res_insample_{}.png".format(num_index))

print(calcu_r_squared(x_test, decoded_return))
# 0.3435499165585959
