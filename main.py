import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.chdir('D:\\GitHub\\DS2')

# 小数点以下桁数の表示設定
np.set_printoptions(precision=3)
pd.options.display.precision = 3

# データ読み込み
df = pd.read_excel('test.xlsx', sheet_name='train')

# 読み込み結果の確認
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
df.head()

# 読み込みデータの情報確認
df.info()

# 各変数の要約統計量
df.describe()

# 多変量連関図の描画
sns.pairplot(df, plot_kws={'alpha':0.3})
plt.show()

# 各変数間の相関係数
df.corr()

# 列名の取得
col_df = df.columns

# 注目変数と他変数の相関を把握
df_c = df.loc[:, [col_df[1], col_df[3], col_df[5], col_df[6]]]
fig =plt.figure(figsize=(15,5))
ax = []
for i in np.arange(1,4):
    ax_add = fig.add_subplot(1, 3, i)
    ax.append(ax_add)
x_no = 3
color = ['red', 'green', 'blue']
for i in np.arange(0, 3):
    ax[i].scatter(df_c.iloc[:, i+1], df_c.iloc[:, 0], label=df_c.columns[i], color=color[i])
    ax[i].legend()
plt.show()


