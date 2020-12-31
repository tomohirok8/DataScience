import numpy as np
import pandas as pd
from pandas_datareader import data as wb
import matplotlib.pyplot as plt
import datetime



####### 株価を読み込み #######
start = datetime.datetime(2020, 1, 5)
end = datetime.datetime(2020, 2, 10)

df_yahoo = wb.DataReader('^IXIC', 'yahoo', start, end)
df_stooq = wb.DataReader('^GSPC', 'stooq', start, end)
site = 'stooq'

# ポートフォリオの株価をDataFrame化
tickers = ['PG', 'MSFT', 'F', 'GE']
mydata = pd.DataFrame()
for t in tickers:
    mydata[t] = wb.DataReader(t, site, start, end)['Close']
mydata = mydata.sort_values('Date')

# マーケットインデックスをDataFrame化
tickers_ind = ['^GSPC', '^IXIC', '^GDAXI']
ind_data = pd.DataFrame()
for t in tickers_ind:
    ind_data[t] = wb.DataReader(t, site, start, end)['Close']

# ポートフォリオの重み
weights = np.array([0.25, 0.25, 0.25, 0.25])

# 期間の初めの日を100として正規化する
(mydata / mydata.iloc[0] * 100).plot(figsize = (15, 6));
plt.show()
(ind_data / ind_data.iloc[0] * 100).plot(figsize=(15, 6));
plt.show()

# 利益率
returns = (mydata / mydata.shift(1)) - 1
returns_log = np.log(mydata / mydata.shift(1))
ind_returns = (ind_data / ind_data.shift(1)) - 1

# 利益率の平均（1年分＝1日分×250）
annual_returns = returns.mean() * 250
annual_returns_log = returns_log.mean() * 250
annual_ind_returns = ind_returns.mean() * 250

# 利益率の分散（1年分＝1日分×250**0.5）
annual_returns_log_std = returns_log.std() * 250 ** 0.5

# ポートフォリオの年間単純利益率
print(str(round(np.dot(annual_returns, weights), 5) * 100) + ' %')


####### 利益率の共分散、相関係数を計算 #######
# 共分散、相関係数を計算するために2つの項目を取り出す
corr_returns = returns_log[['PG', 'MSFT']]

# 利益率の分散を計算（1年分＝1日分×250）
returns_var1 = returns_log['PG'].var() * 250
returns_var2 = returns_log['MSFT'].var() * 250

# 共分散を計算（1年分＝1日分×250）
cov_matrix = corr_returns.cov() * 250

# 相関係数を計算
corr_matrix = corr_returns.corr()


####### ポートフォリオのリスクの計算 #######
# ポートフォリオの分散の計算
pfolio_var = np.dot(weights.T, np.dot(returns_log.cov() * 250, weights))

# ポートフォリオのボラティリティの計算
pfolio_vol = (np.dot(weights.T, np.dot(returns_log.cov() * 250, weights))) ** 0.5
print(str(round(pfolio_vol, 5) * 100) + ' %')

# 分散可能なリスクの計算（ポートフォリオの分散　－　重みを加味したそれぞれの株式の分散）
dr = pfolio_var
for i in range(len(tickers)):
    dr = dr - weights[i] ** 2 * returns_log.iloc[:,i].var() * 250
print(str(round(dr*100, 3)) + ' %')

# 分散不可能なリスクの計算
n_dr = pfolio_var - dr
print(str(round(n_dr*100, 3)) + ' %')





