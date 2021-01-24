import numpy as np
import pandas as pd
from pandas_datareader import data as wb
import matplotlib.pyplot as plt
import datetime
from scipy.stats import norm



####### 株価を読み込み #######
start = datetime.datetime(2018, 1, 5)
end = datetime.datetime(2021, 1, 15)

site = 'stooq'

# ポートフォリオの株価をDataFrame化
assets = [#'APPL',
          'BAC',
          'CPRT',
          'KHC',
          'KO',
          'NVDA',
          'PG',
          'T']
mydata = pd.DataFrame()
for t in assets:
    mydata[t] = wb.DataReader(t, site, start, end)['Close']
mydata = mydata.sort_values('Date')

# マーケットインデックスをDataFrame化
assets_ind = ['^NKX', '^DJI', '^NDQ', '^SPX']
ind_data = pd.DataFrame()
for t in assets_ind:
    ind_data[t] = wb.DataReader(t, site, start, end)['Close']
ind_data = ind_data.sort_values('Date')

# ポートフォリオの重み
weights = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.4])

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
print('ポートフォリオのボラティリティ : ', str(round(pfolio_vol, 5) * 100) + ' %')

# 分散可能なリスクの計算（ポートフォリオの分散　－　重みを加味したそれぞれの株式の分散）
dr = pfolio_var
for i in range(len(assets)):
    dr = dr - weights[i] ** 2 * returns_log.iloc[:,i].var() * 250
print('分散可能なリスク : ', str(round(dr*100, 3)) + ' %')

# 分散不可能なリスクの計算
n_dr = pfolio_var - dr
print('分散不可能なリスク : ', str(round(n_dr*100, 3)) + ' %')


####### 効率的フロンティアの計算 #######
num_assets = len(assets)

pfolio_returns = []
pfolio_volatilities = []
for x in range (1000):
    # 重みの組み合わせをランダムに作成し、合計が1となるように調整
    weights_a = np.random.random(num_assets)
    weights_a /= np.sum(weights_a)
    # ポートフォリオの予想利益率
    pfolio_returns.append(np.sum(weights_a * returns_log.mean()) * 250)
    # ポートフォリオの予想されるボラティリティ
    pfolio_volatilities.append(np.sqrt(np.dot(weights_a.T,np.dot(returns_log.cov() * 250, weights_a))))
    
pfolio_returns = np.array(pfolio_returns)
pfolio_volatilities = np.array(pfolio_volatilities)

portfolios = pd.DataFrame({'Return': pfolio_returns, 'Volatility': pfolio_volatilities})

portfolios.plot(x='Volatility', y='Return', kind='scatter', figsize=(10, 6));
plt.xlabel('Expected Volatility')
plt.ylabel('Expected Return')


####### 資本資産評価モデルの計算 #######
# 直近5年で計算するのが一般的
df_beta = pd.concat([mydata['KO'], ind_data['^SPX']], axis=1)

beta_returns = np.log(df_beta / df_beta.shift(1))
beta_cov = beta_returns.cov() * 250
cov_with_market = beta_cov.iloc[0,1]
market_var = beta_returns['^SPX'].var() * 250

# ベータの計算
beta = cov_with_market / market_var
print('ベータ : ', str(round(beta, 3)))

# 株式の期待利益率を計算
rf = 0.6 # リスクフリーレート（10年債の利率）
rp = 5.5 # マーケットのリスクプレミアム
er = rf / 100 + beta * (rp - rf) / 100
print('株式の期待利益率 : ', str(round(er*100, 3)) + ' %')

# シャープレシオの計算
sharpe_ratio = (er - rf / 100) / (beta_returns['KO'].std() * 250 ** 0.5)
print('シャープレシオ : ', str(round(sharpe_ratio, 3)))


####### モンテカルロシミュレーションによる株価予測 #######
u_MC = returns_log['KO'].mean()
var_MC = returns_log['KO'].var()
drift_MC = u_MC - (0.5 * var_MC)
stdev_MC = returns_log['KO'].std()

t_intervals = 1000
iterations = 10

daily_returns = np.exp(drift_MC + stdev_MC * norm.ppf(np.random.rand(t_intervals, iterations)))

S0 = mydata['KO'].iloc[-1]

price_list = np.zeros_like(daily_returns)
price_list[0] = S0
for t in range(1, t_intervals):
    price_list[t] = price_list[t - 1] * daily_returns[t]

plt.figure(figsize=(10,6))
plt.plot(price_list)


####### ブラックショールズ方程式によるオプション価格 #######
def d1(S, K, r, stdev, T):
    return (np.log(S / K) + (r + stdev ** 2 / 2) * T) / (stdev * np.sqrt(T))
 
def d2(S, K, r, stdev, T):
    return (np.log(S / K) + (r - stdev ** 2 / 2) * T) / (stdev * np.sqrt(T))

def BSM(S, K, r, stdev, T):
    return (S * norm.cdf(d1(S, K, r, stdev, T))) - (K * np.exp(-r * T) * norm.cdf(d2(S, K, r, stdev, T)))

# 現在の株価
S = mydata['KO'].iloc[-1]

stdev_BS = returns_log['KO'].std() * 250 ** 0.5
r = 0.006 # リスクフリーレート
K = 50.0 # ストライクプライス
T = 1 # 経過時間（年）

value_BSM = BSM(S, K, r, stdev_BS, T)
print('オプション価格 : ', str(round(value_BSM, 3)))


####### モンテカルロシミュレーションによるオプション価格 #######
stdev_MCBS = returns_log['KO'].std() * 250 ** 0.5
T_MCBS = 1.0
t_intervals_MCBS = 250
delta_t = T_MCBS / t_intervals_MCBS
iterations_MCBS = 10000  
Z_MCBS = np.random.standard_normal((t_intervals_MCBS + 1, iterations_MCBS))  
S_MCBS = np.zeros_like(Z_MCBS) 
S0_MCBS = mydata['KO'].iloc[-1]
S_MCBS[0] = S0_MCBS

for t in range(1, t_intervals_MCBS + 1):
    S_MCBS[t] = S_MCBS[t-1] * np.exp((r - 0.5 * stdev_MCBS ** 2) * delta_t + stdev_MCBS * delta_t ** 0.5 * Z_MCBS[t])

plt.figure(figsize=(10, 6))
plt.plot(S_MCBS[:, :10])

p = np.maximum(S_MCBS[-1] - 50, 0)

C = np.exp(-r * T_MCBS) * np.sum(p) / iterations_MCBS
print('オプション価格 : ', str(round(C, 3)))










