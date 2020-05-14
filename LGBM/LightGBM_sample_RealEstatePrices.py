import os
os.chdir('D:\\GitHub\\datascience')
import sys 
import numpy as np
import pandas as pd

import lightgbm as lgb
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns



# data load
data = pd.read_csv("dummy_data.csv")
#train test split
train_data, test_data = train_test_split(data, test_size=5000, random_state=0)
print(f'num of train data : {len(train_data)}')
print(f'num of test data : {len(test_data)}')


# 特徴作成
def create_feature(data):
    feature = data[["駅名", "建物構造", "徒歩分", "専有面積", "間取り", "部屋数", "築年", "所在階"]].copy()
    feature["築年"] = pd.to_datetime(feature["築年"].fillna(0).astype(int).astype(str), format="%Y%m", errors="coerce")
    feature["築年"] = (feature["築年"] - pd.to_datetime("1900-01-01")).dt.total_seconds() #日付を連続値に変換
    feature["所在階"] = feature["所在階"].str.replace('B', '-').astype(int) #地下表記を数字に変換
    
    cat_cols = ['駅名', '建物構造', '間取り']
    for cat in cat_cols:
        feature[cat] = feature[cat].astype("category")
    feature = feature.rename(columns={'駅名': 'station', '建物構造': 'structure', '徒歩分': 'walk_min', '専有面積': 'area', '間取り': 'room_type', '部屋数': 'room', '築年': 'age', '所在階': 'floor'})
    
    return feature
    
X_train = create_feature(train_data)
y_train = train_data["成約価格"]


# LightGBMの学習
X_trn, X_val, y_trn, y_val = train_test_split(X_train, y_train, test_size=5000, random_state=0)

lgb_dataset_trn = lgb.Dataset(X_trn, label=y_trn, categorical_feature='auto')
lgb_dataset_val = lgb.Dataset(X_val, label=y_val, categorical_feature='auto')

params = {
    'objective' : 'rmse', 
    'learning_rate' : 0.1, 
    'max_depth' : 4, 
}

def mape_func(y_pred, data):
    y_true = data.get_label()
    mape = calc_mape(y_true, y_pred)
    return 'mape', mape, False

result_dic ={}
model = lgb.train(
    params=params, 
    train_set=lgb_dataset_trn, 
    valid_sets=[lgb_dataset_trn, lgb_dataset_val], 
    feval=mape_func, 
    num_boost_round=10000, 
    early_stopping_rounds=100, 
    verbose_eval=100,
    evals_result=result_dic
)

# 学習経過を表示
result_df = pd.DataFrame(result_dic['training']).add_prefix('train_').join(pd.DataFrame(result_dic['valid_1']).add_prefix('valid_'))
fig, ax = plt.subplots(figsize=(10, 6))
result_df[['train_mape', 'valid_mape']].plot(ax=ax)
ax.set_ylabel('MAPE [%]')
ax.set_xlabel('num of iteration')
ax.set_ylim(2, 8)
ax.grid()

# validの確認
def calc_mape(y_true, y_pred):
    data_num = len(y_true)
    mape = (np.sum(np.abs(y_pred-y_true)/y_true)/data_num)*100
    return mape

train_pred = model.predict(X_train)
train_mape = calc_mape(y_train.values, train_pred)
val_pred = model.predict(X_val)
val_mape = calc_mape(y_val.values, val_pred)
print(f'train mape : {train_mape:.3f}%')
print(f'valid mape : {val_mape:.3f}%')


# testデータの予測
X_test = create_feature(test_data)
y_test = test_data["成約価格"]

test_pred = model.predict(X_test)
test_mape = calc_mape(y_test.values, test_pred)
print(f'test mape : {test_mape:.3f}%')


plt.rcParams["font.family"] = "IPAexGothic"
feature_importance = pd.DataFrame({
    'feature_name' : model.feature_name(),
    'importance' : model.feature_importance(importance_type='gain'), 
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

plt.figure(figsize = (6, 6))
sns.barplot(data=feature_importance, x='importance', y='feature_name')
plt.savefig('feature_importance.png')


data = data.append(pd.DataFrame.from_dict({
    "駅名": ["新橋"],
    "建物構造": ["SRC"],
    "徒歩分": [10],
    "専有面積": [30],
    "間取り": ["R"],
    "部屋数": [1],
    "築年": ["198001"],
    "所在階": ["5"],
    "成約価格": [-1]
})).reset_index(drop=True)
X = create_feature(data)
model.predict(X[-1:])
