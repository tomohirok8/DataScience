import os
os.chdir('D:\\GitHub\\datascience')
import sys 
import numpy as np
import pandas as pd
from collections import Counter

import lightgbm as lgb
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns



# データ読み込み
train_data = pd.read_table("D:/GitHub/SIGNATE/exercises_mpg/train.tsv")
test_data = pd.read_table("D:/GitHub/SIGNATE/exercises_mpg/test.tsv")

# 車種名からブランドを抽出
def get_brand(data):
    brand = []
    brand_name = []
    name_top = []
    car_name = data["car name"]
    for name in car_name:
        brand.append(name.split())
        brand_name.append(name)
        name_top.append(name.split()[0])
    return brand, brand_name, name_top

# カウント
def word_count(word):
    wordcounter = Counter(word)
    wordcommon = wordcounter.most_common()
    return wordcommon

# ブランド名を修正
def modify_brand_name(brand_name):
    brand_name_mod = []
    for name in brand_name:
        name_mod = name.replace("vw", "volkswagen")
        name_mod = name_mod.replace("vokswagen", "volkswagen")
        name_mod = name_mod.replace("chevy", "chevrolet")
        name_mod = name_mod.replace("chevroelt", "chevrolet")
        name_mod = name_mod.replace("toyouta", "toyota")
        name_mod = name_mod.replace("maxda", "mazda")
        name_mod = name_mod.replace("mercedes-benz", "mercedes")
        brand_name_mod.append(name_mod)
    brand_name_mod_common = word_count(brand_name_mod)
    return brand_name_mod, brand_name_mod_common

brand_word_train, brand_name_train, name_top_train = get_brand(train_data)
brand_word_test, brand_name_test, name_top_test = get_brand(test_data)

top_common_train = word_count(name_top_train)
top_common_test = word_count(name_top_test)

name_top_mod_train, name_top_mod_common_train = modify_brand_name(name_top_train)
#brand_name_mod_train, _ = modify_brand_name(brand_name_train)
name_top_mod_test, name_top_mod_common_test = modify_brand_name(name_top_test)
#brand_name_mod_test, _ = modify_brand_name(brand_name_test)

# ブランドリスト作成
brand_list_train = []
for brand in name_top_mod_train:
    if brand not in brand_list_train:
        brand_list_train.append(brand)

brand_list_test = []
for brand in name_top_mod_test:
    if brand not in brand_list_test:
        brand_list_test.append(brand)

# ブランドリストを結合
import copy
brand_list_all = copy.copy(brand_list_train)
for brand in brand_list_test:
    if brand not in brand_list_all:
        brand_list_all.append(brand)

# ブランドリストをID割り当て
brand_to_id = {}
id_to_brand = {}
for brand in brand_list_all:
    if brand not in brand_to_id:
        new_id = len(brand_to_id)
        brand_to_id[brand] = new_id
        id_to_brand[new_id] = brand

# ブランドリストをID変換
def brand_id_change(brand_to_id, brand_list):
    brand_id = []
    for brand in brand_list:
        brand_id.append(brand_to_id[brand])
    return brand_id

train_data["brand"] = brand_id_change(brand_to_id, name_top_mod_train)
test_data["brand"] = brand_id_change(brand_to_id, name_top_mod_test)

# horsepowerの?を平均値に変換
def modify_horsepower(data):
    num_HP = data[data["horsepower"] != '?'].loc[:,"horsepower"].astype('float')
    HP_ave = num_HP.mean()
    data["horsepower"] = data["horsepower"].replace("?", HP_ave).astype('float')
    return data


# 特徴作成
def create_feature(data):
    feature = data[["cylinders", "displacement", "horsepower", "weight", "acceleration", "model year", "origin", "brand"]].copy()
    feature = modify_horsepower(feature)
#    cat_cols = ['駅名', '建物構造', '間取り']
#    for cat in cat_cols:
#        feature[cat] = feature[cat].astype("category")
    return feature
    
X_train = create_feature(train_data)
X_test = create_feature(test_data)
Y_train = train_data["mpg"]


# LightGBMの学習
X_trn, X_val, Y_trn, Y_val = train_test_split(X_train, Y_train, test_size=20, random_state=0)

lgb_dataset_trn = lgb.Dataset(X_trn, label=Y_trn, categorical_feature='auto')
lgb_dataset_val = lgb.Dataset(X_val, label=Y_val, categorical_feature='auto')

# validの確認
def calc_mape(y_true, y_pred):
    data_num = len(y_true)
    mape = (np.sum(np.abs(y_pred-y_true)/y_true)/data_num)*100
    return mape

def mape_func(y_pred, data):
    y_true = data.get_label()
    mape = calc_mape(y_true, y_pred)
    return 'mape', mape, False

# グリッドサーチ
rate_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
depth_list = [3, 4, 5, 6, 7, 8, 9, 10]

best_score = 100
best_parameters = {}
for rate in rate_list:
    for depth in depth_list:
        params = {'objective' : 'rmse',
                  'learning_rate' : rate,
                  'max_depth' : depth,
                  }

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

        train_pred = model.predict(X_train)
        train_mape = calc_mape(Y_train.values, train_pred)
        val_pred = model.predict(X_val)
        val_mape = calc_mape(Y_val.values, val_pred)
        print("rate  = ", rate)
        print("depth = ", depth)
        print(f'train mape : {train_mape:.3f}%')
        print(f'valid mape : {val_mape:.3f}%')
        
        # 最も良いスコアのパラメータとスコアを更新
        score = val_mape
        if score < best_score:
            best_score = score
            best_parameters = {'rate' : rate, 'depth' : depth}

print('Best score: {}'.format(best_score))
print('Best parameters: {}'.format(best_parameters))



params = {'objective' : 'rmse',
          'learning_rate' : best_parameters["rate"],
          'max_depth' : best_parameters["depth"],
          }

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
#ax.set_ylim(2, 8)
ax.grid()


# testデータの予測
Y_pred = model.predict(X_test)


plt.rcParams["font.family"] = "IPAexGothic"
feature_importance = pd.DataFrame({
    'feature_name' : model.feature_name(),
    'importance' : model.feature_importance(importance_type='gain'), 
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

plt.figure(figsize = (6, 6))
sns.barplot(data=feature_importance, x='importance', y='feature_name')
plt.savefig('feature_importance.png')



# 提出用データを作成
submission = pd.concat([test_data.loc[:,"id"], pd.Series(Y_pred, name='label')], axis=1)
submission.to_csv('submission.csv', header=False, index=False)


