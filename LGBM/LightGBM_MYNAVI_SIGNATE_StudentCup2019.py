import os
os.chdir('D:\\GitHub\\datascience')
import sys 
import numpy as np
import pandas as pd
from collections import Counter
import datetime

import lightgbm as lgb
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns
import copy
import re


# 変数をID変換
def variables_id_change(variables_to_id, variables_list):
    variables_id = []
    for vari in variables_list:
        variables_id.append(variables_to_id[vari])
    return variables_id


############## データ読み込み ##############
train_data = pd.read_csv("D:/GitHub/SIGNATE/MYNAVI_SIGNATE_StudentCup2019/train.csv")
test_data = pd.read_csv("D:/GitHub/SIGNATE/MYNAVI_SIGNATE_StudentCup2019/test.csv")


############## 所在地を分析 ##############
location_train = train_data['所在地']
c_location_train = location_train.value_counts()
location_test = test_data['所在地']
c_location_test = location_test.value_counts()

# 所在地リスト作成
location_list_train = []
for loc in location_train:
    if loc not in location_list_train:
        location_list_train.append(loc)

location_list_test = []
for loc in location_test:
    if loc not in location_list_test:
        location_list_test.append(loc)

# 所在地リストを結合
location_list_all = copy.copy(location_list_train)
for loc in location_list_test:
    if loc not in location_list_all:
        location_list_all.append(loc)

# 所在地にID割り当て
location_to_id = {}
id_to_location = {}
for loc in location_list_all:
    if loc not in location_to_id:
        new_id = len(location_to_id)
        location_to_id[loc] = new_id
        id_to_location[new_id] = loc

# 所在地をIDへ変換
train_data["所在地"] = variables_id_change(location_to_id, location_train)
test_data["所在地"] = variables_id_change(location_to_id, location_test)


############## アクセスを分析 ##############
access_train = train_data['アクセス']
c_access_train = access_train.value_counts()
access_test = test_data['アクセス']
c_access_test = access_test.value_counts()

# アクセスリスト作成
access_list_train = list(access_train)
access_list_test = list(access_test)

# バス使用時のバス乗車時間と徒歩時間を分離
def bus_walk_time(acc_element):
    if ')' in acc_element:
        time_pos = acc_element.find(')')
    else:
        time_pos = acc_element.find('分')
    time_bus = acc_element[:time_pos+1]
    time_walk = acc_element[time_pos+1:]
    return time_bus, time_walk

# アクセスリストを修正
def modify_acc(acc):
    acc_mod = acc.replace('/','')
    acc_mod = acc_mod.replace('徒歩　','徒歩')
    acc_mod = acc_mod.replace('東京メトロ','')
    acc_mod = acc_mod.replace('有楽町線月島駅','有楽町線　月島駅')
    acc_mod = acc_mod.replace('山手線渋谷駅','山手線 渋谷駅')
    acc_mod = acc_mod.replace('渋谷駅バス','渋谷駅 バス')
    acc_mod = acc_mod.replace('東急目黒線武蔵小山駅','東急目黒線 武蔵小山駅')
    acc_mod = acc_mod.replace('丸ノ内線荻窪駅','丸ノ内線 荻窪駅')
    acc_mod = acc_mod.replace('関東バス／善福寺緑地公園　7分','中央線 高円寺駅 関東バス(13分)善福寺緑地公園下車徒歩7分')
    acc_mod = acc_mod.replace('京王線仙川駅','京王線 仙川駅')
    acc_mod = acc_mod.replace('西武バス：長久保バス停から荻窪駅まで直通バスあり。','')
    acc_mod = acc_mod.replace('関東バス／成田東三丁目　2分','中央線 中野駅 関東バス(14分)成田東三丁目下車徒歩2分')
    acc_mod = acc_mod.replace('駅徒歩','駅 徒歩')
    acc_mod = acc_mod.replace('徒歩圏内','徒歩10分')
    acc_mod = acc_mod.replace('都営バス／堀ノ内　2分','山手線 新宿駅 都営バス(24分)堀ノ内下車徒歩2分')
    acc_mod = acc_mod.replace('西武新宿線下井草駅','西武新宿線 下井草駅')
    acc_mod = acc_mod.replace('関東バス／日産自動車　6分','中央線 荻窪駅 関東バス(10分)日産自動車下車徒歩6分')
    acc_mod = acc_mod.replace('バス停まで約140ｍ。中野駅・幡ヶ谷・渋谷・新宿西口行きあり！','')
    acc_mod = acc_mod.replace('総武線・中央線（各停）	東中野駅	徒歩7分		山手線	高田馬場駅	徒歩15分		東西線	落合(東京都)駅	徒歩8分		バス','総武線・中央線（各停）	東中野駅	徒歩7分		山手線	高田馬場駅	徒歩15分		東西線	落合(東京都)駅	徒歩8分	')
    acc_mod = acc_mod.replace('付近にバス停有。','')
    acc_mod = acc_mod.replace('京浜東北・根岸線王子駅バス25分常盤台教会下車徒歩2分','京浜東北根岸線 王子駅 バス25分常盤台教会下車徒歩2分')
    acc_mod = acc_mod.replace('中央本線吉祥寺駅バス8分関町南２丁目下車徒歩8分','中央本線 吉祥寺駅 バス8分関町南２丁目下車徒歩8分')
    acc_mod = acc_mod.replace('車2.8km(5分)','徒歩50分')
    acc_mod = acc_mod.replace('下井草駅バス8分下石神井坂下下車徒歩2分','下井草駅 バス8分下石神井坂下下車徒歩2分')
    acc_mod = acc_mod.replace('小田急電鉄小田原線成城学園前駅バス9分上祖師谷四丁目下車徒歩4分','小田急電鉄小田原線 成城学園前駅 バス9分上祖師谷四丁目下車徒歩4分')
    acc_mod = acc_mod.replace('京浜東北・根岸線蒲田駅バス19分高畑神社下車徒歩4分','京浜東北根岸線 蒲田駅 バス19分高畑神社下車徒歩4分')
    acc_mod = acc_mod.replace('京浜東北・根岸線蒲田駅バス15分税務署前下車徒歩5分','京浜東北根岸線 蒲田駅 バス15分税務署前下車徒歩5分')
    acc_mod = acc_mod.replace('山手線東京駅バス20分新島橋下車徒歩1分','山手線 東京駅 バス20分新島橋下車徒歩1分')
    acc_mod = acc_mod.replace('山手線東京駅バス19分晴海三丁目下車徒歩2分','山手線 東京駅 バス19分晴海三丁目下車徒歩2分')
    acc_mod = acc_mod.replace('車2.58km(7分)','徒歩70分')
    acc_mod = acc_mod.replace('中央本線中野駅バス15分成田東三丁目下車徒歩1分','中央本線 中野駅 バス15分成田東三丁目下車徒歩1分')
    acc_mod = acc_mod.replace('中央本線中野駅バス7分大新横丁下車徒歩4分','中央本線 中野駅 バス7分大新横丁下車徒歩4分')
    acc_mod = acc_mod.replace('車3.9km(11分)','徒歩110分')
    acc_mod = acc_mod.replace('山手線恵比寿駅バス19分下馬六丁目下車徒歩3分','山手線 恵比寿駅 バス19分下馬六丁目下車徒歩3分')
    acc_mod = acc_mod.replace('千代田・常磐緩行線綾瀬駅バス9分中之台小学校下車徒歩3分','千代田線常磐緩行線 綾瀬駅 バス9分中之台小学校下車徒歩3分')
    acc_mod = acc_mod.replace('山手線新宿駅バス14分笹塚こども図書館下車徒歩1分','山手線　新宿駅　バス14分笹塚こども図書館下車徒歩1分')
    acc_mod = acc_mod.replace('京浜東北・根岸線大井町駅バス6分大井第一小学校下車徒歩3分','京浜東北根岸線　大井町駅　バス6分大井第一小学校下車徒歩3分')
    acc_mod = acc_mod.replace('山手線東京駅バス26分晴海三丁目下車徒歩5分','山手線　東京駅　バス26分晴海三丁目下車徒歩5分')
    acc_mod = acc_mod.replace('有楽町線要町駅バス19分志村坂上下車徒歩2分','有楽町線　要町駅　バス19分志村坂上下車徒歩2分')
    acc_mod = acc_mod.replace('荻窪駅バス10分南荻窪二丁目下車徒歩2分','荻窪駅　バス10分南荻窪二丁目下車徒歩2分')
    acc_mod = acc_mod.replace('車2.8km(10分)','徒歩100分')
    acc_mod = acc_mod.replace('中央本線三鷹駅バス9分北裏下車徒歩4分','中央本線　三鷹駅　バス9分北裏下車徒歩4分')
    acc_mod = acc_mod.replace('東急目黒線多摩川駅バス3分雪が谷下車徒歩6分','東急目黒線　多摩川駅　バス3分雪が谷下車徒歩6分')
    acc_mod = acc_mod.replace('総武本線錦糸町駅バス20分小松川警察署前下車徒歩6分','総武本線　錦糸町駅　バス20分小松川警察署前下車徒歩6分')
    acc_mod = acc_mod.replace('車3.1km(10分)','徒歩100分')
    return acc_mod

# アクセスリストを路線、駅、徒歩時間に分解
def make_line_station_walk(access_list):
    access_parts1 = []
    for acc in access_list:
        acc_mod = modify_acc(acc)
        acc_element = acc_mod.split()
        if 'バス' not in acc_mod:
            acc_num = int(len(acc_element) /3)
            access_parts = []
            for i in range(acc_num):
                access_disassy = []
                access_disassy.append(acc_element[i*3])
                access_disassy.append(acc_element[i*3+1])
                acc_time = int(re.sub("\\D", "", acc_element[i*3+2])) # アクセス時間を数値化
                access_disassy.append(acc_time)
                access_parts.append(access_disassy)
        else:
            access_parts = []
            access_disassy = []
            acc_time = 0
            for i in range(len(acc_element)):
                if 'バス' not in acc_element[i]:
                    access_disassy.append(acc_element[i])
                    if '徒歩' in acc_element[i]:
                        access_disassy[2] = int(re.sub("\\D", "", access_disassy[2])) + acc_time + 10 # バス乗り換えで+10分
                        access_parts.append(access_disassy)
                        access_disassy = []
                        acc_time = 0
                elif 'バス' in acc_element[i] and '下車' not in acc_element[i] and '徒歩' not in acc_element[i]:
                    access_disassy[1] = access_disassy[1] + 'バス'
                    time_element_bus, _ = bus_walk_time(acc_element[i])
                    acc_time = int(re.sub("\\D", "", time_element_bus))
                elif 'バス' in acc_element[i] and '下車徒歩' in acc_element[i]:
                    time_element_bus, time_element_walk = bus_walk_time(acc_element[i])
                    acc_time_bus = int(re.sub("\\D", "", time_element_bus)) + 10 + int(re.sub("\\D", "", time_element_walk))
                    access_disassy.append(acc_time_bus)
                    access_parts.append(access_disassy)
                    access_disassy = []
                elif 'バス' in acc_element[i] and '下車' in acc_element[i] and '徒歩' not in acc_element[i]:
                    time_element_bus, _ = bus_walk_time(acc_element[i])
                    acc_time_bus = int(re.sub("\\D", "", time_element_bus)) + 10
                    access_disassy.append(acc_time_bus)
                    access_parts.append(access_disassy)
                    access_disassy = []
        access_parts.sort(key=lambda x: x[2], reverse=False) # アクセス時間が短い順にソート
        access_parts1.append(access_parts[0])
    
    return np.array(access_parts1)
                
access_parts1_train = make_line_station_walk(access_list_train)
access_parts1_test = make_line_station_walk(access_list_test)

df_acc_part_train = pd.DataFrame(access_parts1_train, columns=['line', 'station', 'walk'])
df_acc_part_test = pd.DataFrame(access_parts1_test, columns=['line', 'station', 'walk'])


############## 間取りを分析 ##############
type_train = train_data['間取り']
c_type_train = type_train.value_counts()
type_test = test_data['間取り']
c_type_test = type_test.value_counts()

# 間取りリスト作成
type_list_train = []
for ty in type_train:
    if ty not in type_list_train:
        type_list_train.append(ty)

type_list_test = []
for ty in type_test:
    if ty not in type_list_test:
        type_list_test.append(ty)

# 間取りリストを結合
type_list_all = copy.copy(type_list_train)
for ty in type_list_test:
    if ty not in type_list_all:
        type_list_all.append(ty)

# 間取りにID割り当て
type_to_id = {}
id_to_type = {}
for ty in type_list_all:
    if ty not in type_to_id:
        new_id = len(type_to_id)
        type_to_id[ty] = new_id
        id_to_type[new_id] = ty

# 間取りをIDへ変換
train_data["間取り"] = variables_id_change(type_to_id, type_train)
test_data["間取り"] = variables_id_change(type_to_id, type_test)


############## 築年数を分析 ##############
age_train = train_data['築年数']
age_test = test_data['築年数']

# 築年数を月表示に変換
def age_count(age_list_org):
    age_list = []
    for age in age_list_org:
        if age == '新築':
            time = 0
        else:
            year_pos = age.find('年')
            time_year = age[:year_pos]
            month_pos = age.find('ヶ月')
            time_month = age[year_pos+1:month_pos]
            time = int(time_year) * 12 + int(time_month)
        age_list.append(time)
    
    return age_list

age_list_train = age_count(age_train)
age_list_test = age_count(age_test)

train_data["築年数"] = age_list_train
test_data["築年数"] = age_list_test


############## 方角を分析 ##############
direction_train = train_data['方角']
direction_test = test_data['方角']

direction_train.fillna('不明', inplace=True)
direction_test.fillna('不明', inplace=True)

# 方角リスト作成
direction_list_train = []
for direction in direction_train:
    if direction not in direction_list_train:
        direction_list_train.append(direction)

direction_list_test = []
for direction in direction_test:
    if direction not in direction_list_test:
        direction_list_test.append(direction)

# 方角リストを結合
direction_list_all = copy.copy(direction_list_train)
for direction in direction_list_test:
    if direction not in direction_list_all:
        direction_list_all.append(direction)

# 方角にID割り当て
direction_to_id = {}
id_to_direction = {}
for direction in direction_list_all:
    if direction not in direction_to_id:
        new_id = len(direction_to_id)
        direction_to_id[direction] = new_id
        id_to_direction[new_id] = direction

# 方角をIDへ変換
train_data["方角"] = variables_id_change(direction_to_id, direction_train)
test_data["方角"] = variables_id_change(direction_to_id, direction_test)


############## 面積を分析 ##############
area_train = train_data['面積']
area_test = test_data['面積']

def del_unit_m2(area_list):
    area_without_unit = []
    for area in area_list:
        area_without_unit.append(float(area.replace('m2','')))
    return area_without_unit

area_without_unit_train = del_unit_m2(area_train)
area_without_unit_test = del_unit_m2(area_test)

train_data["面積"] = area_without_unit_train
test_data["面積"] = area_without_unit_test


############## 所在階を分析 ##############
floor_train = train_data['所在階']
floor_test = test_data['所在階']

floor_train.fillna('floor_ave', inplace=True)
floor_test.fillna('floor_ave', inplace=True)

# 間取りリスト作成
def floor_height_list(floor):
    floor_list = []
    height_list = []
    for fl in floor:
        fl_mod = fl.replace('地下4階／5階建','4階／5階建')
        fl_mod = fl_mod.replace('地下2階／6階建','2階／6階建')
        fl_mod = fl_mod.replace('地下1階／4階建','1階／4階建')
        fl_mod = fl_mod.replace('地下1階／3階建（地下1階）','1階／3階建（地下1階）')
        fl_mod = fl_mod.replace('地下1階／5階建','1階／5階建')
        fl_mod = fl_mod.replace('地下1階／3階建','1階／3階建')
        fl_mod = fl_mod.replace('地下3階／4階建','3階／4階建')
        fl_mod = fl_mod.replace('地下1階／2階建（地下1階）','1階／2階建（地下1階）')
        fl_mod = fl_mod.replace('地下2階／7階建','2階／7階建')
        fl_mod = fl_mod.replace('地下1階／6階建（地下1階）','1階／6階建（地下1階）')
        fl_mod = fl_mod.replace('地下8階／15階建','8階／15階建')
        fl_mod = fl_mod.replace('地下3階／5階建','3階／5階建')
        fl_mod = fl_mod.replace('地下1階／10階建','1階／10階建')
        fl_mod = fl_mod.replace('地下9階／10階建','9階／10階建')
        fl_mod = fl_mod.replace('地下2階／3階建','2階／3階建')
        fl_mod = fl_mod.replace('地下1階／8階建','1階／8階建')
        fl_mod = fl_mod.replace('地下2階／9階建（地下2階）','2階／9階建（地下2階）')
        fl_mod = fl_mod.replace('地下10階／15階建','10階／15階建')
        fl_mod = fl_mod.replace('地下2階／2階建','2階／2階建')
        fl_mod = fl_mod.replace('地下7階／15階建','7階／15階建')
        fl_mod = fl_mod.replace('地下3階／3階建','3階／3階建')
        fl_mod = fl_mod.replace('地下4階／4階建','4階／4階建')
        fl_mod = fl_mod.replace('地下1階／2階建','1階／2階建')
        fl_mod = fl_mod.replace('地下15階／15階建','15階／15階建')
        fl_mod = fl_mod.replace('地下2階／4階建（地下2階）','2階／4階建（地下2階）')
        fl_mod = fl_mod.replace('地下1階／6階建','1階／6階建')
        fl_mod = fl_mod.replace('地下1階／7階建','1階／7階建')
        fl_mod = fl_mod.replace('地下1階／11階建','1階／11階建')
        fl_mod = fl_mod.replace('地下5階／6階建','5階／6階建')
        fl_mod = fl_mod.replace('地下9階／15階建','9階／15階建')
        fl_mod = fl_mod.replace('地下2階／4階建','2階／4階建')
        fl_mod = fl_mod.replace('地下6階／15階建','6階／15階建')
        if fl_mod == 'floor_ave':
            floor_list.append(fl_mod)
            height_list.append(fl_mod)
        elif '／' in fl_mod:
            slash_pos = fl_mod.find('／')
            if '地下' in fl_mod:
                ug_pos = fl_mod.find('地下')
                if slash_pos == 0:
                    floor_list.append(int(re.sub("\\D", "", fl_mod[slash_pos:ug_pos]))/2)
                    height_list.append(int(re.sub("\\D", "", fl_mod[slash_pos:ug_pos])) + int(re.sub("\\D", "", fl_mod[ug_pos:])))
                else:
                    floor_list.append(int(re.sub("\\D", "", fl_mod[:slash_pos])))
                    height_list.append(int(re.sub("\\D", "", fl_mod[slash_pos:ug_pos])) + int(re.sub("\\D", "", fl_mod[ug_pos:])))
                    
            else:
                if slash_pos == 0:
                    floor_list.append(int(re.sub("\\D", "", fl_mod[slash_pos:]))/2)
                    height_list.append(int(re.sub("\\D", "", fl_mod[slash_pos:])))
                else:
                    floor_list.append(int(re.sub("\\D", "", fl_mod[:slash_pos])))
                    height_list.append(int(re.sub("\\D", "", fl_mod[slash_pos:])))
        else:
            if '階建' in fl_mod:
                floor_list.append(int(re.sub("\\D", "", fl_mod))/2)
                height_list.append(int(re.sub("\\D", "", fl_mod)))
            else:
                floor_list.append(int(re.sub("\\D", "", fl_mod)))
                height_list.append(int(re.sub("\\D", "", fl_mod)))
    return floor_list, height_list

floor_list_train, height_list_train = floor_height_list(floor_train)
floor_list_test, height_list_test = floor_height_list(floor_test)

floor_list_train_ave = np.mean([x for x in floor_list_train if x != 'floor_ave'])
height_list_train_ave = np.mean([x for x in height_list_train if x != 'floor_ave'])
floor_list_test_ave = np.mean([x for x in floor_list_test if x != 'floor_ave'])
height_list_test_ave = np.mean([x for x in height_list_test if x != 'floor_ave'])

def floor_height_mod(floor, ave):
    floor_list = []
    for fl in floor:
        if fl == 'floor_ave':
            floor_list.append(ave)
        else:
            floor_list.append(fl)
    return floor_list

floor_list_train_mod = floor_height_mod(floor_list_train, floor_list_train_ave)
height_list_train_mod = floor_height_mod(height_list_train, height_list_train_ave)
floor_list_test_mod = floor_height_mod(floor_list_test, floor_list_test_ave)
height_list_test_mod = floor_height_mod(height_list_test, height_list_test_ave)

floor_height_train = np.concatenate([np.array(floor_list_train_mod).reshape(-1,1), np.array(height_list_train_mod).reshape(-1,1)], axis=1)
floor_height_test = np.concatenate([np.array(floor_list_test_mod).reshape(-1,1), np.array(height_list_test_mod).reshape(-1,1)], axis=1)

df_floor_height_train = pd.DataFrame(floor_height_train, columns=['floor', 'height'])
df_floor_height_test = pd.DataFrame(floor_height_test, columns=['floor', 'height'])


############## バス・トイレを分析 ##############
bath_train = train_data['バス・トイレ']
bath_test = test_data['バス・トイレ']

bath_train.fillna('不明', inplace=True)
bath_test.fillna('不明', inplace=True)

# バス・トイレの要素リスト作成
bath_ele_list_train = []
for bath in bath_train:
    bath_mod = bath.replace('／','')
    bath_element = bath_mod.split()
    bath_ele_list_train.append(bath_element)

bath_ele_list_test = []
for bath in bath_test:
    bath_mod = bath.replace('／','')
    bath_element = bath_mod.split()
    bath_ele_list_test.append(bath_element)

bath_element_list = []
for bath in bath_ele_list_train:
    for ele in bath:
        if ele not in bath_element_list:
            bath_element_list.append(ele)
for bath in bath_ele_list_test:
    for ele in bath:
        if ele not in bath_element_list:
            bath_element_list.append(ele)

# バス・トイレリストを要素に分解
def bath_toilet_list(bath_ele_list):
    personal_bath_list = [0] * len(bath_ele_list)
    personal_toilet_list = [0] * len(bath_ele_list)
    separate_list = [0] * len(bath_ele_list)
    shower_list = [0] * len(bath_ele_list)
    bathroom_dryer_list = [0] * len(bath_ele_list)
    washing_toilet_list = [0] * len(bath_ele_list)
    washstand_list = [0] * len(bath_ele_list)
    dressing_room_list = [0] * len(bath_ele_list)
    refire_list = [0] * len(bath_ele_list)
    shared_toilet_list = [0] * len(bath_ele_list)
    shared_bath_list = [0] * len(bath_ele_list)
    bath_unknown_list = [0] * len(bath_ele_list)
    for i in range(len(bath_ele_list)):
        if '専用バス' in bath_ele_list[i]:
            personal_bath_list[i] = 1
        if '専用トイレ' in bath_ele_list[i]:
            personal_toilet_list[i] = 1
        if 'バス・トイレ別' in bath_ele_list[i]:
            separate_list[i] = 1
        if 'シャワー' in bath_ele_list[i]:
            shower_list[i] = 1
        if '浴室乾燥機' in bath_ele_list[i]:
            bathroom_dryer_list[i] = 1
        if '温水洗浄便座' in bath_ele_list[i]:
            washing_toilet_list[i] = 1
        if '洗面台独立' in bath_ele_list[i]:
            washstand_list[i] = 1
        if '脱衣所' in bath_ele_list[i]:
            dressing_room_list[i] = 1
        if '追焚機能' in bath_ele_list[i]:
            refire_list[i] = 1
        if '共同トイレ' in bath_ele_list[i]:
            shared_toilet_list[i] = 1
        if '共同バス' in bath_ele_list[i]:
            shared_bath_list[i] = 1
        if '不明' in bath_ele_list[i]:
            bath_unknown_list[i] = 1
    
    bath_toilet = np.concatenate([np.array(personal_bath_list).reshape(-1,1),
                                  np.array(personal_toilet_list).reshape(-1,1),
                                  np.array(separate_list).reshape(-1,1),
                                  np.array(shower_list).reshape(-1,1),
                                  np.array(bathroom_dryer_list).reshape(-1,1),
                                  np.array(washing_toilet_list).reshape(-1,1),
                                  np.array(washstand_list).reshape(-1,1),
                                  np.array(dressing_room_list).reshape(-1,1),
                                  np.array(refire_list).reshape(-1,1),
                                  np.array(shared_toilet_list).reshape(-1,1),
                                  np.array(shared_bath_list).reshape(-1,1),
                                  np.array(bath_unknown_list).reshape(-1,1)], axis=1)
    
    df_bath = pd.DataFrame(bath_toilet,
                           columns=['personal_bath',
                                    'personal_toilet',
                                    'separate',
                                    'shower',
                                    'bathroom_dryer',
                                    'washing_toilet',
                                    'washstand',
                                    'dressing_room',
                                    'refire',
                                    'shared_toilet',
                                    'shared_bath',
                                    'bath_unknown'])
    return df_bath

df_bath_train = bath_toilet_list(bath_ele_list_train)
df_bath_test = bath_toilet_list(bath_ele_list_test)


############## キッチンを分析 ##############
kitchen_train = train_data['キッチン']
kitchen_test = test_data['キッチン']

kitchen_train.fillna('不明', inplace=True)
kitchen_test.fillna('不明', inplace=True)

# キッチンの要素リスト作成
kitchen_ele_list_train = []
for kitchen in kitchen_train:
    kitchen_mod = kitchen.replace('／','')
    kitchen_element = kitchen_mod.split()
    kitchen_ele_list_train.append(kitchen_element)

kitchen_ele_list_test = []
for kitchen in kitchen_test:
    kitchen_mod = kitchen.replace('／','')
    kitchen_element = kitchen_mod.split()
    kitchen_ele_list_test.append(kitchen_element)

kitchen_element_list = []
for kitchen in kitchen_ele_list_train:
    for ele in kitchen:
        if ele not in kitchen_element_list:
            kitchen_element_list.append(ele)
for kitchen in kitchen_ele_list_test:
    for ele in kitchen:
        if ele not in kitchen_element_list:
            kitchen_element_list.append(ele)

# キッチンリストを要素に分解
def kitchen_list(kitchen_ele_list):
    stove_list = [0] * len(kitchen_ele_list)
    stove2_list = [0] * len(kitchen_ele_list)
    system_kitchen_list = [0] * len(kitchen_ele_list)
    hot_water_supply_list = [0] * len(kitchen_ele_list)
    independent_kitchen_list = [0] * len(kitchen_ele_list)
    stove3_list = [0] * len(kitchen_ele_list)
    stove_IH_list = [0] * len(kitchen_ele_list)
    stove1_list = [0] * len(kitchen_ele_list)
    refrigerator_list = [0] * len(kitchen_ele_list)
    stove2_can_list = [0] * len(kitchen_ele_list)
    counter_kitchen_list = [0] * len(kitchen_ele_list)
    L_kitchen_list = [0] * len(kitchen_ele_list)
    stove_can_unknown_list = [0] * len(kitchen_ele_list)
    electric_stove_list = [0] * len(kitchen_ele_list)
    stove3_can_list = [0] * len(kitchen_ele_list)
    stove4over_list = [0] * len(kitchen_ele_list)
    stove1_can_list = [0] * len(kitchen_ele_list)
    stove4over_can_list = [0] * len(kitchen_ele_list)
    kitchen_unknown_list = [0] * len(kitchen_ele_list)
    for i in range(len(kitchen_ele_list)):
        if 'ガスコンロ' in kitchen_ele_list[i]:
            stove_list[i] = 1
        if 'コンロ2口' in kitchen_ele_list[i]:
            stove2_list[i] = 1
        if 'システムキッチン' in kitchen_ele_list[i]:
            system_kitchen_list[i] = 1
        if '給湯' in kitchen_ele_list[i]:
            hot_water_supply_list[i] = 1
        if '独立キッチン' in kitchen_ele_list[i]:
            independent_kitchen_list[i] = 1
        if 'コンロ3口' in kitchen_ele_list[i]:
            stove3_list[i] = 1
        if 'IHコンロ' in kitchen_ele_list[i]:
            stove_IH_list[i] = 1
        if 'コンロ1口' in kitchen_ele_list[i]:
            stove1_list[i] = 1
        if '冷蔵庫あり' in kitchen_ele_list[i]:
            refrigerator_list[i] = 1
        if 'コンロ設置可（コンロ2口）' in kitchen_ele_list[i]:
            stove2_can_list[i] = 1
        if 'カウンターキッチン' in kitchen_ele_list[i]:
            counter_kitchen_list[i] = 1
        if 'L字キッチン' in kitchen_ele_list[i]:
            L_kitchen_list[i] = 1
        if 'コンロ設置可（口数不明）' in kitchen_ele_list[i]:
            stove_can_unknown_list[i] = 1
        if '電気コンロ' in kitchen_ele_list[i]:
            electric_stove_list[i] = 1
        if 'コンロ設置可（コンロ3口）' in kitchen_ele_list[i]:
            stove3_can_list[i] = 1
        if 'コンロ4口以上' in kitchen_ele_list[i]:
            stove4over_list[i] = 1
        if 'コンロ設置可（コンロ1口）' in kitchen_ele_list[i]:
            stove1_can_list[i] = 1
        if 'コンロ設置可（コンロ4口以上）' in kitchen_ele_list[i]:
            stove4over_can_list[i] = 1
        if '不明' in kitchen_ele_list[i]:
            kitchen_unknown_list[i] = 1
    
    kitchen_equipment = np.concatenate([np.array(stove_list).reshape(-1,1),
                                        np.array(stove2_list).reshape(-1,1),
                                        np.array(system_kitchen_list).reshape(-1,1),
                                        np.array(hot_water_supply_list).reshape(-1,1),
                                        np.array(independent_kitchen_list).reshape(-1,1),
                                        np.array(stove3_list).reshape(-1,1),
                                        np.array(stove_IH_list).reshape(-1,1),
                                        np.array(stove1_list).reshape(-1,1),
                                        np.array(refrigerator_list).reshape(-1,1),
                                        np.array(stove2_can_list).reshape(-1,1),
                                        np.array(counter_kitchen_list).reshape(-1,1),
                                        np.array(L_kitchen_list).reshape(-1,1),
                                        np.array(stove_can_unknown_list).reshape(-1,1),
                                        np.array(electric_stove_list).reshape(-1,1),
                                        np.array(stove3_can_list).reshape(-1,1),
                                        np.array(stove4over_list).reshape(-1,1),
                                        np.array(stove1_can_list).reshape(-1,1),
                                        np.array(stove4over_can_list).reshape(-1,1),
                                        np.array(kitchen_unknown_list).reshape(-1,1)], axis=1)
    
    df_kitchen = pd.DataFrame(kitchen_equipment,
                              columns=['stove_list',
                                       'stove2_list',
                                       'system_kitchen_list',
                                       'hot_water_supply_list',
                                       'independent_kitchen_list',
                                       'stove3_list',
                                       'stove_IH_list',
                                       'stove1_list',
                                       'refrigerator_list',
                                       'stove2_can_list',
                                       'counter_kitchen_list',
                                       'L_kitchen_list',
                                       'stove_can_unknown_list',
                                       'electric_stove_list',
                                       'stove3_can_list',
                                       'stove4over_list',
                                       'stove1_can_list',
                                       'stove4over_can_list',
                                       'kitchen_unknown_list'])
    return df_kitchen

df_kitchen_train = kitchen_list(kitchen_ele_list_train)
df_kitchen_test = kitchen_list(kitchen_ele_list_test)


############## 放送・通信を分析 ##############
broadcast_train = train_data['放送・通信']
broadcast_test = test_data['放送・通信']

broadcast_train.fillna('不明', inplace=True)
broadcast_test.fillna('不明', inplace=True)

# 放送・通信の要素リスト作成
broadcast_ele_list_train = []
for broadcast in broadcast_train:
    broadcast_mod = broadcast.replace('／','')
    broadcast_element = broadcast_mod.split()
    broadcast_ele_list_train.append(broadcast_element)

broadcast_ele_list_test = []
for broadcast in broadcast_test:
    broadcast_mod = broadcast.replace('／','')
    broadcast_element = broadcast_mod.split()
    broadcast_ele_list_test.append(broadcast_element)

broadcast_element_list = []
for broadcast in broadcast_ele_list_train:
    for ele in broadcast:
        if ele not in broadcast_element_list:
            broadcast_element_list.append(ele)
for broadcast in broadcast_ele_list_test:
    for ele in broadcast:
        if ele not in broadcast_element_list:
            broadcast_element_list.append(ele)

# 放送・通信リストを要素に分解
def broadcast_list(broadcast_ele_list):
    internet_list = [0] * len(broadcast_ele_list)
    CATV_list = [0] * len(broadcast_ele_list)
    CS_list = [0] * len(broadcast_ele_list)
    BS_list = [0] * len(broadcast_ele_list)
    optical_fiber_list = [0] * len(broadcast_ele_list)
    HS_internet_list = [0] * len(broadcast_ele_list)
    free_internet_list = [0] * len(broadcast_ele_list)
    cable_broadcasting_list = [0] * len(broadcast_ele_list)
    broadcast_unknown_list = [0] * len(broadcast_ele_list)
    for i in range(len(broadcast_ele_list)):
        if 'インターネット対応' in broadcast_ele_list[i]:
            internet_list[i] = 1
        if 'CATV' in broadcast_ele_list[i]:
            CATV_list[i] = 1
        if 'CSアンテナ' in broadcast_ele_list[i]:
            CS_list[i] = 1
        if 'BSアンテナ' in broadcast_ele_list[i]:
            BS_list[i] = 1
        if '光ファイバー' in broadcast_ele_list[i]:
            optical_fiber_list[i] = 1
        if '高速インターネット' in broadcast_ele_list[i]:
            HS_internet_list[i] = 1
        if 'インターネット使用料無料' in broadcast_ele_list[i]:
            free_internet_list[i] = 1
        if '有線放送' in broadcast_ele_list[i]:
            cable_broadcasting_list[i] = 1
        if '不明' in broadcast_ele_list[i]:
            broadcast_unknown_list[i] = 1
    
    broadcast_equipment = np.concatenate([np.array(internet_list).reshape(-1,1),
                                        np.array(CATV_list).reshape(-1,1),
                                        np.array(CS_list).reshape(-1,1),
                                        np.array(BS_list).reshape(-1,1),
                                        np.array(optical_fiber_list).reshape(-1,1),
                                        np.array(HS_internet_list).reshape(-1,1),
                                        np.array(free_internet_list).reshape(-1,1),
                                        np.array(cable_broadcasting_list).reshape(-1,1),
                                        np.array(broadcast_unknown_list).reshape(-1,1)], axis=1)
    
    df_broadcast = pd.DataFrame(broadcast_equipment,
                              columns=['internet',
                                       'CATV',
                                       'CS',
                                       'BS',
                                       'optical_fiber',
                                       'HS_internet',
                                       'free_internet',
                                       'cable_broadcasting',
                                       'broadcast_unknown'])
    return df_broadcast

df_broadcast_train = broadcast_list(broadcast_ele_list_train)
df_broadcast_test = broadcast_list(broadcast_ele_list_test)


############## 室内設備を分析 ##############
installation_train = train_data['室内設備']
installation_test = test_data['室内設備']

installation_train.fillna('不明', inplace=True)
installation_test.fillna('不明', inplace=True)

# 室内設備の要素リスト作成
installation_ele_list_train = []
for installation in installation_train:
    installation_mod = installation.replace('／','')
    installation_element = installation_mod.split()
    installation_ele_list_train.append(installation_element)

installation_ele_list_test = []
for installation in installation_test:
    installation_mod = installation.replace('／','')
    installation_element = installation_mod.split()
    installation_ele_list_test.append(installation_element)

installation_element_list = []
for installation in installation_ele_list_train:
    for ele in installation:
        if ele not in installation_element_list:
            installation_element_list.append(ele)
for installation in installation_ele_list_test:
    for ele in installation:
        if ele not in installation_element_list:
            installation_element_list.append(ele)

# 室内設備リストを要素に分解
def installation_list(installation_ele_list):
    air_conditioner_list = [0] * len(installation_ele_list)
    indoor_washing_machine_list = [0] * len(installation_ele_list)
    flooring_list = [0] * len(installation_ele_list)
    citygas_list = [0] * len(installation_ele_list)
    balcony_list = [0] * len(installation_ele_list)
    sewage_list = [0] * len(installation_ele_list)
    public_water_supply_list = [0] * len(installation_ele_list)
    shoesbox_list = [0] * len(installation_ele_list)
    garbagestorage_onsite_list = [0] * len(installation_ele_list)
    elevator_list = [0] * len(installation_ele_list)
    tiling_list = [0] * len(installation_ele_list)
    hour24_ventilation_system_list = [0] * len(installation_ele_list)
    two_sided_lighting_list = [0] * len(installation_ele_list)
    air_conditioning_list = [0] * len(installation_ele_list)
    walkin_closet_list = [0] * len(installation_ele_list)
    for i in range(len(installation_ele_list)):
        if 'エアコン付' in installation_ele_list[i]:
            air_conditioner_list[i] = 1
        if '室内洗濯機置場' in installation_ele_list[i]:
            indoor_washing_machine_list[i] = 1
        if 'フローリング' in installation_ele_list[i]:
            flooring_list[i] = 1
        if '都市ガス' in installation_ele_list[i]:
            citygas_list[i] = 1
        if 'バルコニー' in installation_ele_list[i]:
            balcony_list[i] = 1
        if '下水' in installation_ele_list[i]:
            sewage_list[i] = 1
        if '公営水道' in installation_ele_list[i]:
            public_water_supply_list[i] = 1
        if 'シューズボックス' in installation_ele_list[i]:
            shoesbox_list[i] = 1
        if '敷地内ごみ置き場' in installation_ele_list[i]:
            garbagestorage_onsite_list[i] = 1
        if 'エレベーター' in installation_ele_list[i]:
            elevator_list[i] = 1
        if 'タイル張り' in installation_ele_list[i]:
            tiling_list[i] = 1
        if '24時間換気システム' in installation_ele_list[i]:
            hour24_ventilation_system_list[i] = 1
        if '2面採光' in installation_ele_list[i]:
            two_sided_lighting_list[i] = 1
        if '冷房' in installation_ele_list[i]:
            air_conditioning_list[i] = 1
        if 'ウォークインクローゼット' in installation_ele_list[i]:
            walkin_closet_list[i] = 1
    
    installation_equipment = np.concatenate([np.array(air_conditioner_list).reshape(-1,1),
                                             np.array(indoor_washing_machine_list).reshape(-1,1),
                                             np.array(flooring_list).reshape(-1,1),
                                             np.array(citygas_list).reshape(-1,1),
                                             np.array(balcony_list).reshape(-1,1),
                                             np.array(sewage_list).reshape(-1,1),
                                             np.array(public_water_supply_list).reshape(-1,1),
                                             np.array(shoesbox_list).reshape(-1,1),
                                             np.array(garbagestorage_onsite_list).reshape(-1,1),
                                             np.array(elevator_list).reshape(-1,1),
                                             np.array(tiling_list).reshape(-1,1),
                                             np.array(hour24_ventilation_system_list).reshape(-1,1),
                                             np.array(two_sided_lighting_list).reshape(-1,1),
                                             np.array(air_conditioning_list).reshape(-1,1),
                                             np.array(walkin_closet_list).reshape(-1,1)], axis=1)
    
    df_installation = pd.DataFrame(installation_equipment,
                                   columns=['air_conditioner',
                                            'indoor_washing_machine',
                                            'flooring',
                                            'citygas',
                                            'balcony',
                                            'sewage',
                                            'public_water_supply',
                                            'shoesbox',
                                            'garbagestorage_onsite',
                                            'elevator',
                                            'tiling',
                                            'hour24_ventilation_system',
                                            'two_sided_lighting',
                                            'air_conditioning',
                                            'walkin_closet'])
    return df_installation

df_installation_train = installation_list(installation_ele_list_train)
df_installation_test = installation_list(installation_ele_list_test)


############## 駐車場を分析 ##############
parking_train = train_data['駐車場']
parking_test = test_data['駐車場']

parking_train.fillna('不明', inplace=True)
parking_test.fillna('不明', inplace=True)

# 駐車場リストを駐車場、駐輪場、バイク置き場と値段に分解
def make_park_cost(park_list):
    df_park = pd.DataFrame(index=[], columns=['parking', 'parking_cost', 'bike', 'bike_cost' ,'bicycle'])
    for park in park_list:
        park_element = park.split()
        tmp_park = 0
        tmp_park_cost = 0
        tmp_bike = 0
        tmp_bike_cost = 0
        tmp_bicycle = 0
        for i in range(len(park_element)):
            if park_element[i] == '駐車場':
                if i+1 <= len(park_element)-1 and '空有' in park_element[i+1]:
                    tmp_park = 1
                elif i+1 <= len(park_element)-1 and '近隣' in park_element[i+1]:
                    tmp_park = 1
                else:
                    tmp_park = 0
                    tmp_park_cost = 1000000
                tmp_cost = 0
                if tmp_park == 1 and i+2 <= len(park_element)-1:
                    if re.sub("\\D", "", park_element[i+2]) != '':
                        tmp_cost = int(re.sub("\\D", "", park_element[i+2]))
                    else:
                        tmp_cost = 0
                if tmp_cost > 0:
                    tmp_park_cost = tmp_cost
                else:
                    tmp_park_cost = 'parking_cost_ave'
            if park_element[i] == 'バイク置き場':
                if i+1 <= len(park_element)-1 and '空有' in park_element[i+1]:
                    tmp_bike = 1
                elif i+1 <= len(park_element)-1 and '近隣' in park_element[i+1]:
                    tmp_bike = 1
                else:
                    tmp_bike = 0
                    tmp_bike_cost = 100000
                tmp_cost = 0
                if tmp_park == 1 and i+2 <= len(park_element)-1:
                    if re.sub("\\D", "", park_element[i+2]) != '':
                        tmp_cost = int(re.sub("\\D", "", park_element[i+2]))
                    else:
                        tmp_cost = 0
                if tmp_cost > 0:
                    tmp_bike_cost = tmp_cost
                else:
                    tmp_bike_cost = 'bike_cost_ave'
            if park_element[i] == '駐輪場':
                if i+1 <= len(park_element)-1 and '空有' in park_element[i+1]:
                    tmp_bicycle = 1
                else:
                    tmp_bicycle = 0
                    
        df_park = df_park.append(pd.Series([tmp_park, tmp_park_cost, tmp_bike, tmp_bike_cost, tmp_bicycle],
                                           index=df_park.columns), ignore_index=True)
    
    return df_park
               
df_parking_train = make_park_cost(parking_train)
df_parking_test = make_park_cost(parking_test)

park_cost_train_ave = np.mean([x for x in list(df_parking_train.loc[:,'parking_cost']) if x != 'parking_cost_ave'])
bike_cost_train_ave = np.mean([x for x in list(df_parking_train.loc[:,'bike_cost']) if x != 'bike_cost_ave'])
park_cost_test_ave = np.mean([x for x in list(df_parking_test.loc[:,'parking_cost']) if x != 'parking_cost_ave'])
bike_cost_test_ave = np.mean([x for x in list(df_parking_test.loc[:,'bike_cost']) if x != 'bike_cost_ave'])

def park_bike_mod(park, ave, cost_ave):
    park_list = []
    for pk in park:
        if pk == cost_ave:
            park_list.append(ave)
        else:
            park_list.append(pk)
    return park_list

park_cost_mod_train = park_bike_mod(list(df_parking_train.loc[:,'parking_cost']), park_cost_train_ave, 'parking_cost_ave')
bike_cost_mod_train = park_bike_mod(list(df_parking_train.loc[:,'bike_cost']), bike_cost_train_ave, 'bike_cost_ave')
park_cost_mod_test = park_bike_mod(list(df_parking_test.loc[:,'parking_cost']), park_cost_test_ave, 'parking_cost_ave')
bike_cost_mod_test = park_bike_mod(list(df_parking_test.loc[:,'bike_cost']), bike_cost_test_ave, 'bike_cost_ave')

df_parking_train['parking_cost'] = park_cost_mod_train
df_parking_train['bike_cost'] = bike_cost_mod_train
df_parking_test['parking_cost'] = park_cost_mod_test
df_parking_test['bike_cost'] = bike_cost_mod_test


############## 周辺環境を分析 ##############
environment_train = train_data['周辺環境']
environment_test = test_data['周辺環境']

environment_train.fillna('不明', inplace=True)
environment_test.fillna('不明', inplace=True)

# 周辺環境の要素リスト作成
environment_ele_list_train = []
for env in environment_train:
    environment_mod = env.replace('／','')
    environment_element = environment_mod.split()
    environment_ele_list_train.append(environment_element)

environment_ele_list_test = []
for env in environment_test:
    environment_mod = env.replace('／','')
    environment_element = environment_mod.split()
    environment_ele_list_test.append(environment_element)

environment_element_list = []
for env in environment_ele_list_train:
    for ele in env:
        if ele not in environment_element_list:
            environment_element_list.append(ele)
for env in environment_ele_list_test:
    for ele in env:
        if ele not in environment_element_list:
            environment_element_list.append(ele)

# 周辺環境リストを施設と距離に分解
def make_facility_distance(environment_list):
    df_environment = pd.DataFrame(index=[],
                                  columns=['supermarket', 'supermarket_dist',
                                           'convenience', 'convenience_dist',
                                           'restaurant', 'restaurant_dist',
                                           'drugstore', 'drugstore_dist',
                                           'hospital', 'hospital_dist',
                                           'park', 'park_dist',
                                           'postoffice', 'postoffice_dist',
                                           'elementary', 'elementary_dist',
                                           'university', 'university_dist'])
    for env in environment_list:
        env_element = env.split()
        tmp_supermarket = 0
        tmp_supermarket_dist = 10000
        tmp_min_supermarket_dist = 10000
        tmp_convenience = 0
        tmp_convenience_dist = 10000
        tmp_min_convenience_dist = 10000
        tmp_restaurant = 0
        tmp_restaurant_dist = 10000
        tmp_min_restaurant_dist = 10000
        tmp_drugstore = 0
        tmp_drugstore_dist = 10000
        tmp_min_drugstore_dist = 10000
        tmp_hospital = 0
        tmp_hospital_dist = 10000
        tmp_min_hospital_dist = 10000
        tmp_park = 0
        tmp_park_dist = 10000
        tmp_min_park_dist = 10000
        tmp_postoffice = 0
        tmp_postoffice_dist = 10000
        tmp_min_postoffice_dist = 10000
        tmp_elementary = 0
        tmp_elementary_dist = 10000
        tmp_min_elementary_dist = 10000
        tmp_university = 0
        tmp_university_dist = 10000
        tmp_min_university_dist = 10000
        for i in range(len(env_element)):
            if env_element[i] == '【スーパー】':
                tmp_supermarket = 1
                tmp_dist = 0
                if i+1 <= len(env_element)-1 and 'm' in env_element[i+1]:
                    if re.sub("\\D", "", env_element[i+1]) != '':
                        tmp_dist = int(re.sub("\\D", "", env_element[i+1]))
                        if tmp_min_supermarket_dist > tmp_dist:
                            tmp_min_supermarket_dist = tmp_dist
                else:
                    tmp_supermarket_dist = 'supermarket_dist_ave'
            if env_element[i] == '【コンビニ】':
                tmp_convenience = 1
                tmp_dist = 0
                if i+1 <= len(env_element)-1 and 'm' in env_element[i+1]:
                    if re.sub("\\D", "", env_element[i+1]) != '':
                        tmp_dist = int(re.sub("\\D", "", env_element[i+1]))
                        if tmp_min_convenience_dist > tmp_dist:
                            tmp_min_convenience_dist = tmp_dist
                else:
                    tmp_convenience_dist = 'convenience_dist_ave'
            if env_element[i] == '【飲食店】':
                tmp_restaurant = 1
                tmp_dist = 0
                if i+1 <= len(env_element)-1 and 'm' in env_element[i+1]:
                    if re.sub("\\D", "", env_element[i+1]) != '':
                        tmp_dist = int(re.sub("\\D", "", env_element[i+1]))
                        if tmp_min_restaurant_dist > tmp_dist:
                            tmp_min_restaurant_dist = tmp_dist
                else:
                    tmp_restaurant_dist = 'restaurant_dist_ave'
            if env_element[i] == '【ドラッグストア】':
                tmp_drugstore = 1
                tmp_dist = 0
                if i+1 <= len(env_element)-1 and 'm' in env_element[i+1]:
                    if re.sub("\\D", "", env_element[i+1]) != '':
                        tmp_dist = int(re.sub("\\D", "", env_element[i+1]))
                        if tmp_min_drugstore_dist > tmp_dist:
                            tmp_min_drugstore_dist = tmp_dist
                else:
                    tmp_drugstore_dist = 'drugstore_dist_ave'
            if env_element[i] == '【病院】':
                tmp_hospital = 1
                tmp_dist = 0
                if i+1 <= len(env_element)-1 and 'm' in env_element[i+1]:
                    if re.sub("\\D", "", env_element[i+1]) != '':
                        tmp_dist = int(re.sub("\\D", "", env_element[i+1]))
                        if tmp_min_hospital_dist > tmp_dist:
                            tmp_min_hospital_dist = tmp_dist
                else:
                    tmp_hospital_dist = 'hospital_dist_ave'
            if env_element[i] == '【公園】':
                tmp_park = 1
                tmp_dist = 0
                if i+1 <= len(env_element)-1 and 'm' in env_element[i+1]:
                    if re.sub("\\D", "", env_element[i+1]) != '':
                        tmp_dist = int(re.sub("\\D", "", env_element[i+1]))
                        if tmp_min_park_dist > tmp_dist:
                            tmp_min_park_dist = tmp_dist
                else:
                    tmp_park_dist = 'park_dist_ave'
            if env_element[i] == '【郵便局】':
                tmp_postoffice = 1
                tmp_dist = 0
                if i+1 <= len(env_element)-1 and 'm' in env_element[i+1]:
                    if re.sub("\\D", "", env_element[i+1]) != '':
                        tmp_dist = int(re.sub("\\D", "", env_element[i+1]))
                        if tmp_min_postoffice_dist > tmp_dist:
                            tmp_min_postoffice_dist = tmp_dist
                else:
                    tmp_postoffice_dist = 'postoffice_dist_ave'
            if env_element[i] == '【小学校】':
                tmp_elementary = 1
                tmp_dist = 0
                if i+1 <= len(env_element)-1 and 'm' in env_element[i+1]:
                    if re.sub("\\D", "", env_element[i+1]) != '':
                        tmp_dist = int(re.sub("\\D", "", env_element[i+1]))
                        if tmp_min_elementary_dist > tmp_dist:
                            tmp_min_elementary_dist = tmp_dist
                else:
                    tmp_elementary_dist = 'elementary_dist_ave'
            if env_element[i] == '【大学】':
                tmp_university = 1
                tmp_dist = 0
                if i+1 <= len(env_element)-1 and 'm' in env_element[i+1]:
                    if re.sub("\\D", "", env_element[i+1]) != '':
                        tmp_dist = int(re.sub("\\D", "", env_element[i+1]))
                        if tmp_min_university_dist > tmp_dist:
                            tmp_min_university_dist = tmp_dist
                else:
                    tmp_university_dist = 'university_dist_ave'
        tmp_supermarket_dist = tmp_min_supermarket_dist
        tmp_convenience_dist = tmp_min_convenience_dist
        tmp_restaurant_dist = tmp_min_restaurant_dist
        tmp_drugstore_dist = tmp_min_drugstore_dist
        tmp_hospital_dist = tmp_min_hospital_dist
        tmp_park_dist = tmp_min_park_dist
        tmp_postoffice_dist = tmp_min_postoffice_dist
        tmp_elementary_dist = tmp_min_elementary_dist
        tmp_university_dist = tmp_min_university_dist
                    
        df_environment = df_environment.append(pd.Series([tmp_supermarket, tmp_supermarket_dist,
                                                          tmp_convenience, tmp_convenience_dist,
                                                          tmp_restaurant, tmp_restaurant_dist,
                                                          tmp_drugstore, tmp_drugstore_dist,
                                                          tmp_hospital, tmp_hospital_dist,
                                                          tmp_park, tmp_park_dist,
                                                          tmp_postoffice, tmp_postoffice_dist,
                                                          tmp_elementary, tmp_elementary_dist,
                                                          tmp_university, tmp_university_dist],
                                                         index=df_environment.columns), ignore_index=True)
    
    return df_environment

df_environment_train = make_facility_distance(environment_train)
df_environment_test = make_facility_distance(environment_test)

supermarket_dist_mod_train = park_bike_mod(list(df_environment_train.loc[:,'supermarket_dist']), park_cost_train_ave, 'supermarket_dist_ave')
convenience_dist_mod_train = park_bike_mod(list(df_environment_train.loc[:,'convenience_dist']), park_cost_train_ave, 'convenience_dist_ave')
restaurant_dist_mod_train = park_bike_mod(list(df_environment_train.loc[:,'restaurant_dist']), park_cost_train_ave, 'restaurant_dist_ave')
drugstore_dist_mod_train = park_bike_mod(list(df_environment_train.loc[:,'drugstore_dist']), park_cost_train_ave, 'drugstore_dist_ave')
hospital_dist_mod_train = park_bike_mod(list(df_environment_train.loc[:,'hospital_dist']), park_cost_train_ave, 'hospital_dist_ave')
park_dist_mod_train = park_bike_mod(list(df_environment_train.loc[:,'park_dist']), park_cost_train_ave, 'park_dist_ave')
postoffice_dist_mod_train = park_bike_mod(list(df_environment_train.loc[:,'postoffice_dist']), park_cost_train_ave, 'postoffice_dist_ave')
elementary_dist_mod_train = park_bike_mod(list(df_environment_train.loc[:,'elementary_dist']), park_cost_train_ave, 'elementary_dist_ave')
university_dist_mod_train = park_bike_mod(list(df_environment_train.loc[:,'university_dist']), park_cost_train_ave, 'university_dist_ave')
supermarket_dist_mod_test = park_bike_mod(list(df_environment_test.loc[:,'supermarket_dist']), park_cost_test_ave, 'supermarket_dist_ave')
convenience_dist_mod_test = park_bike_mod(list(df_environment_test.loc[:,'convenience_dist']), park_cost_test_ave, 'convenience_dist_ave')
restaurant_dist_mod_test = park_bike_mod(list(df_environment_test.loc[:,'restaurant_dist']), park_cost_test_ave, 'restaurant_dist_ave')
drugstore_dist_mod_test = park_bike_mod(list(df_environment_test.loc[:,'drugstore_dist']), park_cost_test_ave, 'drugstore_dist_ave')
hospital_dist_mod_test = park_bike_mod(list(df_environment_test.loc[:,'hospital_dist']), park_cost_test_ave, 'hospital_dist_ave')
park_dist_mod_test = park_bike_mod(list(df_environment_test.loc[:,'park_dist']), park_cost_test_ave, 'park_dist_ave')
postoffice_dist_mod_test = park_bike_mod(list(df_environment_test.loc[:,'postoffice_dist']), park_cost_test_ave, 'postoffice_dist_ave')
elementary_dist_mod_test = park_bike_mod(list(df_environment_test.loc[:,'elementary_dist']), park_cost_test_ave, 'elementary_dist_ave')
university_dist_mod_test = park_bike_mod(list(df_environment_test.loc[:,'university_dist']), park_cost_test_ave, 'university_dist_ave')

df_environment_train['supermarket_dist'] = supermarket_dist_mod_train
df_environment_train['convenience_dist'] = convenience_dist_mod_train
df_environment_train['restaurant_dist'] = restaurant_dist_mod_train
df_environment_train['drugstore_dist'] = drugstore_dist_mod_train
df_environment_train['hospital_dist'] = hospital_dist_mod_train
df_environment_train['park_dist'] = park_dist_mod_train
df_environment_train['postoffice_dist'] = postoffice_dist_mod_train
df_environment_train['elementary_dist'] = elementary_dist_mod_train
df_environment_train['university_dist'] = university_dist_mod_train
df_environment_test['supermarket_dist'] = supermarket_dist_mod_test
df_environment_test['convenience_dist'] = convenience_dist_mod_test
df_environment_test['restaurant_dist'] = restaurant_dist_mod_test
df_environment_test['drugstore_dist'] = drugstore_dist_mod_test
df_environment_test['hospital_dist'] = hospital_dist_mod_test
df_environment_test['park_dist'] = park_dist_mod_test
df_environment_test['postoffice_dist'] = postoffice_dist_mod_test
df_environment_test['elementary_dist'] = elementary_dist_mod_test
df_environment_test['university_dist'] = university_dist_mod_test


############## 建物構造を分析 ##############
structure_train = train_data['建物構造']
structure_test = test_data['建物構造']

structure_train.fillna('不明', inplace=True)
structure_test.fillna('不明', inplace=True)

# 建物構造リスト作成
structure_list_train = []
for st in structure_train:
    if st not in structure_list_train:
        structure_list_train.append(st)

structure_list_test = []
for st in structure_test:
    if st not in structure_list_test:
        structure_list_test.append(st)

# 建物構造リストを結合
structure_list_all = copy.copy(structure_list_train)
for st in structure_list_test:
    if st not in structure_list_all:
        structure_list_all.append(st)

# 建物構造にID割り当て
structure_to_id = {}
id_to_structure = {}
for st in structure_list_all:
    if st not in structure_to_id:
        new_id = len(structure_to_id)
        structure_to_id[st] = new_id
        id_to_structure[new_id] = st

# 建物構造をIDへ変換
train_data["建物構造"] = variables_id_change(structure_to_id, structure_train)
test_data["建物構造"] = variables_id_change(structure_to_id, structure_test)


############## 契約期間を分析 ##############
period_train = train_data['契約期間']
period_test = test_data['契約期間']

period_train.fillna('不明', inplace=True)
period_test.fillna('不明', inplace=True)

# 契約期間の要素リスト作成
period_ele_list_train = []
for pe in period_train:
    period_mod = pe.replace('／','')
    period_element = period_mod.split()
    period_ele_list_train.append(period_element)

period_ele_list_test = []
for pe in period_test:
    period_mod = pe.replace('／','')
    period_element = period_mod.split()
    period_ele_list_test.append(period_element)

period_element_list = []
for pe in period_ele_list_train:
    for ele in pe:
        if ele not in period_element_list:
            period_element_list.append(ele)
for pe in period_ele_list_test:
    for ele in pe:
        if ele not in period_element_list:
            period_element_list.append(ele)

# 契約期間リストを期間と定期借家に分解
def make_period_fixedterm(period_list):
    df_period = pd.DataFrame(index=[],
                             columns=['period', 'fixedterm'])
    for pe in period_list:
        tmp_period = 0
        if '※この物件は' in pe:
            tmp_fixedterm = 1
        elif '不明' in pe:
            tmp_fixedterm = 0
        else:
            tmp_fixedterm = 2
        pe_element = pe.split()
        for i in range(len(pe_element)):
            if '年間' in pe_element[i]:
                tmp_period = int(re.sub("\\D", "", pe_element[i])) *12
            elif 'ヶ月間' in pe_element[i] and '年' not in pe_element[i]:
                tmp_period = int(re.sub("\\D", "", pe_element[i]))
            elif 'まで' in pe_element[i]:
                year_pos = pe_element[i].find('年')
                year = int(pe_element[i][:year_pos])
                month_pos = pe_element[i].find('月')
                month = int(pe_element[i][year_pos+1:month_pos])
                dt_end = datetime.datetime(year=year, month=month, day=1, hour=0)
                dt_start = datetime.datetime(year=2019, month=8, day=1, hour=0)
                tmp_period = dt_end - dt_start
                tmp_period = tmp_period.total_seconds() /60/60/24/30
            elif '年' in pe_element[i] and 'ヶ月間' in pe_element[i]:
                year_pos = pe_element[i].find('年')
                year = int(pe_element[i][:year_pos])
                month_pos = pe_element[i].find('ヶ')
                month = int(pe_element[i][year_pos+1:month_pos])
                tmp_period = year *12 + month
                    
        df_period = df_period.append(pd.Series([tmp_period, tmp_fixedterm],
                                               index=df_period.columns), ignore_index=True)
    
    return df_period

df_period_train = make_period_fixedterm(period_train)
df_period_test = make_period_fixedterm(period_test)


#import collections
#all_list = []
#for s in period_ele_list_train:
#    for w in s:
#        all_list.append(w)
#for s in period_ele_list_train:
#    for w in s:
#        all_list.append(w)
#all_count = collections.Counter(all_list)
#all_common = all_count.most_common()
#print(all_common)
#
#c__train = period_train.value_counts()

                        
############## 新しく作った説明変数を追加 ##############
train_data_mod = pd.concat([train_data,
                            df_acc_part_train,
                            df_floor_height_train,
                            df_bath_train,
                            df_kitchen_train,
                            df_broadcast_train,
                            df_installation_train,
                            df_parking_train,
                            df_environment_train,
                            df_period_train], axis=1)
test_data_mod = pd.concat([test_data,
                           df_acc_part_test,
                           df_floor_height_test,
                           df_bath_test,
                           df_kitchen_test,
                           df_broadcast_test,
                           df_installation_test,
                           df_parking_test,
                           df_environment_test,
                           df_period_test], axis=1)

# タイプ変換
def change_type(data):
    cat_int = ['walk', 'parking', 'bike', 'bicycle', 'supermarket', 'convenience', 'restaurant', 'drugstore', 'hospital', 'park',
                'postoffice', 'elementary', 'university', 'period', 'fixedterm']
    for cat in cat_int:
        data[cat] = data[cat].astype('int')
    cat_float = ['period']
    for cat in cat_float:
        data[cat] = data[cat].astype('float')
    return data

train_data_mod = change_type(train_data_mod)
test_data_mod = change_type(test_data_mod)

# 特徴作成
# 200507_1 20385.406665991788
def create_feature(data):
    feature = data[['所在地', '間取り', '築年数', '方角', '面積', '建物構造',
                    'line', 'station', 'walk', 'floor', 'height', 'personal_bath', 'personal_toilet', 'separate', 'shower',
                    'bathroom_dryer', 'washing_toilet', 'washstand', 'dressing_room', 'refire', 'shared_toilet', 'shared_bath',
                    'bath_unknown', 'stove_list', 'stove2_list', 'system_kitchen_list', 'hot_water_supply_list', 'independent_kitchen_list',
                    'stove3_list', 'stove_IH_list', 'stove1_list', 'refrigerator_list', 'stove2_can_list', 'counter_kitchen_list',
                    'L_kitchen_list', 'stove_can_unknown_list', 'electric_stove_list', 'stove3_can_list', 'stove4over_list',
                    'stove1_can_list', 'stove4over_can_list', 'kitchen_unknown_list', 'internet', 'CATV', 'CS', 'BS', 'optical_fiber',
                    'HS_internet', 'free_internet', 'cable_broadcasting', 'broadcast_unknown', 'air_conditioner', 'indoor_washing_machine',
                    'flooring', 'citygas', 'balcony', 'sewage', 'public_water_supply', 'shoesbox', 'garbagestorage_onsite', 'elevator',
                    'tiling', 'hour24_ventilation_system', 'two_sided_lighting', 'air_conditioning', 'walkin_closet', 'parking', 'parking_cost',
                    'bike', 'bike_cost', 'bicycle', 'supermarket', 'supermarket_dist', 'convenience', 'convenience_dist', 'restaurant',
                    'restaurant_dist', 'drugstore', 'drugstore_dist', 'hospital', 'hospital_dist', 'park', 'park_dist', 'postoffice',
                    'postoffice_dist', 'elementary', 'elementary_dist', 'university', 'university_dist', 'period', 'fixedterm']].copy()
    cat_cols = ['line', 'station']
    for cat in cat_cols:
        feature[cat] = feature[cat].astype("category")
    feature = feature.rename(columns={'所在地': 'location',
                                      '間取り': 'type',
                                      '築年数': 'age',
                                      '方角': 'direction',
                                      '面積': 'area',
                                      '建物構造': 'structure',})
    return feature

# 200507_2 20385.406665991788
# 200508_2 20661.105510628804
#def create_feature(data):
#    feature = data[['所在地', '間取り', '築年数', '方角', '面積', '建物構造',
#                    'line', 'station', 'walk', 'floor', 'height', 'personal_bath', 'personal_toilet', 'separate', 'shower',
#                    'bathroom_dryer', 'washing_toilet', 'washstand', 'dressing_room', 'refire', 'shared_toilet',
#                    'stove_list', 'stove2_list', 'system_kitchen_list', 'hot_water_supply_list', 'independent_kitchen_list',
#                    'stove3_list', 'stove_IH_list', 'refrigerator_list', 'counter_kitchen_list',
#                    'L_kitchen_list', 'stove4over_list',
#                    'kitchen_unknown_list', 'internet', 'CATV', 'CS', 'BS', 'optical_fiber',
#                    'free_internet', 'broadcast_unknown', 'indoor_washing_machine',
#                    'flooring', 'citygas', 'balcony', 'sewage', 'public_water_supply', 'shoesbox', 'garbagestorage_onsite', 'elevator',
#                    'tiling', 'hour24_ventilation_system', 'two_sided_lighting', 'air_conditioning', 'walkin_closet', 'parking', 'parking_cost',
#                    'bike', 'bike_cost', 'bicycle', 'supermarket', 'supermarket_dist', 'convenience', 'convenience_dist', 'restaurant',
#                    'restaurant_dist', 'drugstore_dist', 'hospital', 'hospital_dist', 'park_dist',
#                    'postoffice_dist', 'elementary', 'elementary_dist', 'university', 'university_dist', 'period', 'fixedterm']].copy()
#    cat_cols = ['line', 'station']
#    for cat in cat_cols:
#        feature[cat] = feature[cat].astype("category")
#    feature = feature.rename(columns={'所在地': 'location',
#                                      '間取り': 'type',
#                                      '築年数': 'age',
#                                      '方角': 'direction',
#                                      '面積': 'area',
#                                      '建物構造': 'structure',})
#    return feature

# 200508_1 20555.17812696607
#def create_feature(data):
#    feature = data[['所在地', '間取り', '築年数', '方角', '面積', '建物構造',
#                    'line', 'station', 'walk', 'floor', 'height',
#                    'bathroom_dryer', 'washing_toilet', 'washstand', 'dressing_room', 'refire',
#                    'stove_list', 'system_kitchen_list', 'hot_water_supply_list', 'independent_kitchen_list',
#                    'stove3_list', 'stove_IH_list', 'counter_kitchen_list',
#                    'L_kitchen_list',
#                    'kitchen_unknown_list', 'internet', 'CATV', 'CS', 'BS', 'optical_fiber',
#                    'free_internet', 'indoor_washing_machine',
#                    'flooring', 'citygas', 'balcony', 'public_water_supply', 'garbagestorage_onsite', 'elevator',
#                    'tiling', 'hour24_ventilation_system', 'two_sided_lighting', 'air_conditioning', 'walkin_closet', 'parking', 'parking_cost',
#                    'bicycle', 'supermarket', 'supermarket_dist', 'convenience_dist', 'restaurant',
#                    'restaurant_dist', 'drugstore_dist', 'park_dist',
#                    'elementary_dist', 'period', 'fixedterm']].copy()
#    cat_cols = ['line', 'station']
#    for cat in cat_cols:
#        feature[cat] = feature[cat].astype("category")
#    feature = feature.rename(columns={'所在地': 'location',
#                                      '間取り': 'type',
#                                      '築年数': 'age',
#                                      '方角': 'direction',
#                                      '面積': 'area',
#                                      '建物構造': 'structure',})
#    return feature

# 200509_1 21936.25332374267
#def create_feature(data):
#    feature = data[['所在地', '間取り', '築年数', '面積', '建物構造',
#                    'line', 'station', 'walk', 'floor', 'height',
#                    'bathroom_dryer', 'washing_toilet', 'washstand', 'dressing_room', 'refire',
#                    'stove_list', 'system_kitchen_list',
#                    'stove3_list', 'stove_IH_list',
#                    'kitchen_unknown_list', 'internet',
#                    'indoor_washing_machine',
#                    'citygas', 'garbagestorage_onsite', 'elevator',
#                    'tiling', 'two_sided_lighting', 'air_conditioning', 'walkin_closet', 'parking', 'parking_cost',
#                    'bicycle', 'supermarket', 'supermarket_dist', 'convenience_dist', 'restaurant',
#                    'drugstore_dist',
#                    'period', 'fixedterm']].copy()
#    cat_cols = ['line', 'station']
#    for cat in cat_cols:
#        feature[cat] = feature[cat].astype("category")
#    feature = feature.rename(columns={'所在地': 'location',
#                                      '間取り': 'type',
#                                      '築年数': 'age',
#                                      '面積': 'area',
#                                      '建物構造': 'structure',})
#    return feature


X_train = create_feature(train_data_mod)
X_test = create_feature(test_data_mod)
Y_train = train_data_mod['賃料']


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
rate_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4, 0.41, 0.42, 0.42, 0.43, 0.44, 0.45, 0.5]
depth_list = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

esr = 100

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
                early_stopping_rounds=esr, 
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
        early_stopping_rounds=esr, 
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

plt.figure(figsize = (10, 8))
sns.barplot(data=feature_importance, x='importance', y='feature_name')
plt.savefig('feature_importance.png')



# 提出用データを作成
submission = pd.concat([test_data.loc[:,"id"], pd.Series(Y_pred, name='label')], axis=1)
submission.to_csv('submission.csv', header=False, index=False)


