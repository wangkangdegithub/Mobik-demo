# coding: utf-8

# In[ ]:

'''
2017年8月14日
algorithm：Knn
@author：wangkang
'''

# In[ ]:

import numpy as np
import pandas as pd
from pandas import Series, DataFrame

# In[22]:

train = pd.read_csv('F:/mobike/train.csv')
train.dtypes

# In[23]:

train.isnull().sum()

# In[24]:

# 首先对日期时间转化
import datetime

train.loc[:, 'starttime'] = pd.to_datetime(train.starttime)
train['weekday_time'] = train.starttime.dt.weekday
train['hour_time'] = train.starttime.dt.hour
train['minute_time'] = train.starttime.dt.minute

# In[25]:

# 对经纬度解码
import Geohash

start = list(train.geohashed_start_loc)
start_jw = list(map(lambda x: Geohash.decode_exactly(x), start))
end = list(train.geohashed_end_loc)
end_jw = list(map(lambda x: Geohash.decode_exactly(x), end))
train[['start_lat', 'start_lon', 'start_lat_exactly', 'start_lon_exactly']] = pd.DataFrame(start_jw,
                                                                                           columns=['start_lat',
                                                                                                    'start_lon',
                                                                                                    'start_lat_exactly',
                                                                                                    'start_lon_exactly'])
train[['end_lat', 'end_lon', 'end_lat_exactly', 'end_lon_exactly']] = pd.DataFrame(end_jw,
                                                                                   columns=['end_lat', 'end_lon',
                                                                                            'exactly_lat',
                                                                                            'exactly_lon'])

# In[26]:

from math import *


# In[27]:

# 求两点经纬度之间的距离
def getDistance(latA, lonA, latB, lonB):
    ra = 6378140  # 赤道半径 单位m
    rb = 6356755  # 极半径  单位m
    flatten = (ra - rb) / ra  # 地球扁率
    # change angle to radians
    radLatA = radians(latA)  # 经度转化为弧度
    radLonA = radians(lonA)  # 纬度转化为弧度
    radLatB = radians(latB)
    radLonB = radians(lonB)

    pA = atan(rb / ra * tan(radLatA))
    pB = atan(rb / ra * tan(radLatB))
    x = acos(sin(pA) * sin(pB) + cos(pA) * cos(pB) * cos(radLonA - radLonB))
    c1 = (sin(x) - x) * (sin(pA) + sin(pB)) ** 2 / cos(x / 2) ** 2
    c2 = (sin(x) + x) * (sin(pA) - sin(pB)) ** 2 / sin(x / 2) ** 2
    dr = flatten / 8 * (c1 - c2)
    distance = ra * (x + dr)
    return distance


# In[28]:

# 求两点经纬度之间的方位角
def getDegree(latA, lonA, latB, lonB):
    """
    Args:
        point p1(latA, lonA)
        point p2(latB, lonB)
    Returns:
        bearing between the two GPS points,
        default: the basis of heading direction is north
    """
    radLatA = radians(latA)
    radLonA = radians(lonA)
    radLatB = radians(latB)
    radLonB = radians(lonB)
    dLon = radLonB - radLonA
    y = sin(dLon) * cos(radLatB)
    x = cos(radLatA) * sin(radLatB) - sin(radLatA) * cos(radLatB) * cos(dLon)
    brng = degrees(atan2(y, x))
    brng = (brng + 360) % 360
    return brng


# In[29]:

# dataframe 转list
jw_info = train[['start_lat', 'start_lon', 'end_lat', 'end_lon']].values.tolist()

# 求经纬距离
jw_distance = list(map(lambda x: getDistance(float(x[0]), float(x[1]), float(x[2]), float(x[3])), jw_info))

# 计算start 和 end 的经纬度间的方位角
jw_degree = list(map(lambda x: getDegree(float(x[0]), float(x[1]), float(x[2]), float(x[3])), jw_info))

# 把jw_distance和jw_degree 由列表list形式转化为dataframe形式
train['distance'] = pd.DataFrame(jw_distance)
train['degree'] = pd.DataFrame(jw_degree)

# In[30]:

import math


# 修改start_lat、start_lon
def start_xyz(start_lat, start_lon):
    x_start = math.cos(start_lat) * math.cos(start_lon)
    y_start = math.cos(start_lat) * math.sin(start_lon)
    z_start = math.sin(start_lat)
    return x_start, y_start, z_start


# In[31]:

'''
0~6分别表示一周中的第几天，其中0表示周一，6表示周日
把周一早上0点记为0，周日晚上24点记为1

'''


def day_num(weekday_time, hour_time, minute_time):
    a = weekday_time
    b = hour_time
    c = minute_time / 60
    day_num = (a + b + c) / 7
    # day_num = (weekday_time + hour_time+(minute_time/60))/7
    return day_num


# In[32]:

# 新增一列day_num
df = train.loc[:, ['weekday_time', 'hour_time', 'minute_time']]
list_df = df.values.tolist()
# 把starttime转换为day_num
output = list(map(lambda x: day_num(x[0], x[1], x[2]), list_df))
train['day_num'] = pd.DataFrame(output)

# In[33]:

# 新增三列'x_start','y_start','z_start'
df_zb = train.loc[:, ['start_lat', 'start_lon', 'end_lat', 'end_lon']]
list_zb = df_zb.values.tolist()
output_zb = list(map(lambda x: start_xyz(x[0], x[1]), list_zb))
train[['x_start', 'y_start', 'z_start']] = pd.DataFrame(output_zb)

# In[34]:

# 查看字段index
train.columns

# In[35]:

# 选择X，y作为特征字段
X = train[['day_num', 'x_start', 'y_start', 'z_start']]
y = train[['end_lat', 'end_lon']]

# In[36]:

from sklearn.neighbors import KNeighborsRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import train_test_split

# In[38]:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
tuned_parameters = [{'n_neighbors': [5, 10, 20, 30]}]
'''
scores = ['r2']
for score in scores:   
    clf = GridSearchCV(KNeighborsRegressor(), tuned_parameters, cv=5, scoring=score)
    clf.fit(X_train, y_train)
    print('计算最佳参数：')
    #best_estimator_ returns the best estimator chosen by the search
    print(clf.best_estimator_)
    print("得分分别是:")
    for params, mean_score, scores in clf.grid_scores_:
        print('%0.3f (+/-%0.03f) for %r'% (mean_score, scores.std() / 2, params))
'''

# In[39]:

model = KNeighborsRegressor(n_neighbors=10).fit(X_train, y_train)
result = model.predict(X_test)

# In[42]:

result

# In[59]:

output_pred = pd.DataFrame(list(map(lambda x: Geohash.encode(x[0], x[1], precision=7), result)),
                           columns=['output_pred'])

# In[60]:

output_true = pd.DataFrame(list(map(lambda x: Geohash.encode(x[0], x[1], precision=7), y_test.values.tolist())),
                           columns=['output_true'])

# In[66]:

'''
sklearn.metrics.precision_score
评判标准为Precision准确率
'''
true = output_true.as_matrix()
pred = output_pred.as_matrix()

# In[67]:

'''
from sklearn.metrics import precision_score
precision_score(true, pred, average='samples')  

#precision_score不适用于multi-labels,因此不可行。
'''

# In[70]:

from sklearn.metrics import accuracy_score

'''
normalize == True,返回分类正确的样本比例；
normalize == False,返回分类正确的样本数；
'''
accuracy_score(true, pred, normalize=False)

# In[71]:

'''
修改test.csv的表结构，同train.csv一样
'''
test = pd.read_csv('F:/mobike/test.csv')

test.loc[:, 'starttime'] = pd.to_datetime(test.starttime)
test['weekday_time'] = test.starttime.dt.weekday
test['hour_time'] = test.starttime.dt.hour
test['minute_time'] = test.starttime.dt.minute

start = list(test.geohashed_start_loc)
start_jw = list(map(lambda x: Geohash.decode_exactly(x), start))

test[['start_lat', 'start_lon', 'start_lat_exactly', 'start_lon_exactly']] = pd.DataFrame(start_jw,
                                                                                          columns=['start_lat',
                                                                                                   'start_lon',
                                                                                                   'start_lat_exactly',
                                                                                                   'start_lon_exactly'])

# 新增一列day_num
df = test.loc[:, ['weekday_time', 'hour_time', 'minute_time']]
list_df = df.values.tolist()
# 把starttime转换为day_num
output = list(map(lambda x: day_num(x[0], x[1], x[2]), list_df))
test['day_num'] = pd.DataFrame(output)

# 新增三列'x_start','y_start','z_start'
df_zb = test.loc[:, ['start_lat', 'start_lon', 'end_lat', 'end_lon']]
list_zb = df_zb.values.tolist()
output_zb = list(map(lambda x: start_xyz(x[0], x[1]), list_zb))
test[['x_start', 'y_start', 'z_start']] = pd.DataFrame(output_zb)

# In[73]:

test_index = test[['day_num', 'x_start', 'y_start', 'z_start']]

# In[75]:

prediction = model.predict(test_index)

predictions = pd.DataFrame(list(map(lambda x: Geohash.encode(x[0], x[1], precision=7), prediction)),
                           columns=['predictions'])

# In[92]:

predictions.to_csv('F:/mobike/predictions.csv', sep=',', header=None)  # 导出为csv，其中,header =None 表示导出的数据不含表头（字段）。


# In[ ]:



