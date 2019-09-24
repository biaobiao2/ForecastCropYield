# coding=utf-8
#!/usr/bin/python3
'''
Created on 2019年9月19日

@author: liushouhua
'''

import os
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 1000,'display.max_columns', 1000,"display.max_colwidth",1000,'display.width',1000)
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)

import matplotlib.pyplot as plt 
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False

import warnings
warnings.filterwarnings('ignore')


from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold,GridSearchCV
import xgboost as xgb
from xgboost import plot_importance
from sklearn.metrics import mean_squared_error
from scipy.interpolate import lagrange
from scipy.stats import norm,skew
import seaborn as sns
path = "../data/"
from scipy import stats
def changeArr(x):
    try:
        r = float(x)
    except:
        r = np.nan
    return r

def deal_loss():
    """处理缺失值，牛顿插值法，因为只用年中数据对于插值产生的年尾几天数据的准确程度忽略
    """
    
    rows = ['日照时数（单位：h)', '日平均风速(单位：m/s)', '日降水量（mm）', '日最高温度（单位：℃）', '日最低温度（单位：℃）', '日平均温度（单位：℃）', '日相对湿度（单位：%）', '日平均气压（单位：hPa）']
    df_rice = pd.read_csv(path+"train_rice.csv",encoding="gbk")
    df_rice["later"] = (df_rice["2015年晚稻"]+df_rice["2016年晚稻"]+df_rice["2017年晚稻"])/3
    df = pd.read_csv(path+"train_weather.csv",encoding="gbk")

    for row in rows:
        df[row] = df[row].apply(changeArr)
    
    df['日降水量（mm）'][list(df[df['日降水量（mm）']>=150].index)] = np.nan
    for row in rows:
        print(row)
        for j in range(len(df)-1):
            if df[row].isnull()[j]:
                df[row][j] = ployinterrp(df[row], j)
    
    df.to_csv(path + 'change_data_1.csv', index=False,encoding='gbk')

def change():
    """处理数据，降低纬度,按照月维度
    """
    if not os.path.exists(path+"train_rice.csv",):
        deal_loss()
        
    df_rice = pd.read_csv(path+"train_rice.csv",encoding="gbk")
    df_rice["later"] = (df_rice["2015年晚稻"]+df_rice["2016年晚稻"]+df_rice["2017年晚稻"])/3
    countyId = set(list(df_rice['区县id'].values))
    
    df = pd.read_csv(path+"change_data_1.csv",encoding="gbk")
    
    years=[2015,2016,2017,2018]
    months=[7,8,9,10]

    #删除站点为2的数据
    df = df.drop(df[df["站名id"]==2].index)
    df["日降水量（mm）"] = np.log1p( df["日降水量（mm）"])
    
#     df["日平均气压（单位：hPa）"] = np.log1p( df["日平均气压（单位：hPa）"])
    def test(row):
        print(df[row].skew())
        sns.distplot(df[row])
        fig=plt.figure()
        res = stats.probplot(df[row],plot=plt)
        plt.show()
#     test("日平均气压（单位：hPa）")
    
    
    rows = ['日照时数（单位：h)', '日平均风速(单位：m/s)', '日降水量（mm）', '日最高温度（单位：℃）', '日最低温度（单位：℃）', '日平均温度（单位：℃）', '日相对湿度（单位：%）', '日平均气压（单位：hPa）']

    rows_sum = ['日照时数（单位：h)','日降水量（mm）']
    rows_min = ['日最低温度（单位：℃）']
    rows_max = ['日最高温度（单位：℃）']
    df_mean = df.groupby(['区县id','站名id','年份','月份'])[rows].mean()
    df_sum = df.groupby(['区县id','站名id','年份','月份'])[rows_sum].sum()
    df_min = df.groupby(['区县id','站名id','年份','月份'])[rows_min].min()
    df_max = df.groupby(['区县id','站名id','年份','月份'])[rows_max].max()
    
    feature = ["产量","地区特征"]
    for m in months:
        for i in rows:
            feature.append(str(m)+"月份"+i)
        feature.extend([str(m)+"月份日照时数",str(m)+"月份降水量"])
        feature.extend([str(m)+"月份最低温度",str(m)+"月份最高温度"])

    res = pd.DataFrame()
    res["feature"] = feature
    for cid in countyId:
        for year in years:
            label = 0
            later_rice = df_rice.loc[df_rice['区县id'] == cid]["later"].values[0]
            if year != 2018 :
                label = df_rice.loc[df_rice['区县id'] == cid][str(year)+'年晚稻'].values[0]
            features = [label,later_rice]
            for m in months:
                features += df_mean.loc[(cid, 1, year, m)].values.tolist()
                features += df_sum.loc[(cid, 1, year, m)].values.tolist()
                features += df_min.loc[(cid, 1, year, m)].values.tolist()
                features += df_max.loc[(cid, 1, year, m)].values.tolist()
            res[cid + '_' + str(year)] = features
    
    
    res = res.set_index("feature")
    columns = res.columns

    res = res.T
    res.insert(0,'columns',columns)
    res.to_csv('create_feature.csv', index=False,encoding='gbk')


def ployinterrp(s,n,k=5):
    """拉格朗日插值法
    """
    y = s[list(range(n-k,n))+list(range(n+1,n+1+k)) ]
    y = y[y.notnull()]
    return y.mean()
#     return lagrange(y.index, list(y))(n)


def change_week():
    
    rows = ['日照时数（单位：h)', '日平均风速(单位：m/s)', '日降水量（mm）', '日最高温度（单位：℃）', '日最低温度（单位：℃）', '日平均温度（单位：℃）', '日相对湿度（单位：%）', '日平均气压（单位：hPa）']
    df_rice = pd.read_csv(path+"train_rice.csv",encoding="gbk")
    df_rice["later"] = (df_rice["2015年晚稻"]+df_rice["2016年晚稻"]+df_rice["2017年晚稻"])/3
    df = pd.read_csv(path+"train_weather.csv",encoding="gbk")
    df["week"] = df["年份"].map(str)+"-"+df["月份"].map(str)+"-"+df["日期"].map(str)
    df["week"] = pd.to_datetime(df["week"])
    df["week"] = df["week"].dt.weekofyear
    
    countyId = set(list(df_rice['区县id'].values))
    years=[2015,2016,2017,2018]
    weeks=[i for i in range(30,44)]
    
    for row in rows:
        df[row] = df[row].apply(changeArr)
    df['日降水量（mm）'][list(df[df['日降水量（mm）']>=150].index)] = np.nan
    for row in rows:
        print(row)
        for j in range(len(df)-1):
            if df[row].isnull()[j]:
                df[row][j] = ployinterrp(df[row], j)
        
    #删除站点为2的数据
    df = df.drop(df[df["站名id"]==2].index)

    
    rows_mean = rows 
    rows_sum = ['日照时数（单位：h)','日降水量（mm）']
    rows_min = ['日最低温度（单位：℃）']
    rows_max = ['日最高温度（单位：℃）']
    df_mean = df.groupby(['区县id','站名id','年份',"week"])[rows_mean].mean()
    df_sum = df.groupby(['区县id','站名id','年份',"week"])[rows_sum].sum()
    df_min = df.groupby(['区县id','站名id','年份',"week"])[rows_min].min()
    df_max = df.groupby(['区县id','站名id','年份',"week"])[rows_max].max()
    
    feature = ["产量","地区特征"]
    for m in weeks:
        for i in rows_mean:
            feature.append(str(m)+"周"+i)
        feature.extend([str(m)+"周日照时数",str(m)+"周降水量"])
        feature.extend([str(m)+"周最低温度",str(m)+"周最高温度"])
    print(len(feature))
    res = pd.DataFrame()
    res["feature"] = feature
    postfix='年晚稻'
    for cid in countyId:
        for year in years:
            label = 0
            early_rice = df_rice.loc[df_rice['区县id'] == cid]["later"].values[0]
            if year != 2018 :
                label = df_rice.loc[df_rice['区县id'] == cid][str(year)+postfix].values[0]
            features = [label,early_rice]
            for m in weeks:
                features += df_mean.loc[(cid, 1, year, m)].values.tolist()
                features += df_sum.loc[(cid, 1, year, m)].values.tolist()
                features += df_min.loc[(cid, 1, year, m)].values.tolist()
                features += df_max.loc[(cid, 1, year, m)].values.tolist()
            res[cid + '_' + str(year)] = features
    
    res = res.set_index("feature")
    columns = res.columns
    res = res.T
    res.insert(0,'columns',columns)
    res.to_csv('create_feature.csv', index=False,encoding='gbk')
    
def get_data():
    """获取训练数据以及测试数据
    """
    if not os.path.exists('create_feature.csv'):
        change()
    df = pd.read_csv("create_feature.csv",encoding="gbk")
    
    testst = df[df["产量"]==0.0]
    prediction = pd.DataFrame()
    prediction["columns"] = testst["columns"]
    prediction["columns"] = prediction["columns"].apply(arr)
    
    df= df.drop('columns', 1)
    testst = testst.drop('columns', 1)
    testst = testst.drop('产量', 1)
    
    #删除预测数据
    df = df.drop(df[df["产量"]==0.0].index)
        
    y_label = df["产量"]
    X_data = df.drop('产量', 1)
    return X_data,y_label,testst,prediction




def model(X_data,y_label,testst,prediction):
    """模型搭建
    """
    global params_xgb  #模型参数，设置全局变量便于调参
    n_splits = 25
    res = []
    kf = KFold(n_splits = n_splits, shuffle=True, random_state=520)
    for i, (train_index, test_index) in enumerate(kf.split(X_data)):
        print('第{}次训练...'.format(i+1))
       
        train_data = X_data.iloc[train_index]
        train_label = y_label.iloc[train_index]
          
        valid_data = X_data.iloc[test_index]
        valid_label = y_label.iloc[test_index]

        xgb_train = xgb.DMatrix(train_data, label=train_label)
        xgb_valid = xgb.DMatrix(valid_data, valid_label)
        evallist = [(xgb_valid, 'eval'), (xgb_train, 'train')]
        cgb_model = xgb.train(params_xgb, xgb_train, num_boost_round=500 , evals=evallist, verbose_eval=500, early_stopping_rounds=300, feval=myFeval)
        
        valid = cgb_model.predict(xgb_valid, ntree_limit=cgb_model.best_ntree_limit)
        valid_score = mean_squared_error(valid_label,valid)*0.5
        if valid_score > 0.01:
            #验证集分数不好的模型丢弃
            continue
        xgb_test = xgb.DMatrix(testst)
        preds = cgb_model.predict(xgb_test, ntree_limit=cgb_model.best_ntree_limit)
          
        res.append(preds)
        
        print("\n")

#     plot_importance(cgb_model,max_num_features=20)
#     plt.show()

    res = np.array(res)
    print(res.shape)
    res = res.mean(axis=0)
    prediction["pre_xgboost"] = res
    prediction.to_csv('result.csv', index=False,encoding='gbk')                  

def myFeval(preds, xgbtrain):
    """模型评分 0.5*mse
    """
    xgbtrain = xgbtrain.get_label() #将xgboost.core.DMatrix类转化为ndarray类别
    score = mean_squared_error(xgbtrain,preds)*0.5
    return 'myFeval', score

def arr(a):
    return a.split("_")[0]

class XGBoostre(object):
    """封装类用于网格调参
    """
    def __init__(self,**kwargs):
        self.params = kwargs
        if "num_boost_round" in self.params:
            self.num_boost_round = self.params["num_boost_round"]
        self.params.update({'objective': 'reg:squarederror','silent': 0,'seed': 1000})
        
    def fit(self,x_train,y_train):
        xgb_train = xgb.DMatrix(x_train, label=y_train)
        self.bst = xgb.train(params=self.params, dtrain=xgb_train, num_boost_round=self.num_boost_round,  verbose_eval=100, feval=myFeval)
    
    def predict(self,x_pred):
        dpred = xgb.DMatrix(x_pred)
        return self.bst.predict(dpred)
    
    def kfold(self,x_train,y_train,n_fold=5):
        xgb_train = xgb.DMatrix(x_train, label=y_train)
        bst_cv = xgb.cv(params=self.params, dtrain=xgb_train,feval=myFeval,num_boost_round=self.num_boost_round, nfold=n_fold,)
        return bst_cv.iloc[-1,:]
    
    def plt_feature_importance(self):
        feat = pd.Series(self.bst.get_fscore()).sort_values(ascending=False)
        feat.plot(title = "Feature_importance")
    
    def get_params(self,deep=True):
        return self.params
    
    def set_params(self,**params):
        self.params.update(params)
        return self
    
if __name__ == "__main__":
    deal_loss()
    change()
    change_week()
    params_xgb = {
        'booster': 'gbtree',
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',  # 对于回归问题,默认值是rmse,对于分类问题,默认值是error
        'gamma': 0.1,     #损失下降多少才分裂
        'max_depth': 4,  
        'lambda': 1.2,    #控制模型复杂度的权重值的L2曾泽化参数，参数越大越不容易过拟合
        'subsample': 0.9,   #随机采样的训练样本
        'colsample_bytree': 0.9,    #生成树时特征采样比例
        'min_child_weight': 3,  # 6
        'silent': 0,    #信息输出设置成1则没有信息输出
        'eta': 0.12,   #类似学习率
        'seed': 1000,  
        'nthread': 9,
    }
    X_data,y_label,testst,prediction = get_data()
       
    model(X_data,y_label,testst,prediction)
    
    
    df = pd.read_csv("result.csv",encoding="gbk")
    df["区县id"] =  df["columns"].apply(arr)
       
    df1 = pd.read_csv(path+"train_rice.csv",encoding="gbk")
    d = df1[["区县id","2017年晚稻"]]
    la = pd.merge(df,d,on="区县id")
    score = mean_squared_error(la["pre_xgboost"],la["2017年晚稻"])*0.5
    print(score)
    s = la[["区县id","pre_xgboost"]]
    s.to_csv('x_prediction.csv', index=False,encoding='gbk',header=False)

    
#     xgb_param_grid = {"lambda":[1.2],"max_depth":[3,4,5,6],"min_child_weight":[i for i in range(1,6)]}
#     grid = GridSearchCV(XGBoostre(max_depth=3,min_child_weight=3,eta=0.12,gamma=0.1,subsample=0.9,colsample_bytree=0.9,num_boost_round=500,early_stopping_rounds=300,),param_grid=xgb_param_grid,scoring="neg_mean_squared_error",cv=5)
#     grid.fit(X_data,y_label)
#     means = grid.cv_results_['mean_test_score']
#     params = grid.cv_results_['params']
#     for mean,param in zip(means,params):
#         print("%f  with:   %r" % (mean,param))
#     print(grid.best_params_,"\n",grid.best_score_)
    
    
    
    
 




