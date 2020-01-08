#https://www.kaggle.com/mgmarques/houses-prices-complete-solution/
import os
from datetime import datetime
import warnings
warnings.simplefilter(action = 'ignore', category=FutureWarning)
warnings.filterwarnings('ignore')
def ignore_warn(*args, **kwargs):
    pass

warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)

import numpy as np
import pandas as pd
pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) #Limiting floats output to 3 decimal points
from matplotlib import pyplot as plt
#%matplotlib inline
import seaborn as sns
sns.set(style="ticks", color_codes=True, font_scale=1.5)
color = sns.color_palette()
sns.set_style('darkgrid')
import mpl_toolkits
from mpl_toolkits.mplot3d import Axes3D
import pylab
from sklearn.feature_selection import RFECV
from scipy import stats
from scipy.stats import skew, norm, probplot, boxcox
from scipy.special import boxcox1p
from patsy import dmatrices
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import RobustScaler, PolynomialFeatures, StandardScaler, LabelEncoder
from sklearn.feature_selection import f_regression, mutual_info_regression, SelectKBest, RFECV, SelectFromModel
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from sklearn.feature_extraction import FeatureHasher
from sklearn.decomposition import PCA, KernelPCA
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression, OrthogonalMatchingPursuit, Lasso, LassoLarsIC, ElasticNet, ElasticNetCV
from sklearn.linear_model import SGDRegressor, PassiveAggressiveRegressor, HuberRegressor, BayesianRidge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, BaggingRegressor, ExtraTreesRegressor
import xgboost as xgb
from xgboost import XGBRegressor, plot_importance
import lightgbm as lgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
import sklearn.model_selection
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold, cross_val_predict, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#Load dataset
features_train_df = pd.read_csv('G:/DataScienceProject/Datadriven-DengAI/dengue_features_train.csv')
labels_train_df = pd.read_csv('G:/DataScienceProject/Datadriven-DengAI/dengue_labels_train.csv')
features_test_df = pd.read_csv('G:/DataScienceProject/Datadriven-DengAI/dengue_features_test.csv')

features_train_df['total_cases'] = labels_train_df['total_cases']
features_test_df['total_cases'] = 0

#Exploratory Data Analysis (EDA)
def rstr(df, pred=None):
    obs = df.shape[0]
    types = df.dtypes
    counts = df.apply(lambda x: x.count())
    uniques = df.apply(lambda x: [x.unique()])
    nulls = df.apply(lambda x: x.isnull().sum())
    distincts = df.apply(lambda x: x.unique().shape[0])
    missing_ration = (df.isnull().sum()/ obs) * 100
    skewness = df.skew()
    kurtosis = df.kurt()
    print('Data shape:', df.shape)

    if pred is None:
        cols = ['types', 'counts', 'distincts', 'nulls', 'missing ration', 'uniques', 'skewness', 'kurtosis']
        str = pd.concat([types, counts, distincts, nulls, missing_ration, uniques, skewness, kurtosis], axis = 1)

    else:
        corr = df.corr()[pred]
        str = pd.concat([types, counts, distincts, nulls, missing_ration, uniques, skewness, kurtosis, corr], axis = 1, sort=False)
        corr_col = 'corr '  + pred
        cols = ['types', 'counts', 'distincts', 'nulls', 'missing_ration', 'uniques', 'skewness', 'kurtosis', corr_col ]

    str.columns = cols
    dtypes = str.types.value_counts()
    print('__\nData types:\n',str.types.value_counts())
    print('__')
    return str

details = rstr(features_train_df, 'total_cases')

#Check total_cases distribution
def QQ_plot(data, measure):
    fig = plt.figure(figsize=(20,7))

    #Get the fitted parameters used by the function
    (mu, sigma) = norm.fit(data)

    #Kernel Density plot
    fig1 = fig.add_subplot(121)
    sns.distplot(data, fit=norm)
    fig1.set_title(measure + ' Distribution ( mu = {:.2f} and sigma = {:.2f} )'.format(mu, sigma), loc='center')
    fig1.set_xlabel(measure)
    fig1.set_ylabel('Frequency')

    #QQ plot
    fig2 = fig.add_subplot(122)
    res = probplot(data, plot=fig2)
    fig2.set_title(measure + ' Probability Plot (skewness: {:.6f} and kurtosis: {:.6f} )'.format(data.skew(), data.kurt()), loc='center')

    plt.tight_layout()
    plt.show()

QQ_plot(features_train_df.total_cases, 'total_cases')

#We use the numpy fuction log1p which applies log(1+x) to all elements of the column
features_train_df.total_cases = np.log1p(features_train_df.total_cases)
QQ_plot(features_train_df.total_cases, 'Log1P of total_cases')

#NA
naList = list(features_train_df)
naList = naList[4:]
for i, value in enumerate(naList):
    features_train_df[value] = features_train_df[value].fillna(features_train_df[value].median())

features = list(features_test_df.columns)
RawNaList = features_test_df.loc[features_test_df.isna().any(axis=1)].index
for i in range(0, len(RawNaList)):
    for j in range(0, len(features)):
        if pd.isnull(features_test_df[features[j]][RawNaList[i]]) == True:
            features_test_df[features[j]][RawNaList[i]] = features_train_df[features[j]].median()


def featureAdjust(dfName):
    dfName.station_min_temp_c = dfName.station_min_temp_c + 273
    dfName.year = 2013 - dfName.year
    dfName.reanalysis_relative_humidity_percent = abs(dfName.reanalysis_relative_humidity_percent - dfName.reanalysis_relative_humidity_percent.max())
    dfName['tdtr2maxk'] =  dfName.reanalysis_max_air_temp_k /dfName.reanalysis_tdtr_k
    dfName['min2rng'] = dfName.station_min_temp_c /dfName.station_diur_temp_rng_c
    dfName['ndvi'] = (dfName['ndvi_ne'] + dfName['ndvi_nw'] + dfName['ndvi_se'] + dfName['ndvi_sw'])/4
    dfName.drop(['ndvi_ne', 'ndvi_nw', 'ndvi_se', 'ndvi_sw', 'week_start_date', 'station_avg_temp_c', 'reanalysis_precip_amt_kg_per_m2', 'precipitation_amt_mm', 'reanalysis_sat_precip_amt_mm', 'station_precip_mm'],inplace=True,axis=1)

featureAdjust(dfName=features_train_df)
featureAdjust(dfName=features_test_df)
reorderList= list(features_train_df)[0:14] + list(features_train_df)[15:18] + list(features_train_df)[14:15]
features_train_df = features_train_df[reorderList]
features_test_df = features_test_df[reorderList]

#Dummies
citycode = {}
citycode['sj'] = 1
citycode['iq'] = 2
features_train_df.city = features_train_df.city.map(citycode)
features_test_df.city = features_test_df.city.map(citycode)

#Convert col int float32 & int32
features = list(features_test_df.columns)
def convertCol(df):
    for i, value in enumerate(features):
        if df[value].dtypes == 'float64':
            df[value] = df[value].astype('float32')
        else:
            df[value] = df[value].astype('int32')

convertCol(df=features_train_df)
convertCol(df=features_test_df)

#Plot all col heatmap - for dectect colinearity
train_data = features_train_df[0:14]
target = features_train_df['total_cases']

C_mat = features_train_df.corr()
fig = plt.figure(figsize = (15,15))

sb.heatmap(C_mat, vmax = .9, square = True)
plt.show()

#Remove colinearity
corrTable = features_train_df.corr()
corrTable.to_csv('G:/DataScienceProject/Datadriven-DengAI/corrTable-Complete.csv')
features_train_df.drop(['reanalysis_avg_temp_k', 'reanalysis_specific_humidity_g_per_kg', 'reanalysis_tdtr_k'],inplace=True,axis=1)
features_test_df.drop(['reanalysis_avg_temp_k', 'reanalysis_specific_humidity_g_per_kg', 'reanalysis_tdtr_k'],inplace=True,axis=1)

#Multicollinearity
def VRF(predict, data, y):

    scale = StandardScaler(with_std=False)
    df = pd.DataFrame(scale.fit_transform(data), columns= cols)
    features = "+".join(cols)
    df['total_cases'] = y.values

    # get y and X dataframes based on this regression:
    y, X = dmatrices(predict + ' ~' + features, data = df, return_type='dataframe')

   # Calculate VIF Factors
    # For each X, calculate VIF and save in dataframe
    vif = pd.DataFrame()
    vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif["features"] = X.columns

    # Inspect VIF Factors
    display(vif.sort_values('VIF Factor'))
    return vif

# Remove the higest correlations and run a multiple regression
cols = features_train_df.columns
cols = cols.drop(['total_cases'])
vif = VRF('total_cases', features_train_df.loc[features_train_df.total_cases>0, cols], features_train_df.total_cases[features_train_df.total_cases>0])

#Lets remove as it cause for high VIF
features_train_df.drop(['reanalysis_dew_point_temp_k'],inplace=True,axis=1)
features_test_df.drop(['reanalysis_dew_point_temp_k'],inplace=True,axis=1)

#boxcox to handle skewness [81-83]
featuresList = list(features_train_df)
skewList= list(features_train_df.skew())
featuresList = featuresList[3:13]
for i, value in enumerate(featuresList):
    if abs(features_train_df.skew()[i]) > 0.7:
        print(value, "  ", features_train_df.skew()[i])

features_train_df['tdtr2maxk'] = features_train_df['tdtr2maxk'] **0.5
features_test_df['tdtr2maxk'] = features_test_df['tdtr2maxk'] **0.5
features_train_df['station_diur_temp_rng_c'] = 1/(features_train_df['station_diur_temp_rng_c'])
features_test_df['station_diur_temp_rng_c'] = 1/(features_test_df['station_diur_temp_rng_c'])

'''
Let handle the col skew acccording to followed params:
Lambda value (λ)	Transformed data (Y')
-3	                Y**-3 = 1/Y**3
-2	                Y**-2 = 1/Y**2
-1	                Y**-1 = 1/Y
-0.5	            Y**-0.5 = 1/(√(Y))
0	                log(Y)(*)
0.5	                Y0.5 = √(Y)
1	                Y**1 = Y
2	                Y**2
3	                Y**3
'''
#Adding ^2 & ^3 features
featuresList = list(features_train_df)
newFeature = featuresList[3:13]
for i, value in enumerate(newFeature):
    colName1 = value + str("^2")
    features_train_df[colName1] = features_train_df[value]**2
    features_test_df[colName1] = features_test_df[value]**2
    colName2 = value + str("^3")
    features_train_df[colName2] = features_train_df[value]**3
    features_test_df[colName2] = features_test_df[value]**3

features_train_df.drop(['Unnamed: 0'],inplace=True,axis=1)
features_test_df.drop(['Unnamed: 0'],inplace=True,axis=1)
reorderCol = list(features_train_df)
reorderList= list(features_train_df)[0:13] + list(features_train_df)[14:33] + reorderCol[13:14]
features_train_df = features_train_df[reorderList]
features_test_df = features_test_df[reorderList]

# split data into train and test sets
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
df = features_train_df.iloc[:,0:32]
y_train = features_train_df['total_cases']
#Create cross-validation
X_train, X_cv, y_train, y_cv = train_test_split(features_train_df.iloc[:,0:32], features_train_df['total_cases'], test_size = 0.2, random_state = 0)

#PCA
#Plot components number
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.title('Pulsar Dataset Explained Variance')
plt.show()

#Chcek explained variant between X_train & X_test
pca = PCA(n_components = 3)
X_train = pca.fit_transform(X_train)
X_cv = pca.transform(X_cv)
explained_variance = pca.explained_variance_ratio_
explained_variance

#Use XGBoost for train & fit
model =  XGBRegressor(base_score=0.5, colsample_bylevel=1, colsample_bytree=1, gamma=0, max_delta_step=0,
                      random_state=101, min_child_weight=1, missing=None, n_jobs=4,
                      scale_pos_weight=1, seed=None, silent=True, subsample=1)

model.fit(X_train, y_train)

#Prediction cross-validation
y_pred = model.predict(X_cv)

# making RMS in cross-validation
# test set of Y and predicted value.
from sklearn.metrics import mean_squared_error
from math import sqrt
rms = sqrt(mean_squared_error(y_cv, y_pred))

#
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
X_test = features_test_df.iloc[:,0:32]
X_test = pca.transform(X_test)
y_test = features_test_df['total_cases']
y_test = model.predict(X_test)

#Convert y_test from log into "real cases"
y_test = np.exp(y_test) + 1

submission = pd.read_csv('G:/DataScienceProject/Datadriven-DengAI/dengue_features_test.csv', usecols=['city', 'year', 'weekofyear'])
submission['total_cases'] = y_test
submission['total_cases'] = round(submission['total_cases']).astype('int64')
submission.to_csv('G:/DataScienceProject/Datadriven-DengAI/submission3.csv', index=False)
