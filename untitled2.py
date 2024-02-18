import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import openpyxl
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_selection import VarianceThreshold,RFE,f_regression
import warnings
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings("ignore")
from matplotlib import rcParams
from matplotlib.cm import rainbow
from sklearn.svm import SVR
from sklearn.feature_selection import chi2
import sklearn.metrics
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor, BaggingRegressor, GradientBoostingRegressor
from sklearn.svm import LinearSVR, NuSVR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import BayesianRidge, ARDRegression
from sklearn.linear_model import Ridge, RidgeCV, SGDRegressor, ElasticNet
from sklearn.cross_decomposition import PLSRegression
import time
from matplotlib.colors import Normalize
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
data = pd.read_csv("C:/Users/91849/Desktop/MAJOR PROJECT/latest.csv")
data
plt.figure(figsize=(15,15))
sns.heatmap(data.corr(),cmap='PiYG',annot=True)
#Before Feature Scaling
x = data.drop(['Dhi'],axis=1)
y= data['Dhi']
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=100)
regressor = LinearRegression()
regressor.fit(x_train, y_train)
# predicting the test set result
y_pred_logreg = regressor.predict(x_test)
from sklearn.metrics import max_error
from sklearn.metrics import median_absolute_error
EVC=explained_variance_score(y_test, y_pred_logreg)
ME=max_error(y_test, y_pred_logreg)
MAE=mean_absolute_error(y_test, y_pred_logreg)
MSE=mean_squared_error(y_test, y_pred_logreg)
MDAE=median_absolute_error(y_test, y_pred_logreg)
RS=r2_score(y_test, y_pred_logreg)
datareg=pd.DataFrame(columns=['Regressor','EVC','ME','MAE','MSE','MDAE','RS'])
L2=["LinearRegressor",EVC,ME,MAE,MSE,MDAE,RS]
datareg.loc[len(datareg),:]=L2
adaregressor = AdaBoostRegressor()
adaregressor.fit(x_train,y_train)
y_pred_adareg = adaregressor.predict(x_test)
EVC=explained_variance_score(y_test, y_pred_adareg)
ME=max_error(y_test, y_pred_adareg)
MAE=mean_absolute_error(y_test, y_pred_adareg)
MSE=mean_squared_error(y_test, y_pred_adareg)
MDAE=median_absolute_error(y_test, y_pred_adareg)
RS=r2_score(y_test, y_pred_adareg)
L2=["AdaBoostRegressor",EVC,ME,MAE,MSE,MDAE,RS]
datareg.loc[len(datareg),:]=L2
