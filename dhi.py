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
baregressor = BaggingRegressor()
baregressor.fit(x_train, y_train)
y_pred_bareg = baregressor.predict(x_test)
EVC=explained_variance_score(y_test, y_pred_bareg)
ME=max_error(y_test, y_pred_bareg)
MAE=mean_absolute_error(y_test, y_pred_bareg)
MSE=mean_squared_error(y_test, y_pred_bareg)
MDAE=median_absolute_error(y_test, y_pred_bareg)
RS=r2_score(y_test, y_pred_bareg)
L2=["BaggingRegressor",EVC,ME,MAE,MSE,MDAE,RS]
datareg.loc[len(datareg),:]=L2
etregressor = ExtraTreesRegressor()
etregressor.fit(x_train,y_train)
y_pred_etreg = etregressor.predict(x_test)
EVC=explained_variance_score(y_test, y_pred_etreg)
ME=max_error(y_test, y_pred_etreg)
MAE=mean_absolute_error(y_test, y_pred_etreg)
MSE=mean_squared_error(y_test, y_pred_etreg)
MDAE=median_absolute_error(y_test, y_pred_etreg)
RS=r2_score(y_test, y_pred_etreg)
L2=["ExtraTreesRegressor",EVC,ME,MAE,MSE,MDAE,RS]
datareg.loc[len(datareg),:]=L2
gbregressor = GradientBoostingRegressor()
gbregressor.fit(x_train,y_train)
y_pred_gbreg = gbregressor.predict(x_test)
EVC=explained_variance_score(y_test, y_pred_gbreg)
ME=max_error(y_test, y_pred_gbreg)
MAE=mean_absolute_error(y_test, y_pred_gbreg)
MSE=mean_squared_error(y_test, y_pred_gbreg)
MDAE=median_absolute_error(y_test, y_pred_gbreg)
RS=r2_score(y_test, y_pred_gbreg)
L2=["GradientBoostingRegressor",EVC,ME,MAE,MSE,MDAE,RS]
datareg.loc[len(datareg),:]=L2
rfregressor = RandomForestRegressor()
rfregressor.fit(x_train,y_train)
y_pred_rfreg = rfregressor.predict(x_test)
EVC=explained_variance_score(y_test, y_pred_rfreg)
ME=max_error(y_test, y_pred_rfreg)
MAE=mean_absolute_error(y_test, y_pred_rfreg)
MSE=mean_squared_error(y_test, y_pred_rfreg)
MDAE=median_absolute_error(y_test, y_pred_rfreg)
RS=r2_score(y_test, y_pred_rfreg)
L2=["RandomForestRegressor",EVC,ME,MAE,MSE,MDAE,RS]
datareg.loc[len(datareg),:]=L2
from sklearn.datasets import make_regression
regr = ElasticNet(random_state=0)
regr.fit(x_train,y_train)
y_pred_enreg= regr.predict(x_test)
EVC=explained_variance_score(y_test, y_pred_enreg)
ME=max_error(y_test, y_pred_enreg)
MAE=mean_absolute_error(y_test, y_pred_enreg)
MSE=mean_squared_error(y_test, y_pred_enreg)
MDAE=median_absolute_error(y_test, y_pred_enreg)
RS=r2_score(y_test, y_pred_enreg)
L2=["ElasticNet",EVC,ME,MAE,MSE,MDAE,RS]
datareg.loc[len(datareg),:]=L2
tree_regressor = DecisionTreeRegressor(random_state =0)
tree_regressor.fit(x_train,y_train)
y_pred_dtreg=tree_regressor.predict(x_test)
EVC=explained_variance_score(y_test, y_pred_dtreg)
ME=max_error(y_test, y_pred_dtreg)
MAE=mean_absolute_error(y_test, y_pred_dtreg)
MSE=mean_squared_error(y_test, y_pred_dtreg)
MDAE=median_absolute_error(y_test, y_pred_dtreg)
RS=r2_score(y_test, y_pred_dtreg)
L2=["DecisionTreeRegressor",EVC,ME,MAE,MSE,MDAE,RS]
datareg.loc[len(datareg),:]=L2
curr_data=data
# with feature scaling
x = curr_data.drop(['Dhi'],axis=1)
y= curr_data['Dhi']
sc_x=StandardScaler()
x=sc_x.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=100)
sx_train=x_train
sx_test=x_test
regressor = LinearRegression()
regressor.fit(sx_train,y_train)
y_pred_logreg = regressor.predict(sx_test)
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
adaregressor.fit(sx_train,y_train)
y_pred_adareg = adaregressor.predict(sx_test)
EVC=explained_variance_score(y_test, y_pred_adareg )
ME=max_error(y_test, y_pred_adareg )
MAE=mean_absolute_error(y_test, y_pred_adareg )
MSE=mean_squared_error(y_test, y_pred_adareg )
MDAE=median_absolute_error(y_test, y_pred_adareg )
RS=r2_score(y_test, y_pred_adareg )
datareg=pd.DataFrame(columns=['Regressor','EVC','ME','MAE','MSE','MDAE','RS'])
L2=["AdaBoostRegressor",EVC,ME,MAE,MSE,MDAE,RS]
baregressor = BaggingRegressor()
baregressor.fit(sx_train, y_train)
y_pred_bareg = baregressor.predict(sx_test)
EVC=explained_variance_score(y_test, y_pred_bareg)
ME=max_error(y_test, y_pred_bareg)
MAE=mean_absolute_error(y_test, y_pred_bareg)
MSE=mean_squared_error(y_test, y_pred_bareg)
MDAE=median_absolute_error(y_test, y_pred_bareg)
RS=r2_score(y_test, y_pred_bareg)
L2=["BaggingRegressor",EVC,ME,MAE,MSE,MDAE,RS]
datareg.loc[len(datareg),:]=L2
etregressor = ExtraTreesRegressor()
etregressor.fit(sx_train,y_train)
y_pred_etreg = etregressor.predict(sx_test)
EVC=explained_variance_score(y_test, y_pred_etreg)
ME=max_error(y_test, y_pred_etreg)
MAE=mean_absolute_error(y_test, y_pred_etreg)
MSE=mean_squared_error(y_test, y_pred_etreg)
MDAE=median_absolute_error(y_test, y_pred_etreg)
RS=r2_score(y_test, y_pred_etreg)
L2=["ExtraTreesRegressor",EVC,ME,MAE,MSE,MDAE,RS]
datareg.loc[len(datareg),:]=L2
gbregressor = GradientBoostingRegressor()
gbregressor.fit(sx_train,y_train)
y_pred_gbreg = gbregressor.predict(sx_test)
EVC=explained_variance_score(y_test, y_pred_gbreg)
ME=max_error(y_test, y_pred_gbreg)
MAE=mean_absolute_error(y_test, y_pred_gbreg)
MSE=mean_squared_error(y_test, y_pred_gbreg)
MDAE=median_absolute_error(y_test, y_pred_gbreg)
RS=r2_score(y_test, y_pred_gbreg)
L2=["GradientBoostingRegressor",EVC,ME,MAE,MSE,MDAE,RS]
datareg.loc[len(datareg),:]=L2
rfregressor = RandomForestRegressor()
rfregressor.fit(sx_train,y_train)
y_pred_rfreg = rfregressor.predict(sx_test)
EVC=explained_variance_score(y_test, y_pred_rfreg)
ME=max_error(y_test, y_pred_rfreg)
MAE=mean_absolute_error(y_test, y_pred_rfreg)
MSE=mean_squared_error(y_test, y_pred_rfreg)
MDAE=median_absolute_error(y_test, y_pred_rfreg)
RS=r2_score(y_test, y_pred_rfreg)
L2=["RandomForestRegressor",EVC,ME,MAE,MSE,MDAE,RS]
datareg.loc[len(datareg),:]=L2
from sklearn.datasets import make_regression
regr = ElasticNet(random_state=0)
regr.fit(sx_train,y_train)
y_pred_enreg= regr.predict(sx_test)
EVC=explained_variance_score(y_test, y_pred_enreg)
ME=max_error(y_test, y_pred_enreg)
MAE=mean_absolute_error(y_test, y_pred_enreg)
MSE=mean_squared_error(y_test, y_pred_enreg)
MDAE=median_absolute_error(y_test, y_pred_enreg)
RS=r2_score(y_test, y_pred_enreg)
L2=["ElasticNet",EVC,ME,MAE,MSE,MDAE,RS]
datareg.loc[len(datareg),:]=L2
tree_regressor = DecisionTreeRegressor(random_state =0)
tree_regressor.fit(sx_train,y_train)
y_pred_dtreg=tree_regressor.predict(sx_test)
etregressor = ExtraTreesRegressor()
etregressor.fit(x_train,y_train)
y_pred_etreg = etregressor.predict(x_test)
EVC=explained_variance_score(y_test, y_pred_etreg)
ME=max_error(y_test, y_pred_etreg)
MAE=mean_absolute_error(y_test, y_pred_etreg)
MSE=mean_squared_error(y_test, y_pred_etreg)
MDAE=median_absolute_error(y_test, y_pred_etreg)
RS=r2_score(y_test, y_pred_etreg)
L2=["ExtraTreesRegressor",EVC,ME,MAE,MSE,MDAE,RS]
datareg.loc[len(datareg),:]=L2
gbregressor = GradientBoostingRegressor()
gbregressor.fit(x_train,y_train)
y_pred_gbreg = gbregressor.predict(x_test)
EVC=explained_variance_score(y_test, y_pred_gbreg)
ME=max_error(y_test, y_pred_gbreg)
MAE=mean_absolute_error(y_test, y_pred_gbreg)
MSE=mean_squared_error(y_test, y_pred_gbreg)
MDAE=median_absolute_error(y_test, y_pred_gbreg)
RS=r2_score(y_test, y_pred_gbreg)
L2=["GradientBoostingRegressor",EVC,ME,MAE,MSE,MDAE,RS]
datareg.loc[len(datareg),:]=L2
rfregressor = RandomForestRegressor()
rfregressor.fit(x_train,y_train)
y_pred_rfreg = rfregressor.predict(x_test)
EVC=explained_variance_score(y_test, y_pred_rfreg)
ME=max_error(y_test, y_pred_rfreg)
MAE=mean_absolute_error(y_test, y_pred_rfreg)
MSE=mean_squared_error(y_test, y_pred_rfreg)
MDAE=median_absolute_error(y_test, y_pred_rfreg)
RS=r2_score(y_test, y_pred_rfreg)
L2=["RandomForestRegressor",EVC,ME,MAE,MSE,MDAE,RS]
datareg.loc[len(datareg),:]=L2
from sklearn.datasets import make_regression
regr = ElasticNet(random_state=0)
regr.fit(x_train,y_train)
y_pred_enreg= regr.predict(x_test)
EVC=explained_variance_score(y_test, y_pred_enreg)
ME=max_error(y_test, y_pred_enreg)
MAE=mean_absolute_error(y_test, y_pred_enreg)
MSE=mean_squared_error(y_test, y_pred_enreg)
MDAE=median_absolute_error(y_test, y_pred_enreg)
RS=r2_score(y_test, y_pred_enreg)
L2=["ElasticNet",EVC,ME,MAE,MSE,MDAE,RS]
datareg.loc[len(datareg),:]=L2
tree_regressor = DecisionTreeRegressor(random_state =0)
tree_regressor.fit(sx_train,y_train)
y_pred_dtreg=tree_regressor.predict(sx_test)
EVC=explained_variance_score(y_test, y_pred_dtreg)
ME=max_error(y_test, y_pred_dtreg)
MAE=mean_absolute_error(y_test, y_pred_dtreg)
MSE=mean_squared_error(y_test, y_pred_dtreg)
MDAE=median_absolute_error(y_test, y_pred_dtreg)
RS=r2_score(y_test, y_pred_dtreg)
L2=["DecisionTreeRegressor",EVC,ME,MAE,MSE,MDAE,RS]
datareg.loc[len(datareg),:]=L2
svr_regressor = SVR(kernel='rbf',gamma='auto')
svr_regressor.fit(sx_train,y_train)
y_pred_svreg=svr_regressor.predict(sx_test)
EVC=explained_variance_score(y_test, y_pred_svreg)
ME=max_error(y_test, y_pred_svreg)
MAE=mean_absolute_error(y_test, y_pred_svreg)
MSE=mean_squared_error(y_test, y_pred_svreg)
MDAE=median_absolute_error(y_test, y_pred_svreg)
RS=r2_score(y_test, y_pred_svreg)
L2=["SVR",EVC,ME,MAE,MSE,MDAE,RS]
datareg.loc[len(datareg),:]=L2