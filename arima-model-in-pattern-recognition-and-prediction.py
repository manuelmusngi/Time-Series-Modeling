#!/usr/bin/env python
# coding: utf-8

#  ![image.png](attachment:image.png)

# 
# ### Objective
# The objective of this project is to evaluate the production predictability in the following economic sector:
# Industrial Production: Manufacturing: Non-Durable Goods: Sugar and Confectionery Product (NAICS = 3113) (IPG3113N)	
# 
# 
# ### Data 
# Data on the monthly candy production was retrieved from the Federal Bank of St.Louis ECONOMIC RESEARCH:
# 
# https://fred.stlouisfed.org/series/IPG3113N
# 
# The data is also available on Kaggle:
# 
# https://www.kaggle.com/rtatman/us-candy-production-by-month
# 
# 
# ### Modeling and Evaluation
# The evaluation models that will be used are:
# * Autoregressive Integrated Moving Average (ARIMA) model
# * Seasonal Autoregressive Integrated Moving Average (SARIMAXmodel) model

# ### Import Libraries
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# ### Import Data
original_production_df = pd.read_csv('../input/us-candy-production-by-month/candy_production.csv')

# make a copy to retain the data integrity of the original
df=original_production_df.copy()

# a brief glance at the data
df

# rename column 'IPG3113N' as candy_production
df.rename(columns={'IPG3113N':'candy_production'}, inplace=True)
df.head()

# ### Data Preprocessing
df.info()
df['observation_date'] = pd.to_datetime(df['observation_date'])
df.set_index('observation_date',inplace=True)

# check for missing data
df.isnull().sum()

# ### Data Visualization

df.plot(figsize=(20,10));
timeseries = df['candy_production']
timeseries.rolling(12).mean().plot(label='12 Month Rolling Mean')
timeseries.rolling(12).std().plot(label='12 Month Rolling Std')
timeseries.plot(figsize=(20,10))
plt.legend();

# ### Decomposition
# 
# ETS (Error, Trend, Seasonality) Decomposition model gives another view of the data.
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(df['candy_production'], period=12)  
fig = plt.figure()   
fig = fig.set_size_inches(10, 5)
fig = decomposition.plot();


# ### Test stationarity of time series data
# 
# In statistics and econometrics, the augmented Dickey–Fuller test (ADF) tests the null hypothesis that a unit root 
# is present in a time series sample. The alternative hypothesis is different depending on which version of the test 
# is used, but is usually stationarity or trend-stationarity.
# 
# To accept the Null Hypothesis **H0** (that the time series has a unit root, indicating it is non-stationary) or 
# reject **H0** and go with the Alternative Hypothesis (that the time series has no unit root and is stationary).
# 
# The p-value will be the factor to decide the with Null Hypothesis or the Alternative Hypothesis in terms of stationarity:
# 
# * A small p-value (typically ≤ 0.05) indicates strong evidence against the null hypothesis, so you reject the null hypothesis.
# 
# * A large p-value (> 0.05) indicates weak evidence against the null hypothesis, so you fail to reject the null hypothesis.
# 

from statsmodels.tsa.stattools import adfuller
result = adfuller(df['candy_production'])
print('Augmented Dickey-Fuller Test:')
labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']
for value,label in zip(result,labels):
    print(label+' : '+str(value) )   
if result[1] <= 0.05:
    print("strong evidence against the null hypothesis, reject the null hypothesis. Data has no unit root and is stationary")
else:
    print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")

# Store in a function for later use!
def adf_check(time_series):
    """
    Pass in a time series, returns ADF report
    """
    result = adfuller(time_series)
    print('Augmented Dickey-Fuller Test:')
    labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']
    for value,label in zip(result,labels):
        print(label+' : '+str(value) ) 
    if result[1] <= 0.05:
        print("strong evidence against the null hypothesis, reject the null hypothesis. Data has no unit root and is stationary")
    else:
        print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")


# ## Differencing
# 
# Differencing is a method of transforming a non-stationary time series into a stationary one. This is an important step 
# in preparing data to be used in an ARIMA model. The first differencing value is the difference between the current time 
# period and the previous time period. You continue to take the second difference, third difference, and so on until your 
# data is stationary.
#
# ### First Difference 
# 
# The first difference of a time series is the series of changes from one period to the next. If Yt denotes the value of 
# the time series Y at period t, then the first difference of Y at period t is equal to Yt-Yt-1.

df['First Difference'] = df['candy_production'] - df['candy_production'].shift(1)
adf_check(df['First Difference'].dropna())
df['First Difference'].plot(figsize=(20,10));


# ### Second Difference  
# Sometimes it would be necessary to do a second difference 
# This is just for show, we didn't need to do a second difference in our case
df['Second Difference'] = df['First Difference'] - df['First Difference'].shift(1)
adf_check(df['Second Difference'].dropna())
df['Second Difference'].plot(figsize=(20,10));

# ### Seasonal Difference 
df['Seasonal Difference'] = df['candy_production'] - df['candy_production'].shift(12)
df['Seasonal Difference'].plot(figsize=(20,10));

# Seasonal Difference by itself was not enough!
adf_check(df['Seasonal Difference'].dropna())

# ### Seasonal First Difference 
# You can also do seasonal first difference
df['Seasonal First Difference'] = df['First Difference'] - df['First Difference'].shift(12)
df['Seasonal First Difference'].plot(figsize=(20,10));
adf_check(df['Seasonal First Difference'].dropna())

# ## Autocorrelation and Partial Autocorrelation Plots
# 
# ### Autocorrelation 
# 
# Autocorrelation represents the degree of similarity between a given time series and a lagged version of itself 
# over successive time intervals. It measures the relationship between a variable's current value and its past value.
# 
# The interpretation of Autocorrelation is how it relates to the ARIMA model and understand which of the, AR or MA, 
# components does the ARIMA model use or both, as well as, how many lags to use. 
# 
# In general one would use either AR or MA of the model, using both is less common.
# 
# * If the autocorrelation plot shows positive autocorrelation at the first lag (lag-1), this suggests using the AR 
#   terms in relation to the lag
# 
# * If the autocorrelation plot shows negative autocorrelation at the first lag, this suggests using MA terms.
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf

# Duplicate plots
fig_first = plot_acf(df["First Difference"].dropna())
fig_seasonal_first = plot_acf(df["Seasonal First Difference"].dropna())
from pandas.plotting import autocorrelation_plot
autocorrelation_plot(df['Seasonal First Difference'].dropna());


# ### Partial Autocorrelation
# 
# In general, a partial correlation is a conditional correlation. It is the correlation between two variables 
# under the assumption that we know and take into account the values of some other set of variables. For instance, 
# consider a regression context in which y is the response variable and x1,x2, and x3 are predictor variables. 
# The partial correlation between y and x3 is the correlation between the variables determined taking into account 
# how both y and x3 are related to x1 and x2.
# 
# More formally, we can define the partial correlation just described as:
# 
# ## $\frac{\text{Covariance}(y, x_3|x_1, x_2)}{\sqrt{\text{Variance}(y|x_1, x_2)\text{Variance}(x_3| x_1, x_2)}}$
# 
# this relationship is plotted as:
result = plot_pacf(df["Seasonal First Difference"].dropna())

# ### Autocorrelation and Partial Autocorrelation: Identifying Model to Employ
# 
# * Identification of an AR model is often observed with the PACF.
#     * For an AR model, the theoretical PACF “shuts off” past the order of the model.  The phrase “shuts off” means 
#       that in theory the partial autocorrelations are equal to 0 beyond that point.  Put another way, the number 
#       of non-zero partial autocorrelations gives the order of the AR model.  By the “order of the model” we mean 
#       the most extreme lag of x that is used as a predictor.
#     
#     
# * Identification of an MA model is often observed with the ACF rather than the PACF.
#     * For an MA model, the theoretical PACF does not shut off, but instead tapers toward 0 in some manner.  
#       A clearer pattern for an MA model is in the ACF.  The ACF will have non-zero autocorrelations only at lags 
#       involved in the model.
#
# ### Autocorrelation Interpretation
#  
# * A sharp drop after lag "k" suggests an AR-k model should be used. 
# * A gradual decline, suggests an MA model should be employed.
#
# ### Final ACF and PACF Visualization Plots
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(df['Seasonal First Difference'].iloc[13:], lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df['Seasonal First Difference'].iloc[13:], lags=40, ax=ax2)


# ## ARIMA modeling 
# For non-seasonal data
from statsmodels.tsa.arima_model import ARIMA

# ### p,d,q parameters
# 
# * p: The number of lag observations included in the model.
# * d: The number of times that the raw observations are differenced, also called the degree of differencing.
# * q: The size of the moving average window, also called the order of moving average.
df.describe()
df.index = pd.DatetimeIndex(df.index, freq='MS')

# modelling with seasonal data
model = sm.tsa.statespace.SARIMAX(df['candy_production'],
                                  order=(0,1,0),
                                  freq = 'MS',
                                  seasonal_order=(1,1,1,12))
results = model.fit()
print(results.summary())
results.resid.plot(figsize=(20,10));
results.resid.plot(kind='kde',figsize=(20,10));


# ### Projection of Future Values 
# 
# We can get an idea of how well our model performs by just predicting for values that we already have:
df
df['prediction'] = results.predict(start = '2015-01-01', end='2017-08-01', dynamic= True)  
df[['candy_production','prediction']].plot(figsize=(15,8));


# References: PierienData, Wikipedia
