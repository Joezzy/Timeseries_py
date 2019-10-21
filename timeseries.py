import numpy as np 
import pandas as pd 
import matplotlib.pylab as plt
from datetime import datetime
#  import ARIMA 
# from statesmodels.tsa.arima_model import arima_model
from matplotlib.pylab import rcParams
rcParams['figure.figsize']=10,6
dataset=pd.read_csv("airpassengers.csv")

dataset['Month']=pd.to_datetime(dataset['Month'], infer_datetime_format=True)
indexedDataset=dataset.set_index(['Month'])

rolmean=indexedDataset.rolling(window=12).mean()#mean
rolstd=indexedDataset.rolling(window=12).std()#standard deviation
#print(rolmean,rolstd)


#plot the rolling statistics
# orig=plt.plot(indexedDataset, color='blue', label='Original' )
# mean=plt.plot(rolmean, color='red', label='Rolling Mean' )
# std=plt.plot(rolstd, color='black', label='Rolling STD' )
# plt.legend(loc='best')
# plt.title('Rolling Mean & Standard deviation')
# plt.show(block=False)

#Perform Dickey-Fuller test
from statsmodels.tsa.stattools import adfuller
print('Dickey-Fuller Test Result')
dftest=adfuller(indexedDataset['#Passengers'],autolag='AIC')
dfoutput=pd.Series(dftest[0:4],index=['Test Statistic', 'p-value', '#Lags Used','Number of Observations Used'])
for key,value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key]=value
print(dfoutput)

#Estimating trend
indexedDataset_logScale = np.log(indexedDataset)
# plt.plot(indexedDataset_logScale)
# plt.show()

#MOVING AVERAGE
movingAverage=indexedDataset_logScale.rolling(window=12).mean()
movingSTD=indexedDataset_logScale.rolling(window=12).std()
# plt.plot(indexedDataset_logScale)
# plt.plot(movingAverage,color='red')
# plt.show()
 
#difference between the moving average and the original no. of passengers
datasetLogScaleMinusMovingAverage=indexedDataset_logScale-movingAverage
datasetLogScaleMinusMovingAverage.head(2)
#Remove Nan Values
datasetLogScaleMinusMovingAverage.dropna(inplace=True)
print(datasetLogScaleMinusMovingAverage.head(10))

def test_stationary(timeseries):
    #determine rolling statistics
    movingAverage=timeseries.rolling(window=12).mean()
    movingSTD=timeseries.rolling(window=12).std()

#plot the rolling statistics
    orig=plt.plot(timeseries, color='blue', label='Original' )
    mean=plt.plot(rolmean, color='red', label='Rolling Mean' )
    std=plt.plot(rolstd, color='black', label='Rolling STD' )
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard deviation')
    plt.show(block=False)
    #Perform Dickey-Fuller test:
    print('Dickey-Fuller Test Result')
    dftest=adfuller(timeseries['#Passengers'],autolag='AIC')
    dfoutput=pd.Series(dftest[0:4],index=['Test Statistic', 'p-value', '#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key]=value
    print(dfoutput)

#test_stationary(datasetLogScaleMinusMovingAverage)
#test_stationary(datasetLogScaleMinusMovingAverage)
datasetLogDiffShifting=indexedDataset_logScale-indexedDataset_logScale.shift()
#plt.plot(datasetLogDiffShifting)
#plt.show() #show well

#datasetLogDiffShifting.dropna(inplace=True)
#test_stationary(datasetLogDiffShifting)
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition=seasonal_decompose(indexedDataset_logScale)

trend=decomposition.trend
seasonal=decomposition.seasonal
residual=decomposition.resid

# plt.subplot(411)
# plt.plot(indexedDataset_logScale,label='Original')
# plt.legend(loc='best')
# plt.subplot(412)
# plt.plot(trend,label='Trend')
# plt.legend(loc='best')
# plt.subplot(413)
# plt.plot(seasonal,label='Seasonality')
# plt.legend(loc='best')
# plt.subplot(414)
# plt.plot(residual,label='Residuals')
# plt.legend(loc='best')
# plt.tight_layout()
#plt.show()

decomposedLogData=residual
decomposedLogData.dropna(inplace=True)
#test_stationary(decomposedLogData)
#autocorrelation /Partial ACF
from statsmodels.tsa.stattools import acf,pacf
datasetLogDiffShifting.dropna(inplace=True)
lag_acf=acf(datasetLogDiffShifting, nlags=20)
lag_pacf=pacf(datasetLogDiffShifting, nlags=20, method='ols')
# PLOT THIS TO GET UR (P,B/D,Q) values i.e (2,1,2) below in AR and MA model
#plot ACF
# plt.subplot(121)
# plt.plot(lag_acf)
# plt.axhline(y=0,linestyle='--',color='gray')
# plt.axhline(y=-1.96/np.sqrt(len(datasetLogDiffShifting)),linestyle='--',color='gray')
# plt.axhline(y=1.96/np.sqrt(len(datasetLogDiffShifting)),linestyle='--',color='gray')
# plt.title('Autocorrelation Function')

# #plot PACF
# plt.subplot(122)
# plt.plot(lag_pacf)
# plt.axhline(y=0,linestyle='--',color='gray')
# plt.axhline(y=-1.96/np.sqrt(len(datasetLogDiffShifting)),linestyle='--',color='gray')
# plt.axhline(y=1.96/np.sqrt(len(datasetLogDiffShifting)),linestyle='--',color='gray')
# plt.title('Partial Autocorrelation Function')
# plt.tight_layout()
#plt.show()

from statsmodels.tsa.arima_model import ARIMA
#AR MOdel
model=ARIMA(indexedDataset_logScale, order=(2,1,2))
results_AR=model.fit(disp=-1)
# plt.plot(datasetLogDiffShifting)
# plt.plot(results_AR.fittedvalues, color='red')
# plt.title('RSS: %.4f'% sum((results_AR.fittedvalues-datasetLogDiffShifting["#Passengers"])**2))
# print('Plottting AR model')
# #plt.show()

#MA MODEL
model=ARIMA(indexedDataset_logScale, order=(0,1,2))
results_MA=model.fit(disp=-1)
# plt.plot(datasetLogDiffShifting)
# plt.plot(results_MA.fittedvalues, color='red')
# plt.title('RSS: %.4f'% sum((results_MA.fittedvalues-datasetLogDiffShifting["#Passengers"])**2))
# print('Plottting MA model')
# #plt.show()

model=ARIMA(indexedDataset_logScale, order=(2,1,2))
results_ARIMA=model.fit(disp=-1)
# plt.plot(datasetLogDiffShifting)
# plt.plot(results_ARIMA.fittedvalues, color='red')
# plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-datasetLogDiffShifting["#Passengers"])**2))
# print('Plottting ARIMA model')
# #plt.show()

predictions_ARIMA_diff=pd.Series(results_ARIMA.fittedvalues, copy=True)
print('Print ARIMA DIFFERENCE')
print(predictions_ARIMA_diff.head())
#covert TO CUMMULAT
# IVE SUM
predictions_ARIMA_diffsum=predictions_ARIMA_diff.cumsum()
print('Cummulative Sum')
print(predictions_ARIMA_diffsum.head())

predictions_ARIMA_log=pd.Series(indexedDataset_logScale['#Passengers'].ix[0], indexedDataset_logScale.index)
predictions_ARIMA_log=predictions_ARIMA_log.add(predictions_ARIMA_diffsum,fill_value=0)
predictions_ARIMA_log.head()

#View Fittingnew s with Original data
predictions_ARIMA=np.exp(predictions_ARIMA_log)
# plt.plot(indexedDataset)
# plt.plot(predictions_ARIMA)
# plt.show()
#print(indexedDataset_logScale)# check how many row

results_ARIMA.plot_predict(1, 264)
#x=results_ARIMA.forecast(steps=120)
#Predict using the above in Jupyter
plt.show()
