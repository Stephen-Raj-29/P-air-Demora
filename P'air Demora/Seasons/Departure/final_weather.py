import warnings
import itertools
import pandas as pd
import numpy as np
import csv
import datetime
import statsmodels.api as sm
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

print("Enter Source :")
source = input()
print("Enter Destination :")
destination = input()
print("Enter Date :")
dat=input()
datee = datetime.datetime.strptime(dat, "%d-%m-%Y")
month=datee.month
date=datee.day

if(month== 6 or  month== 7 or month== 8):
  data=pd.read_csv("JUN.csv")
elif(month== 9 or month== 10 or  month== 11):
  data=pd.read_csv("SEP.csv")
elif(month== 3 or month== 4 or month== 5):
  data=pd.read_csv("MAR.csv")
else:
  data=pd.read_csv("JAN.csv")
data.query('ORIGIN== "{}" and DEST== "{}"'.format(source,destination),inplace=True)

data = data.loc[:, ~data.columns.str.contains('ORIGIN')]
data = data.loc[:, ~data.columns.str.contains('^DEST')]
data = data.replace(np.nan, 0)

a=data["WEATHER_DELAY"].mean()
data = data.replace(0, a)

length = len(data.index)
data = data.set_index(['FL_DATE'])
# data.head(5)
# data.plot(figsize=(19, 4))
# plt.show()

from pylab import rcParams
rcParams['figure.figsize'] = 18, 8
decomposition = sm.tsa.seasonal_decompose(data, model='additive',freq=3)
fig = decomposition.plot()
# plt.show()

p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

# for param in pdq:
#     for param_seasonal in seasonal_pdq:
#         try:
#             mod = sm.tsa.statespace.SARIMAX(data,order=param,seasonal_order=param_seasonal,enforce_stationarity=False,enforce_invertibility=False)
#             results = mod.fit()
#             print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
#         except:
#             continue

mod = sm.tsa.statespace.SARIMAX(data,
                                order=(0, 0, 1),
                                seasonal_order=(0, 1, 1,12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results=mod.fit()

results.plot_diagnostics(figsize=(18, 8))
# plt.show()

start=data.index[0]

pred = results.get_prediction(start, dynamic=False)
pred_ci = pred.conf_int()
ax = data[start:].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 4))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Time')
ax.set_ylabel('WEATHER_DELAY')
plt.legend()
plt.ylim(-5,40)
# plt.show()

pred_uc = results.get_forecast(steps=92)
pred_ci = pred_uc.conf_int()
ax = data.plot(label='observed', figsize=(14, 4))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('FL_DATE')
ax.set_ylabel('WEATHER_DELAY')
plt.legend()
# plt.show()

data_forecasted = pred.predicted_mean
data_truth = data[start:]

data_forecasted = pred.predicted_mean
# data_forecasted.head(12)

# data_truth.head(5)

# pred_ci.head(5)

forecast = pred_uc.predicted_mean

if(month=="January"):
  predict=date
elif(month=="February"):
  predict=31+date
elif(month=="Macrh"):
  predict=date
elif(month=="April"):
  predict=31+date
elif(month=="May"):
  predict=61+date
elif(month=="June"):
  predict=date
elif(month=="July"):
  predict=30+date
elif(month=="August"):
  predict=61+date
elif(month=="September"):
  predict=date
elif(month=="October"):
  predict=30+date
elif(month=="November"):
  predict=61+date
else:
  predict=59+date

predict=predict+length
result = forecast.get(key = predict)

if(result<0):
  frac=str(result)
  for digit in frac:
    if digit !='-'and digit!='0' and digit!=".":
      lokes=float("0.{}".format(digit))
      print(lokes)
      break
else:
  print(round(result,1))

