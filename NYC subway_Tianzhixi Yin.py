import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
import numpy as np
import statsmodels.api as sm
from ggplot import *

subway = pd.read_csv("turnstile_weather_v2.csv")

plt.ylim(0, 2500)
plt.xlim(0, 15000)
fig = subway[subway['rain']==1]['ENTRIESn_hourly'].hist(bins = 200)
fig.set_title('With Rain')
fig.set_xlabel('ENTRIESn_hourly')
fig.set_ylabel('Frequency')

plt.xlim(0, 15000)
fig = subway[subway['rain']==0]['ENTRIESn_hourly'].hist(bins = 200)
fig.set_title('Without Rain')
fig.set_xlabel('ENTRIESn_hourly')
fig.set_ylabel('Frequency')

x = subway[subway['rain']==1]['ENTRIESn_hourly']
y = subway[subway['rain']==0]['ENTRIESn_hourly']
U, p = scipy.stats.mannwhitneyu(x, y)
print U, p

m_u = len(x)*len(y)/2
sigma_u = np.sqrt(len(x)*len(y)*(len(x)+len(y)+1)/12)
z = (U - m_u)/sigma_u
print z
pval = 2*scipy.stats.norm.cdf(z)
print pval

with_rain_mean=np.mean(x)
without_rain_mean=np.mean(y)
print with_rain_mean
print without_rain_mean

with_rain_median=np.median(x)
without_rain_median=np.median(y)
print with_rain_median
print without_rain_median

print np.percentile(x, 25)
print np.percentile(x, 50)
print np.percentile(x, 75)
print np.percentile(y, 25)
print np.percentile(y, 50)
print np.percentile(y, 75)


features = subway[['rain', 'hour', 'weekday', 'fog', 'tempi', 'wspdi']]
dummy_units = pd.get_dummies(subway['UNIT'], prefix='unit')
features = features.join(dummy_units)
values = subway['ENTRIESn_hourly']

features_cons = sm.add_constant(features)
model = sm.OLS(values, features_cons)
results = model.fit()
print results.summary()
'''print results.rsquared'''

intercept = results.params[0]
params = results.params[1:]
predictions = intercept + np.dot(features, params)

(values - predictions).hist(bins=500)

results.params
results.tvalues

Stations = ['R003', 'R004', 'R005', 'R006', 'R007', 'R008', 'R009', 'R010', 'R011', 'R012', 'R013', 'R014', 'R015', 'R016', 'R017', 'R018', 'R019', 'R020', 'R021', 'R022']
subset = subway[subway['UNIT'].isin(Stations)][['hour', 'UNIT', 'ENTRIESn_hourly']]
subset_hourly = subset.groupby(['hour', 'UNIT']).sum()
pd.DataFrame(subset_hourly).reset_index(inplace=True)
ggplot(subset_hourly, aes('hour', 'ENTRIESn_hourly', color='UNIT'))+geom_point()+geom_line()+xlim(0,20)+ylim(0,)+ggtitle('Ridership by time-of-day of UNITS R003 to R022')+xlab('Time-of-day')+ylab('ENTRIESn_hourly')
