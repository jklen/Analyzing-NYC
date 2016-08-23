# -*- coding: utf-8 -*-
"""
Created on Wed Jul 01 23:14:13 2015

@author: Jaroslav Klen
"""


import pandas
import scipy.stats
import numpy
import matplotlib.pyplot as plt
import statsmodels.api as sm

weather_df = pandas.read_csv('C:\Users\IBM_ADMIN\Desktop\PY\\turnstile_data_master_with_weather.csv')
weather_df['DATEn'] = pandas.to_datetime(weather_df['DATEn'])
weather_df['weekday'] = weather_df['DATEn'].dt.weekday
weather_df['day'] = weather_df['DATEn'].dt.day

def normalize_features(features):
    
    means = numpy.mean(features, axis=0)
    std_devs = numpy.std(features, axis=0)
    normalized_features = (features - means) / std_devs
    
    return means, std_devs, normalized_features

def plots(weather_df):
    
    plot1 = weather_df['ENTRIESn_hourly'][(weather_df['rain']==1) & (weather_df['ENTRIESn_hourly']<=10000)].plot(kind = 'hist', bins = 25, alpha = 0.8, label = 'Rain', color = 'green')
    plot2 = weather_df['ENTRIESn_hourly'][(weather_df['rain']==0) & (weather_df['ENTRIESn_hourly']<=10000)].plot(kind = 'hist', bins = 25, alpha = 0.3, label = 'No rain', color = 'blue')
    plt.legend()
    plt.title('Rain vs. No rain histogram')
    plt.xlabel('Hourly entries')
    plt.ylabel('Occurencies')
    plt.show()
    
    mean_by_weekday = pandas.DataFrame(weather_df.groupby(['weekday', 'rain'], as_index = False)['ENTRIESn_hourly'].mean())
    mean_by_weekday['rain'] = mean_by_weekday['rain'].replace(to_replace = 0, value = 'No')
    mean_by_weekday['rain'] = mean_by_weekday['rain'].replace(to_replace = 1, value = 'Yes')
    mean_by_weekday = pandas.pivot_table(mean_by_weekday, values = ['ENTRIESn_hourly'], index = ['weekday'], columns = ['rain'])
    
    plot3 = mean_by_weekday['ENTRIESn_hourly'].plot(kind='bar', stacked = False, alpha = 0.5)
    plt.title('Average hourly entries by weekday')
    plt.xlabel('Weekday')
    plt.ylabel('Mean entries')
    labels = ['Mon', 'Tue', 'Wen', 'Thu', 'Fri', 'Sat', 'Sun']
    plt.xticks( [0,1,2,3,4,5,6], labels)
    plt.show()
    
    mean_by_day = pandas.DataFrame(weather_df.groupby(['day'], as_index = False)['ENTRIESn_hourly'].mean())
    for_plot = pandas.DataFrame(weather_df.groupby(['day', 'rain'], as_index = False)['ENTRIESn_hourly'].mean())
    for_plot['ENTRIESn_hourly'][for_plot['rain']==0] = 0
    for_plot.rename(columns={'ENTRIESn_hourly': 'Rainy days'}, inplace = True)
    
    plot4 = mean_by_day['ENTRIESn_hourly'].plot(kind = 'line', x = 'day', y = 'ENTRIESn_hourly', marker = 'o', markersize = 5, legend = False)
    plot5 = for_plot['Rainy days'].plot(kind='bar', x='day', y = 'Rainy days', color = 'green', width = 0.17, alpha = 0.3, legend = True)
    plt.title('Average hourly entries by day')
    plt.xlabel('Day')
    plt.ylabel('Mean entries')
    plt.show()
    
    mean_by_hour = pandas.DataFrame(weather_df.groupby(['Hour'], as_index = False)['ENTRIESn_hourly'].mean())
    plot6 = mean_by_hour['ENTRIESn_hourly'].plot(kind = 'line', legend = True, marker = 'o', markersize = 5)
    plt.title('Average hourly entries by hour')
    plt.ylabel('Mean entries')
    plt.xlabel('Hour')
    plt.show()
    
    print plot1, plot2, plot3, plot4, plot5, plot6
   
def OLS(weather_df):
    
    features = weather_df[['rain', 'meantempi', 'meanpressurei', 'meanwindspdi']]
    m, std, features = normalize_features(features)
    
    values = weather_df['ENTRIESn_hourly']
    
    dummy_units = pandas.get_dummies(weather_df['UNIT'], prefix='unit')
    features = features.join(dummy_units)
    
    dummy_hours = pandas.get_dummies(weather_df['Hour'], prefix='hour')
    features = features.join(dummy_hours)
    
    dummy_weekdays = pandas.get_dummies(weather_df['weekday'], prefix = 'day')
    features = features.join(dummy_weekdays)
    
    features.drop(['day_6'], axis = 1, inplace = True)
    features.drop(['unit_R404'], axis = 1, inplace = True)
    features.drop(['hour_6'], axis = 1, inplace = True)
    features.drop(['hour_7'], axis = 1, inplace = True)
    
    model = sm.OLS(values, features)
    results = model.fit()
    
    params_norm_all = results.params
    predictions = numpy.dot(features, results.params)
    residuals = (predictions - values).values
    params_nondummy = params_norm_all[:4]/std
    rsquared = results.rsquared
    cnumber = results.condition_number
    
    return predictions, residuals, params_nondummy, rsquared, cnumber

def stat_test(weather_df):
    
    rain_df = pandas.DataFrame(weather_df['ENTRIESn_hourly'][weather_df['rain']==1])
    rain_mean = rain_df['ENTRIESn_hourly'].mean()
    norain_df = pandas.DataFrame(weather_df['ENTRIESn_hourly'][weather_df['rain']==0])
    norain_mean = norain_df['ENTRIESn_hourly'].mean()
    
    up = scipy.stats.mannwhitneyu(rain_df, norain_df)
    U = up[0]
    two_tailed_p = up[1] * 2
    
    return rain_mean, norain_mean, U, two_tailed_p

def residuals_plots(predictions, values):
    
    residuals = predictions - values
    res_norm_mean, res_norm_std, residuals_norm = normalize_features(residuals)
    
    forplot_df = pandas.DataFrame(predictions, columns = ['predictions'])
    forplot_df['values'] = pandas.Series(values)
    forplot_df['residuals'] = pandas.Series(residuals)
    forplot_df['normalized_residuals'] = pandas.Series(abs(residuals_norm))
    
    residuals_hist = residuals.hist(bins=50)
    plt.title('Residual histogram')
    plt.xlabel('Residuals value')
    plt.ylabel('Occurencies')
    plt.show()
    
    pred_vs_val = forplot_df.plot(kind = 'scatter', x = 'values', y = 'predictions', s = 20, facecolors = 'none')
    plt.title('Predictions vs. observed values')
    plt.xlabel('Observed values')
    plt.ylabel('Predicted values')
    plt.show()
    
    res_vs_pred = forplot_df.plot(kind = 'scatter', x = 'predictions', y = 'residuals', s = 20, facecolors = 'none')
    plt.title('Residuals vs. predictions')
    plt.xlabel('Predicted value')
    plt.ylabel('Residuals')
    plt.show()
    
    res_vs_val = forplot_df.plot(kind = 'scatter', x = 'values', y = 'residuals', s = 20, facecolors = 'none')
    plt.title('Residuals vs. observed values')
    plt.xlabel('Observed values')
    plt.ylabel('Residuals')
    plt.show()
    
    pplot = scipy.stats.probplot(residuals, plot = plt)
    plt.title('Residuals - normal probability plot')
    plt.show()
    
    print residuals_hist, pred_vs_val, res_vs_pred, res_vs_val, pplot, residuals.describe()
    

#plots(weather_df)
residuals_plots(OLS(weather_df)[0], weather_df['ENTRIESn_hourly'])
#print OLS(weather_df)
#print stat_test(weather_df)
