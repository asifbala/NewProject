import numpy as np
import pandas as pd 
import os
import matplotlib.pyplot as plt
import json
import datetime 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from xgboost import XGBClassifier

#Convert Json File into Pandas DF

time_format = '%Y-%m-%d'

openfile=open('/Users/asifbala/Documents/ultimate_challenge/ultimate_data_challenge.json')

jsondata=json.load(openfile)

df=pd.DataFrame(jsondata)

openfile.close()

#Create Active Target Variable

df['signup_date'] = pd.to_datetime(df['signup_date'], format=time_format)

df['six_months'] = df['signup_date'] + datetime.timedelta(days=181)

df['last_trip_date'] = pd.to_datetime(df['last_trip_date'], format=time_format)

df['diff'] = list(df['six_months'] - df['last_trip_date'])

df['diff'] = df['diff'].dt.days

df['active'] = np.where(df['diff'] <= 30,1,0)

#Fix Class Imbalance

#Find Number of samples which are active
no_active = len(df[df['active'] == 1])

#Get indices of non active samples
non_active_indices = df[df.active == 0].index

#Random sample non active indices
random_indices = np.random.choice(non_active_indices,no_active, replace=False)

#Find the indices of active samples
active_indices = df[df.active == 1].index

#Concat active indices with sample non-active ones
under_sample_indices = np.concatenate([active_indices,random_indices])

#Get Balanced Dataframe
under_sample = df.loc[under_sample_indices]

print(under_sample.shape)

#Visual Exploratory Data Analysis

class_counts = under_sample['active'].value_counts()

print(class_counts)

class_counts.plot(kind='bar')

plt.show()

avg_dist_active = pd.crosstab(df['avg_dist'], df['active'])

print(avg_dist_active)

avg_dist_active.plot()

plt.show()

avg_rating_by_driver_active = pd.crosstab(df['avg_rating_by_driver'],under_sample['active'])

print(avg_rating_by_driver_active)

avg_rating_by_driver_active.plot()

plt.show()

avg_rating_of_driver_active = pd.crosstab(df['avg_rating_of_driver'],under_sample['active'])

print(avg_rating_of_driver_active)

avg_rating_of_driver_active.plot()

plt.show()

avg_surge_active = pd.crosstab(df['avg_surge'],under_sample['active'])

print(avg_surge_active)

avg_surge_active.plot()

plt.show()

city_active = pd.crosstab(df['city'],under_sample['active'],normalize='index')

print(city_active)

city_active.plot(kind='bar')

plt.show()

surge_pct_active = pd.crosstab(df['surge_pct'],under_sample['active'])

print(surge_pct_active)

surge_pct_active.plot()

plt.show()

trips_in_first_30_days_active = pd.crosstab(df['trips_in_first_30_days'],under_sample['active'])

print(trips_in_first_30_days_active)

trips_in_first_30_days_active.plot()

plt.show()

ultimate_black_user_active = pd.crosstab(df['ultimate_black_user'],under_sample['active'])

print(ultimate_black_user_active)

ultimate_black_user_active.plot(kind='bar')

plt.show()

weekday_pct_active = pd.crosstab(df['weekday_pct'],under_sample['active'])

print(weekday_pct_active)

weekday_pct_active.plot()

plt.show()

#Features found from EDA

#Number of drivers with 100 weekday pct
#number of drivers with 0 weekday pct
#number of drivers with average distance of 0 in first 30 days
#true or false ultimate_black_user
#number of 0 surge percentage
#city
#number of avg_surge being 1
#number of avg_rating_of_driver being 5
#number of avg_rating_by_driver being 5

#Create Features DF to input into model

under_sample['weekday_pct_100'] = np.where(under_sample['weekday_pct'] == 100,1,0)

under_sample['weekday_pct_0'] = np.where(under_sample['weekday_pct'] == 0,1,0)

under_sample['avg_dist_0'] = np.where(under_sample['avg_dist'] == 0,1,0)

under_sample['ultimate_black_user'] = np.where(under_sample['ultimate_black_user'] == True,1,0)

under_sample['surge_pct_0'] = np.where(under_sample['surge_pct'] == 0,1,0)

def city_function(df):
    if df['city'] == "King's Landing":          
        return 0
    if df['city'] == 'Astapor':
        return 1
    else:
        return 2

under_sample['city_numeric'] = under_sample.apply(city_function,axis=1)

under_sample['avg_surge_1'] = np.where(under_sample['avg_surge'] == 1,1,0)

under_sample['avg_rating_of_driver_5'] = np.where(under_sample['avg_rating_of_driver'] == 5,1,0)

under_sample['avg_rating_by_driver_5'] = np.where(under_sample['avg_rating_by_driver'] == 5,1,0)

feature_df = under_sample[['avg_dist_0','weekday_pct_100','weekday_pct_0','ultimate_black_user','surge_pct_0','city_numeric','avg_surge_1','avg_rating_of_driver_5','avg_rating_by_driver_5','active']]

#Split into Train and Test For Model

x = feature_df.drop('active',axis=1)

y = feature_df['active']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=21)

#XGBoost Model

xgb = XGBClassifier(max_depth=3,n_estimators=250,learning_rate=0.1)

xgb = xgb.fit(x_train,y_train)

y_pred_xgb = xgb.predict(x_test)

print(y_pred_xgb)

cr_xgb = classification_report(y_test,y_pred_xgb)

print(cr_xgb)

cv_scores_xgb = cross_val_score(xgb,x,y,cv=5)

mean_cv_scores_xgb = np.mean(cv_scores_xgb)

print(mean_cv_scores_xgb)

y_pred_proba_xgb = xgb.predict_proba(x_test)

y_pred_proba_xgb_df = pd.DataFrame(y_pred_proba_xgb)

#73% accuracy on Test Set