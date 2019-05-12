import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
import datetime
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from xgboost import XGBClassifier

df_timestamp = pd.read_csv('/Users/asifbala/Springboard-Data-Science/data_science_take_home_challenge_relax_inc/takehome_user_engagement.csv',encoding='latin-1')
df_users = pd.read_csv('/Users/asifbala/Springboard-Data-Science/data_science_take_home_challenge_relax_inc/takehome_users.csv', encoding='latin-1')

date_list = list(df_timestamp['time_stamp'])

date = pd.to_datetime(date_list)

df_timestamp['time_stamp'] = list(date)

df_timestamp['time_stamp'] = df_timestamp['time_stamp'].dt.floor('d').astype(np.int64)

df_timestamp = df_timestamp.sort_values(['user_id', 'time_stamp']).drop_duplicates()

rolling_timestamp = df_timestamp.groupby('user_id')['time_stamp'].rolling(window=3)

days_rolling_timestamp = pd.to_timedelta((rolling_timestamp.max()- rolling_timestamp.min())).dt.days

print(days_rolling_timestamp.head())

adopted_users = days_rolling_timestamp[days_rolling_timestamp < 8].index.get_level_values('user_id').tolist()

adopted_users_set = set(adopted_users)

adopted_users_list = list(adopted_users_set)

new_df = pd.DataFrame({'range': range(len(adopted_users_list))}, index = adopted_users_list)

print(new_df.index)

df_users['adopted'] = np.where(df_users['object_id'].isin(new_df.index),1,0)

print(df_users['adopted'].head())

print(df_users.head())

class_counts = df_users['adopted'].value_counts()

print(class_counts)

#Find Number of samples which are adopted
no_adopted = len(df_users[df_users['adopted'] == 1])

#Get indices of non adopted samples
non_adopted_indices = df_users[df_users.adopted == 0].index

#Random sample non adopted indices
random_indices = np.random.choice(non_adopted_indices,no_adopted, replace=False)

#Find the indices of adopted samples
adopted_indices = df_users[df_users.adopted == 1].index

#Concat adopted indices with sample non-adopted ones
under_sample_indices = np.concatenate([adopted_indices,random_indices])

#Get Balanced Dataframe
under_sample = df_users.loc[under_sample_indices]

class_counts = under_sample['adopted'].value_counts()

print(class_counts)

class_counts.plot(kind='bar')

plt.show()

print(under_sample.head())

under_sample['creation_time'] = pd.to_datetime(under_sample['creation_time'])
under_sample['last_session_creation_time'] = under_sample['last_session_creation_time'].map(lambda data: 
                                    datetime.datetime.utcfromtimestamp(int(data)).strftime('%Y-%m-%d %H:%M:%S'),na_action='ignore')
    
print(under_sample[['creation_time','last_session_creation_time']].head())

under_sample['last_session_creation_time'] = pd.to_datetime(under_sample['last_session_creation_time'])
under_sample['time_active'] = under_sample['last_session_creation_time'] - under_sample['creation_time']
under_sample['time_active'] = [x.total_seconds() for x in under_sample['time_active']]
under_sample['time_active'] = under_sample['time_active'].fillna(0)
print(under_sample['time_active'].head())

under_sample['email_domain'] = [x.split('@')[1] for x in under_sample.email]
top_emails = under_sample.email_domain.value_counts().index[:6]
under_sample['email_domain'] = [x if x in top_emails else 'domain_other' for x in under_sample.email_domain]
print(under_sample['email_domain'].head())

under_sample['invited_by_user_id'] = under_sample['invited_by_user_id'].fillna(0)
print(under_sample['invited_by_user_id'].head())

le = LabelEncoder()

creation_source_labels = le.fit_transform(under_sample['creation_source'])
under_sample['creation_source'] = creation_source_labels

email_domain_labels = le.fit_transform(under_sample['email_domain'])
under_sample['email_domain'] = email_domain_labels

print(under_sample.head())

feature_df = under_sample[['creation_source','time_active','email_domain','org_id','invited_by_user_id','enabled_for_marketing_drip','opted_in_to_mailing_list','adopted']]

print(feature_df.head())

x = feature_df.drop('adopted',axis=1)

y = feature_df['adopted']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=21,stratify=y)

print(y_train.value_counts())

print(y_test.value_counts())

xgb = XGBClassifier(max_depth=3,n_estimators=250,learning_rate=0.1)

cv_scores_xgb = cross_val_score(xgb,x,y,cv=5)

print(cv_scores_xgb)

mean_cv_scores_xgb = np.mean(cv_scores_xgb)

print(mean_cv_scores_xgb)

xgb = xgb.fit(x_train,y_train)

y_pred_xgb = xgb.predict(x_test)

print(y_pred_xgb[:10])

cr_xgb = classification_report(y_test,y_pred_xgb)

print(cr_xgb)

y_pred_proba_xgb = xgb.predict_proba(x_test)

print(type(y_pred_proba_xgb))

y_pred_proba_xgb_df = pd.DataFrame(y_pred_proba_xgb)

print(y_pred_proba_xgb_df[:10])

feature_importance = pd.DataFrame()
feature_importance['coef'] = xgb.feature_importances_
feature_importance = feature_importance.set_index(x.columns)
feature_importance['coef'].nlargest(10)

plt.figure(figsize=(10,5))
(feature_importance['coef']).nlargest(10).plot(kind='bar', x=feature_importance.index)
plt.title('XGBoost Classifier Feature Coefficients')
plt.ylabel('coefficient value')
plt.show()