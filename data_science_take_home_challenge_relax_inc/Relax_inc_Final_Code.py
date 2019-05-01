import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression

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

creation_source_adopted = pd.crosstab(under_sample['creation_source'],under_sample['adopted'])

print(creation_source_adopted)

creation_source_adopted.plot(kind='bar')

plt.show()

org_id_adopted = pd.crosstab(under_sample['org_id'],under_sample['adopted'])

print(type(org_id_adopted))

print(org_id_adopted.head(30))

#There isnt any significant difference between adopted and not adopted for the many organizations.

opted_mailing_list_adopted = pd.crosstab(under_sample['opted_in_to_mailing_list'],under_sample['adopted'])

print(opted_mailing_list_adopted)

opted_mailing_list_adopted.plot(kind='bar')

plt.show()

enabled_marketing_drip_adopted = pd.crosstab(under_sample['enabled_for_marketing_drip'],under_sample['adopted'])

print(enabled_marketing_drip_adopted)

enabled_marketing_drip_adopted.plot(kind='bar')

plt.show()

invited_by_user_id_adopted = pd.crosstab(under_sample['invited_by_user_id'],under_sample['adopted'])

print(invited_by_user_id_adopted.head(10))

under_sample['creation_time'] = pd.to_datetime(under_sample['creation_time'])

under_sample2 = under_sample.set_index('creation_time')

under_sample_monthly_sum = under_sample2.resample('M').sum()

adopted_monthly = under_sample_monthly_sum['adopted']

print(adopted_monthly.head(5))

under_sample_monthly_count = under_sample2.resample('M').count()

total_monthly = under_sample_monthly_count['adopted']

not_adopted_monthly = total_monthly - adopted_monthly

not_adopted_monthly.name = 'not_adopted'

print(not_adopted_monthly.head(5))

monthly_dataset = pd.concat([adopted_monthly,not_adopted_monthly],axis=1)

print(monthly_dataset)

monthly_dataset.plot()

plt.show() 