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
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

data = pd.read_csv('/Users/asifbala/Documents/Data Science/CapstoneProject/CSV Files/WorldCupMatches.csv',index_col='Year')
data_nona = data.dropna()
df = data_nona.loc[2002.0:2014.0]
pd.options.mode.chained_assignment = None

df2 = data_nona.loc[2002.0]

df2['Tie'] = np.where(df2['Home Team Goals'] == df2['Away Team Goals'],1, 0)

df2['HomeTeamWins'] = np.where(df2['Home Team Goals'] > df2['Away Team Goals'],1, 0)

df2['AwayTeamWins'] = np.where(df2['Away Team Goals'] > df2['Home Team Goals'],1, 0)

hometeamwins_sum2 = df2.groupby('Home Team Name')['HomeTeamWins'].sum()

awayteamwins_sum2 = df2.groupby('Away Team Name')['AwayTeamWins'].sum()

totalwins_sum2 = hometeamwins_sum2 + awayteamwins_sum2

total_hometeams2 = df2.groupby('Home Team Name')['Home Team Initials'].count()

total_awayteams2 = df2.groupby('Away Team Name')['Away Team Initials'].count()

total_teams2 = total_hometeams2 + total_awayteams2

team_winpercentage2 = totalwins_sum2/total_teams2

#Team winning percentages for only 2006 world cup

df3 = data_nona.loc[2006.0]

df3['Tie'] = np.where(df3['Home Team Goals'] == df3['Away Team Goals'],1, 0)

df3['HomeTeamWins'] = np.where(df3['Home Team Goals'] > df3['Away Team Goals'],1, 0)

df3['AwayTeamWins'] = np.where(df3['Away Team Goals'] > df3['Home Team Goals'],1, 0)

hometeamwins_sum3 = df3.groupby('Home Team Name')['HomeTeamWins'].sum()

awayteamwins_sum3 = df3.groupby('Away Team Name')['AwayTeamWins'].sum()

totalwins_sum3 = hometeamwins_sum3 + awayteamwins_sum3

total_hometeams3 = df3.groupby('Home Team Name')['Home Team Initials'].count()

total_awayteams3 = df3.groupby('Away Team Name')['Away Team Initials'].count()

total_teams3 = total_hometeams3 + total_awayteams3

team_winpercentage3 = totalwins_sum3/total_teams3

#Team winning percentages for only 2010 world cup

df4 = data_nona.loc[2010.0]

df4['Tie'] = np.where(df4['Home Team Goals'] == df4['Away Team Goals'],1, 0)

df4['HomeTeamWins'] = np.where(df4['Home Team Goals'] > df4['Away Team Goals'],1, 0)

df4['AwayTeamWins'] = np.where(df4['Away Team Goals'] > df4['Home Team Goals'],1, 0)

hometeamwins_sum4 = df4.groupby('Home Team Name')['HomeTeamWins'].sum()

awayteamwins_sum4 = df4.groupby('Away Team Name')['AwayTeamWins'].sum()

totalwins_sum4 = hometeamwins_sum4 + awayteamwins_sum4

total_hometeams4 = df4.groupby('Home Team Name')['Home Team Initials'].count()

total_awayteams4 = df4.groupby('Away Team Name')['Away Team Initials'].count()

total_teams4 = total_hometeams4 + total_awayteams4

team_winpercentage4 = totalwins_sum4/total_teams4

#Team winning percentages for only 2014 world cup

df5 = data_nona.loc[2014.0]

df5['Tie'] = np.where(df5['Home Team Goals'] == df5['Away Team Goals'],1, 0)

df5['HomeTeamWins'] = np.where(df5['Home Team Goals'] > df5['Away Team Goals'],1, 0)

df5['AwayTeamWins'] = np.where(df5['Away Team Goals'] > df5['Home Team Goals'],1, 0)

hometeamwins_sum5 = df5.groupby('Home Team Name')['HomeTeamWins'].sum()

awayteamwins_sum5 = df5.groupby('Away Team Name')['AwayTeamWins'].sum()

totalwins_sum5 = hometeamwins_sum5 + awayteamwins_sum5

total_hometeams5 = df5.groupby('Home Team Name')['Home Team Initials'].count()

total_awayteams5 = df5.groupby('Away Team Name')['Away Team Initials'].count()

total_teams5 = total_hometeams5 + total_awayteams5

team_winpercentage5 = totalwins_sum5/total_teams5

#Concatenate dataframes for 2002,2006,2010,and 2014

winningpercentage_byyear = pd.concat([team_winpercentage2,team_winpercentage3,team_winpercentage4,team_winpercentage5],axis=1)

#Drop teams that didnt participate in all 4 world cups

winningpercentage_byyear = winningpercentage_byyear.dropna()

winningpercentage_byyear.columns = ['2002','2006','2010','2014']

winningpercentage_byyear = winningpercentage_byyear.T

print(winningpercentage_byyear)

winningpercentage_byyear_plot = winningpercentage_byyear.plot(subplots=True,figsize=(20,20))

plt.show()

winningpercentage_byyear_plot = winningpercentage_byyear.plot()

winningpercentage_byyear_plot.legend(loc='center left', bbox_to_anchor=(1, 0.5))

winningpercentage_byyear_plot.set(xlabel='Year', ylabel='Winning Percentage')

plt.show()

winningpercentage_byyear.plot(kind='box',figsize=(20,10))

plt.show()

df['Tie'] = np.where(df['Home Team Goals'] == df['Away Team Goals'],1, 0)

df['HomeTeamWins'] = np.where(df['Home Team Goals'] > df['Away Team Goals'],1, 0)

df['AwayTeamWins'] = np.where(df['Away Team Goals'] > df['Home Team Goals'],1, 0)

hometeamwins_sum = df.groupby('Home Team Name')['HomeTeamWins'].sum()

awayteamwins_sum = df.groupby('Away Team Name')['AwayTeamWins'].sum()

home_ties_sum = df.groupby('Home Team Name')['Tie'].sum()

away_ties_sum = df.groupby('Away Team Name')['Tie'].sum()

totalwins_sum = hometeamwins_sum + awayteamwins_sum

total_ties_sum = home_ties_sum + away_ties_sum

total_hometeams = df.groupby('Home Team Name')['Home Team Initials'].count()

total_awayteams = df.groupby('Away Team Name')['Away Team Initials'].count()

total_teams = total_hometeams + total_awayteams

team_tp = total_ties_sum/total_teams

team_wp = totalwins_sum/total_teams

print(team_wp.head())

df = df.assign(id=(df['Home Team Name']).astype('category').cat.codes)

df = df.assign(id2=(df['Away Team Name']).astype('category').cat.codes)

df['Winning Team'] = np.where(df['Home Team Goals'] > df['Away Team Goals'], df['Home Team Name'],df['Away Team Name'])

df['Home Team Name'],df['Away Team Name'] = np.where(df['id'] > df['id2'],[df['Away Team Name'],df['Home Team Name']],[df['Home Team Name'], df['Away Team Name']])

df['Home Team Goals'],df['Away Team Goals'] = np.where(df['id'] > df['id2'],[df['Away Team Goals'],df['Home Team Goals']],[df['Home Team Goals'], df['Away Team Goals']])

df['HomeTeamWins'],df['AwayTeamWins'] = np.where(df['id'] > df['id2'],[df['AwayTeamWins'],df['HomeTeamWins']],[df['HomeTeamWins'], df['AwayTeamWins']])

df.rename(columns={'Home Team Name': 'Team0', 'Away Team Name': 'Team1', 'HomeTeamWins':'Team0Wins','AwayTeamWins':'Team1Wins', 'Home Team Goals': 'Team 0 Goals','Away Team Goals':'Team 1 Goals'}, inplace=True)

total_matches = df.groupby(['Team0','Team1'])['Team0'].count()

team1wins_sum = df.groupby(['Team0','Team1'])['Team0Wins'].sum()

team2wins_sum = df.groupby(['Team0','Team1'])['Team1Wins'].sum()

ties_sum = df.groupby(['Team0','Team1'])['Tie'].sum()

team1_winpercent = team1wins_sum/total_matches

team2_winpercent = team2wins_sum/total_matches

tiespercent = ties_sum/total_matches

wp_df = pd.concat([team1_winpercent,team2_winpercent,tiespercent],axis=1)

wp_df.columns = ['Team0_Win%_InMatchUp','Team1_Win%_InMatchUp','BothTeams_Tie%_InMatchUp']

print(wp_df.head())

df = df.reset_index()

def newfunction(df):
    if df['Team 0 Goals'] > df['Team 1 Goals']:          
        return 1
    if df['Team 0 Goals'] == df['Team 1 Goals']:
        return 2
    else:
        return 0
    
df['Winning Team'] = df.apply(newfunction,axis=1)

base_df = df[['Team0','Team1','Winning Team']]

base_df = base_df.set_index(['Team0','Team1'])

base_df = base_df.sort_index()

team_tp = team_tp.reset_index()

team_tp.columns = ['Team0','Team0_Tie%_Overall']

team_tp = team_tp.set_index('Team0')

team_tp2 = team_tp.reset_index()

team_tp2.columns = ['Team1','Team1_Tie%_Overall']

team_tp2 = team_tp2.set_index('Team1')

team_wp = team_wp.reset_index()

team_wp.columns = ['Team0','Team0_Win%_Overall']

team_wp = team_wp.set_index('Team0')

team_wp2 = team_wp.reset_index()

team_wp2.columns = ['Team1','Team1_Win%_Overall']

team_wp2 = team_wp2.set_index('Team1')

base_df = base_df.join([team_wp,team_wp2,wp_df,team_tp,team_tp2])

print(base_df.head())

df = data_nona.loc[2010.0:2014.0]

df['Tie'] = np.where(df['Home Team Goals'] == df['Away Team Goals'],1, 0)

df['HomeTeamWins'] = np.where(df['Home Team Goals'] > df['Away Team Goals'],1, 0)

df['AwayTeamWins'] = np.where(df['Away Team Goals'] > df['Home Team Goals'],1, 0)

hometeamwins_sum = df.groupby('Home Team Name')['HomeTeamWins'].sum()

awayteamwins_sum = df.groupby('Away Team Name')['AwayTeamWins'].sum()

home_ties_sum = df.groupby('Home Team Name')['Tie'].sum()

away_ties_sum = df.groupby('Away Team Name')['Tie'].sum()

totalwins_sum = hometeamwins_sum + awayteamwins_sum

total_ties_sum = home_ties_sum + away_ties_sum

total_hometeams = df.groupby('Home Team Name')['Home Team Initials'].count()

total_awayteams = df.groupby('Away Team Name')['Away Team Initials'].count()

total_teams = total_hometeams + total_awayteams

team_tp = total_ties_sum/total_teams

team_tp = team_tp.reset_index()

team_tp.columns = ['Team0','Team0_Tie%_Last2WC']

team_tp = team_tp.set_index('Team0')

team_tp2 = team_tp.reset_index()

team_tp2.columns = ['Team1','Team1_Tie%_Last2WC']

team_tp2 = team_tp2.set_index('Team1')

team_wp = totalwins_sum/total_teams

team_wp = team_wp.reset_index()

team_wp.columns = ['Team0','Team0_Win%_Last2WC']

team_wp = team_wp.set_index('Team0')

team_wp2 = team_wp.reset_index()

team_wp2.columns = ['Team1','Team1_Win%_Last2WC']

team_wp2 = team_wp2.set_index('Team1')

base_df = base_df.join([team_wp,team_wp2,team_tp,team_tp2])

base_df['Team0_Win%_Last2WC'] = np.where(base_df['Team0_Win%_Last2WC'].isnull(), base_df['Team0_Win%_Overall'],base_df['Team0_Win%_Last2WC'])

base_df['Team1_Win%_Last2WC'] = np.where(base_df['Team1_Win%_Last2WC'].isnull(), base_df['Team1_Win%_Overall'],base_df['Team1_Win%_Last2WC'])

base_df['Team0_Tie%_Last2WC'] = np.where(base_df['Team0_Tie%_Last2WC'].isnull(), base_df['Team0_Tie%_Overall'],base_df['Team0_Tie%_Last2WC'])

base_df['Team1_Tie%_Last2WC'] = np.where(base_df['Team1_Tie%_Last2WC'].isnull(), base_df['Team1_Tie%_Overall'],base_df['Team1_Tie%_Last2WC'])

print(base_df.head())

x = base_df.drop('Winning Team',axis=1)

print(x.head())

y = base_df['Winning Team']

print(y.head())

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=21,stratify=y)

print(y_train.value_counts())

print(y_test.value_counts())

rf = RandomForestClassifier(max_depth=3,n_estimators=250)

cv_scores_rf = cross_val_score(rf,x,y,cv=5)

mean_cv_scores_rf = np.mean(cv_scores_rf)

print(mean_cv_scores_rf)

rf = rf.fit(x_train,y_train)

y_pred_rf = rf.predict(x_test)

print(y_pred_rf)

cr_rf = classification_report(y_test,y_pred_rf)

print(cr_rf)

y_pred_proba_rf = rf.predict_proba(x_test)

y_pred_proba_rf_df = pd.DataFrame(y_pred_proba_rf)

print(y_pred_proba_rf_df.head())

logreg = LogisticRegression(multi_class='multinomial',solver='saga')

cv_scores_logreg = cross_val_score(logreg,x,y,cv=5)

mean_cv_scores_logreg = np.mean(cv_scores_logreg)

print(mean_cv_scores_logreg)

logreg = logreg.fit(x_train,y_train)

y_pred_logreg = logreg.predict(x_test)

print(y_pred_logreg)

cr_logreg = classification_report(y_test,y_pred_logreg)

print(cr_logreg)

y_pred_proba_logreg = logreg.predict_proba(x_test)

y_pred_proba_logreg_df = pd.DataFrame(y_pred_proba_logreg)

print(y_pred_proba_logreg_df.head())

xgb = XGBClassifier(max_depth=3,n_estimators=250,learning_rate=0.1)

cv_scores_xgb = cross_val_score(xgb,x,y,cv=5)

mean_cv_scores_xgb = np.mean(cv_scores_xgb)

print(mean_cv_scores_xgb)

xgb = xgb.fit(x_train,y_train)

y_pred_xgb = xgb.predict(x_test)

print(y_pred_xgb)

cr_xgb = classification_report(y_test,y_pred_xgb)

print(cr_xgb)

y_pred_proba_xgb = xgb.predict_proba(x_test)

y_pred_proba_xgb_df = pd.DataFrame(y_pred_proba_xgb)

print(y_pred_proba_xgb_df.head())

y2 = y_test.reset_index()

ensemble_output= 0.15 * y_pred_proba_xgb + 0.7 * y_pred_proba_logreg + 0.15 * y_pred_proba_rf

new_df = pd.DataFrame({0: ensemble_output[:,0],1: ensemble_output[:,1],2: ensemble_output[:,2]})

joined_df = pd.concat([y2,new_df],axis=1)

joined_df = joined_df.set_index(['Team0','Team1'])

print(joined_df.head())

joined_df['label_highestprob'] = joined_df[[0,1,2]].idxmax(axis=1)

print(joined_df['label_highestprob'].head())

cr_ensemble = classification_report(joined_df['Winning Team'],joined_df['label_highestprob'])

print(cr_ensemble)