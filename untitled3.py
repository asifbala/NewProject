#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 16:02:32 2018
@author: asifbala
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#Drop missing values, specify world cups from 2002 to 2014 only

pd.set_option('display.max_rows', None)
data = pd.read_csv('/Users/asifbala/Downloads/WorldCupMatches.csv',index_col='Year')
data_nona = data.dropna()
df = data_nona.loc[2002.0:2014.0]

pd.options.mode.chained_assignment = None

#Find overall winning percentage of each team

df['Tie'] = np.where(df['Home Team Goals'] == df['Away Team Goals'],1, 0)

df['HomeTeamWins'] = np.where(df['Home Team Goals'] > df['Away Team Goals'],1, 0)

df['AwayTeamWins'] = np.where(df['Away Team Goals'] > df['Home Team Goals'],1, 0)

hometeamwins_sum = df.groupby('Home Team Name')['HomeTeamWins'].sum()

awayteamwins_sum = df.groupby('Away Team Name')['AwayTeamWins'].sum()

totalwins_sum = hometeamwins_sum + awayteamwins_sum

total_hometeams = df.groupby('Home Team Name')['Home Team Initials'].count()

total_awayteams = df.groupby('Away Team Name')['Away Team Initials'].count()

total_teams = total_hometeams + total_awayteams

team_winpercentage = totalwins_sum/total_teams

print(team_winpercentage)

#For every match combination of two teams, find winning percentage of both teams

df = df.assign(id=(df['Home Team Name']).astype('category').cat.codes)

df = df.assign(id2=(df['Away Team Name']).astype('category').cat.codes)

df['Winning Team'] = np.where(df['Home Team Goals'] > df['Away Team Goals'], df['Home Team Name'],df['Away Team Name'])

df['Home Team Name'],df['Away Team Name'] = np.where(df['id'] > df['id2'],[df['Away Team Name'],df['Home Team Name']],[df['Home Team Name'], df['Away Team Name']])

df['Home Team Goals'],df['Away Team Goals'] = np.where(df['id'] > df['id2'],[df['Away Team Goals'],df['Home Team Goals']],[df['Home Team Goals'], df['Away Team Goals']])

df['HomeTeamWins'],df['AwayTeamWins'] = np.where(df['id'] > df['id2'],[df['AwayTeamWins'],df['HomeTeamWins']],[df['HomeTeamWins'], df['AwayTeamWins']])

df.rename(columns={'Home Team Name': 'Team1', 'Away Team Name': 'Team2', 'HomeTeamWins':'Team1Wins','AwayTeamWins':'Team2Wins', 'Home Team Goals': 'Team 1 Goals','Away Team Goals':'Team 2 Goals'}, inplace=True)

total_matches = df.groupby(['Team1','Team2'])['Team1'].count()

team1wins_sum = df.groupby(['Team1','Team2'])['Team1Wins'].sum()

team2wins_sum = df.groupby(['Team1','Team2'])['Team2Wins'].sum()

ties_sum = df.groupby(['Team1','Team2'])['Tie'].sum()

team1_winpercent = team1wins_sum/total_matches

team2_winpercent = team2wins_sum/total_matches

tiespercent = ties_sum/total_matches

new_df = pd.concat([team1_winpercent,team2_winpercent,tiespercent],axis=1)

new_df.columns = ['Team1Win%','Team2Win%','Tie%']

print(new_df)

#Exploratory Data Analysis

print(new_df.describe())

print(df.describe())

#For only teams that participated in all worlds cup from 2002 to 2014, I plotted any insights on winning percentage per world cup for each team.

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

#line plot showcasing winning percentage by year for teams
winningpercentage_byyear_plot = winningpercentage_byyear.plot(subplots=True,figsize=(20,20))

plt.show()

winningpercentage_byyear_plot = winningpercentage_byyear.plot()

winningpercentage_byyear_plot.legend(loc='center left', bbox_to_anchor=(1, 0.5))

winningpercentage_byyear_plot.set(xlabel='Year', ylabel='Winning Percentage')

plt.show()

#box plot showcasing winning percentage distribution from 2002-2014 for teams

winningpercentage_byyear.plot(kind='box',figsize=(20,10))

plt.show()

#Inferential Statistics/Hypothesis Testing

#Test for Difference in Means:

#Ho = Population mean of home goals = Population mean of away goals.Thus, a team playing at home or away is not statistically significant.

#Ha = Population mean of away goals is NOT equal to Population mean of away goals. Thus a team playing at home or away is statistically significant.

def bootstrap_replicate_1d(data, func):
    return func(np.random.choice(data, size=len(data)))

def draw_bs_reps(data, func, size=1):
    bs_replicates = np.empty(size)
    for i in range(size):
        bs_replicates[i] = bootstrap_replicate_1d(data,func)

    return bs_replicates


new_df = data_nona.loc[2002.0:2014.0]

print(new_df.columns)

home_goals = new_df.groupby('Home Team Name')['Home Team Goals'].sum()

away_goals = new_df.groupby('Away Team Name')['Away Team Goals'].sum()

df_concat_goals = pd.concat([home_goals,away_goals],axis=1)

print(df_concat_goals)

emp_diff_means = np.mean(home_goals.values) - np.mean(away_goals.values)

goals_concat = np.concatenate([home_goals.values,away_goals.values])

mean_goals = np.mean(goals_concat)

home_goals_shifted = home_goals.values - np.mean(home_goals.values) + mean_goals

away_goals_shifted = away_goals.values - np.mean(away_goals.values) + mean_goals

bs_replicates_home = draw_bs_reps(home_goals_shifted,np.mean,10000)

plt.hist(bs_replicates_home,color='red')

bs_replicates_away = draw_bs_reps(away_goals_shifted,np.mean,10000)

plt.hist(bs_replicates_away,color='blue')

bs_replicates_diff =  bs_replicates_home - bs_replicates_away

plt.hist(bs_replicates_diff,color='green')

plt.show()

p = np.true_divide(np.sum(bs_replicates_diff >= emp_diff_means),len(bs_replicates_diff))

print('p-val =', p)

#There is a p value of 0.2928 which indicates that we fail to reject(accept) the null hypothesis that there is no difference in the population mean of home goals and away goals and thus whetheror not a team plays at home or not is not statistically significant in how many goals they will score.

#New Hypothesis Test for Correlation:

#Ho = There is no population correlation(pearson_r = 0) between home goals by world cup teams and away goals by world cup teams

#Ha = There is a population correlation(pearson_r not 0) between home goals by world cup teams and away goals by world cup teams

df_concat_goals.plot(kind='scatter', x='Home Team Goals',y='Away Team Goals',figsize=(15,5))

plt.xticks(np.arange(0,40))

plt.xlabel('Home Goals- all teams')

plt.ylabel('Away Goals- all teams')

plt.margins(0.02)

plt.show()

sns.regplot(home_goals.values,away_goals.values)

plt.xlabel('Home Goals- all teams')

plt.ylabel('Away Goals- all teams')

plt.show()

def pearson_r(x, y):
    corr_mat = np.corrcoef(x,y)
    return corr_mat[0,1]

r_obs = pearson_r(home_goals.values,away_goals.values)

print(r_obs)

perm_replicates = np.empty(10000)

for i in range(10000):
    home_goals_permuted = np.random.permutation(home_goals.values)
    perm_replicates[i] = pearson_r(home_goals_permuted,away_goals.values)
    
p = np.true_divide(np.sum(perm_replicates >= r_obs),len(perm_replicates))

print('p-val =', p)

#There is a p value of 0, which indicates that we can reject the null hypothesis that there is no correlation between home goals and away goals. Thus, we accept the alternative 
#hypothesis that there is a correlation between home goals and away goals.
