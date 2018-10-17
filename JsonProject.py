import pandas as pd
import json
from pandas.io.json import json_normalize
import numpy as np

'''Load Json and Normalize it'''

data = json.load((open('/Users/asifbala/Downloads/data_wrangling_json/data/world_bank_projects.json')))
n_json = json_normalize(data,'mjtheme_namecode',['countryshortname',['_id','$oid'],['theme1', 'Name'],['theme1','Percent']])
df = pd.DataFrame(n_json)

'''1.) Find the 10 countries with most projects'''

x = df.countryshortname.value_counts()
print(x.head(10))

'''2.) Find the top 10 major project themes (using column 'mjtheme_namecode')'''

z = df.name.value_counts()
print(z.head(10))

'''3.)In 2. above you will notice that some entries have only the code and the name is missing. Create a dataframe with the missing names filled in.'''

new_df = df.replace('',np.nan)

df['name'] = new_df.groupby('code')['name'].ffill().bfill()
print(df)

''' Values for top 10 major project themes has now updated to include the the previously missing values'''

w = df.name.value_counts()
print(w.head(10))