import datetime
import math
import pdb
import pickle

import pandas as pd


# clean up date columns
def dateparse(time_in_secs):
    if math.isnan(time_in_secs):
        return float('nan')
    return datetime.datetime.fromtimestamp(float(time_in_secs))


lambda_dateparse = lambda x: dateparse(x)

# 10 minute intervals of outage snapshots
df = pd.read_csv('./outages_snapshots.csv')
parse_dates = ['lastUpdateTime', 'currentEtor']
for col in parse_dates:
    df[col] = df[col].apply(lambda_dateparse)

# lookup table
snapshot_df = pd.read_csv('./snapshots.csv')
snapshot_df['when'] = snapshot_df['when'].apply(lambda_dateparse)

# more lookup tables
cause_df = pd.read_csv('./cause.csv')
crew_status_df = pd.read_csv('./crew_status.csv')
region_df = pd.read_csv('./region.csv')

# df.head()
# snapshot_df.head()
# cause_df.head()
# crew_status_df.head()
# region_df.head()


# joins to do lookups
def join(df, right_df, right_cols, left_on, right_on):
    pre_join_len = len(df)

    df = df.merge(right_df[right_cols],
                  how='inner',
                  left_on=left_on,
                  right_on=right_on)

    post_join_len = len(df)
    assert (pre_join_len == post_join_len)

    df.drop(columns='id_y', inplace=True)
    df.rename(columns={'id_x': 'id'}, inplace=True)
    return df


snapshot_cols = ['id', 'when']
df = join(df, snapshot_df, snapshot_cols, 'snapshot', 'id')

# df.columns

df = join(df, cause_df, cause_df.columns, 'cause', 'id')

df.rename(columns={'name': 'cause'}, inplace=True)

# df.columns

df = join(df, crew_status_df, crew_status_df.columns, 'crewCurrentStatus',
          'id')

df.rename(columns={'name': 'crewCurrentStatus'}, inplace=True)

# df.columns

df = join(df, region_df, region_df.columns, 'regionName', 'id')

df.rename(columns={'name': 'regionName'}, inplace=True)

# df.columns

# len(df)

# save for downstream consumption
pickle_file = './merged_data.pkl'
with open(pickle_file, 'wb') as f:
    pickle.dump(df, f)

old_df = df

with open(pickle_file, 'rb') as f:
    df = pickle.load(f)

assert (old_df.equals(df))