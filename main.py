import os
import sys
import time

import pandas as pd
import numpy as np

start = time.time()

TIME_START = "2015-01-01 00:00:00"
time_start = pd.to_datetime(TIME_START).timestamp()

# Read the data
DATA_DIR = './dataset/'


def print_df(df, msg):
    print(
        f'\n-------------------------------------------------------------- {msg} --------------------------------------------------------------\n')
    print(df)
    print(f'\n----------------------------------------------------------------------------------------------------------------------------------------------------------\n')


def datetime_func(dt):
    """ Converts timestamp in milliseconds to hours since time constant TIME_START """
    dt = dt // 10 ** 9  # Milliseconds to seconds
    dt = dt - time_start    # Seconds since TIME_START
    dt = dt // 60 ** 2  # Converts to hours
    return int(dt)


def time_since_error(r, error):
    res = errors_df.loc[(errors_df['datetime'] <= r['datetime']) &
                        (errors_df['machineID'] == r['machineID']) &
                        (errors_df['errorID'] == error)]

    if res.empty:
        return np.nan
    else:
        return r['datetime'] - res.iloc[-1]['datetime']
 
telemetry_df = pd.read_csv(f'{DATA_DIR}/PdM_telemetry.csv')
errors_df = pd.read_csv(f'{DATA_DIR}/PdM_errors.csv')
maint_df = pd.read_csv(f'{DATA_DIR}/PdM_maint.csv')
failures_df = pd.read_csv(f'{DATA_DIR}/PdM_failures.csv')
machines_df = pd.read_csv(f'{DATA_DIR}/PdM_machines.csv')

# Format date & time. Sort based on date for better readability
tables = [telemetry_df, maint_df, failures_df, errors_df]
for df in tables:
    df['datetime'] = pd.to_datetime(
        df['datetime'], format='%Y-%m-%d %H:%M:%S').values.astype(np.int64)
    df['datetime'] = df['datetime'].apply(datetime_func)
    df.sort_values(['datetime', 'machineID'], inplace=True, ignore_index=True)

tables.append(machines_df)
tables_names = ['telemetry_df', 'maint_df',
                'failures_df', 'errors_df', 'machines_df']

for df, df_name in zip(tables, tables_names):
    print_df(df, f'Initial {df_name} datafrane')
    # df.to_csv(f'./dataset_with_periods/{df_name}.csv')

# Creates new dataframe merging telemetry and machines
df = machines_df.merge(telemetry_df, on='machineID')

print_df(df, 'Dataframe after merging with machines')

# One hot encoding of the model feature
df = pd.get_dummies(df)

print_df(df, 'Dataframe after one hot encoding model')

# Reading result from csv since it takes to long (~10 min for each)
""" 
# Creating columns time_since_last_error
df['time_since_last_error1'] = df.apply(
    time_since_error, engine='numba', raw=True, args=('error1',), axis=1).astype(pd.Int64Dtype())
print('Finished error 1 in {:.2f} seconds.'.format(time.time()-start))

df['time_since_last_error2'] = df.apply(
    time_since_error, engine='numba', raw=True, args=('error2',), axis=1).astype(pd.Int64Dtype())
print('Finished error 2 in {:.2f} seconds.'.format(time.time()-start))

df['time_since_last_error3'] = df.apply(
    time_since_error, engine='numba', raw=True, args=('error3',), axis=1).astype(pd.Int64Dtype())
print('Finished error 3 in {:.2f} seconds.'.format(time.time()-start))

df['time_since_last_error4'] = df.apply(
    time_since_error, engine='numba', raw=True, args=('error4',), axis=1).astype(pd.Int64Dtype())
print('Finished error 4 in {:.2f} seconds.'.format(time.time()-start))

df['time_since_last_error5'] = df.apply(
    time_since_error, engine='numba', raw=True, args=('error5',), axis=1).astype(pd.Int64Dtype())
print('Finished error 5 in {:.2f} seconds.'.format(time.time()-start))

df.to_csv('./output/output.csv')  
"""

df = pd.read_csv(f'./output/df.csv')

print_df(df, 'Dataframe with time_since_last_error columns')


#TODO time_since_last_replacement_componentX
#TODO time_since_last_failure_componentX
#TODO time_to_errorX


print('Finished in {:.2f} seconds.'.format(time.time()-start))