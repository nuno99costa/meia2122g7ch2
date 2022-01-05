import os
import sys
import time

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.style.use('Solarize_Light2')

TIME_START = "2015-01-01 00:00:00"
time_start = pd.to_datetime(TIME_START).timestamp()

# Read the data
DATA_DIR = './dataset/'

def print_df(df, msg):
    print(f'---------------- {msg} ----------------\n')
    print(df)
    print(f'\n------------------------------------------------------------------------------\n\n')


def datetime_func(dt):
    """ Converts timestamp in milliseconds to hours since time constant TIME_START """
    dt = dt // 10 ** 9  # Milliseconds to seconds
    dt = dt - time_start    # Seconds since TIME_START
    dt = dt // 60 ** 2  # Converts to hours
    return int(dt)

start = time.time()

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

# Creates new dataframe merging telemetry and machines
df = machines_df.merge(telemetry_df, on='machineID')

print_df(df, 'Dataframe after merging with machines')

# One hot encoding of the model feature
df = pd.get_dummies(df)

print_df(df, 'Dataframe after one hot encoding model')

print('Finished in {:.2f} seconds.'.format(time.time()-start))