import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle

# get the link file
df_link = pd.read_csv('links (table 3).csv')

# get the route file
df_routes = pd.read_csv('routes (table 4).csv')

# get the df_trajectories file
df_trajectories = pd.read_csv('trajectories(table 5)_training.csv')
# extract the travel sequence for routes
df_travel_seq = pd.DataFrame(df_trajectories.iloc[:,4].str.replace(';','#').str.split('#'))
#df_travel_seq.iloc[0][0]

# get the volume file
df_volume = pd.read_csv('volume(table 6)_training.csv')
# turns the nan to 0
df_volume['vehicle_type'] = df_volume['vehicle_type'].fillna(0)

# get the weather file
df_weather = pd.read_csv('weather (table 7)_training_update.csv')

df_travel_seq = pd.DataFrame(df_trajectories.iloc[:,4].str.strip("[]").str.replace(';','#').str.split('#'))


df_trajectories = df_trajectories.rename(columns={"starting_time":"time"})
df_trajectories['time'] = pd.to_datetime(df_trajectories['time'], format='%Y-%m-%d %H:%M')
df_trajectories['time'] = df_trajectories['time'].map(lambda t: t.strftime('%Y-%m-%d %H:%M'))
df_trajectories['time'] = pd.to_datetime(df_trajectories['time'], format='%Y-%m-%d %H:%M')
df_volume['time'] = pd.to_datetime(df_volume['time'], format='%d/%m/%Y %H:%M')


df_trajectories_volume = pd.merge(df_trajectories, df_volume, on=['time'])
df_trajectories_volume.to_pickle('trajectories_volume.pkl')
