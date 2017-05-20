import pandas as pd
import numpy as np
import datetime
import readDataUtil


datapath = "../../../../data/original_dataset/"
#======================================================
##train data
##df_trajectories, df_travel_segment = readDataUtil.read_trajectory("df_trajectories.pkl", "df_travel_segment.pkl")
#df_trajectories, df_travel_segment = readDataUtil.read_trajectory(datapath+"/training/trajectories_table_5_training.csv")
#df_volume = readDataUtil.read_volume(datapath+"/training/volume_table_6_training.csv")
#df_weather = readDataUtil.read_weather(datapath+"/training/weather_table_7_training_update.csv")
##outname = 'phase1_training_vol_route_interpolated_weather_joined_table'
##times = pd.date_range('09/20/2016' , '10/18/2016', freq="3H")
#outname = 'phase1_training_vol_route_weather_joined_table_interpolated_per20min'
#vol_sampling_times = pd.date_range('09/20/2016' , '10/18/2016', freq="20min")
#route_sampling_times = pd.date_range('07/19/2016' , '10/18/2016', freq="20min")
#======================================================
#test data
df_trajectories, df_travel_segment = readDataUtil.read_trajectory(datapath+"/testing_phase1/trajectories_table_5_test1.csv")
df_volume = readDataUtil.read_volume(datapath+"/testing_phase1/volume_table_6_test1.csv")
df_weather = readDataUtil.read_weather(datapath+"/testing_phase1/weather_table_7_test1.csv")
outname = 'phase1_testing_vol_route_weather_joined_table'
vol_sampling_times = pd.date_range('10/18/2016' , '10/25/2016', freq="20min")
route_sampling_times = pd.date_range('10/18/2016' , '10/25/2016', freq="20min")
#======================================================

def is_holiday(t):
  rdate = t.date()
  if rdate>=datetime.date(2016, 9,15) and rdate<=datetime.date(2016, 9,17): return 1 #mid autum holiday
  if rdate==datetime.date(2016, 9,18): return 0
  if rdate>=datetime.date(2016,10, 1) and rdate<=datetime.date(2016,10, 7): return 1 #national holiday
  if rdate>=datetime.date(2016,10, 8) and rdate<=datetime.date(2016,10, 9): return 0
  if t.dayofweek == 0 or t.dayofweek == 6: return 1 # sun or sat
  return 0 #weekdays

vols = [(1,0), (1,1), (2,0), (3,0), (3,1)]
routes = [ ("A",2), ("A",3), ("B",1), ("B",3), ("C",1), ("C",3) ]
weather_fields = ['pressure', 'sea_pressure', 'wind_direction',
                  'wind_speed', 'temperature', 'rel_humidity', 'precipitation']
vehicle_class = ['motorcycle', 'cargocar', 'privatecar', 'unknowncar']

#======================================================

# prepare volume data
def get_vehicle_class(row):
  vehicle_type  = row.vehicle_type
  vehicle_model = row.vehicle_model
  if vehicle_type==1 : return 'cargocar'
  if vehicle_model==1 and vehicle_type==0 : return 'motorcycle'
  if vehicle_model>1 and vehicle_type==0 : return 'privatecar'
  return 'unknowncar'

df_volume['vehicle_type'] = df_volume['vehicle_type'].replace(np.nan,-1)
df_volume['vehicle_class'] = df_volume.apply( get_vehicle_class, axis=1)
vehicle_class_volume = df_volume.groupby( ['tollgate_id','direction','vehicle_class',pd.TimeGrouper('20min')]).size()
etc_volume = df_volume.groupby( ['tollgate_id','direction','has_etc',pd.TimeGrouper('20min')]).size()
tot_volume = etc_volume.sum(level=[0,1,3])

tmp = {}
for (p,q) in vols:
  for aclass in vehicle_class:
    try:
      tmp2 = vehicle_class_volume[(p,q,aclass)].reindex(vol_sampling_times)
      tmp2.fillna(0, inplace=True)
    except KeyError as e:
      tmp2 = pd.Series(0, index=vol_sampling_times)

    tmp[(p,q,aclass)] = tmp2

  tmp2 = etc_volume[(p,q,1)].reindex(vol_sampling_times)
  tmp2.fillna(0, inplace=True)
  tmp[(p,q,'etc')] = tmp2

  tmp2 = tot_volume[(p,q)].reindex(vol_sampling_times)
  tmp2.fillna(0, inplace=True)
  tmp[(p,q,'tot')] = tmp2

df_cartype_volume = pd.DataFrame(tmp)

# prepare route median data
trajectories_median = df_trajectories.set_index('starting_time') \
                                     .groupby(['intersection_id', 'tollgate_id',pd.TimeGrouper('20min')]) \
                                     .travel_time \
                                     .median()
tmp = {}
for aroute in routes:
  tmp2 = trajectories_median[aroute].reindex(route_sampling_times) #fill missing times with NA
  tmp2 = tmp2.interpolate() #interpolate NA from nearby data
  tmp[aroute]=tmp2
df_trajectories_median = pd.DataFrame(tmp)

# preapre weather data
df_weather['wind_direction'].replace(999017,np.NaN, inplace=True)
df_weather = df_weather.reindex(vol_sampling_times) 
df_weather = df_weather.apply(pd.Series.interpolate) #interpolate 3hr interval data to 20min interval

# combine and add extra columns
df_combined = pd.concat( [df_cartype_volume, df_trajectories_median.loc[vol_sampling_times] , df_weather] , axis=1)
df_combined["date"] = df_combined.index
df_combined["dayofweek"] = df_combined['date'].apply( lambda x: x.dayofweek )
df_combined["hour"] = df_combined['date'].apply( lambda x: x.hour )
df_combined["is_holiday"] = df_combined['date'].apply( is_holiday )

df_combined.to_csv('%s.csv'%outname, index=False)
df_combined.to_pickle('%s.pkl'%outname)
