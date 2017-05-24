import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

#note:
#sth is strange with (2,0,'tot') and others vol from 1 Oct 00:00 to 7Oct 23:59

df_merged_volume = pd.read_pickle("../../exploreData/run/phase1_training_vol_route_weather_joined_table_interpolated_per20min.pkl")
df_merged_volume['timeofday'] = df_merged_volume.date.apply( lambda d : d.hour+d.minute/60.)

useful_cols = [
              #(1, 0, 'cargocar'),
              #(1, 0, 'etc'),
              #(1, 0, 'motorcycle'),
              #(1, 0, 'privatecar'),
               (1, 0, 'tot'),
              #(1, 0, 'unknowncar'),
              #(1, 1, 'cargocar'),
              #(1, 1, 'etc'),
              #(1, 1, 'motorcycle'),
              #(1, 1, 'privatecar'),
               (1, 1, 'tot'),
              #(1, 1, 'unknowncar'),
              #(2, 0, 'cargocar'),
              #(2, 0, 'etc'),
              #(2, 0, 'motorcycle'),
              #(2, 0, 'privatecar'),
               (2, 0, 'tot'),
              #(2, 0, 'unknowncar'),
              #(3, 0, 'cargocar'),
              #(3, 0, 'etc'),
              #(3, 0, 'motorcycle'),
              #(3, 0, 'privatecar'),
               (3, 0, 'tot'),
              #(3, 0, 'unknowncar'),
              #(3, 1, 'cargocar'),
              #(3, 1, 'etc'),
              #(3, 1, 'motorcycle'),
              #(3, 1, 'privatecar'),
               (3, 1, 'tot'),
              #(3, 1, 'unknowncar'),
               ('A', 2),
               ('A', 3),
               ('B', 1),
               ('B', 3),
               ('C', 1),
               ('C', 3),
              #'date',
              #'hour',
              #'pressure',
              #'sea_pressure',
              #'wind_direction',
              #'wind_speed',
              #'temperature',
               'rel_humidity',
               'precipitation',
               'dayofweek',
               'is_holiday',
               'timeofday'
              ]

sel_rows = df_merged_volume[ lambda r : ((r.timeofday>= 6) & (r.timeofday<10)) |
		                        ((r.timeofday>=15) & (r.timeofday<19))
		           ]
sel_rows = sel_rows[ useful_cols ]

#split to train and test set
train_rows = sel_rows[: -24*7]
test_rows = sel_rows[-24*7:] #reserve 1 week for test

#get numpy array from panda dataframe
train_arr = train_rows.values
test_arr = test_rows.values

#scale feature array to range -1 to 1
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler = scaler.fit(train_arr)
train_scaled_arr = scaler.transform(train_arr)
test_scaled_arr = scaler.transform(test_arr)

#sample subsequence from the time series
train_seqs = []
nSegments = train_arr.shape[0]//12 # each segment holds 4hr data (12 datapoints, 20min each)
for segment in range(nSegments):
  for t in range(6):
    startIdx = segment*12 + t
    train_seqs.append(train_scaled_arr[startIdx: startIdx+7])
train_seqs = np.stack(train_seqs)

test_seqs = []
nSegments = test_arr.shape[0]//12 # each segment holds 4hr data (12 datapoints, 20min each)
for segment in range(nSegments):
  for t in range(6):
    startIdx = segment*12 + t
    test_seqs.append(test_scaled_arr[startIdx: startIdx+7])
test_seqs = np.stack(test_seqs)

#keras
#https://keras.io/getting-started/sequential-model-guide/#examples
input_dim = len(useful_cols)
output_dim = len(useful_cols)
timesteps = 6 # use 6 timesteps to predict the 7th

x_train, y_train = train_seqs[:, 0:-1], train_seqs[:, -1]
x_test , y_test  =  test_seqs[:, 0:-1],  test_seqs[:, -1]

model = Sequential()
model.add(LSTM(16, input_shape=(timesteps, input_dim)))
model.add(Dense(output_dim))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=500, batch_size=64, validation_data=(x_test,y_test))
