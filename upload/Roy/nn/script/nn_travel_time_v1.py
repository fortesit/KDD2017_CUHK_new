import pandas as pd
import numpy as np
import argparse
import pickle
import utils
from keras.models import Sequential
from keras.layers import LSTM, Dense, Lambda
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
import theano

prefix = "nn_travel_time_v1"

parser = argparse.ArgumentParser()
parser.add_argument('--train', action='store_true',
                               help='train')
parser.add_argument('--savePred', action='store_true')
args = parser.parse_args()

vols = [(1,0), (1,1), (2,0), (3,0), (3,1)]
routes = [ ("A",2), ("A",3), ("B",1), ("B",3), ("C",1), ("C",3) ]

# T: travel time
in_cols_T = [
                 ('A', 2),
                 ('A', 3),
                 ('B', 1),
                 ('B', 3),
                 ('C', 1),
                 ('C', 3),
	        ]

# W: weather and other
in_cols_W = [
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

# T: travel time
out_cols_T = [
                 ('A', 2),
                 ('A', 3),
                 ('B', 1),
                 ('B', 3),
                 ('C', 1),
                 ('C', 3),
	        ]

input_dim  = (len(in_cols_T) + len(in_cols_W))*6
output_dim = len(out_cols_T)*6

def preproc( df, inScalerT=None, inScalerW=None ):
  """
  preprocess input data
  for time cols, apply log then MinMaxScaler
  for other cols, just apply MinMaxScaler
  """
  #for aCol in useful_cols_A:
  #aCol = (aCol,)
  #  df.loc[:,aCol] = df.loc[:,aCol].apply(np.log)
  dfT = df[in_cols_T]
  dfW = df[in_cols_W]

  dfT = dfT.values.copy()
  dfW = dfW.values.copy()

  dfT = np.log(dfT)

  if not inScalerT:
    inScalerT = MinMaxScaler(feature_range=(-1, 1))
    partT = inScalerT.fit_transform(dfT)
  else:
    partT = inScalerT.transform(dfT)

  if not inScalerW:
    inScalerW = MinMaxScaler(feature_range=(-1, 1))
    partW = inScalerW.fit_transform(dfW)
  else:
    partW = inScalerW.transform(dfW)

  scaled_df = np.concatenate((partT,partW), axis=1)
  return scaled_df, inScalerT, inScalerW

def getOutputScaler(df):
  """
  get scaler to map output (~-1 to 1) back to travel time
  """
  #for aCol in useful_cols_A:
  #  aCol = (aCol,)
  #  df.loc[:,aCol] = df.loc[:,aCol].apply(np.log)
  dfT = df[out_cols_T]
  dfT = dfT.values.copy()
  dfT = np.log(dfT)
  outScalerT = MinMaxScaler(feature_range=(-1, 1))
  outScalerT.fit(dfT)
  return outScalerT

def get_model( outScalerT, weightsFile = None):
  """
  build keras model
  load saved weights if given
  """
  if type(outScalerT)==str: outScalerT = pickle.load( open(outScalerT, "rb") )

  tv_outputScaler_min   = theano.tensor.as_tensor_variable(outScalerT.data_min)
  tv_outputScaler_scale = theano.tensor.as_tensor_variable(outScalerT.scale_)
  
  def postproc( nnOutput ):
    nnOutput = nnOutput.reshape( (-1, len(out_cols_T)) )
    nnOutput = (nnOutput - outScalerT.feature_range[0])/tv_outputScaler_scale + tv_outputScaler_min
    #nnOutput = scalerA.inverse_transform(nnOutput)
    nnOutput = nnOutput.reshape( (-1, output_dim) )
    nnOutput = np.exp(nnOutput)
    return nnOutput
  
  model = Sequential()
  model.add(Dense( 3, activation='relu', input_shape=(input_dim,)))
  model.add(Dense( 3, activation='relu'))
  model.add(Dense( 3, activation='relu'))
  model.add(Dense( 3, activation='relu'))
  #model.add(Dense( 3, input_shape=(input_dim,)))
  model.add(Dense(output_dim))
  model.add(Lambda(lambda x: postproc(x), output_shape=(output_dim,)))
  if weightsFile: model.load_weights(weightsFile)
  model.compile(loss='mean_absolute_percentage_error', optimizer='adam')

  return model
  
  
def train_model(df_traj):
  input_rows = df_traj[ lambda r : ((r.timeofday>= 6) & (r.timeofday< 8)) |
  		                 ((r.timeofday>=15) & (r.timeofday<17))
  	    ]
  output_rows = df_traj[ lambda r : ((r.timeofday>= 8) & (r.timeofday<10)) |
                                     ((r.timeofday>=17) & (r.timeofday<19))
  		    ]
  #input_rows = df_traj[ lambda r : ((r.timeofday>= 6) & (r.timeofday< 8)) ]
  #output_rows = df_traj[ lambda r : ((r.timeofday>= 8) & (r.timeofday<10)) ]
  
  input_rows  =  input_rows[ in_cols_T + in_cols_W ]
  output_rows = output_rows[ out_cols_T ]
  
  train_input_rows  = input_rows[: -24*7]
  train_output_rows = output_rows[: -24*7]
  test_input_rows   = input_rows[-24*7:] #reserve 1 week for test
  test_output_rows  = output_rows[-24*7:] #reserve 1 week for test
  
  #no need preproc output col, the network will undo the transform and compare directly with output
  scaled_train_input_rows, inScalerT, inScalerW = preproc(train_input_rows)
  scaled_test_input_rows,_,_ = preproc(test_input_rows, inScalerT, inScalerW)
  outScalerT = getOutputScaler(train_output_rows)
  pickle.dump(  inScalerT, open("inScalerT.pkl", "wb") )
  pickle.dump(  inScalerW, open("inScalerW.pkl", "wb") )
  pickle.dump( outScalerT, open("outScalerT.pkl", "wb") )
  
  train_output_rows = train_output_rows.values
  test_output_rows  = test_output_rows.values
  
  #reshape to have each 2hr segment as a row
  scaled_train_input_rows = scaled_train_input_rows.reshape( (-1,input_dim) )
  scaled_test_input_rows  =  scaled_test_input_rows.reshape( (-1,input_dim) )
  train_output_rows = train_output_rows.reshape( (-1,output_dim) )
  test_output_rows  =  test_output_rows.reshape( (-1,output_dim) )

  model = get_model(outScalerT)

  filepath="weights-{val_loss:.4f}-%s-{epoch:02d}.hdf5" % prefix
  checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
  callbacks_list = [checkpoint]
  history = model.fit(scaled_train_input_rows, train_output_rows, callbacks=callbacks_list, batch_size=10000, epochs=1000, validation_data=(scaled_test_input_rows,test_output_rows))
  print (min(history.history['val_loss']))

  return model

def use_model(model, inScalerT, inScalerW, df_traj):
  input_rows = df_traj[ lambda r : ((r.timeofday>= 6) & (r.timeofday< 8)) |
  		                 ((r.timeofday>=15) & (r.timeofday<17))
  	    ]
  #input_rows = df_traj[ lambda r : ((r.timeofday>= 6) & (r.timeofday< 8)) ]
  input_rows  =  input_rows[ in_cols_T + in_cols_W ]
  scaled_input_rows,_,_ = preproc(input_rows, inScalerT, inScalerW)
  
  #reshape to have each 2hr segment as a row
  scaled_input_rows  =  scaled_input_rows.reshape( (-1,input_dim) )

  return model.predict(scaled_input_rows)

if args.train:
  df_traj = pd.read_pickle("../../exploreData/run/phase1_training_route_weather_joined_table_interpolated_per20min.pkl")
  df_traj['timeofday'] = df_traj.date.apply( lambda d : d.hour+d.minute/60.)
  model = train_model(df_traj)
else:
  #df_traj = pd.read_pickle("../../exploreData/run/phase1_testing_route_weather_joined_table_interpolated_per20min.pkl")
  df_traj = pd.read_pickle("../../exploreData/run/phase1_training_vol_route_weather_joined_table_interpolated_per20min.pkl")
  df_traj['timeofday'] = df_traj.date.apply( lambda d : d.hour+d.minute/60.)

  inScalerT = pickle.load( open("inScalerT.pkl", "rb") )
  inScalerW = pickle.load( open("inScalerW.pkl", "rb") )
  model = get_model("outScalerT.pkl", "weights-18.1267-nn_travel_time_v1-240.hdf5")

  pred = use_model(model, inScalerT, inScalerW, df_traj)

  #frame pred back to a panda dataframe
  pred_times = [x for x in df_traj.index if x.minute==0 and (x.hour==6 or x.hour==15)]
  pred_colnames = []
  for dt in range(120,240,20):
    for (startPt, endPt) in routes:
      pred_colnames.append("dt%i_%s%s_routetime_median"%(dt,startPt, endPt))
  df_pred = pd.DataFrame( pred, index=pred_times, columns=pred_colnames)

  if args.savePred:
    pickle.dump( df_pred, open("%s_predict_traveltime.pkl"%prefix, "wb") )
    utils.save_traveltime_prediction( df_pred, prefix)
