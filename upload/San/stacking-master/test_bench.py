import pandas as pd


train_dataset = pd.read_csv('/home/user/san/KDD/stacking-master/data/preprocessed_input_interpolate_20min.csv')

# change "Date" to datetime object
train_dataset['date'] = pd.to_datetime(train_dataset['date'])

# construct "time of day"
train_dataset['timeofday'] = train_dataset.date.apply( lambda d : d.hour+d.minute/60.)


#train_data_dropped = train_dataset.drop([(1,0,'tot'),'''(1, 0, 'cargocar')''','''(1, 0, 'etc')''','''(1, 0, 'motorcycle')'''],axis=1)

train_dataset_copy = train_dataset.copy()

using_cols = [
              #                 "(1, 0, 'cargocar')",
              #                 "(1, 0, 'etc')",
              #                 "(1, 0, 'motorcycle')",
              #                 "(1, 0, 'privatecar')",
              #                 "(1, 0, 'tot')",
              #                 "(1, 0, 'unknowncar')",
              #                 "(1, 1, 'cargocar')",
              #                 "(1, 1, 'etc')",
              #                 "(1, 1, 'motorcycle')",
              #                 "(1, 1, 'privatecar')",
              #                 "(1, 1, 'tot')",
              #                 "(1, 1, 'unknowncar')",
              #                 "(2, 0, 'cargocar')",
              "(2, 0, 'etc')",
              #                 "(2, 0, 'motorcycle')",
              #                 "(2, 0, 'privatecar')",
              "(2, 0, 'tot')",
              #                 "(2, 0, 'unknowncar')",
              #                 "(3, 0, 'cargocar')",
              #                 "(3, 0, 'etc')",
              #                 "(3, 0, 'motorcycle')",
              #                 "(3, 0, 'privatecar')",
              #                 "(3, 0, 'tot')",
              #                 "(3, 0, 'unknowncar')",
              #                 "(3, 1, 'cargocar')",
              #                 "(3, 1, 'etc')",
              #                 "(3, 1, 'motorcycle')",
              #                 "(3, 1, 'privatecar')",
              #                 "(3, 1, 'tot')",
              #                 "(3, 1, 'unknowncar')",
              "('A', 2)",
              #                 "('A', 3)",
              #                 "('B', 1)",
              #                 "('B', 3)",
              #                 "('C', 1)",
              #                 "('C', 3)",
              'date',
              'hour',
              'pressure',
              'sea_pressure',
              'wind_direction',
              'wind_speed',
              'temperature',
              'rel_humidity',
              'precipitation',
              'dayofweek',
              'is_holiday',
              'timeofday'
              ]




train_dataset_dropped = train_dataset_copy[using_cols]
train_dataset_dropped.head(30)


train_dataset_target = train_dataset_copy["(1, 0, 'tot')"]
