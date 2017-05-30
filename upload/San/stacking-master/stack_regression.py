'''
Author: San Wong

The following code is a trail code for KDD 2017
Traffic Volume Prediction with Stacking Methods

Goal: Predict Traffic Volume (Focus on 1_0_tot for now)



Reference: Stacking (stacked generalization)
https://github.com/ikki407/stacking

'''
# ----- for creating dataset -----
from sklearn.datasets import load_boston
from sklearn.cross_validation import train_test_split

# ----- general import -----
import pandas as pd
import numpy as np

# ----- stacking library -----
from stacking.base import FOLDER_NAME, PATH, INPUT_PATH, TEMP_PATH,\
        FEATURES_PATH, OUTPUT_PATH, SUBMIT_FORMAT
# ----- utils -----
from stacking.base import load_data, save_pred_as_submit_format, create_cv_id, \
        eval_pred
# ----- classifiers -----
from stacking.base import BaseModel, XGBRegressor, KerasRegressor

# ----- keras -----
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l1, l2, l1_l2

# ----- scikit-learn -----
from sklearn.svm import SVR
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor

# ----- Set problem type!! -----
problem_type = 'regression'
classification_type = ''
eval_type = 'rmse'

BaseModel.set_prob_type(problem_type, classification_type, eval_type)

# -- Import class and function created by teammates --------------------------------------
from pong_time_series import seasonal


    # -- How to use this function
    # -- model1 = seasonal(1, 1, 1, 1, 1, 0)
    # -- model1.get_data(training, p1_training, p1_testing)
    # -- model1.complete_training('''(1, 0, 'tot')''')
    # -- model1.mape()



# DATA ===================================================================================
'''
During Training. Our data includes 1_0_tot, which is the total traffic volume exiting gate 1
Therefore, we are extracting it from the dataset and set it to be our Y (i.e: Data that we are trying to predict)
And the remaining data (i.e. Data that drop '1_0_tot') to be our X. We are aiming to learn the weight matrix of it

Therefore, the same data processing need to be done for test (also known as the validation period)
'''

# -- TRAINING -------------------------------------------------------------------------------------------------
train_dataset = pd.read_csv('/home/user/san/KDD/stacking-master/data/preprocessed_input_interpolate_20min.csv')

# change "Date" to datetime object
train_dataset['date'] = pd.to_datetime(train_dataset['date'])
# construct "time of day"
train_dataset['timeofday'] = train_dataset.date.apply( lambda d : d.hour+d.minute/60.)
train_dataset_copy = train_dataset.copy()

feature_cols = [
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
                               "(1, 1, 'tot')",
              #                 "(1, 1, 'unknowncar')",
              #                 "(2, 0, 'cargocar')",
              #                 "(2, 0, 'etc')",
              #                 "(2, 0, 'motorcycle')",
              #                 "(2, 0, 'privatecar')",
                               "(2, 0, 'tot')",
              #                 "(2, 0, 'unknowncar')",
              #                 "(3, 0, 'cargocar')",
              #                 "(3, 0, 'etc')",
              #                 "(3, 0, 'motorcycle')",
              #                 "(3, 0, 'privatecar')",
                               "(3, 0, 'tot')",
              #                 "(3, 0, 'unknowncar')",
              #                 "(3, 1, 'cargocar')",
              #                 "(3, 1, 'etc')",
              #                 "(3, 1, 'motorcycle')",
              #                 "(3, 1, 'privatecar')",
                               "(3, 1, 'tot')",
              #                 "(3, 1, 'unknowncar')",
                               "('A', 2)",
                               "('A', 3)",
                               "('B', 1)",
                               "('B', 3)",
                               "('C', 1)",
                               "('C', 3)",
              #'date',
              'hour',
              'pressure',
              #'sea_pressure',
              'wind_direction',
              'wind_speed',
              'temperature',
              'rel_humidity',
              'precipitation',
              'dayofweek',
              'is_holiday',
              'timeofday'
              ]

target_label = "(1, 0, 'tot')"
        

train_dataset_dropped = train_dataset_copy[feature_cols]
#train_dataset_target = train_dataset_copy["(1, 0, 'tot')"]
train_dataset_target = train_dataset_copy[target_label]

# -- TESTING DATA ---------------------------------------------------------------------------------------------
test_dataset = pd.read_csv('../data/preprocessed_input_interpolate_20min_phase1_test.csv')

# change "Date" to datetime object
test_dataset['date'] = pd.to_datetime(test_dataset['date'])
# construct "time of day"
test_dataset['timeofday'] = test_dataset.date.apply( lambda d : d.hour+d.minute/60.)
test_dataset_copy = test_dataset.copy()


test_dataset_dropped = test_dataset_copy[feature_cols]
test_dataset_target = test_dataset_copy[target_label]


# concat data as pandas' dataframe format
# Flip the naming to identify the DataFrame object
dropped_train_dataset = pd.DataFrame(train_dataset_dropped)

# Try to change target col name before turn it into DataFrame and hence a csv file
train_dataset_target.columns = ['target']


target_train_dataset = pd.DataFrame(train_dataset_target)
#target_train_dataset = pd.DataFrame(train_dataset_target, columns=target_label)

dropped_test_dataset = pd.DataFrame(test_dataset_dropped)


# Update col name
test_dataset_target.columns = ['target']
target_test_dataset = pd.DataFrame(test_dataset_target)
#target_test_dataset = pd.DataFrame(test_dataset_target, columns=target_label)

# save data under /data/input.
dropped_train_dataset.to_csv(INPUT_PATH + 'train.csv', index=False)
target_train_dataset.to_csv(INPUT_PATH + 'target.csv', index=False)
dropped_test_dataset.to_csv(INPUT_PATH + 'test.csv', index=False)
target_test_dataset.to_csv(INPUT_PATH + 'label_test.csv', index=False)


# END OF DATA ==============================================================================



# -----create features -----

# -- Apply a function to a DataFrame that is intended to operate elementwise,
# -- i.e. like doing map(func, series) for each series in the DataFrame
# -- LOG files store 13 continuous attributes (we originally have 47 features, we have dropped some features and we have 21 left)



train_log = dropped_train_dataset.iloc[:, :20].applymap(lambda x: np.log(x+1))
test_log = dropped_test_dataset.iloc[:, :20].applymap(lambda x: np.log(x+1))

train_log.columns = map(str, train_log.columns)
test_log.columns = map(str, test_log.columns)

train_log.columns += '_log'
test_log.columns += '_log'

# save data under /data/output/features/.
train_log.to_csv(FEATURES_PATH + 'train_log.csv', index=False)
test_log.to_csv(FEATURES_PATH + 'test_log.csv', index=False)

# ----- END create features -----




# STAGE ONE of MODEL STACKING =============================================

# FEATURE LISTS in Stage 1.
FEATURE_LIST_stage1 = {
                'train':(
                         INPUT_PATH + 'train.csv',
                         FEATURES_PATH + 'train_log.csv',
                        ),

                'target':(
                         INPUT_PATH + 'target.csv',
                        ),

                'test':(
                         INPUT_PATH + 'test.csv',
                         FEATURES_PATH + 'test_log.csv',
                        ),
                }

# need to get input shape for NN now
X,y,test  = load_data(flist=FEATURE_LIST_stage1, drop_duplicates=False)
assert((False in X.columns == test.columns) == False)
nn_input_dim_NN = X.shape[1:]
del X, y, test



# Models in Stage 1

PARAMS_V1 = {
        'colsample_bytree':0.5,
        'learning_rate':0.1,'gamma':0,
        'max_depth':5, 'min_child_weight':1,
        'nthread':4,'reg_lambda':0,'reg_alpha':0,
        'objective':'reg:linear', 'seed':407,
        'silent':1, 'subsample':0.65
         }

class ModelV1(BaseModel):
        def build_model(self):
            return XGBRegressor(params=self.params, num_round=1)
#return XGBRegressor(params=self.params, num_round=10)

PARAMS_V2 = {
            'batch_size':32,
            'nb_epoch':5,
            'verbose':1, 
            'callbacks':[],
            'validation_split':0.,
            'validation_data':None,
            'shuffle':True,
            'class_weight':None,
            'sample_weight':None,
            'normalize':True,
            'categorize_y':False,
            }

class ModelV2(BaseModel):
        def build_model(self):
            model = Sequential()
            model.add(Dense(64, input_shape=nn_input_dim_NN))
            #model.add(Dropout(0.5))
            model.add(Dense(1))
            model.add(Activation('linear'))
            sgd = SGD(lr=0.001)
            model.compile(loss='mean_squared_error', optimizer=sgd, metrics=["accuracy"])

            return KerasRegressor(nn=model,**self.params)
        

PARAMS_V3 = {'kernel':'rbf', 
             'C':1e3, 
             'gamma':0.1
             }

class ModelV3(BaseModel):
        def build_model(self):
            return SVR(**self.params)

PARAMS_V4 = {'n_neighbors':5,
             'weights':'distance', 
             'p':2, 
             'n_jobs':4
             }

class ModelV4(BaseModel):
        def build_model(self):
            return KNeighborsRegressor(**self.params)

PARAMS_V5 = {'n_estimators':150,
             'learning_rate':0.1, 
             'max_depth':18, 
             'random_state':0, 
             'loss':'lad', 
             'subsample':0.7, 
             'verbose':1
            }

class ModelV5(BaseModel):
        def build_model(self):
            return GradientBoostingRegressor(**self.params)

PARAMS_V6 = {}

class ModelV6(BaseModel):
        def build_model(self):
            # Direct passing model parameters can be used
            return linear_model.BayesianRidge(normalize=True, verbose=True, compute_score=True)


        
# ----- END first stage stacking model -----

# ----- Second stage stacking model -----

PARAMS_V1_stage2 = {
        'colsample_bytree':0.6,
        'learning_rate':0.1,'gamma':0,
        'max_depth':5, 'min_child_weight':1,
        'nthread':4,'reg_lambda':0,'reg_alpha':0,
        'objective':'reg:linear', 'seed':407,
        'silent':1, 'subsample':0.8
         }

class ModelV1_stage2(BaseModel):
        def build_model(self):
            return XGBRegressor(params=self.params, num_round=50)


# ----- END second stage stacking model -----

if __name__ == "__main__":
    
    # Create cv-fold index
    target = pd.read_csv(INPUT_PATH + 'target.csv')
    # we dont need CV for this traffic data set
    create_cv_id(target, n_folds_ = 2, cv_id_name='cv_id', seed=407)

    ######## stage1 Models #########
    print 'Start stage 1 training'

    m = ModelV1(name="v1_stage1",
                flist=FEATURE_LIST_stage1,
                params = PARAMS_V1,
                kind = 'st'
                )
    m.run()


    m = ModelV2(name="v2_stage1",
                flist=FEATURE_LIST_stage1,
                params = PARAMS_V2,
                kind = 'st'
                )
    m.run()

    m = ModelV3(name="v3_stage1",
                flist=FEATURE_LIST_stage1,
                params = PARAMS_V3,
                kind = 'st'
                )
    m.run()

    m = ModelV4(name="v4_stage1",
                flist=FEATURE_LIST_stage1,
                params = PARAMS_V4,
                kind = 'st'
                )
    m.run()

    m = ModelV5(name="v5_stage1",
                flist=FEATURE_LIST_stage1,
                params = PARAMS_V5,
                kind = 'st'
                )
    m.run()

    m = ModelV6(name="v6_stage1",
                flist=FEATURE_LIST_stage1,
                params = PARAMS_V6,
                kind = 'st'
                )
    m.run()



    print 'Done stage 1'
    print 
    ######## stage2 Models #########

    print 'Start stage 2 training'

    # FEATURE LISTS in Stage 2.
    # Need to define here because the outputs for NN dim. haven't been created yet.
    FEATURE_LIST_stage2 = {
                'train':(INPUT_PATH + 'train.csv',
                         FEATURES_PATH + 'train_log.csv',
                         
                         TEMP_PATH + 'v1_stage1_all_fold.csv',
                         TEMP_PATH + 'v2_stage1_all_fold.csv',
                         TEMP_PATH + 'v3_stage1_all_fold.csv',
                         TEMP_PATH + 'v4_stage1_all_fold.csv',
                         TEMP_PATH + 'v5_stage1_all_fold.csv',
                         TEMP_PATH + 'v6_stage1_all_fold.csv',
                        ),

                'target':(
                         INPUT_PATH + 'target.csv',
                        ),

                'test':(INPUT_PATH + 'test.csv',
                         FEATURES_PATH + 'test_log.csv',
                         
                         TEMP_PATH + 'v1_stage1_test.csv',
                         TEMP_PATH + 'v2_stage1_test.csv',
                         TEMP_PATH + 'v3_stage1_test.csv',
                         TEMP_PATH + 'v4_stage1_test.csv',
                         TEMP_PATH + 'v5_stage1_test.csv',
                         TEMP_PATH + 'v6_stage1_test.csv',                       
                        ),
                }

    X,y,test  = load_data(flist=FEATURE_LIST_stage2, drop_duplicates=False)
    assert((False in X.columns == test.columns) == False)
    nn_input_dim_NN2 = X.shape[1]
    del X, y, test


    # Models
    m = ModelV1_stage2(name="v1_stage2",
                    flist=FEATURE_LIST_stage2,
                    params = PARAMS_V1_stage2,
                    kind = 'st',
                    )
    m.run()

    print 'Done stage 2'
    print 
    
    # averaging
    print 'Saving as submission format'
    #sample_sub = pd.read_csv('data/input/sample_submission.csv')
    label = pd.read_csv(INPUT_PATH + 'label_test.csv')
    testID = range(len(label))
    testID = pd.DataFrame(testID, columns=['ID'])
    pred = pd.read_csv(TEMP_PATH + 'v1_stage2_TestInAllTrainingData.csv')

    print 'Test evaluation'
    auc = eval_pred(label.target, pred.iloc[:,0], eval_type=eval_type)
    pred = pd.concat([testID, pred], axis=1)
    pred.to_csv(TEMP_PATH + 'final_submission.csv', index=False)

    


