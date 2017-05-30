from datetime import datetime, time, timedelta
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from datetime import time

class seasonal:
    def __init__(self, p, d, q, P, D, Q):
        self.p = p
        self.d = d
        self.q = q
        self.P = P
        self.D = D
        self.Q = Q
        
    def get_data(self, training, p1_training, p1_testing):
        self.training = training
        self.p1_training = p1_training
        self.p1_testing = p1_testing
    
    def convert_date(self):
        
        def convert(date):

            return datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
        
        self.training.index = self.training['date'].apply(convert)
        self.p1_training.index = self.p1_training['date'].apply(convert)
        self.p1_testing.index = self.p1_testing['date'].apply(convert)
        
        
    def train_session(self, training):
        
        self.model = SARIMAX(training, trend='n', order=(self.p, self.d, self.q), seasonal_order=(self.P,self.D,self.Q,72))
        self.results = self.model.fit()
        
    def complete_training(self, route):
        #changing index to date
        self.convert_date()
        self.route = route
        
        #get corresponding column
        self.training_set = self.training[route]
        p1_train = self.p1_training[route]
        self.p1_testing = self.p1_testing[route]
        
        #create timestamps on different sessions
        start_stamps = [self.training_set.index[-1] + timedelta(minutes = 20)]
        end_stamps = [p1_train.index[0] - timedelta(minutes = 20)]

        previous_timestamp = p1_train.index[0]
        for i in range(1, len(p1_train)):
            timestamp = p1_train.index[i]
            if timestamp - previous_timestamp != timedelta(minutes = 20):
                start_stamps.append(previous_timestamp + timedelta(minutes = 20))
                end_stamps.append(timestamp - timedelta(minutes = 20))
            previous_timestamp = timestamp
            
            
        self.start_stamps = start_stamps
        self.end_stamps = end_stamps
        
        #train using the original training data
        self.train_session(self.training_set)
        
        #training and form new training set from forecasting
        self.total_prediction = []
        for i in range(len(start_stamps)):
            forecasting = self.results.predict(start = start_stamps[i] - timedelta(minutes = 20), end = end_stamps[i], dynamic = True)
            self.training_set = pd.concat([self.training_set, forecasting[1:], p1_train[end_stamps[i] + timedelta(minutes = 20):end_stamps[i] + timedelta(minutes = 20)*6]])
            self.train_session(self.training_set)
            prediction = self.results.predict(start = end_stamps[i] + timedelta(minutes = 20)*6, end = end_stamps[i] + timedelta(minutes = 20)*13, dynamic = True)
            self.total_prediction.append(prediction[1:-1])
            
    def mape(self):
        y_pred = np.array(self.total_prediction).flatten()
        y_true = np.array(self.p1_testing)
        
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    def get_ape(self):
        y_pred = np.array(self.total_prediction).flatten()
        y_true = np.array(self.p1_testing)
        
        return np.abs((y_true - y_pred) / y_true)*100
    
    def session_mape(self):
        ape = self.get_ape()
        self.session_scores = []
        
        for i in range(len(self.total_prediction)/7):
            self.session_scores.append(np.mean(ape[i*7:(i+1)*7]))
            
        return self.session_scores
    
    def simple_training(self, route):
        #changing index to date
        self.convert_date()
        self.route = route
        
        #get corresponding column
        dataset = self.training[route]
        training_set = dataset.iloc[:-72]
        testing_set = dataset.iloc[-72:]
        
        self.train_session(training_set)
        validation_set = testing_set.between_time(start_time='06:00', end_time='07:40')
        self.p1_testing = testing_set.between_time(start_time='08:00', end_time='10:00')
        forecasting = self.results.predict(start = training_set.index[-1], end = validation_set.index[0] - timedelta(minutes = 20), dynamic = False)
        self.total_training = pd.concat([training_set, forecasting[1:], validation_set])
        self.train_session(self.total_training)
        self.total_prediction = self.results.predict(start = self.p1_testing.index[0] - timedelta(minutes = 20), end = self.p1_testing.index[-1], dynamic = False)[1:]
        
        

def get_training_data():
    return pd.read_csv('/home/user/san/KDD/stacking-master/data/preprocessed_input_interpolate_20min.csv')
#return pd.read_csv(r'preprocessed_input_interpolate_20min.csv')

def get_phase_1():
    
    def convert(date):

        return datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
    
        
    
    #testing_dataset = pd.read_csv(r'preprocessed_input_interpolate_20min_phase1and2_train.csv')
    testing_dataset = pd.read_csv('/home/user/san/KDD/stacking-master/data/preprocessed_input_interpolate_20min_phase1and2_train.csv')
    testing_dataset = testing_dataset.iloc[2016:]


    
    criteria_1 = ((testing_dataset['hour'] >= 6) & (testing_dataset['hour']<8)) | ((testing_dataset['hour']>=15) & (testing_dataset['hour']<17))
    p1_training = testing_dataset[criteria_1]

    criteria_2 = ((testing_dataset['hour'] >=8) & (testing_dataset['hour']<10)) | ((testing_dataset['hour']>=17) & (testing_dataset['hour']<19))
    p1_testing = testing_dataset[criteria_2]
    
    
    
    return pd.DataFrame(p1_training), pd.DataFrame(p1_testing)



training = get_training_data()
p1_training, p1_testing = get_phase_1()


# -- model1 = seasonal(1, 1, 1, 1, 1, 0)
# -- model1.get_data(training, p1_training, p1_testing)
# -- model1.complete_training('''(1, 0, 'tot')''')
# -- model1.mape()

