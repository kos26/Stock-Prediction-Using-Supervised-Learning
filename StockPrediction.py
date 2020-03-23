import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
# For reading stock data from yahoo
from pandas_datareader.data import DataReader

# For time stamps
from datetime import datetime

# For reading stock data from yahoo
from pandas_datareader.data import DataReader

#Splitting data into train and test
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

#Statistical Models for predicting target value.
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.svm import SVR 

#Deep Learning Models for predicting target values
from keras.models import Sequential
from keras.layers import Dense, LSTM

class StockPrediction:
    def __init__(self, name, start_date):
        self.name = name
        self.start_date = start_date  
        
    def stock_analysis(self):
        name = self.name
        start_date = self.start_date
        df = DataReader(name, data_source='yahoo', start=start_date, end=datetime.now())
        print("Stock Information")
        print(df)
        print("\n\n\n")
        print("**********" * 11)
        #Comparing the features with the target.
        plt.figure(figsize=(16, 8))
        plt.plot(df[['Open','High','Low','Close']])
        plt.ylabel('price in USD', fontsize=20)
        plt.title('Historical daily Open, Close, High and Low price', fontsize=20)
        plt.legend(['Open','High','Low','Close'])
        print(plt.show())
        print("\n\n\n")
        print("**********" * 11)
        #Display Volume of stocks traded each day
        plt.figure(figsize=(16, 8))
        plt.plot(df[['Volume']])
        plt.ylabel('Volume', fontsize=20)
        plt.title('Volume of stock traded each day', fontsize=20)
        plt.legend(['Volume'])
        print(plt.show())
        print("\n\n\n")
        print("**********" * 11)
        df.hist(figsize=(12, 12));
        print("\n\n\n")
        print("**********" * 11)
        df['Daily Return'] = df['Adj Close'].pct_change()
        plt.figure(figsize=(16, 8))
        plt.plot(df[['Daily Return']], linestyle='--', marker='o')
        plt.ylabel('Returns', fontsize=20)
        plt.title('Daily returns from stock traded each day', fontsize=20)
        plt.legend(['Volume'])
        print(plt.show())
        print("\n\n\n")
        print("**********" * 11)
        
        sns.distplot(df['Daily Return'].dropna(), bins=100, color='purple')
        plt.ylabel('Daily Return', fontsize=20)
        plt.title(f'Daily returns from {self.name}\'s stock traded each day')


    def predict_stock(self):
        name = self.name
        start_date = self.start_date
        df = DataReader(name, data_source='yahoo', start=start_date, end=datetime.now())
        print("Stock Information")
        print(df)
        print("\n\n\n")
        print("**********" * 11)
        #Comparing the features with the target.
        plt.figure(figsize=(16, 8))
        plt.plot(df[['Open','High','Low','Close']])
        plt.ylabel('price in USD')
        plt.title('Historical daily Open, Close, High and Low price')
        plt.legend(['Open','High','Low','Close'])
        print(plt.show())
        n = int(df.shape[0]*0.8)
        m = int(df.shape[0]-n)
        X_train_and_test = df.head(n)
        forcast = df.tail(m)
        X_forcast = forcast.drop(['Close', 'Adj Close'], axis=1)
        X = X_train_and_test.drop(['Close', 'Adj Close'], axis=1)
        #X = df.drop(['High', 'Low', 'Volume', 'Close', 'Adj Close', 'Daily_avg'], axis=1)
        y = X_train_and_test['Close']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
        print("\n\n\n")
        print("**********" * 11)
        print("Predictions on testing samples to compare their accuracies.\n\n")
        Regressor = {
            'Linear Regression': LinearRegression(),
            'Support Vector Regressor': SVR(kernel='rbf', C=1e3, gamma=0.1),
            'Bayesian Ridge': BayesianRidge(),
            'ExtraTrees Regressor': ExtraTreesRegressor(n_estimators=500, min_samples_split=20),
            'Random Forest Regressor': RandomForestRegressor(n_estimators=200),
            'Gradient Boosting Regressor': GradientBoostingRegressor(n_estimators=500)
        }
        for name, clf in Regressor.items():
            print(name)
            clf.fit(X_train, y_train) 
            mse = mean_squared_error(y_test, clf.predict(X_test))
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, clf.predict(X_test))
            mae = mean_absolute_error(y_test, clf.predict(X_test))

            print(f'Mean Square Error: {mse:.2f}')
            print(f'Root Mean Square Error: {rmse:.2f}')
            print(f'R2 Score: {r2:.2f}')
            print(f'Mean Absolute Error: {mae:.2f}')
            print()
        model = LinearRegression()
        model.fit(X, y)
        target = model.predict(X_forcast)
        forecasted_X = target
        forcast['Predictions'] = forecasted_X
        plt.figure(figsize=(15,8))
        print("\n\n\n")
        print("**********" * 11)
        print("Visualization of Forcasted Prediction vs Actual Closing Values.\n\n")
        plt.xlabel('Time', fontsize=20)
        plt.ylabel('Closing Price in $USD', fontsize=20)
        plt.title('Historic Daily Closing Price + Predictions', fontsize=20)
        plt.plot(X_train_and_test['Close'])
        plt.plot(forcast[['Close', 'Predictions']])
        plt.legend(['Train', 'Target Closing Price', 'Predictions for Closing Price'], loc='upper left')
        print(plt.show())
        print(forcast[['Close', 'Predictions']])

    def LSTM_predict_stock(self):
        name = self.name
        start_date = self.start_date
        df = DataReader(name, data_source='yahoo', start=start_date, end=datetime.now())
        df['Daily_avg'] = (df['Open'] + df['High'] + df['Low']) / 3
        print("Stock Information")
        print(df)
        print("\n\n\n")
        print("**********" * 11)
        # Comparing the daily average vs the closing amount.
        plt.figure(figsize=(16,8))
        plt.title('Average Price = (Opening Price + High + Low)/3', fontsize=20)
        plt.plot(df['Daily_avg'])
        plt.plot(df['Close'])
        plt.xlabel('Date', fontsize=20)
        plt.ylabel('Average Price USD ($)', fontsize=20)
        plt.legend(['Daily Average Price', 'Daily Closing Price'], loc='upper left')
        print(plt.show())
        #Create a new dataframe with only the 'Close column
        data = df.filter(['Daily_avg'])
        target = df.filter(['Close'])
        dataset = data.values #Convert the dataframe to a numpy array
        targetset = target.values #Convert the dataframe to a numpy array
        training_data_len = int(np.ceil( len(dataset) * .8 )) #Get the number of rows to train the model on
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = scaler.fit_transform(dataset)
        scaled_target = scaler.fit_transform(targetset)
        train_data = scaled_data[0:int(training_data_len), :] #Create the scaled training data set
        train_target = scaled_target[0:int(training_data_len), :] #Create the scaled training data set
        x_train = [] #Splitting the data into x_train and y_train data sets
        y_train = []
        for i in range(60, len(train_data)):
            x_train.append(train_data[i-60:i, 0])
            y_train.append(train_target[i, 0])
        x_train, y_train = np.array(x_train), np.array(y_train) # Convert the x_train and y_train to numpy arrays 
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1)) #reshape the data
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape= (x_train.shape[1], 1))) #Buildning LSTM model
        model.add(LSTM(50, return_sequences= False))
        model.add(Dense(25))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error') # Compile the model
        model.fit(x_train, y_train, batch_size=1, epochs=1) #Train the model
        test_data = scaled_data[training_data_len - 60: , :]
        x_test = [] #Create the data sets x_test and y_test
        y_test = targetset[training_data_len:, :]
        for i in range(60, len(test_data)):
            x_test.append(test_data[i-60:i, 0])
        x_test = np.array(x_test) # Convert the data to a numpy array
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 )) # Reshape the data
        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions) # Get the models predicted price values 
        mse = np.mean(((predictions - y_test) ** 2))
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        print("\n\n\n")
        print("**********" * 11)
        print(f'Mean Square Error: {mse:.2f}')
        print(f'Root Mean Square Error: {rmse:.2f}')
        print(f'R2 Score: {r2:.2f}')
        print(f'Mean Absolute Error: {mae:.2f}')
        print('\n\n\n')
        print("**********" * 11)
        train = target[:training_data_len] # Plot the data
        valid = target[training_data_len:]
        valid['Predictions'] = predictions
        valid['DailyAvg'] = dataset[training_data_len:, :]
        plt.figure(figsize=(16,8)) # Visualize the data
        plt.title('Historical Closing Price + Predictions', fontsize=20)
        plt.xlabel('Date', fontsize=20)
        plt.ylabel('Closing Price $USD', fontsize=20)
        plt.plot(train['Close'])
        plt.plot(valid[['Close', 'Predictions']])
        plt.legend(['Train', 'Target Closing Price', 'Predictions for Closing Price'], loc='upper left')
        plt.show()
        print(valid)
