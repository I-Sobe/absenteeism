# import all libraries needed
import numpy as np
import pandas as pd
import pickle
import datetime as dt
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

# the custom scaler class
class CustomScaler(BaseEstimator, TransformerMixin):

    def __init__(self,columns,copy=True,with_mean=True,with_std=True):
        self.scaler = StandardScaler(copy,with_mean,with_std)
        self.columns = columns
        self.mean_ = None
        self.var_ = None

    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns], y)
        self.mean_ = np.mean(X[self.columns])
        self.var_ = np.var(X[self.columns])
        return self

    def transform(self, X, y=None, copy=None):
        init_col_order = X.columns
        X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]), columns=self.columns)
        X_not_scaled = X.loc[:,~X.columns.isin(self.columns)]
        return pd.concat([X_not_scaled, X_scaled], axis=1)[init_col_order]


# create the special class that we are going to use from here on to predict new data
# This is a modularization of the 'machine_learning.ipynb' and 'preprocessing.ipynb' files
class absenteeism_model():

        def __init__(self, model_file, scaler_file):
            # read the 'model' and 'scaler'files which were saved
            with open('model', 'rb') as model_file, open('scaler', 'rb') as scaler_file:
                self.reg = pickle.load(model_file)
                self.scaler = pickle.load(scaler_file)
                self.data = None

        # take a data file (*.csv) and preprocess it in the same way in the preprocessing.ipynb file
        def load_and_clean_data(self, data_file):

            # import the data
            df = pd.read_csv(data_file, delimiter=',')
            # store the data in a new variable for later use
            self.df_with_predictions = df.copy()
            # drop the 'ID' column
            df = df.drop(['ID'], axis =1)
            # to preserve the code created in the previous section, add a column with 'NaN' strings
            # df['Absenteeism Time in Hours'] = 0

            # create a separate dataframe, containing dummy values for ALL available reasons
            reason_columns = pd.get_dummies(df['Reason for Absence'], drop_first = True)

            # spilt reason_columns into 4 types
            reason_type_1 = reason_columns.loc[:, '1':'14'].max(axis=1)
            reason_type_2 = reason_columns.loc[:, '15':'17'].max(axis=1)
            reason_type_3 = reason_columns.loc[:, '18':'21'].max(axis=1)
            reason_type_4 = reason_columns.loc[:, '22':].max(axis=1)

            # to avoid multicollinearity, drop the 'reason for absence' column from df
            df = df.drop(['Reason for Absence'], axis = 1)

            # concatenate df and the 4 types of reason for absence
            df = pd.concat([df, reason_type_1, reason_type_2, reason_type_3, reason_type_4], axis = 1)

            df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

            # Extracting the 'Year','Month' and 'day' from Date Column.
            df['Month'] = df['Date'].dt.month_name()
            df['Day'] = df['Date'].dt.day_name()

            # converting Month and Day columns to intergers, this will help in StandardScaler
            df['Month'] = df['Month'].replace(['January', 'February', 'March', 'April', 'May',
                                'June', 'July', 'August', 'September', 'October', 'November', 'December'], 
                                ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'])
            df['Day'] = df['Day'].replace(['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'], 
                                ['1', '2', '3', '4', '5', '6', '7'])

            # dropping the 'Date' Column
            df = df.drop(['Date'], axis = 1)
            
            column_change = ['Transportation Expense', 'Distance to Work', 'Age', 'Daily Work Load Average', 'Body Mass Index', 
            'Education', 'Children', 'Pets', 'Reason_1', 'Reason_2', 'Reason_3', 'Reason_4', 'Month', 'Day']
            
            df.columns = column_change

            # reordering the columns
            column_names_reodered = ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4', 'Month', 'Day', 'Transportation Expense',
                        'Distance to Work', 'Age', 'Daily Work Load Average', 'Body Mass Index',
                        'Education', 'Children', 'Pets']
            df = df[column_names_reodered]
                  
           
            # using the 'map' method to reassign the values in the Education column
            df['Education'] = df['Education'].map({1:0, 2:1, 3:1, 4:1})

            # replace the NaN values
            df = df.fillna(value=0) 

            # drop the variables we decide we don't need
            df = df.drop(['Day', 'Daily Work Load Average', 'Distance to Work'], axis = 1)

            # we have included this line of code if you want to call the 'preprocessed data'
            self.preprocessed_data = df.copy()

            # we need this line so we can use it in the next functions
            self.data = self.scaler.transform(df)

        # a function which outputs the probability of a data point to be 1
        def predicted_probability(self):
            if (self.data is not None):
                pred_outputs = self.reg.predict(self.data)
                return pred_outputs

        # predict the outputs and the probabilities and add columns with these values at the end of new data
        def predicted_outputs(self):
            if (self.data is not None):
                self.preprocessed_data['Probability'] = self.reg.predict_proba(self.data)[:,1]
                self.preprocessed_data['Prediction'] = self.reg.predict(self.data)
                return self.preprocessed_data
