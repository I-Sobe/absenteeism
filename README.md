# Employee Absenteeism dataset
## a Machine Learning project

## Dataset
The raw dataset(Absenteeism_data) consists of absenteeism data from a company with 700 instances of absenteeism over the monitored period. It consists of 700 rows and 12 columns. The columns consists of ID, Reason for absence, Transportation Expense, DIstance to Work, Age, Daily Work Load Average, Body MAss Index, Body Mass Index, Education, Children, Pets, Absenteeism Time in Hours. The aim is to create a model to predict further absenteeism in the work place. 

## Environment
This project was written in python and mySQL and visualization in Tableau. the following packages need to be installed in other to effectively run:
* import numpy
* import pandas
* import datetime
* import pickle
* from sklearn.preprocessing import StandardScaler
* from sklearn.base import BaseEstimator, TransformerMixin
* pymysql

## Context
For this Machine learning project, i preprocessed the Absenteeism_data using the 'preprocessing.ipynb' notebook this outputs 'preprocessed.csv'.
This 'preprocessed.csv' file was used in a 'machine_learning.ipynb' operation using linear regression to predict absenteeism. This process created the 'model' and 'scaler' files.
Next, A python module of the model was created in the 'absenteeism_module' file.
This python module was used in the 'absenteeism_exercise_integration.ipynb' to predict the absenteeism in the 'absenteeism_new_data.csv' file.
A connection was created to mywork bench, where a smiliar predicted table was created (db file) and data from python workbook exported to MySQL.
This table was exported from MySQL to Tableau as 'absenteeism_prediction.csv', where visualizations and data relationships was carried out.

## Key Insights fro presentation
In Tableau, i focused on:
* Age vs probability
* Reason vs Probability
* Transportation Expense and Children vs Probability