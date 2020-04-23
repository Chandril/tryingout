
#Step 1: Importing the libraries
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

#Step 2: Loading the data file
dataset = pd.read_csv('C:/Users/ghosh/Downloads/rent.csv')

#Step 3: Extracting the column and row numbers
X = dataset.iloc[:, :3]

y = dataset.iloc[:, -1]

#Step 4: Build the model
regressor_model = LinearRegression()

#Fitting model with trainig data
regressor_model.fit(X, y)

# Saving model to in pickle format
pickle.dump(regressor_model, open('reg_model.pkl','wb'))


