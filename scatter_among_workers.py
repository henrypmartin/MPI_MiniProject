from mpi4py import MPI # Importing mpi4py package from MPI module
import pandas as pd
# Importing Numpy
import numpy as np
# Importing sqrt function from the Math
from math import sqrt
# Importing Decimal, ROUND_HALF_UP functions from the decimal package
from decimal import Decimal, ROUND_HALF_UP
import time

from sklearn.model_selection import train_test_split
import pickle

#Define a function to split the data frame into features and target
def split_features_and_target(df):
  # Separate features and target
  X = pwr_df.drop(columns=['PE'])  # Features
  y = pwr_df['PE']  # Target

  return (X, y)

# Define a function to calculate the error (root mean squared error)
def calculate_rmse(y_actual, y_pred):
  # Compute the squared differences between true and predicted values
  squared_errors = (y_actual - y_pred) ** 2

  # Compute the mean of squared errors
  mean_squared_error = np.mean(squared_errors)

  # Compute the square root of mean squared error to get RMSE
  rmse = np.sqrt(mean_squared_error)

  return rmse

# fucntion to predict the values
def predict(x, intercept, coefficients):
  '''
  y = b_0 + b_1*x + ... + b_i*x_i
  '''
  # Add a column of ones to the features for the intercept term
  features_with_intercept = np.c_[np.ones((x.shape[0], 1)), x]
  predictions = np.dot(features_with_intercept, np.concatenate([[intercept], coefficients]))

  return predictions

def compute_coefficients(X, y):

    # Compute X transpose
    X_transpose = np.transpose(X)

    # Compute X transpose times X
    X_transpose_X = np.dot(X_transpose, X)

    # Compute the inverse of X transpose times X
    X_transpose_X_inv = np.linalg.inv(X_transpose_X)

    # Compute X transpose times y
    X_transpose_y = np.dot(X_transpose, y)

    # Compute the coefficients
    coefficients = np.dot(X_transpose_X_inv, X_transpose_y)

    return coefficients

# defining a fit function
def fit(features, target):
    # Extract features (X) and target (y) from the DataFrame
    X = features.values
    y = target.values

    # Add a column of ones to the feature matrix X for the intercept term
    X_with_intercept = np.c_[np.ones((X.shape[0], 1)), X]

    coefficients = compute_coefficients(X_with_intercept, y)

    # Extract intercept and coefficients
    intercept = coefficients[0]
    coeffs = coefficients[1:]

    return intercept, coeffs

# define function to divide data for each worker
def dividing_data(x_train, y_train, size_of_workers):

    # Size of the slice
    slice_for_each_worker = int(Decimal(x_train.shape[0]/size_of_workers).quantize(Decimal('1.'), rounding = ROUND_HALF_UP))

    #sliced x train and y train data and picked it for serialization
    sliced_train = [pickle.dumps((x_train[i:i+slice_for_each_worker], y_train[i:i+slice_for_each_worker])) for i in range(0, len(x_train), slice_for_each_worker)]

    return sliced_train

#define function to invoke MPI commands and process features and target and calculate RMSE in distributed manner
def process_data_using_MPI(features, target):

    # Creating a Communicator
    comm = MPI.COMM_WORLD
    #number of the process running the code
    rank = comm.Get_rank()
    # total number of processes running
    size = comm.Get_size()
    # master process
    sliced_train = None
    
    #if master i:e in MPI usually process 0 or rank 0 is termed as master
    #master is responsible for slicing the data to be processed by all the workers including master itself
    if rank == 0:
        
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = 0.30, random_state = 42)

        sliced_train = dividing_data(X_train, y_train, size)        

    #wait until all workers reached this point
    print(f"Rank {rank}: waiting at barrier")
    comm.Barrier()

    #scatter data to all workers once they are ready
    print(f"Rank {rank}: outside barrier")
    recvbuf = comm.scatter(sliced_train, root=0)

    #data received by each worker including the master
    rec_x_train, rec_y_train = pickle.loads(recvbuf)

    #each worker fits, predict and calculates RMSE on the sliced data
    intercept, coefficients = fit(rec_x_train, rec_y_train)

    y_train_pred = predict(rec_x_train, intercept, coefficients)

    train_rmse = calculate_rmse(rec_y_train, y_train_pred)

    print(f'Root Mean Squared Error for training data in {rank} is {train_rmse}')

    value = np.array(train_rmse,'d')

    value_sum = np.array(0.0,'d')

    #take a sum of all the RMSE's calculated by each worker as a MPI reduce operation
    comm.Reduce(value, value_sum, op=MPI.SUM, root=0)

    #master will take the mean of RMSE's from all workers as the final RMSE for the entire data
    if rank == 0:
      print(f'Sum of Root Mean Squared Error is {value_sum}')
      mean = value_sum / size
      print(f'Mean of Root Mean Squared Error is {mean}')


FILENAME = "/content/PowerPlantData.csv" # File path
pwr_df = pd.read_csv(FILENAME)

pwr_df = pwr_df.drop_duplicates(keep='first')

X = pwr_df.drop(columns=['PE'])  # Features
y = pwr_df['PE']  # Target

X_standardized_df = X.apply(lambda x: (x - x.mean())/x.std())

# Convert back to DataFrame
X_scaled_df = pd.DataFrame(X_standardized_df, columns=X.columns)

# Concatenate standardized features with the target column
pwr_standardized_df = pd.concat([X_scaled_df, y], axis=1)

features, target = split_features_and_target(pwr_standardized_df)

print(f'Invoking MPi process')
process_data_using_MPI(features, target)
