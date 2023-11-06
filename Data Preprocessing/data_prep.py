import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler

def standardization(df, cols):
    scaler = StandardScaler()
    df[cols] = scaler.fit_transform(df[cols])

def data_prep():
# Read txt
    train_df = pd.read_table('traininingdata.txt')
    test_df = pd.read_table('testdata.txt')

    # Find 3 kinds of attributes
    numeric_var = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
    binary_var = ['default', 'housing', 'loan', 'y']
    categorical_var = ['job', 'marital', 'education', 'contact', 'month', 'poutcome']

    # Replace the pdays -1 with maximized pdays
    max_pdays_train = max(train_df['pdays'])
    max_pdays_test = max(test_df['pdays'])
    max_pdays = max(max_pdays_train, max_pdays_test)
    train_df['pdays'] = train_df['pdays'].replace({-1: max_pdays})
    test_df['pdays'] = test_df['pdays'].replace({-1: max_pdays})

    # Replace labels Yes and No with 1 and 0
    train_df = train_df.replace({'yes': 1, 'no': 0})
    test_df = test_df.replace({'yes': 1, 'no': 0})

    # Split data to train and test parts
    x_train = train_df.iloc[:, :-1]
    y_train = train_df.iloc[:, -1]
    x_test = test_df.iloc[:, :-1]
    y_test = test_df.iloc[:, -1]

    # Transfrom categorical attributes with one-hot encoding 
    x_train = pd.get_dummies(x_train)
    x_test = pd.get_dummies(x_test)
    # Find the columns that in x_train but not in x_test
    missing_cols = set(x_train.columns) - set(x_test.columns)
    for c in missing_cols:
        x_test[c] = 0
    # Ensure the column order of x_train and x_test are the same
    x_test = x_test[x_train.columns]

    # Standarize numerical attributes
    standardization(x_train, numeric_var)
    standardization(x_test, numeric_var)

    return x_train, y_train, x_test, y_test

# if __name__ == "__main__":
#     x_train, y_train, x_test, y_test = data_prep()
#     print(x_train.shape)




