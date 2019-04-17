import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score as accuracy
import numpy as np
import seaborn as sb
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

FEATURES = {'MonthlyIncome': 4, 'DebtRatio': 4, 'RevolvingUtilizationOfUnsecuredLines': 4,
            'age': 4, 'NumberOfTime30-59DaysPastDueNotWorse': 4, 
            'NumberOfOpenCreditLinesAndLoans': 4, 'NumberOfTimes90DaysLate': 4,
            'NumberRealEstateLoansOrLines': 4, 'NumberOfTime60-89DaysPastDueNotWorse': 4,
            'NumberOfDependents': 4, 'zipcode_1': 2, 'zipcode_60601': 2, 'zipcode_60618': 2,
            'zipcode_60625': 2, 'zipcode_60629': 2, 'zipcode_60637': 2, 'zipcode_60644': 2}
TARGET = 'SeriousDlqin2yrs'

def pre_process(filename):
    '''
    Reads a csv file as a dataframe and fills in missing
    values with the mean of that column.
    '''
    df = pd.read_csv(filename)
    df.fillna(df.mean(), inplace=True)
    return df

def discretize(df):
    '''
    Takes a dataframe and discretizes continuous variables by
    converting columns within the values of the BOUNDS 
    dictionary into categorical variables.
    '''
    for colname, bins in FEATURES.items():
        label_list = []
        for index in range(bins - 1):
            label_list.append("quantile " + str(index + 1))
        df[colname] = pd.qcut(
            df[colname], bins, duplicates='drop', labels=label_list)
            
def create_dummies(df, colnames=None):
    '''
    Takes a dataframe and list of column name and converts categorical
    variables into dummy/indicator variables. If no columns are
    specified, all categorical columns will be converted.
    '''
    if colnames:
        return pd.get_dummies(df, columns=colnames)
    else:
        return pd.get_dummies(df)

def split_df(df):
    '''
    Given a dataframe, uses TARGET and FEATUREs to split
    df into X and y, training and testing sets.
    '''
    y = df[TARGET]
    X = df[FEATURES.keys()]
    return train_test_split(
        X, y, test_size=0.33, random_state=42)

def build_classifier(X_train, y_train, depth=4):
    '''
    Given a training set and optional depth value,
    creates and returns a fitted decision tree.
    '''
    dt = DecisionTreeClassifier(max_depth=depth, random_state=99)
    dt.fit(X_train, y_train)
    return dt    

def evaluate(X_test, y_test, dt, threshold):
    '''
    Given a testing set, decision tree, and threshold,
    returns the accuracy of the decision tree based on
    that threshold.
    '''
    predicted_y = dt.predict_proba(X_test)[:,1]
    calc_threshold = lambda x, y: 0 if x < y else 1
    predictions = np.array(
        [calc_threshold(score, threshold) for score in predicted_y])
    return accuracy(predictions, y_test)
