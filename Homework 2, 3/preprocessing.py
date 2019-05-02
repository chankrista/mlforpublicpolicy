import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score as accuracy
import numpy as np
import seaborn as sb
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

FEATURES = {'MonthlyIncome': 4, 'DebtRatio': 4, 'RevolvingUtilizationOfUnsecuredLines': 4,
            'age': 4, 'NumberOfTime30-59DaysPastDueNotWorse': 4, 
            'NumberOfOpenCreditLinesAndLoans': 4, 'NumberOfTimes90DaysLate': 4,
            'NumberRealEstateLoansOrLines': 4, 'NumberOfTime60-89DaysPastDueNotWorse': 4,
            'NumberOfDependents': 4, 'zipcode_1': 2, 'zipcode_60601': 2, 'zipcode_60618': 2,
            'zipcode_60625': 2, 'zipcode_60629': 2, 'zipcode_60637': 2, 'zipcode_60644': 2}
TARGET = 'SeriousDlqin2yrs'

CLASSIFIERS = {'Random Forest': RandomForestClassifier(n_estimators=50, n_jobs=-1),
    'Ada Boost': AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200),
    'Logistic Regression': LogisticRegression(penalty='l1', C=1e5),
    'SVM': svm.SVC(kernel='linear', probability=True, random_state=0),
    'Decision Tree': DecisionTreeClassifier(),
    'KNN': KNeighborsClassifier(n_neighbors=3),
    'BAG': BaggingClassifier(DecisionTreeClassifier(max_depth=1))
            }

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

def fit_clfs(X_train, y_train, 
	clfs=['Random Forest', 'Ada Boost', 'Logistic Regression', 'SVM',\
	'Decision Tree', 'KNN', 'BAG']):
	'''
	Given training sets of features and predictors and a list of names classifiers
	to use, returns a dictionary of the classifiers' names as keys and fitted models
	as corresponding values.
	'''
	fitted_clfs = {}
	for clf in clfs:
		classifier = CLASSIFIERS.get(clf)
		model = classifier.fit(X_train, y_train)
		fitted_clfs['clf'] = model
	return model

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

def accuracy_dt(X_train, y_train):
	for d in range(1, 7):
    dt = pp.build_dt(X_train, y_train, depth = d)
    	for t in range(1, 7):
        	accuracy = pp.evaluate(X_test, y_test, dt, t / 10)
        	print(
        		"With a depth of " + str(d) + " and threshold of " + str(t / 10), \
        		" the model has an accuracy of " + str(accuracy))

def accuracy_knn(X_train, y_train):
	knn = KNeighborsClassifier(
		n_neighbors=10, metric='minkowski', metric_params={'p': 3})
	knn.fit(X_train, y_train)
	pred_scores = knn.predict_proba(x_test)[:1]
	for t in range(1, 7):
		threshold = t / 10
    	pred_label = [1 if x[1]>threshold else 0 for x in pred_scores]
    	print(
    		"With threshold {}, the total number predicted is {}, the accuracy is {:.2f}".format(
        	threshold, sum(pred_label), accuracy(pred_label,y_test)))