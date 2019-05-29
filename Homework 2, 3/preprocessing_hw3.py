import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from mlfunctions import plot_precision_recall_n
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score as accuracy,\
    precision_recall_fscore_support, classification_report, roc_auc_score, \
    precision_recall_curve, confusion_matrix
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, \
    BaggingClassifier

CONT_FEATURES = ['total_price_including_optional_support',
            'students_reached']
CAT_FEATURES = ['school_metro',
                'school_charter',
                'school_magnet',
                'primary_focus_area',
                'teacher_prefix',
                'resource_type',
                'poverty_level',
                'grade_level',
                'eligible_double_your_impact_match']
TARGET = 'over_60'

CLASSIFIERS = {'Random Forest': RandomForestClassifier(),
    'Ada Boost': AdaBoostClassifier(DecisionTreeClassifier(max_depth=1) ),
    'Logistic Regression': LogisticRegression(penalty='l1'),
    'SVM': LinearSVC(random_state=0),
    'Decision Tree': DecisionTreeClassifier(),
    'KNN': KNeighborsClassifier(n_neighbors=3),
    'BAG': BaggingClassifier(DecisionTreeClassifier(max_depth=1))
            }
PARAM_GRID = { 
    'Random Forest':{'n_estimators': [10, 20], 'max_depth': [5, 10], 'max_features': ['sqrt','log2'],'min_samples_split': [2,10], 'n_jobs':[-1]},
    'Logistic Regression': { 'penalty': ['l1','l2'], 'C': [0.001,0.1,1]},
    'Ada Boost': { 'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10]},
    'Decision Tree': {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10], 'max_features': [None,'sqrt','log2'],'min_samples_split': [2,5]},
    'SVM' :{'penalty': ['l2']},
    'KNN' :{'n_neighbors': [1,5,10],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']},
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
    Given a dataframe, uses TARGET and FEATURES to split
    df into X and y, training and testing sets.
    '''
    y = df[TARGET]
    X = df[FEATURES.keys()]
    return train_test_split(
        X, y, test_size=0.33, random_state=42)

def time_split_df(df, date_col, test_sets):
    '''
    Given a dataframe, creates training and testing sets using 
    a time series split on the specified date column and with the
    specified number of test sets. Returns a list of tuples
    each with a train and test df.
    '''
    time_period = (df[date_col].max() - df[date_col].min()) / (test_sets + 1)
    train_test_sets = []
    for i in range(1, test_sets + 1):
        lb = df[date_col].min() + (i * time_period)
        ub = df[date_col].min() + ((i + 1) * time_period)
        train = df.loc[(df[date_col] < lb)]
        test = df.loc[(df[date_col] >= lb + np.timedelta64(60, 'D')) \
            & (df[date_col] < ub)]
        train_test_sets.append((train, test))
    return train_test_sets

def clean_features(project):
    project['students_reached'] = \
        project['students_reached'].fillna(
            project['students_reached'].mean())
    project = create_dummies(
        project, colnames=CAT_FEATURES)
    return project

def process_sets(train_test_sets):
    '''
    Given the train, test sets, cleans features in all dataframes and returns
    a tuple of the new train, test sets and a list of target features.
    '''
    old_cols = train_test_sets[0][0].columns
    for ind, train_test_set in enumerate(train_test_sets):
        train = train_test_set[0]
        train = clean_features(train)
        test = train_test_set[1]
        test = clean_features(test)
        train_test_sets[ind] = (train, test)
    new_cols = train_test_sets[0][0].columns
    features = [col for col in new_cols if col not in old_cols]
    features = features + CONT_FEATURES
    return train_test_sets, features

def eval_metrics(test, features, fitted_clfs, thresholds):
    X_test = test[features]
    y_test = test[TARGET]
    for clf_name, clf in fitted_clfs.items():
        if 'SVM' in clf_name:
            predicted_y = clf.decision_function(X_test)
        else:
            predicted_y = clf.predict_proba(X_test)[:,1]
        calc_threshold = lambda x, y: 0 if x < y else 1
        print(clf_name)
        print(clf.get_params())
        for threshold in thresholds:
            predictions = np.array(
                [calc_threshold(score, threshold) for score in predicted_y])
            c = confusion_matrix(y_test, predictions)
            true_negatives, false_positive, false_negatives, true_positives = c.ravel()
            print()
            print("At threshold " + str(threshold) + ":")
            print()
            print("True negatives: %.2f" % true_negatives)
            print("False positives: %.2f" % false_positive)
            print("False negatives: %.2f" % false_negatives)
            print("True positives: %.2f" % true_positives)
            print()
            print("    accuracy %.2f" % calculate_accuracy_at_threshold(
                predictions, y_test))
            print("    precision %.2f" % calculate_precision_at_threshold(
                predictions, y_test))
            print("    recall %.2f" % calculate_recall_at_threshold(
                predictions, y_test))
            print("    f1 %.2f" % calculate_f1_at_threshold(
                predictions, y_test))

        print("ROC AUC Score %.2f" % roc_auc_score(y_test, predicted_y))           
        plot_precision_recall_n(y_test, predicted_y, clf_name, "show")
        print("----------")
        print()

def threshold_cutoff(train_test, features, fitted_clfs, pop_pct):
    X_test = train_test[1][features]
    y_test = train_test[1][TARGET]
    cutoffs = {}
    for clf_name, clf in fitted_clfs.items():
        if 'SVM' in clf_name:
            predicted_y = clf.decision_function(X_test)
        else:
            predicted_y = clf.predict_proba(X_test)[:,1]
        predicted_y = np.sort(predicted_y)
        predicted_y = list(predicted_y[::-1])
        top_count = round(pop_pct * len(predicted_y))
        cutoffs[clf_name] = predicted_y[top_count]
    return cutoffs

def calculate_accuracy_at_threshold(predicted_scores, y_test):
    true_negatives, false_positive, false_negatives, true_positives = \
        confusion_matrix(y_test, predicted_scores).ravel()
    return 1.0 * (true_positives + true_negatives) / (true_negatives + false_positive + false_negatives + true_positives)

def calculate_precision_at_threshold(predicted_scores, y_test):
    _, false_positive, _, true_positives = \
        confusion_matrix(y_test, predicted_scores).ravel()
    return 1.0 * true_positives / (false_positive + true_positives)

def calculate_recall_at_threshold(predicted_scores, y_test):
    _, _, false_negatives, true_positives = \
        confusion_matrix(y_test, predicted_scores).ravel()
    return 1.0 * true_positives / (false_negatives + true_positives)

def calculate_f1_at_threshold(predicted_scores, y_test):
    precision = calculate_precision_at_threshold(predicted_scores, y_test)
    recall = calculate_recall_at_threshold(predicted_scores, y_test)
    return 2 * (precision * recall) / (precision + recall)

def fit_clfs(train_test, features,
    clfs=['Random Forest', 'Ada Boost', 'Logistic Regression', 'SVM',\
    'Decision Tree', 'KNN', 'BAG'], param_loop=False):
    '''
    Given training sets of features and predictors and a list of names 
    classifiers to use, returns a dictionary of the classifiers' names as
    keys and fitted models as corresponding values.
    '''

    train = train_test[0]
    X_train = train[features]
    y_train = train[TARGET]
    fitted_clfs = {}
    for clf in clfs:
        classifier = CLASSIFIERS.get(clf)
        if param_loop:
            params = PARAM_GRID.get(clf)
            if params:
                for param, val_list in params.items():
                    for val in val_list:
                        classifier = classifier.set_params(**{param: val})
                        model = classifier.fit(X_train, y_train)
                        fitted_clfs[clf + " " +str(param) + ": " + str(val)] = model
        else:
            model = classifier.fit(X_train, y_train)
            fitted_clfs[clf] = model
    return fitted_clfs

def eval_test_sets(train_test_sets, features, thresholds, clf_list=['Random Forest', 'Ada Boost', 'Logistic Regression', 'SVM',\
    'Decision Tree', 'KNN', 'BAG'], loop_params=False):
    for i, train_test in enumerate(train_test_sets):
        print("Training and test set " + str(i + 1))
        fitted_clfs = fit_clfs(train_test, features, clfs=clf_list, param_loop=loop_params)
        eval_metrics(train_test[1], features, fitted_clfs, thresholds)

def plot_precision_recall(predicted_scores, true_labels):
    precision, recall, thresholds = precision_recall_curve(true_labels, predicted_scores)
    plt.plot(recall, precision, marker='.')
    plt.show()

def eval_accuracy(X_test, y_test, clf, threshold):
    '''
    Given a testing set, decision tree, and threshold,
    returns the accuracy of the decision tree based on
    that threshold.
    '''
    predicted_y = clf.predict_proba(X_test)[:,1]
    calc_threshold = lambda x, y: 0 if x < y else 1
    predictions = np.array(
        [calc_threshold(score, threshold) for score in predicted_y])
    return accuracy(predictions, y_test)

def accuracy_dt(X_train, y_train):
    for d in range(1, 7):
        dt = build_dt(X_train, y_train, depth = d)
        for t in range(1, 7):
            accuracy = eval_accuracy(X_test, y_test, dt, t / 10)
            print(
                "With a depth of " + str(d) + " and threshold of " + str(t / 10), \
                " the model has an accuracy of " + str(accuracy))
            print()
