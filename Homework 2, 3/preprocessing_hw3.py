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

FEATURES = {'school_metro_rural': 2, 'school_metro_suburban': 2,
            'school_metro_urban': 2, 'school_charter_t': 2,
            'school_magnet_t': 2, 'primary_focus_area_Applied Learning': 2, 
            'primary_focus_area_Health & Sports': 2, 'primary_focus_area_History & Civics': 2,
            'primary_focus_area_Literacy & Language': 2, 'primary_focus_area_Math & Science': 2,
            'primary_focus_area_Music & The Arts': 2, 'primary_focus_area_Special Needs': 2,
            'resource_type_Books': 2, 'resource_type_Other': 2, 'resource_type_Other': 2,
            'resource_type_Supplies': 2, 'resource_type_Technology': 2, 
            'resource_type_Trips': 2, 'resource_type_Visitors': 2,'poverty_level_high poverty': 2,
            'poverty_level_low poverty': 2, 'poverty_level_moderate poverty': 2,
            'grade_level_Grades 3-5': 2, 'grade_level_Grades 6-8': 2,
            'grade_level_Grades 9-12': 2, 'grade_level_Grades PreK-2': 2,
            'total_price_including_optional_support': 4,
            'students_reached': 4, 'eligible_double_your_impact_match_t': 2}
TARGET = 'over_60'

CLASSIFIERS = {'Random Forest': RandomForestClassifier(),
    'Ada Boost': AdaBoostClassifier(DecisionTreeClassifier(max_depth=1) ),
    'Logistic Regression': LogisticRegression(penalty='l1'),
    'SVM': LinearSVC(random_state=0),
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
        test = df.loc[(df[date_col] >= lb) & (df[date_col] < ub)]
        train_test_sets.append((train, test))
    return train_test_sets

def eval_metrics(test, fitted_clfs, thresholds):
    X_test = test[FEATURES.keys()]
    y_test = test[TARGET]
    for clf_name, clf in fitted_clfs.items():
        if clf_name == 'SVM':
            predicted_y = clf.decision_function(X_test)
        else:
            predicted_y = clf.predict_proba(X_test)[:,1]
        calc_threshold = lambda x, y: 0 if x < y else 1
        print(clf_name)
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
            '''
            print("At threshold " + str(threshold))
            print(classification_report(y_test, predictions))
            '''
        print("ROC AUC Score %.2f" % roc_auc_score(y_test, predicted_y))           
        plot_precision_recall_n(y_test, predicted_y, clf_name, "show")
        print("----------")
        print()

def threshold_cutoff(train_test, fitted_clfs, pop_pct):
    X_test = train_test[1][FEATURES.keys()]
    y_test = train_test[1][TARGET]
    cutoffs = {}
    for clf_name, clf in fitted_clfs.items():
        if clf_name == 'SVM':
            predicted_y = clf.decision_function(X_test)
        else:
            predicted_y = clf.predict_proba(X_test)[:,1]
        predicted_y = np.sort(predicted_y)
        predicted_y = list(predicted_y[::-1])
        print(len(predicted_y))
        top_count = round(pop_pct * len(predicted_y))
        cutoffs[clf_name] = predicted_y[top_count]
    print(cutoffs)
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

def fit_clfs(train_test, 
    clfs=['Random Forest', 'Ada Boost', 'Logistic Regression', 'SVM',\
    'Decision Tree', 'KNN', 'BAG']):
    '''
    Given training sets of features and predictors and a list of names classifiers
    to use, returns a dictionary of the classifiers' names as keys and fitted models
    as corresponding values.
    '''
    train = train_test[0]
    X_train = train[FEATURES.keys()]
    y_train = train[TARGET]
    fitted_clfs = {}
    for clf in clfs:
        classifier = CLASSIFIERS.get(clf)
        model = classifier.fit(X_train, y_train)
        fitted_clfs[clf] = model
    return fitted_clfs

def eval_test_sets(train_test_sets, thresholds, clf_list=['Random Forest', 'Ada Boost', 'Logistic Regression', 'SVM',\
    'Decision Tree', 'KNN', 'BAG']):
    for i, train_test in enumerate(train_test_sets):
        print("Training and test set " + str(i + 1))
        fitted_clfs = fit_clfs(train_test, clfs=clf_list)
        eval_metrics(train_test[1], fitted_clfs, thresholds)

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
