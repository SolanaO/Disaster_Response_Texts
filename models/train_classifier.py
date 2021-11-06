# general libraries
import sys
import os
import re
from time import time
import pickle
import joblib

# linear algebra and numerical libraries
import numpy as np
import pandas as pd

# database manipulation packages
from sqlalchemy import create_engine

# nlp packages
#from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# data processing packages
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfTransformer,
    TfidfVectorizer
)

# machine learning packages
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import AdaBoostClassifier


from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
)

# metrics evaluation packages
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    make_scorer
)
#import local modules
from essentials import tokenize

# create a dataframe for the evaluation metrics of the model
def report_to_dataframe(y_true, y_pred, category_names):
    """
    Function to save the sklearn classification report as a
    pandas dataframe.
    INPUT:
        y_true (pd.Dataframe) - the true labels
        y_pred (np.array) - the predicted labels
        categories_names (list) - the list of predicted labels
    OUTPUT:
        reports (list) - a list of two dataframes

    where:
        reports[0] (pd.DataFrame) - contains precision, recall, f1-score, accuracy for each label
        reports[1] (pd.DataFrame) - contains overall averages of the scores of precision, recall, f1-score, accuracy.
    """
    # save classification report in dictionary form
    report_dict = classification_report(y_true, y_pred, target_names=category_names, zero_division=0, output_dict=True)
    # save the report as a datafarme
    report_df = pd.DataFrame.from_dict(report_dict).T.round(2)
    # drop the last 4 rows as they represent averages
    report = report_df[:-4]
    # save the averages in a separate dataframe
    report_avg = report_df[-4:]

    # convert the index into a column, rename it to Category
    report=report.reset_index().rename({'index':'category'},
                                       axis = 'columns')
    # add the individual labels accuracies
    accuracies=[]
    for i in range(len(category_names)):
        accuracies.append(accuracy_score(y_true.iloc[:, i].values, y_pred[:, i]))
    accuracies = pd.Series(accuracies).round(2)

    report.insert(4, 'accuracy', accuracies)

    # create a list of reports
    reports = [report, report_avg]
    return reports

def load_data(database_filepath):
    """
    Creates and SQL engine and creates Pandas dataframe data from database.
    Splits the dataframe into two dataframes, features and labels.

    INPUT:
        database_filename (str) - path and name for the database
    OUTPUT:
        X (dataframe) - contains the text messages
        Y (dataframe) - contains the pre-processed labels
        categories (list) - labels, categories names
    """

    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql('SELECT * FROM MessagesTable', engine)

    # list the categories
    category_names = list(df.columns[2:])

    # separate the features from the target variables
    X = df['message']
    Y = df[df.columns[2:]]

    return X, Y, category_names


def build_model():
    """
    Build the model as a pipeline.

    INPUT:
        none
    OUTPUT:
        pipeline model
    """

    # create pipeline
    pipeline  = Pipeline([
    #('tokenizer', Tokenizer()),
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    #('clf', MultiOutputClassifier(XGBClassifier(eval_metric='logloss',
                                                #use_label_encoder=False))),
    ('clf', MultiOutputClassifier(AdaBoostClassifier())),
    ])

    parameters = {
    'vect__min_df': [5],
    'vect__max_features': [None],
    'clf__estimator__n_estimators': [50],
    'clf__estimator__base_estimator': [None],
    'clf__estimator__learning_rate': [0.1, 1.0],
}

    scorer = make_scorer(f1_score, average='micro')

    # create the grid search pipeline for the specified parameters
    grid = GridSearchCV(pipeline, param_grid=parameters, scoring=scorer, cv=3, verbose=2)

    return grid


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluates the performance of the model and prints the results.

    INPUT:
        model (pipeline.Pipeline): contains the text processing and the classifier
        X_test (pd.Series): text messages in the test dataset
        Y_test (pd.Dataframe): pre-processed labels in the test dataset
        category_names (list): labels
    """
    # predict the labels
    Y_pred = model.predict(X_test)

    # print the predictions
    reports = report_to_dataframe(Y_test, Y_pred, category_names)
    print('The performance metrics for each label sorted by f1-score:')
    print(reports[0].sort_values(by='f1-score', ascending=False))
    print()
    print("The performance metrics overall averages")
    print(reports[1])


def save_model(model, model_filepath):
    """
    Save the model in a joblib file.

    INPUT:
        model (pipeline.Pipeline): the model to be saved
        model_filepath (str): file destination for joblib model
    """
    #pickle.dump(model, open(model_filepath, 'wb'))
    joblib.dump(model, open(model_filepath, 'wb')) #, compress=9)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
