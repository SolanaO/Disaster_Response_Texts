from distressapp import app

import os
import re
import json
import pickle
import joblib

import numpy as np
import pandas as pd

#from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify

import plotly
from plotly.graph_objs import (
    Bar,
    Pie,
    Scatter,
)

from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
)

from sqlalchemy import create_engine

# import the module to create the graphs
from essentials import tokenize


print('Loading database ...')

# load full data
engine = create_engine('sqlite:///data/DisasterResponse.db')
df = pd.read_sql_table('MessagesTable', engine)

category_names = list(df.columns[2:])


print('Loading model ...')

# load model
model = joblib.load('models/classifier.pkl')

print('Loaded the model.')


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    """
    Prepare plotly graphs and corresponding layouts to dump to json
    for html front-end.
    INPUT:
      None
    OUTPUT:
      None
    """
    # graph 1: genre counts
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # add column to record the message length
    df['message_length'] = df['message'].str.len()

    # graph 2: distribution by category
    # sum along each column to get the number of each category
    df_col_counts = df.sum(axis = 0, skipna = True)
    # get the message categories in decreasing order of frequencies
    df_col_counts_ordered = df_col_counts[2:-1].sort_values(ascending=True)
    # get the list of categories in decreasing order of frequencies
    category_order = df_col_counts_ordered.index

    # graph 3: labels and categories
    # work with the numerical (binary) columns only
    df_num = df[df.columns[2:-1]]
    # sum along each row
    df_row_counts = df_num.sum(axis = 1, skipna = True)
    # create a panda series with the row value counts
    row_counts = pd.Series(df_row_counts.values)

    # create a column that records the row count for labels
    df['label_count'] = row_counts
    # print the frequency of each value count
    row_sum_freq = row_counts.value_counts().sort_values(ascending=False)

    # graph 4: messages with at most 12 labels
    # create a dataframe that contains the average message length for
    # messages with the same number of labels
    df_counts =df.groupby('label_count',as_index=False)['message_length'].mean()
    # add the frequencies for each value count
    df_counts['freqs'] = row_sum_freq
    # sort the entries by frequency
    df_counts.sort_values(by='freqs', ascending=False)

     # TODO: Below is an example - modify to create your own visuals
    graphs = [
        # graph 1: pie
        {
            'data': [
                Pie(
                    labels=genre_names,
                    values=genre_counts,
                    textinfo='label+value',
                    hole=0.3,
                )
            ],

            'layout': {
                'title': 'Distribution of Messages by Genres',
                'text': 'Genres'
                }
        },

    # graph 2: horizontal bar
       {
            'data': [
                Bar(
                    x=df_col_counts_ordered,
                    y=category_order,
                    orientation = 'h',
                    marker=dict(
                        color='#1f77b4',
                        line=dict(color='#d62728', width=2)
                        )
                    )
                ],

            'layout': {
                        'title': 'Distribution of Messages by Category',
                        'autosize': False,
                        'width': 1000,
                        'height': 880,
                        'margin': {
                                'l':120,
                                'r': 20,
                                'b': 80,
                                't': 80,
                                'pad': 3
                                }

                        }
    },

   # graph 3: vertical bar
      {
           'data': [
               Bar(
                   x=row_sum_freq.index,
                   y=row_sum_freq,
                   marker=dict(
                       color='#1f77b4',
                       line=dict(color='#d62728', width=2)
                       )
                   )
               ],

           'layout': {
                       'title': 'Distribution of Messages by the Labels Number',
                       'xaxis': {
                            'title': 'Number of categories per message',
                            },
                        'yaxis': {
                            'title': 'Number of messages',
                        },
                        'autosize': False,
                        'width': 800,
                        'height': 600,
                        'margin': {
                               'l':120,
                               'r': 20,
                               'b': 80,
                               't': 80,
                               'pad': 3
                               }

                       }
   },

   # graph 4: scatter

   {
            'data': [
                Scatter(
                    x=df_counts.freqs[:13],
                    y=df_counts.label_count[:13],
                    mode='markers',
                    marker=dict(
                        color='#006400',
                        size=df_counts.message_length[:13]/3
                    )
                )
            ],

            'layout': {
                        'title': 'Messages with at Most 12 Labels, with Average Message Length Indicator',
                        'xaxis': {
                            'title': 'Number of messages',
                         },
                         'yaxis': {
                            'title': 'Number of categories per message',
                     },
                        'autosize': False,
                        'width': 800,
                        'height': 600,
                        'margin': {
                                'l':120,
                                'r': 20,
                                'b': 80,
                                't': 80,
                                'pad': 3
                            }

                    }
},
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    """
    Save user input in query.
    INPUT:
      None
    OUTPUT:
      None
    """

    query = request.args.get('query', '')
    print(query)

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # this will render the go.html file
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )

# uncomment when running locally with python distress.py
#def main():
    #app.run(host='0.0.0.0', port=3001, debug=True)


#if __name__ == '__main__':
    #main()
