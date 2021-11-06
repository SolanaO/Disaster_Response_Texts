# import packages
import sys
import os

import numpy as np
import pandas as pd

#import pycld2 as cld2

from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    Reads data from two csv files and loads them as pandas
    dataframes, which are merged on the 'id' column.

    INPUT:
        messages_filepath (str): path string to csv file
        categories_filepath (str): path string to csv file
    OUTPUT:
        df (pandas dataframe); resulting dataframe of unprocessed data
    '''

    # read csv files into pandas dataframes
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # merge dataframes on common column
    df = messages.merge(categories, on='id')

    return df


def clean_data(df):
    '''
    Pre-process the data. The following actions are performed:

    -- The 'categories' entries are split, expanded in individual binary columns.
    -- Rows with value  '2' in 'related' category are removed.
    -- Columns 'categories', 'id', 'original', 'child_alone' are droped.
    -- Only the English translations of the messages are retained.
    -- Duplicates are removed.

    INPUT:
        df (pandas dataframe): name of dataframe, must have 'categories' column
    OUTPUT:
        df (pandas dataframe):  dataframe in which the 'categories' column is replaced with 36 binary columns
    '''

    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(';', expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories
    category_colnames = row.apply(lambda x: x[:-2])

    # rename the columns of `categories` with the cleaned names
    categories.columns = category_colnames


    for column in categories:
        # keep only the last character of each string (the 1 or 0)
        categories[column] = categories[column].apply(lambda x: x[-1])

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories], axis=1)

    # drop rows with value 2 in the 'related' column
    df = df[df.related !=2]

    # drop the redundant columns from `df`
    df.drop(['id',  'original', 'categories', 'child_alone'], axis=1, inplace=True)

    # retain only messages that are in English
    #df = df[df['message'].apply(lambda x: cld2.detect(x)[2][0][1]) == 'en']

    # drop duplicates
    df = df.drop_duplicates()

    return df


def save_data(df, database_filename):
    '''
    Saves the clean dataframe df into a sqlite database.

    INPUT:
        df (pandas dataframe): name of dataframe
    OUTPUT:
        database_filename (str) - SQL Alchemy database filename
    '''

    # create SQL-Alchemy engine
    engine=create_engine(f'sqlite:///{database_filename}')
    # save the dataframe to 'MessagesTable' table
    df.to_sql('MessagesTable', engine, if_exists='replace', index=False)


def test_db(database_filename):
    '''
    Tests the saved database by uploading it into a pandas dataframe and printing out its first rows.
    INPUT:
      database_filename (str): name of sqlite database
    '''

    # create SQL-Alchemy engine
    engine=create_engine(f'sqlite:///{database_filename}')
    # avoid freezing while process
    connection = engine.raw_connection()
    # read the table information
    data_db = pd.read_sql("SELECT * FROM MessagesTable", con=connection)
    # print the first two rows of the table
    print(data_db.head(2))

def main():
    '''
    Process the data and save it to a database.
    '''

    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

        print('Testing database...\n    DATABASE: {}'.format(database_filepath))
        test_db(database_filepath)

        print('Read from database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
