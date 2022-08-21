# importing required libraries
import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Load Data Function
    Function to load input data file, merge, and output combined data into 1 dataframe
    
    Inputs:
    messages_filepath - path to disaster messages csv file
    categories_filepath - path to categories csv file
    
    Returns:
    df - merged dataset of messages and categories data
    """

    # reading in csv file and converting to dataframe
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # merging both messages and categories dataframe into 1 dataframe
    df = pd.merge(messages, categories, on='id')
    return df


def clean_data(df):
    """
    Clean Data Function
    Function that takes as input the merged dataset dataframe object from load_data function.
    Purpose is to make 36 additional columns, one for each possible message outut classification.
    The 36 category columns will be our MultiOutput Y prediction vector for our ML model.
    For each input message X, this function will provide a value of 1 if a given Y column indicates 'yes' for that category, a 0 will be 'no' for that category.
    
    Inputs:
    df - DataFrame object returned from load_data function
    
    Returns:
    df - cleaned dataframe object (refer to description above regarding details of cleaned dataframe)
    """
    # expanding categories column from 1 into 36 separate categoreis, with ';' as the delimiter
    categories = df.categories.str.split(';', expand=True)
    row = categories.iloc[0]

    # setting column header name to the string vlaue from the first position up to the second to last position
    category_colnames = row.apply(lambda x: x[0:-2])
    categories.columns = category_colnames

    # setting each of the 36 categorical columns to 1 or 0 values
    for column in categories:
        categories[column] = categories[column].astype(str).str.get(-1)
        categories[column] = categories[column].astype(float)

    # removing original categories column
    df.drop(columns='categories', inplace=True)

    # adding cleaned categories column to existing dataframe
    df = pd.concat([df, categories], axis=1)

    # removing duplicate values in the dataframe
    df.drop_duplicates(inplace=True)

    # removing null values in the dataframe
    df.dropna(subset=['related', 'request', 'offer',
       'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
       'security', 'military', 'child_alone', 'water', 'food', 'shelter',
       'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid',
       'infrastructure_related', 'transport', 'buildings', 'electricity',
       'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
       'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
       'other_weather', 'direct_report'], inplace=True)

    # in the 'related' column, replacing all values of 2 and setting to 1
    df.related.replace(to_replace=2, value=1, inplace=True)
    return df
    

def save_data(df, database_filename):
    # creating SQLite database 'DisasterResponse.db', and posting dataframe df to 'messages'
    engine = create_engine('sqlite:///' + str(database_filename))
    df.to_sql('messages', engine, index=False, if_exists='replace')


def main():
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
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
