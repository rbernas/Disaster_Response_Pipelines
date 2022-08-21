# importing required libraries
import sys
import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

def load_data(database_filepath):
    """
    Load Data Function
    Function to load data from SQLite database and return dataframe
    
    Inputs:
    database_filepath - path to database
    
    Returns:
    X - Input feature 'message' vector
    Y = Output feature 'categories' vector
    df.columns - column category headings for Y
    """
    engine = create_engine('sqlite:////home/workspace/' + str(database_filepath))
    df = pd.read_sql('SELECT * FROM messages', engine)
    X = df['message']
    Y = df.iloc[:,4:]
    return X, Y, df.columns[4:]


def tokenize(text):
    """
    Tokenize Function
    Function that takes in message text and return a lemmatized word list, with the text lowercased, and whitespace removed
    
    Inputs:
    text - raw message text
    
    Returns:
    clean_tokens - message list with text lemmatized, stopwords removed, lowercased, and whitespace removed
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words("english")
    
    clean_tokens = [lemmatizer.lemmatize(word).lower().strip() for word in tokens if word not in stop_words]
    
    return clean_tokens


def build_model():
    """
    Build Model Function
    Function that builds the ML pipeline, returning ML model
    
    Inputs:
    No input
    
    Returns:
    returns ML model (cv), using K-Nearest Neighbours algorithm for MultiOuput Classification, with optimized parameter for number of neighbors using GridSearchCV
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(KNeighborsClassifier()))
    ])
       
    pipeline.get_params() 
    parameters = {
    'clf__estimator__n_neighbors': [5, 10]
    }
    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1, verbose=2)
    
    return pipeline
    

def evaluate_model(model, X_test, y_test, category_names):
    """
    Evaluate Model Function
    Function that tests the ML model on unseen data
    
    Inputs:
    model - ML model created from build_model function
    X_test - data from 'messages' column (X) in dataframe that was not used in training of ML model
    y_test - data from the 36 categories column (y) in dataframe that was not used in training of ML model
    category_names - used in the classifcation report for column headers
    
    Returns:
    y_pred - category predictions for X_test
    classification report - report showing precision, recall, F1 score performance for the test dataset
    """
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names = category_names))
    
    
def save_model(model, model_filepath):
    """
    Save Model Function
    Function that saves the ML model and exports as pickle file
    
    Inputs:
    model - ML model created from build_model function
    model_filepath - path to model
    
    Returns:
    classifier.pkl - pickle file of our ML model that can be used for future unseen data via the web app
    """
    import pickle
    with open('classifier.pkl', 'wb') as f:
        pickle.dump(model, f)


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
