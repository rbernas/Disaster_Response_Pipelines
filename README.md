# Disaster_Response_Pipelines
Project analyzing data from Figure Eight to build a model for an API that classifies disaster messages using a machine learning pipeline that categorizes events so they can be sent to an appropriate disaster relief agency.

## Installation <a name="installation"></a>

There should be no necessary libraries to run the code here beyond the Anaconda distribution of Python.  The code should run with no issues using Python versions 3.*.

## Project Motivation<a name="motivation"></a>

The dataset for this project consists of real disaster messages collected by Figure Eight. Each message is categorized into 1 more of 36 categories (e.g., Aid Related, Weather Related, Search And Rescue, etc.)
The goal is to assist aid workers in contacting the agency that would most help people in need of help. 
 
## File Descriptions <a name="files"></a>

There are 3 main steps for this project, each with an associated file. These steps for the data pipeline are listed below in order:

Data Pipeline

1) <b>ETL (Extract, Transform, Load) process</b>
Loading of the raw dataset, cleaning, and storing into a SQLite database. 
Associated file: <em>workspace/data/process_data.py</em>

2) <b>Machine Learning Pipeline</b>
The dataset consists of ~26,000 disaster messages. 90% of the data is used for the training set, 10% for the test set. A machine learning pipeline is created using NLTK, scikit-learn's Pipeline and GridSearchCV (to
optimize hyperparameters) to outut a machine learning model that uses the message column to predict classifications for 36 categories. This model is exported as a pickle file, to be used in the final step.
Associated file: <em>workspace/models/train_classifier.py</em>
 
3) <b>Flask App</b>
Flask is used to displays results in a web app, along with 3 plots showing distributions for the data set in a bar char and pie charts.
Associated file: <em>workspace/app/run.py</em>
 
## How to Run <a name="run"></a>

Run the following commands in the project's root directory and in below order to setup the database and ML model.

1) <b>ETL</b><br>
	Run <em>python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db</em>
	
2) <b>Machine Learning Pipeline</b><br>
	Run <em>python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl</em>

3) <b>Flask App</b><br>
	Run <em> python run.py</em>

4)	Go to http://0.0.0.0:3001/

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Thank you to Figure Eight for the data and to Udacity for the guidance and help.  

