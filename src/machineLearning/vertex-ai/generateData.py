import pandas as pd
from sklearn.datasets import load_iris
from google.cloud import bigquery
from google.oauth2 import service_account

# Load the Iris dataset
iris = load_iris()
iris_data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
# iris_data.columns = [col.replace(' ', '_') for col in iris_data.columns]
iris_data.columns = ['sepal_length','sepal_width','petal_length','petal_width']
iris_data['target'] = iris.target

# Initialize BigQuery client
# credentials = service_account.Credentials.from_service_account_file('path/to/your/service_account.json')
client = bigquery.Client()

# Define table details
table_id = 'kallollearnmlai.mltraining.iris_table'

# Create BigQuery table schema
schema = [
    bigquery.SchemaField('sepal_length', 'FLOAT'),
    bigquery.SchemaField('sepal_width', 'FLOAT'),
    bigquery.SchemaField('petal_length', 'FLOAT'),
    bigquery.SchemaField('petal_width', 'FLOAT'),
    bigquery.SchemaField('target', 'INTEGER'),
]

# Create BigQuery table
table = bigquery.Table(table_id, schema=schema)
# table = client.create_table(table)  # API request

print(iris_data)
# Load data into BigQuery
job = client.load_table_from_dataframe(iris_data, table_id,)  # Make an API request
job.result()  # Wait for the job to complete

print('Loaded {} rows into {}.'.format(job.output_rows, table_id))
