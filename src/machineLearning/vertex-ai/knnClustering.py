import pandas as pd
from sklearn.cluster import KMeans
from google.cloud import bigquery
from google.oauth2 import service_account
import joblib
from google.cloud import storage
import logging
# The pip install command is: pip install google-cloud-storage

# Initialize BigQuery client
credentials = service_account.Credentials.from_service_account_file('vertex-ai\kallollearnmlai-key.json') 
bq_client = bigquery.Client(credentials=credentials, project='kallollearnmlai')
# client = bigquery.Client(project='kallollearnmlai')

# Define input and output table details
input_table_id = 'kallollearnmlai.mltraining.iris_table'
output_table_id = 'kallollearnmlai.mltraining.iris_clusters'

# Read data from BigQuery
query = f'SELECT * FROM `{input_table_id}`'
iris_data = bq_client.query(query).to_dataframe()


logging.basicConfig(level=logging.INFO)
logging.info(f'iris_data: {iris_data.head(5).to_string()}')
print(f'iris_data: {iris_data.head(5).to_string()}')
# Run K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
iris_data['cluster'] = kmeans.fit_predict(iris_data.drop(columns=['target']))

# Write the results back to BigQuery
job = bq_client.load_table_from_dataframe(iris_data, output_table_id)  # Make an API request
job.result()  # Wait for the job to complete

print(f'Loaded {job.output_rows} rows into {output_table_id}.')

# Save the model to a file
joblib.dump(kmeans, 'kmeans_iris_model.joblib')

# Initialize GCS client
storage_client = storage.Client(credentials=credentials,project='kallollearnmlai')

# Define bucket and destination path
bucket_name = 'kallollearnmlai-model-bucket'
destination_blob_name = 'models/kmeans_iris_model.joblib'

# Upload model to GCS
bucket = storage_client.bucket(bucket_name)
blob = bucket.blob(destination_blob_name)
blob.upload_from_filename('kmeans_iris_model.joblib')

print(f'Model uploaded to gs://{bucket_name}/{destination_blob_name}.')
