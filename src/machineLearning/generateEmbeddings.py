from google.cloud import bigquery
from transformers import AutoTokenizer, AutoModel
import torch

# Initialize BigQuery client
client = bigquery.Client()

# Define your source and destination tables
source_table_id = 'bigquery-public-data.samples.shakespeare'
destination_table_id = 'kallollearnmlai.mltraining.embeddings_table'

# Load data from BigQuery table
query = f'SELECT word, word_count, corpus, corpus_date FROM `{source_table_id}` LIMIT 10'
query_job = client.query(query)
rows = query_job.result()

# Load pre-trained transformer model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def generate_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()
    return embeddings

# Generate embeddings
embeddings_list = []
for row in rows:
    word = row['word']
    word_count = row['word_count']
    corpus = row['corpus']
    corpus_date = row['corpus_date']
    embeddings = generate_embeddings(word)
    embeddings_list.append((word, word_count, corpus, corpus_date, embeddings.tolist()))

# Define schema for destination table
schema = [
    bigquery.SchemaField('word', 'STRING'),
    bigquery.SchemaField('word_count', 'INTEGER'),
    bigquery.SchemaField('corpus', 'STRING'),
    bigquery.SchemaField('corpus_date', 'INTEGER'),
    bigquery.SchemaField('embeddings', 'FLOAT', mode='REPEATED')
]

# Create destination table if it doesn't exist
table = bigquery.Table(destination_table_id, schema=schema)
table = client.create_table(table, exists_ok=True)

# Insert embeddings back into BigQuery table
rows_to_insert = [
    {
        'word': word,
        'word_count': word_count,
        'corpus': corpus,
        'corpus_date': corpus_date,
        'embeddings': embeddings
    }
    for word, word_count, corpus, corpus_date, embeddings in embeddings_list
]

errors = client.insert_rows_json(destination_table_id, rows_to_insert)
if errors == []:
    print('New rows have been added.')
else:
    print('Encountered errors while inserting rows: {}'.format(errors))
