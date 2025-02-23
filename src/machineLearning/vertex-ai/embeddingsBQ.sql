WITH embeddings AS (
  SELECT 
    word,
    ML.PREDICT(MODEL `kallollearnmlai.mltraining.bert_model`, 
      (SELECT AS STRUCT * FROM UNNEST([word]))) AS embeddings
  FROM (
    SELECT DISTINCT word 
    FROM `kallollearnmlai.mltraining.iris_table`
  )
)
SELECT * FROM embeddings;
