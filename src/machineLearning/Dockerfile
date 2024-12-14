FROM python:3.11-slim

WORKDIR /machineLearning

# Copies the trainer code to the docker image.
COPY . /machineLearning

RUN pip install -r requirements.txt

# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python", "-m", "vertex-ai.knnClustering"]