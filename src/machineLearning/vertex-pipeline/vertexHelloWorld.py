import json
from typing import NamedTuple

from google.cloud import aiplatform
from kfp import compiler, dsl
from kfp.dsl import component

PROJECT_ID = "kallollearnmlai"  # @param {type:"string"}
LOCATION = "us-central1"  # @param {type:"string"}
BUCKET_URI = f"gs://kallollearnmlai-vertex-pipeline-bucket"  # @param {type:"string"}
PIPELINE_ROOT = "{}/pipeline_root/control".format(BUCKET_URI)


aiplatform.init(project=PROJECT_ID, location=LOCATION, staging_bucket=BUCKET_URI)


# API service endpoint
API_ENDPOINT = f"{LOCATION}-aiplatform.googleapis.com"
# Pipelne root dir
PIPELINE_ROOT = f"{BUCKET_URI}/pipeline_root/intro"

@component(base_image="python:3.9")
def hello_world(text: str) -> str:
    print(text)
    return text

@component(packages_to_install=["google-cloud-storage"])
def two_outputs(
    text: str,
) -> NamedTuple(
    "Outputs",[
        ("output_one", str),  # Return parameters
        ("output_two", str),
    ],
):
    # the import is not actually used for this simple example, but the import
    # is successful, as it was included in the `packages_to_install` list.
    from google.cloud import storage  # noqa: F401

    o1 = f"output one from text: {text}"
    o2 = f"output two from text: {text}"
    print("output one: {}; output_two: {}".format(o1, o2))
    return (o1, o2)

@component
def consumer(text1: str, text2: str, text3: str) -> str:
    print(f"text1: {text1}; text2: {text2}; text3: {text3}")
    return f"text1: {text1}; text2: {text2}; text3: {text3}"

@dsl.pipeline(
    name="intro-pipeline-unique",
    description="A simple intro pipeline",
    pipeline_root=PIPELINE_ROOT,
)
def pipeline(text: str = "hi there"):
    hw_task = hello_world(text=text)
    two_outputs_task = two_outputs(text=text)
    consumer_task = consumer(  # noqa: F841
        text1=hw_task.output,
        text2=two_outputs_task.outputs["output_one"],
        text3=two_outputs_task.outputs["output_two"],
    )

compiler.Compiler().compile(pipeline_func=pipeline, package_path="intro_pipeline.json")

DISPLAY_NAME = "intro_pipeline_job_unique"

job = aiplatform.PipelineJob(
    display_name=DISPLAY_NAME,
    template_path="intro_pipeline.json",
    pipeline_root=PIPELINE_ROOT,
)

job.run()