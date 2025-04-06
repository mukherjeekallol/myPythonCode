# this program is for learning KubeFlow
# I want to run kubeflow in my local machine

import kfp
import kfp.dsl as dsl
import kfp.gcp as gcp

from typing import NamedTuple

@dsl.pipeline(
    name="intro-pipeline",
    description="A simple intro pipeline",
    pipeline_root="gs://kallollearnmlai-vertex-pipeline-bucket/pipeline_root/intro",
)
def pipeline(text: str = "hi there"):
    hw_task = hello_world(text=text)    # noqa: F841
    two_outputs_task = two_outputs(text=text)
    consumer_task = consumer(  # noqa: F841
        text1=hw_task.output,
        text2=two_outputs_task.outputs["output_one"],
        text3=two_outputs_task.outputs["output_two"],
    )

def hello_world(text: str) -> str:
    print(text)
    return text

def two_outputs(text: str) -> NamedTuple(
    "Outputs",
    [
        ("output_one", str),  # Return parameters
        ("output_two", str),
    ],
):
    o1 = f"output one from text: {text}"
    o2 = f"output two from text: {text}"
    print("output one: {}; output_two: {}".format(o1, o2))
    return (o1, o2)

def consumer(text1: str, text2: str, text3: str) -> str:
    print(f"text1: {text1}; text2: {text2}; text3: {text3}")
    return f"text1: {text1}; text2: {text2}; text3: {text3}"

if __name__ == "__main__":
    compiler.Compiler().compile(pipeline_func=pipeline, package_path="intro_pipeline.yaml")

    DISPLAY_NAME = "intro_pipeline_job"

    job = kfp.Client().create_run_from_pipeline_func(pipeline, arguments={})

    kfp.Client().delete_run(job.run_id)
