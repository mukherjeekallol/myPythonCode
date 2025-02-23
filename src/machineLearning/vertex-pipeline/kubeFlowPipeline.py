
import json

import google.auth

from google.cloud import storage
from google.cloud import aiplatform
from kfp import compiler, dsl
from kfp.dsl import component

PROJECT_ID = "kallollearnmlai"  # @param {type:"string"}
LOCATION = "us-central1"  # @param {type:"string"}
BUCKET_URI = f"gs://kallollearnmlai-vertex-pipeline-bucket"  # @param {type:"string"}
PIPELINE_ROOT = "{}/pipeline_root/control".format(BUCKET_URI)
aiplatform.init(project=PROJECT_ID, staging_bucket=BUCKET_URI)

@component
def args_generator_op() -> str:
    import json

    return json.dumps(
        [{"cats": "1", "dogs": "2"}, {"cats": "10", "dogs": "20"}],
        sort_keys=True,
    )


@component
def print_op(msg: str):
    print(msg)


@component
def flip_coin_op() -> str:
    """Flip a coin and return heads or tails randomly."""
    import random

    result = "heads" if random.randint(0, 1) == 0 else "tails"
    return result


@dsl.pipeline(
    name="control",
    pipeline_root=PIPELINE_ROOT,
)
def pipeline(
    json_string: str = json.dumps(
        [
            {
                "snakes": "anaconda",
                "lizards": "anole",
                "bunnies": [{"cottontail": "bugs"}, {"cottontail": "thumper"}],
            },
            {
                "snakes": "cobra",
                "lizards": "gecko",
                "bunnies": [{"cottontail": "roger"}],
            },
            {
                "snakes": "boa",
                "lizards": "iguana",
                "bunnies": [
                    {"cottontail": "fluffy"},
                    {"fuzzy_lop": "petunia", "cottontail": "peter"},
                ],
            },
        ],
        sort_keys=True,
    )
):

    flip1 = flip_coin_op()

    with dsl.Condition(
        flip1.output != "no-such-result", name="alwaystrue"
    ):  # always true

        args_generator = args_generator_op()
        with dsl.ParallelFor(args_generator.output) as item:
            print_op(msg=json_string)

            with dsl.Condition(flip1.output == "heads", name="heads"):
                print_op(msg=item.cats)

            with dsl.Condition(flip1.output == "tails", name="tails"):
                print_op(msg=item.dogs)

    with dsl.ParallelFor(json_string) as item:
        with dsl.Condition(item.snakes == "boa", name="snakes"):
            print_op(msg=item.snakes)
            print_op(msg=item.lizards)
            print_op(msg=item.bunnies)

    # it is possible to access sub-items
    with dsl.ParallelFor(json_string) as item:
        with dsl.ParallelFor(item.bunnies) as item_bunnies:
            print_op(msg=item_bunnies.cottontail)

compiler.Compiler().compile(
    pipeline_func=pipeline, package_path="control_pipeline.yaml"
)

DISPLAY_NAME = "control"

job = aiplatform.PipelineJob(
    display_name=DISPLAY_NAME,
    template_path="control_pipeline.yaml",
    pipeline_root=PIPELINE_ROOT,
)

job.run()