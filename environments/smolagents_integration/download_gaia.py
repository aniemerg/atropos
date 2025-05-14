#!/usr/bin/env python3
"""
Script to download the GAIA benchmark dataset from HuggingFace.
"""

import argparse
import logging
import os
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Download GAIA benchmark dataset")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/gaia",
        help="Directory to store GAIA dataset",
    )
    parser.add_argument(
        "--use-raw",
        action="store_true",
        help="Use raw dataset instead of annotated version",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=["validation", "test"],
        help="Dataset split to download",
    )
    parser.add_argument(
        "--create-example",
        action="store_true",
        help="Create example dataset if download fails",
    )
    return parser.parse_args()


def download_dataset(
    output_dir, use_raw=False, split="validation", create_example=False
):
    """Download the GAIA dataset from HuggingFace."""
    try:
        # Import required libraries
        import datasets
        from huggingface_hub import snapshot_download

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"Downloading GAIA {'raw' if use_raw else 'annotated'} dataset...")

        # Download the dataset (which is gated/private)
        try:
            repo_id = "gaia-benchmark/GAIA" if use_raw else "smolagents/GAIA-annotated"
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=output_dir,
                ignore_patterns=[".gitattributes", "README.md"],
            )
            logger.info(f"Dataset downloaded to {output_dir}")
        except Exception as e:
            logger.error(f"Error downloading dataset: {e}")
            if not create_example:
                return False
            logger.info("Creating example dataset instead...")

            # Create GAIA.py with example data
            create_example_dataset(output_dir, split)
            return True

        # Create preprocessing function
        with open(os.path.join(output_dir, "GAIA.py"), "w") as f:
            f.write(
                f'''
"""GAIA benchmark dataset."""

import os
import datasets
import json

_DESCRIPTION = "GAIA benchmark dataset for evaluating code generation abilities"
_CITATION = ""

class GAIA(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("2023.0.0")
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="2023_all",
            version=VERSION,
            description="Full GAIA 2023 benchmark",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features({{
                "Question": datasets.Value("string"),
                "Final answer": datasets.Value("string"),
                "Level": datasets.Value("string"),
                "task_id": datasets.Value("string"),
                "file_name": datasets.Value("string"),
            }}),
            homepage="https://huggingface.co/datasets/gaia-benchmark/GAIA",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={{"split": "validation"}},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={{"split": "test"}},
            ),
        ]

    def _generate_examples(self, split):
        """Read the specified data split."""
        try:
            metadata_path = os.path.join(os.path.dirname(__file__), split, "metadata.jsonl")
            with open(metadata_path, "r") as f:
                for i, line in enumerate(f):
                    example = json.loads(line)
                    yield i, example
        except FileNotFoundError:
            # Fallback to example data if file doesn't exist
            for i, example in enumerate(EXAMPLE_DATA.get(split, [])):
                yield i, example

# Example data in case download fails
EXAMPLE_DATA = {{
    "validation": [
        {{
            "Question": "I need to filter a list of strings, keeping only those that contain at least one uppercase letter. Write a Python function called filter_uppercase that takes a list of strings as input and returns a new list containing only the strings with at least one uppercase letter.",
            "Final answer": "def filter_uppercase(strings):\\n    return [s for s in strings if any(c.isupper() for c in s)]",
            "Level": "Programming",
            "task_id": "GAIA2023_P0003",
            "file_name": ""
        }},
        {{
            "Question": "Solve the equation: 3x + 7 = 22",
            "Final answer": "x = 5",
            "Level": "Math",
            "task_id": "GAIA2023_M0001",
            "file_name": ""
        }}
    ],
    "test": []
}}
'''
            )

        logger.info(
            f"GAIA dataset setup complete. Created GAIA.py module in {output_dir}"
        )
        return True

    except ImportError:
        logger.error(
            "Required packages not installed. Run: pip install datasets huggingface_hub"
        )
        return False

    except Exception as e:
        logger.error(f"Error setting up GAIA dataset: {e}")
        return False


def create_example_dataset(output_dir, split="validation"):
    """Create an example dataset with a few tasks."""
    # Create the directory structure
    os.makedirs(os.path.join(output_dir, split), exist_ok=True)

    # Create a sample metadata.jsonl
    with open(os.path.join(output_dir, split, "metadata.jsonl"), "w") as f:
        f.write(
            '{"Question": "I need to filter a list of strings, keeping only those that contain at least one uppercase letter. Write a Python function called filter_uppercase that takes a list of strings as input and returns a new list containing only the strings with at least one uppercase letter.", "Final answer": "def filter_uppercase(strings):\\n    return [s for s in strings if any(c.isupper() for c in s)]", "Level": "Programming", "task_id": "GAIA2023_P0003", "file_name": ""}\n'
        )
        f.write(
            '{"Question": "Solve the equation: 3x + 7 = 22", "Final answer": "x = 5", "Level": "Math", "task_id": "GAIA2023_M0001", "file_name": ""}\n'
        )

    logger.info(f"Created example dataset in {output_dir}")
    return True


def main():
    args = parse_args()
    success = download_dataset(
        args.output_dir,
        use_raw=args.use_raw,
        split=args.split,
        create_example=args.create_example,
    )

    if success:
        logger.info("GAIA dataset setup completed successfully")
        logger.info(
            f"You can now run: python -m environments.smolagents_integration.run_gaia_single_task --task-id GAIA2023_P0003 --model-name gpt-4o --base-url https://api.openai.com/v1"
        )
    else:
        logger.error("GAIA dataset setup failed")
        logger.info("Still creating example dataset for testing...")
        create_example_dataset(args.output_dir, args.split)
        exit(1)


if __name__ == "__main__":
    main()
