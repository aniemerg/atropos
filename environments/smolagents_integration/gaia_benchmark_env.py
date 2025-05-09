import asyncio
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field

# Import SmolaGents components
from smolagents import CodeAgent, Tool

from atroposlib.envs.base import BaseEnv, BaseEnvConfig, Item, ScoredDataGroup
from atroposlib.envs.server_handling.openai_server import OpenaiConfig, OpenAIServer
from atroposlib.envs.server_handling.server_manager import ServerManager

# Import the AtroposServerModel
from environments.smolagents_integration.atropos_smolagents_integration import (
    AtroposServerModel,
)


class GAIABenchmarkConfig(BaseEnvConfig):
    """Configuration for the GAIA benchmark environment."""

    dataset_path: str = Field(
        default="data/gaia", description="Path to the GAIA benchmark data"
    )
    split: str = Field(
        default="validation", description="Dataset split to use (validation, test)"
    )
    use_chat_completion: bool = Field(
        default=False, description="Whether to use chat completion API"
    )
    max_steps: int = Field(
        default=12, description="Maximum number of steps for the agent"
    )


class GAIABenchmarkEnv(BaseEnv):
    """
    Environment for running GAIA benchmark evaluations with SmolaGents CodeAgent.

    This environment bridges Atropos and SmolaGents by:
    1. Creating an AtroposServerModel to wrap Atropos servers
    2. Initializing a CodeAgent with this model
    3. Providing benchmark problems from the GAIA dataset
    """

    name = "gaia_benchmark"
    env_config_cls = GAIABenchmarkConfig

    def __init__(
        self,
        config: GAIABenchmarkConfig,
        server_configs: Union[List[OpenaiConfig], OpenaiConfig],
        slurm=False,
        testing=False,
    ):
        # Initialize base class but skip tokenizer initialization to avoid issues with missing models
        # We'll customize the initialization process by skipping actual tokenization since it's not
        # needed for the GAIA benchmark with OpenAI models
        
        # First check if the model name indicates this is an OpenAI API model
        is_openai_model = False
        if isinstance(server_configs, list) and len(server_configs) > 0:
            model_name = server_configs[0].model_name
            if any(name in model_name for name in ["gpt-4", "gpt-3.5", "claude", "gemini"]):
                is_openai_model = True
        elif hasattr(server_configs, "model_name"):
            model_name = server_configs.model_name
            if any(name in model_name for name in ["gpt-4", "gpt-3.5", "claude", "gemini"]):
                is_openai_model = True
        
        # Call the parent initialization
        super().__init__(config, server_configs, slurm, testing)
        
        # If using an OpenAI model, override the tokenizer with a dummy one to avoid HF errors
        if is_openai_model:
            import logging
            logging.info(f"Using OpenAI model {model_name}, tokenizer initialization skipped.")
            # Set a flag to indicate we're not using the tokenizer
            self._using_tokenizer = False

        # Initialize the dataset and tools
        self.current_index = 0
        self.examples = []
        self.agent = None

    async def setup(self):
        """Set up the environment, load the dataset, and initialize the agent."""
        logging.info("Setting up GAIA benchmark environment")

        # Load the dataset
        try:
            import datasets
            import pandas as pd

            self.dataset = datasets.load_dataset(
                f"{self.config.dataset_path}/GAIA.py",
                name="2023_all",
                split=self.config.split,
            )

            # Convert to standard format
            self.examples = [
                {
                    "question": example["Question"],
                    "true_answer": example["Final answer"],
                    "task": example["Level"],
                    "task_id": (
                        example["task_id"] if "task_id" in example else f"task_{i}"
                    ),
                    "file_name": (
                        f"{self.config.dataset_path}/{self.config.split}/{example['file_name']}"
                        if example["file_name"]
                        else ""
                    ),
                }
                for i, example in enumerate(self.dataset)
            ]

            logging.info(
                f"Loaded {len(self.examples)} examples from GAIA {self.config.split} set"
            )
        except Exception as e:
            logging.error(f"Error loading GAIA dataset: {e}")
            # Create an empty list if dataset loading fails
            self.examples = []

        # Create AtroposServerModel that wraps the Atropos server
        self.model = AtroposServerModel(
            server=self.server,
            use_chat_completion=self.config.use_chat_completion,
            model_id="atropos-gaia",
        )

        # Create tools for the agent
        self.tools = self._create_tools()

        # Initialize CodeAgent
        self.agent = CodeAgent(
            tools=self.tools,
            model=self.model,
            max_steps=self.config.max_steps,
            additional_authorized_imports=["*"],  # Allow all imports for flexibility
            verbosity_level=2,  # Set to INFO level
        )

    def _create_tools(self) -> List[Tool]:
        """Create tools for the CodeAgent."""

        # Define the Python executor tool
        def execute_python_safely(code: str) -> str:
            """Execute Python code in a safe environment."""
            try:
                # In a real implementation, this would use the agent's executor
                return "Code execution placeholder (implement with actual executor)"
            except Exception as e:
                return f"Error executing code: {str(e)}"

        # Define the file reader tool
        def read_file(path: str) -> str:
            """Read contents of a file."""
            try:
                with open(path, "r") as f:
                    return f.read()
            except Exception as e:
                return f"Error reading file at {path}: {str(e)}"

        # Create and return the tools
        return [
            Tool(
                name="python",
                description="Execute Python code safely",
                inputs={
                    "code": {"type": "string", "description": "Python code to execute"}
                },
                function=execute_python_safely,
            ),
            Tool(
                name="file_reader",
                description="Read contents of a file",
                inputs={"path": {"type": "string", "description": "Path to file"}},
                function=read_file,
            ),
        ]

    async def get_next_item(self) -> Item:
        """Get the next item from the GAIA dataset."""
        if not self.examples or self.current_index >= len(self.examples):
            self.current_index = 0
            if not self.examples:
                # Return None if there are no examples
                return None

        example = self.examples[self.current_index]
        self.current_index += 1

        # Construct the prompt
        prompt = example["question"]

        # Add file information if available
        if example["file_name"]:
            prompt += f"\n\nTo solve this task, you can use the file at: {example['file_name']}"

        # Create an Item object
        item = Item(
            prompt=prompt,
            metadata={
                "task_id": example["task_id"],
                "task": example["task"],
                "true_answer": example["true_answer"],
                "file_name": example["file_name"],
            },
        )

        return item

    async def collect_trajectory(self, item: Item) -> Tuple[Any, List[Item]]:
        """Run the agent on a single problem and collect the results."""
        logging.info(f"Running agent on task: {item.metadata['task_id']}")

        try:
            # Execute the agent with the prompt
            result = self.agent.run(item.prompt)

            # Create scored data for this trajectory
            # For simplicity, we're just returning a basic metadata object
            scored_data = {
                "task_id": item.metadata["task_id"],
                "task": item.metadata["task"],
                "prompt": item.prompt,
                "result": result,
                "true_answer": item.metadata["true_answer"],
                # We would actually compute a real score based on the answer
                "score": self._evaluate_solution(result, item.metadata["true_answer"]),
            }

            return scored_data, []
        except Exception as e:
            logging.error(f"Error in agent execution: {e}")
            return None, []

    def _evaluate_solution(self, agent_result: str, reference_solution: str) -> float:
        """Evaluate the agent's solution against the reference solution."""
        # This is a simplistic evaluation - in practice, you'd want a more sophisticated approach
        try:
            # For now, just do a simple check if the result contains the reference
            # A more sophisticated implementation would use proper metrics
            if reference_solution.lower() in agent_result.lower():
                return 1.0
            return 0.0
        except:
            return 0.0

    async def evaluate(self, **kwargs):
        """Evaluate the agent on the GAIA benchmark."""
        logging.info("Starting GAIA benchmark evaluation")

        # This is where you would set up a more comprehensive evaluation
        # For now, we'll simply process a few examples

        eval_examples = self.examples[:10] if len(self.examples) > 10 else self.examples

        results = []
        for example in eval_examples:
            # Create an Item for this example
            item = Item(prompt=example["question"], metadata=example)

            # Use a separate worker for each evaluation
            worker = asyncio.create_task(self.collect_trajectory(item))
            self.eval_workers.add(worker)
            worker.add_done_callback(self.eval_workers.discard)

            # Wait for completion and collect results
            result = await worker
            if result and result[0]:
                results.append(result[0])

        # Log evaluation results
        if results:
            avg_score = sum([r["score"] for r in results]) / len(results)
            logging.info(f"Evaluation complete. Average score: {avg_score:.4f}")

            # Update wandb metrics
            if self.config.use_wandb:
                await self.wandb_log(
                    {"eval/avg_score": avg_score, "eval/num_examples": len(results)}
                )
