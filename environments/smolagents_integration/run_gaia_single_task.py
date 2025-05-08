#!/usr/bin/env python3
"""
Script to run a single GAIA benchmark task with Atropos-SmolaGents integration.

This script allows you to run a specific GAIA benchmark task, showing detailed
output including intermediate steps from the CodeAgent. It also properly handles
the file attachments that are part of many GAIA tasks.
"""

import argparse
import asyncio
import json
import logging
import os
import queue
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from smolagents import CodeAgent, LiteLLMModel
from smolagents.tools import Tool, tool

from atroposlib.envs.server_handling.openai_server import OpenaiConfig
from environments.smolagents_integration.atropos_smolagents_integration import (
    AtroposServerModel,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run a single GAIA benchmark task")

    # Task selection
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="data/gaia",
        help="Path to the GAIA benchmark data",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        help="Dataset split to use (validation, test)",
    )
    parser.add_argument(
        "--task-id", type=str, required=True, help="ID of the task to run"
    )

    # Agent configuration
    parser.add_argument(
        "--max-steps",
        type=int,
        default=12,
        help="Maximum number of steps for the agent",
    )
    parser.add_argument(
        "--use-chat-completion",
        action="store_true",
        help="Use chat completion API instead of completion API",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable additional debug logging",
    )

    # Server configuration
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("OPENAI_API_KEY", "x"),
        help="API key for OpenAI API. Use 'x' for local servers.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8000/v1",
        help="URL of the API endpoint",
    )
    parser.add_argument(
        "--model-name", type=str, default="gpt-3.5-turbo", help="Model name to use"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Timeout for server requests in seconds",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="gaia_results",
        help="Directory to store results",
    )
    parser.add_argument(
        "--use-local-model",
        action="store_true",
        help="Use LiteLLM instead of Atropos for testing purposes",
    )

    return parser.parse_args()


def get_zip_contents(zip_path: str) -> Dict[str, str]:
    """Extract the contents of a zip file as a dictionary mapping filenames to contents."""
    contents = {}
    with zipfile.ZipFile(zip_path, "r") as zip_file:
        for file_name in zip_file.namelist():
            with zip_file.open(file_name) as file:
                try:
                    contents[file_name] = file.read().decode("utf-8")
                except UnicodeDecodeError:
                    contents[file_name] = f"[Binary file: {file_name}]"
    return contents


def load_task(dataset_path: str, split: str, task_id: str) -> Dict[str, Any]:
    """Load a specific task from the GAIA dataset."""
    try:
        import datasets

        # Try direct HuggingFace loading if local fails
        try:
            dataset = datasets.load_dataset(
                f"{dataset_path}/GAIA.py",
                name="2023_all",
                split=split,
            )
        except FileNotFoundError:
            # Fall back to loading directly from HuggingFace
            logger.info(f"Local dataset not found, loading directly from HuggingFace...")
            dataset = datasets.load_dataset(
                "neulab/gaia-dataset",
                split=split,
            )

        # Find the task by ID
        for example in dataset:
            if example.get("task_id", "") == task_id:
                task = {
                    "question": example["Question"],
                    "true_answer": example["Final answer"],
                    "task": example["Level"],
                    "task_id": task_id,
                    "file_name": (
                        f"{dataset_path}/{split}/{example['file_name']}"
                        if example["file_name"]
                        else ""
                    ),
                }
                return task

        # If task not found
        raise ValueError(f"Task ID {task_id} not found in the {split} split")

    except Exception as e:
        logger.error(f"Error loading GAIA task: {e}")
        raise


def create_tools(file_path: Optional[str] = None) -> List[Tool]:
    """Create tools for the CodeAgent, including file access tools."""

    tools = []

    # Create Python execution tool using decorator
    @tool
    def python(code: str) -> str:
        """Execute Python code safely.
        
        Args:
            code: Python code to execute
        """
        try:
            # Create a namespace for execution
            namespace = {}

            # Add file contents to namespace if available
            if file_path:
                if file_path.endswith(".zip"):
                    namespace["zip_contents"] = get_zip_contents(file_path)
                else:
                    try:
                        with open(file_path, "r") as f:
                            namespace["file_contents"] = f.read()
                    except Exception as e:
                        return f"Error reading file: {str(e)}"

            # Execute the code in the namespace
            exec(code, namespace)

            # Check for a result variable
            if "result" in namespace:
                return str(namespace["result"])

            # Otherwise return success
            return "Code executed successfully (no result variable found)"
        except Exception as e:
            return f"Error executing code: {str(e)}"

    tools.append(python)

    # Create file reader tool using decorator
    @tool
    def file_reader(path: Optional[str] = None) -> str:
        """Read contents of a file.
        
        Args:
            path: Path to file (optional)
        """
        try:
            # If no path specified but we have a file from the task, use that
            if not path and file_path:
                path = file_path

            if not path:
                return "No file path specified"

            # Handle zip files
            if path.endswith(".zip"):
                contents = get_zip_contents(path)
                # Format as a list of files with summary
                return (
                    f"ZIP archive containing {len(contents)} files:\n"
                    + "\n".join(f"- {name}" for name in contents.keys())
                    + "\n\nUse the python tool to access specific files by using "
                    + "the zip_contents dictionary."
                )

            # Regular file
            with open(path, "r") as f:
                return f.read()
        except Exception as e:
            return f"Error reading file at {path}: {str(e)}"

    tools.append(file_reader)

    return tools


async def create_atropos_model(args):
    """Create an AtroposServerModel instance."""
    import asyncio

    from atroposlib.envs.server_handling.openai_server import OpenAIServer

    # Create server configuration
    server_config = OpenaiConfig(
        api_key=args.api_key,
        base_url=args.base_url,
        model_name=args.model_name,
        timeout=args.timeout,
    )

    # Create server
    server = OpenAIServer(server_config)

    # Create model
    model = AtroposServerModel(
        server=server,
        use_chat_completion=args.use_chat_completion,
        model_id=f"atropos-{args.model_name}",
    )

    return model


def create_test_model(args):
    """Create a LiteLLM model for testing without Atropos."""
    # Ensure we have a valid API key
    api_key = args.api_key
    if api_key == "x" or not api_key:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("No OpenAI API key provided. Set OPENAI_API_KEY environment variable or use --api-key")
    
    return LiteLLMModel(
        model_id=args.model_name,
        api_key=api_key,
        api_base=(
            args.base_url if not args.base_url.endswith("/v1") else args.base_url[:-3]
        ),
    )


async def run_task(args, task, model):
    """Run the GAIA benchmark task with the provided model."""
    # Create tools with file access
    tools = create_tools(task["file_name"])

    # Initialize CodeAgent with the specified number of steps
    agent = CodeAgent(
        tools=tools,
        model=model,
        max_steps=args.max_steps,  # Use the max steps from args
        additional_authorized_imports=["*"],  # Allow all imports for flexibility
        verbosity_level=2,  # Set to INFO level
    )

    # Construct the prompt
    prompt = task["question"]

    # Add file information if available
    if task["file_name"]:
        if task["file_name"].endswith(".zip"):
            prompt += f"\n\nTo solve this task, you can use the zip file at: {task['file_name']}"
            prompt += "\nYou can access the files in the zip using the 'file_reader' tool or directly in Python using the 'zip_contents' dictionary."
        else:
            prompt += (
                f"\n\nTo solve this task, you can use the file at: {task['file_name']}"
            )
            prompt += "\nYou can read this file using the 'file_reader' tool or access it in Python as 'file_contents'."

    logger.info(f"Running agent on task: {task['task_id']}")
    logger.info(f"Prompt: {prompt}")
    logger.info(f"Running agent with timeout of {args.timeout} seconds")

    # Execute the agent with the prompt
    try:
        # Use a timeout to avoid hanging
        import threading
        
        # We need to run agent.run in a separate thread
        # because it may create event loops internally
        result_queue = queue.Queue()
        
        def run_agent():
            try:
                result = agent.run(prompt)
                result_queue.put(("success", result))
            except Exception as e:
                error_msg = f"Error running agent: {e}"
                logger.error(error_msg)
                result_queue.put(("error", error_msg))
                
        # Start a thread to run the agent
        thread = threading.Thread(target=run_agent)
        thread.daemon = True
        thread.start()
        
        # Wait for the thread to complete or timeout
        thread.join(timeout=args.timeout)  # Use the timeout from args
        
        if thread.is_alive():
            # If the thread is still running after timeout, force quit
            logger.error("Agent execution timed out")
            return {
                "task_id": task["task_id"],
                "question": task["question"],
                "true_answer": task["true_answer"],
                "prediction": "Agent timed out. The solution may be incomplete.",
                "correct": False,
                "agent_memory": [],
                "num_steps": 0,
            }
            
        # Get the result
        if not result_queue.empty():
            status, result = result_queue.get()
            if status == "error":
                logger.error(f"Agent failed: {result}")
                return {
                    "task_id": task["task_id"],
                    "question": task["question"],
                    "true_answer": task["true_answer"],
                    "prediction": f"Agent encountered an error. The solution may be incomplete.",
                    "correct": False,
                    "agent_memory": [],
                    "num_steps": 0,
                }
        else:
            result = "No result from agent."
            
        # Get agent memory for detailed output
        try:
            agent_memory = agent.write_memory_to_messages()
        except Exception as memory_e:
            logger.error(f"Error getting agent memory: {memory_e}")
            agent_memory = []
        
        # Evaluate the result
        # Try more flexible matching since formats might differ slightly
        # First, check for strict containment
        is_correct = task["true_answer"].lower() in result.lower()
        
        # If not correct, try stripping whitespace and checking function bodies
        if not is_correct:
            import re
            
            # Extract function body from both expected and actual
            def normalize_code(code_str):
                # Remove whitespace and comments
                code_str = re.sub(r'\s+', '', code_str)
                code_str = re.sub(r'#.*', '', code_str)
                return code_str
            
            # Try to extract logical parts and compare
            expected_normalized = normalize_code(task["true_answer"])
            result_normalized = normalize_code(result)
            
            # Check if the normalized result contains the normalized expected answer
            is_correct = expected_normalized in result_normalized
        
        # Prepare results
        execution_result = {
            "task_id": task["task_id"],
            "question": task["question"],
            "true_answer": task["true_answer"],
            "prediction": result,
            "correct": is_correct,
            "agent_memory": agent_memory,
            "num_steps": len(agent.memory.steps) if hasattr(agent, 'memory') and hasattr(agent.memory, 'steps') else 0,
        }
        
        return execution_result
            
    except Exception as e:
        logger.error(f"Error in agent execution: {e}")
        # Create a minimal result structure for error case
        return {
            "task_id": task["task_id"],
            "question": task["question"],
            "true_answer": task["true_answer"],
            "prediction": f"Error: {str(e)}",
            "correct": False,
            "agent_memory": [],
            "num_steps": 0,
        }


async def main():
    """Main function to run a single GAIA benchmark task."""
    args = parse_args()

    # Set up logging based on debug flag
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger("httpx").setLevel(logging.DEBUG)
        logging.getLogger("smolagents").setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Enable chat completion for chat-only models like GPT-4o
    if args.model_name.startswith(("gpt-4", "gpt-3.5-turbo", "claude", "gemini")):
        logger.info(f"Automatically enabling chat completion for {args.model_name}")
        args.use_chat_completion = True

    # Load the task
    task = load_task(args.dataset_path, args.split, args.task_id)

    # Create the model
    if args.use_local_model:
        model = create_test_model(args)
    else:
        model = await create_atropos_model(args)

    # Run the task
    result = await run_task(args, task, model)

    # Save the result
    output_path = os.path.join(args.output_dir, f"{args.task_id}_result.json")
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    # Print the result
    logger.info(
        f"Task completed: {'✓ Correct' if result['correct'] else '✗ Incorrect'}"
    )
    logger.info(f"Expected: {task['true_answer']}")
    logger.info(f"Actual: {result['prediction']}")
    logger.info(f"Results saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
