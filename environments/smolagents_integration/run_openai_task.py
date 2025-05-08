#!/usr/bin/env python3
"""
Script to run a GAIA benchmark task using OpenAI directly with SmolaGents.
"""

import argparse
import asyncio
import json
import logging
import os
import zipfile
from typing import Dict, Any, List, Optional
from pathlib import Path

# Import dotenv for loading environment variables
try:
    from dotenv import load_dotenv
    # Load environment variables from .env file
    load_dotenv()
except ImportError:
    print("python-dotenv not installed. Install with: pip install python-dotenv")

from smolagents import CodeAgent, Tool, OpenAIModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run a GAIA benchmark task with OpenAI")
    
    # Task selection
    parser.add_argument("--task-id", type=str, required=True,
                        help="ID of the task to run or a prompt file path")
    parser.add_argument("--is-file", action="store_true",
                        help="Treat task-id as a file path containing a prompt")
    parser.add_argument("--file-path", type=str,
                        help="Optional file path to include with the task")
    
    # Agent configuration
    parser.add_argument("--max-steps", type=int, default=12,
                        help="Maximum number of steps for the agent")
    
    # OpenAI configuration
    parser.add_argument("--api-key", type=str, 
                        default=os.environ.get("OPENAI_API_KEY", ""),
                        help="OpenAI API key (can also be set via OPENAI_API_KEY env variable)")
    parser.add_argument("--model", type=str, 
                        default=os.environ.get("OPENAI_MODEL", "gpt-4o"),
                        help="OpenAI model name to use (can also be set via OPENAI_MODEL env variable)")
    
    # Output
    parser.add_argument("--output-dir", type=str, default="gaia_results",
                        help="Directory to store results")
    
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


def create_tools(file_path: Optional[str] = None) -> List[Tool]:
    """Create tools for the CodeAgent, including file access tools."""
    
    tools = []
    
    # Add the python execution tool
    def execute_python(code: str) -> str:
        """Execute Python code safely."""
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
    
    tools.append(Tool(
        name="python",
        description="Execute Python code safely",
        inputs={"code": {"type": "string", "description": "Python code to execute"}},
        function=execute_python
    ))
    
    # Add file reader tool
    def read_file(path: Optional[str] = None) -> str:
        """Read contents of a file."""
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
    
    tools.append(Tool(
        name="file_reader",
        description="Read contents of a file",
        inputs={"path": {"type": "string", "description": "Path to file (optional)"}},
        function=read_file
    ))
    
    return tools


def run_task(args, prompt, file_path=None):
    """Run a task with the OpenAI model."""
    # Create tools with file access
    tools = create_tools(file_path)
    
    # Create the OpenAI model
    model = OpenAIModel(
        model_id=args.model,
        api_key=args.api_key
    )
    
    # Initialize CodeAgent
    agent = CodeAgent(
        tools=tools,
        model=model,
        max_steps=args.max_steps,
        additional_authorized_imports=["*"],  # Allow all imports for flexibility
        verbosity_level=2,  # Set to INFO level
    )
    
    # Add file information if available
    if file_path:
        if file_path.endswith(".zip"):
            prompt += f"\n\nTo solve this task, you can use the zip file at: {file_path}"
            prompt += "\nYou can access the files in the zip using the 'file_reader' tool or directly in Python using the 'zip_contents' dictionary."
        else:
            prompt += f"\n\nTo solve this task, you can use the file at: {file_path}"
            prompt += "\nYou can read this file using the 'file_reader' tool or access it in Python as 'file_contents'."
    
    logger.info(f"Running agent with prompt: {prompt[:100]}...")
    
    # Execute the agent with the prompt
    result = agent.run(prompt)
    
    # Get agent memory for detailed output
    agent_memory = agent.write_memory_to_messages()
    
    # Prepare results
    execution_result = {
        "prompt": prompt,
        "prediction": result,
        "agent_memory": agent_memory,
        "num_steps": len(agent.memory.steps)
    }
    
    return execution_result


def main():
    """Main function."""
    args = parse_args()
    
    # Verify API key is available
    if not args.api_key:
        print("Error: OpenAI API key is required. Set it using --api-key or OPENAI_API_KEY in .env file")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get prompt
    if args.is_file:
        with open(args.task_id, 'r') as f:
            prompt = f.read()
        task_id = os.path.basename(args.task_id).split('.')[0]
    else:
        try:
            # Try to load from GAIA dataset
            import datasets
            
            dataset = datasets.load_dataset(
                "neulab/gaia-dataset",
                split="validation",
            )
            
            # Find the task by ID
            for example in dataset:
                if example.get("task_id", "") == args.task_id:
                    prompt = example["Question"]
                    task_id = args.task_id
                    
                    # Set file path if not explicitly provided
                    if not args.file_path and example["file_name"]:
                        args.file_path = f"data/gaia/validation/{example['file_name']}"
                    break
            else:
                # If task not found
                raise ValueError(f"Task ID {args.task_id} not found in the validation split")
        except ImportError:
            # If datasets library not available
            prompt = args.task_id
            task_id = "custom_task"
        except Exception as e:
            logger.error(f"Error loading GAIA task: {e}")
            # Fallback to using the task_id as the prompt
            prompt = args.task_id
            task_id = "custom_task"
    
    # Run the task
    result = run_task(args, prompt, args.file_path)
    
    # Save the result
    output_path = os.path.join(args.output_dir, f"{task_id}_result.json")
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    
    # Print the result
    logger.info(f"Agent prediction: {result['prediction']}")
    logger.info(f"Number of steps: {result['num_steps']}")
    logger.info(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()