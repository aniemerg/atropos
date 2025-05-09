#!/usr/bin/env python3
"""
Script to run GAIA benchmark with Atropos and SmolaGents integration.

This script demonstrates how to set up and run the GAIA benchmark environment
using the AtroposServerModel to bridge Atropos and SmolaGents.
"""

import argparse
import asyncio
import logging
import os
from dotenv import load_dotenv
import zipfile
import tempfile

# Load environment variables from .env file
load_dotenv()

from atroposlib.envs.server_handling.openai_server import OpenaiConfig
from atroposlib.envs.server_handling.server_manager import ServerManager
from smolagents import CodeAgent, tool
from smolagents.default_tools import PythonInterpreterTool, FinalAnswerTool

# Import our patched AsyncBridge and apply the patch
from environments.smolagents_integration.patched_async_bridge import patch_asyncbridge
# Patch AsyncBridge with our enhanced version
patch_asyncbridge()

# Import our AtroposServerModel (which will now use the patched AsyncBridge)
from environments.smolagents_integration.atropos_smolagents_integration import AtroposServerModel

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Set less verbose logging for our integration code
logging.getLogger("environments.smolagents_integration").setLevel(logging.INFO)

# Set even less verbose logging for other libraries
logging.getLogger("httpx").setLevel(logging.ERROR)  # Reduce HTTP request noise
logging.getLogger("asyncio").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.ERROR)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run GAIA benchmark with Atropos-SmolaGents integration"
    )

    # Environment configuration
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

    # Server configuration
    parser.add_argument(
        "--api-key",
        type=str,
        default="x",
        help="API key for OpenAI API. Use 'x' to use OPENAI_API_KEY env var.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="https://api.openai.com/v1",
        help="URL of the API endpoint",
    )
    parser.add_argument(
        "--model-name", 
        type=str, 
        default="gpt-4o", 
        help="Model name to use"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="Timeout for agent execution in seconds",
    )
    
    # Output configuration
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/gaia",
        help="Output directory for results",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of tasks to run (for testing)",
    )

    # Wandb configuration
    parser.add_argument(
        "--use-wandb", 
        action="store_true", 
        help="Whether to use wandb for logging"
    )
    parser.add_argument(
        "--wandb-name",
        type=str,
        default="gaia-benchmark",
        help="Name to use for wandb run",
    )

    return parser.parse_args()


def get_zip_contents(zip_path):
    """Extract the contents of a ZIP file and return them as a dictionary."""
    result = {}
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for file_info in zip_ref.infolist():
            if not file_info.is_dir():
                with zip_ref.open(file_info) as file:
                    try:
                        result[file_info.filename] = file.read().decode('utf-8')
                    except UnicodeDecodeError:
                        # For binary files, just note that it's binary
                        result[file_info.filename] = "[Binary content]"
    return result

def create_tools(file_path=None):
    """Create tools for the CodeAgent."""
    tools = []
    
    # Add Python tool
    python_tool = PythonInterpreterTool(authorized_imports=["*"])
    tools.append(python_tool)
    
    # Add final answer tool 
    tools.append(FinalAnswerTool())
    
    # Define file reader tool using decorator
    @tool
    def file_reader(path: str = None) -> str:
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
                    + "\n\nUse the python_interpreter tool to access specific files."
                )

            # Regular file
            with open(path, "r") as f:
                return f.read()
        except Exception as e:
            return f"Error reading file at {path}: {str(e)}"
            
    tools.append(file_reader)
    
    return tools

async def run_gaia_benchmark():
    """Run the GAIA benchmark using SmolaGents directly."""
    args = parse_args()
    logger.info(f"Starting run with arguments: {args}")
    
    # Import here to avoid circular imports
    import datasets
    from datasets import load_dataset
    
    # Get API key from command line or environment variable
    api_key = args.api_key
    if api_key == "YOUR_API_KEY" or api_key == "x":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("No OpenAI API key provided. Set OPENAI_API_KEY environment variable or use --api-key")
    
    # Create server configuration and OpenAI server
    server_config = OpenaiConfig(
        api_key=api_key,
        base_url=args.base_url,
        model_name=args.model_name,
        timeout=args.timeout,
    )
    
    # Create OpenAIServer instance
    from atroposlib.envs.server_handling.openai_server import OpenAIServer
    server = OpenAIServer(server_config)
    
    # Create AtroposServerModel
    model = AtroposServerModel(
        server=server,
        use_chat_completion=args.use_chat_completion,
        model_id=f"atropos-{args.model_name}",
    )
    
    # Load the GAIA dataset
    logger.info(f"Loading GAIA dataset (split: {args.split})")
    try:
        # Calculate project root dynamically based on the script's location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, "../.."))
        gaia_script_path = os.path.join(project_root, "data/gaia/GAIA.py")
        logger.info(f"Project root: {project_root}")
        logger.info(f"Using GAIA script at: {gaia_script_path}")
        
        # Set the data directory environment variable to help GAIA.py find the files
        gaia_data_dir = os.path.join(project_root, "data/gaia/2023")
        os.environ["GAIA_DATA_DIR"] = gaia_data_dir
        logger.info(f"Set GAIA_DATA_DIR to: {gaia_data_dir}")
        
        dataset = load_dataset(
            gaia_script_path,
            name="2023_all",
            split=args.split,
            trust_remote_code=True  # Avoid the prompt asking to run custom code
        )
        
        # Preprocess dataset to add file paths
        def preprocess_file_paths(example):
            if example.get("file_name") and example["file_name"]:
                # Use absolute path to the files in the 2023 subdirectory
                file_path = os.path.join(gaia_data_dir, args.split, example['file_name'])
                example["file_path"] = file_path
                # Only log if the file exists to avoid excessive logging
                if os.path.exists(file_path):
                    logger.info(f"Set file path: {file_path}")
                else:
                    logger.warning(f"File does not exist: {file_path}")
            else:
                example["file_path"] = ""
            return example
            
        dataset = dataset.map(preprocess_file_paths)
        logger.info(f"Loaded {len(dataset)} examples from GAIA dataset")
    
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        # Fallback to a dummy dataset for testing
        logger.info("Using fallback dummy dataset for testing")
        dataset = [
            {
                "task_id": "GAIA2023_P001",
                "Question": "Write a Python function to check if a string is a palindrome",
                "Final answer": "def is_palindrome(s):\n    return s == s[::-1]",
                "Level": "programming",
                "file_name": "",
                "file_path": ""
            }
        ]
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    results = []
    
    # Process each task
    for i, example in enumerate(dataset):
        task_id = example.get("task_id", f"task_{i}")
        question = example.get("Question", example.get("question", ""))
        true_answer = example.get("Final answer", example.get("true_answer", ""))
        file_path = example.get("file_path", "")
        
        logger.info(f"Running agent on task: {task_id}")
        logger.info(f"Prompt: {question}")
        
        # Create tools with the current file path
        tools = create_tools(file_path)
        
        # Create agent
        agent = CodeAgent(
            tools=tools,
            model=model,
            max_steps=args.max_steps,
            additional_authorized_imports=["*"],  # Allow all imports for flexibility
            verbosity_level=2,  # Set to INFO level
        )
        
        # Prepare prompt with file information if available
        prompt = question
        if file_path:
            prompt += f"\n\nTo solve this task, you can use the file at: {file_path}"
        
        # Run agent
        try:
            logger.info(f"Running agent with timeout of {args.timeout} seconds")
            # Set a timeout for the agent
            import threading
            import queue
            
            result_queue = queue.Queue()
            
            def run_agent():
                try:
                    result = agent.run(prompt)
                    result_queue.put(("success", result))
                except Exception as e:
                    result_queue.put(("error", str(e)))
            
            thread = threading.Thread(target=run_agent)
            thread.daemon = True
            thread.start()
            
            # Wait for the thread to complete or timeout
            thread.join(timeout=args.timeout)
            
            if thread.is_alive():
                logger.error("Agent execution timed out")
                result = "Agent timed out"
                agent_memory = []
                is_correct = False
            else:
                # Get the result
                if not result_queue.empty():
                    status, result = result_queue.get()
                    if status == "error":
                        logger.error(f"Agent failed: {result}")
                        result = f"Error: {result}"
                        agent_memory = []
                        is_correct = False
                    else:
                        # Get agent memory for detailed output
                        try:
                            agent_memory = agent.write_memory_to_messages()
                        except Exception as memory_e:
                            logger.error(f"Error getting agent memory: {memory_e}")
                            agent_memory = []
                        
                        # Evaluate the result
                        is_correct = true_answer.lower() in result.lower()
                else:
                    result = "No result from agent"
                    agent_memory = []
                    is_correct = False
            
            # Save the result
            task_result = {
                "task_id": task_id,
                "question": question,
                "true_answer": true_answer,
                "prediction": result,
                "correct": is_correct,
                "agent_memory": agent_memory,
                "num_steps": len(agent.memory.steps) if hasattr(agent, 'memory') and hasattr(agent.memory, 'steps') else 0,
            }
            results.append(task_result)
            
            # Save individual result
            import json
            output_path = os.path.join(args.output_dir, f"{task_id}_result.json")
            with open(output_path, "w") as f:
                json.dump(task_result, f, indent=2)
            
            logger.info(f"Task {task_id} completed: {'✓ Correct' if is_correct else '✗ Incorrect'}")
            logger.info(f"Results saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Error processing task {task_id}: {e}")
    
    # Save all results
    import json
    output_path = os.path.join(args.output_dir, "all_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"All results saved to: {output_path}")
    
    # Calculate overall accuracy
    correct_count = sum(1 for result in results if result["correct"])
    total_count = len(results)
    accuracy = correct_count / total_count if total_count > 0 else 0
    
    logger.info(f"Benchmark complete: {correct_count}/{total_count} correct ({accuracy:.2%} accuracy)")
    
    return results

async def main():
    """Main function entry point."""
    try:
        await run_gaia_benchmark()
    except Exception as e:
        logger.error(f"Error in main: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        # Clean up AsyncBridge to prevent hanging
        from atroposlib.utils.async_bridge import shutdown_bridge
        shutdown_bridge()
        
        # Also attempt to clean up any other resources
        from environments.smolagents_integration.run_gaia_single_task import force_terminate_background_processes
        force_terminate_background_processes()
