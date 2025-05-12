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
    level=logging.WARNING, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
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
    """Create tools for the CodeAgent, including file access tools and web tools."""
    tools = []
    
    # Add file tools
    try:
        from environments.smolagents_integration.tools.file_tools import read_file, write_file, append_to_file
        tools.extend([read_file, write_file, append_to_file])
        logger.info("File tools added successfully")
    except ImportError as e:
        logger.error(f"Failed to import file tools: {e}")
        
        # Fallback to simpler file reader if needed
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
                        + "\n\nUse python code to access specific files."
                    )

                # Regular file
                with open(path, "r") as f:
                    return f.read()
            except Exception as e:
                return f"Error reading file at {path}: {str(e)}"
                
        tools.append(file_reader)
    
    # Add Tavily web tools
    tavily_api_key = os.environ.get("TAVILY_API_KEY")
    if tavily_api_key:
        try:
            # Check for tavily package
            import importlib.util
            if importlib.util.find_spec("tavily") is None:
                logger.warning("Tavily package not installed. Install with: pip install tavily-python")
            else:
                from environments.smolagents_integration.tools.tavily_tools import TavilySearchTool, TavilyExtractTool
                tools.extend([
                    TavilySearchTool(api_key=tavily_api_key),
                    TavilyExtractTool(api_key=tavily_api_key)
                ])
                logger.info("Tavily web tools added successfully")
        except Exception as e:
            logger.warning(f"Error initializing Tavily tools: {e}")
    else:
        logger.info("TAVILY_API_KEY not found in environment, web tools will not be available")
    
    # Add special handling for zip files for the current task
    if file_path and file_path.endswith(".zip"):
        # Add zip contents to be accessible by Python code
        zip_contents = get_zip_contents(file_path)
        
        # Register tool to list zip contents
        @tool
        def list_zip_files() -> str:
            """List all files in the provided zip archive."""
            return (
                f"ZIP archive containing {len(zip_contents)} files:\n"
                + "\n".join(f"- {name}" for name in zip_contents.keys())
            )
        
        tools.append(list_zip_files)
        
        # Register tool to read specific file from zip
        @tool
        def read_zip_file(file_name: str) -> str:
            """Read a specific file from the zip archive.
            
            Args:
                file_name: Name of the file within the zip to read
            """
            if file_name in zip_contents:
                return zip_contents[file_name]
            else:
                return f"File '{file_name}' not found in the zip archive. Available files: {', '.join(zip_contents.keys())}"
                
        tools.append(read_zip_file)
    
    # Note: PythonInterpreterTool is NOT needed as CodeAgent already has Python execution capability
    # Note: FinalAnswerTool is automatically added by the CodeAgent/MultiStepAgent class
    
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


def force_terminate_background_processes():
    """Attempt to clean up any background processes that might be keeping 
    the application hanging after timeout."""
    import gc
    import sys
    import threading
    
    # Call garbage collector to clean up lingering objects
    gc.collect()
    
    # Find and terminate stray threads
    for thread in threading.enumerate():
        if thread != threading.current_thread() and not thread.daemon:
            try:
                if hasattr(thread, '_tstate_lock') and thread._tstate_lock:
                    thread._tstate_lock.release()
            except Exception:
                pass
    
    # Additional cleanup for specific libraries
    # This is a best-effort attempt to clean up resources
    libraries_to_clean = ['httpx', 'openai']
    for module_name in list(sys.modules.keys()):
        if any(library in module_name for library in libraries_to_clean):
            if module_name in sys.modules:
                module = sys.modules[module_name]
                # Look for close or cleanup methods
                for attr_name in dir(module):
                    if 'close' in attr_name.lower() or 'cleanup' in attr_name.lower():
                        try:
                            attr = getattr(module, attr_name)
                            if callable(attr):
                                attr()
                        except Exception:
                            pass


def run_task(args, task, model):
    """Run the GAIA benchmark task with the provided model."""
    # Create tools with file access
    tools = create_tools(task["file_name"])

    # Initialize CodeAgent with the specified number of steps
    # LogLevel enum: 0=NONE, 1=ERROR, 2=INFO, 3=DEBUG, 4=TRACE
    from smolagents.monitoring import LogLevel
    
    agent = CodeAgent(
        tools=tools,
        model=model,
        max_steps=args.max_steps,  # Use the max steps from args
        additional_authorized_imports=["os", "json", "re", "datetime", "requests", "zipfile", "textwrap"],
        verbosity_level=LogLevel.DEBUG if args.debug else LogLevel.INFO,  # Use DEBUG level when --debug flag is used
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
        import signal
        import ctypes
        import inspect
        
        # Define a helper function to force terminate a thread
        def terminate_thread(thread):
            """Terminate a thread forcefully."""
            if not thread.is_alive():
                return
                
            # Approach for CPython, get thread's ID to raise exception in it
            tid = thread.ident
            if tid is not None:
                import sys
                if sys.platform == 'darwin':  # MacOS
                    # On MacOS, use a less extreme approach
                    try:
                        # First attempt: set a flag to check
                        setattr(thread, "_terminate", True)
                    except Exception:
                        pass
                else:  # Linux, Windows
                    try:
                        exc = ctypes.py_object(SystemExit)
                        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(tid), exc)
                        if res > 1:
                            # if it returns a number > 1, we're in trouble
                            ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(tid), ctypes.c_long(0))
                            logger.error("Failed to terminate thread cleanly")
                    except Exception as e:
                        logger.error(f"Error terminating thread: {e}")
        
        thread = threading.Thread(target=run_agent)
        thread.daemon = True
        thread.start()
        
        # Wait for the thread to complete or timeout
        thread.join(timeout=args.timeout)  # Use the timeout from args
        
        if thread.is_alive():
            # If the thread is still running after timeout, try to terminate it
            logger.error("Agent execution timed out, attempting to terminate thread")
            
            # First attempt to terminate gently
            terminate_thread(thread)
            
            # Give a small grace period for the thread to terminate
            thread.join(timeout=1.0)
            
            # If it's still alive, try more aggressive cleanup
            if thread.is_alive():
                logger.warning("Thread still alive after termination attempt, cleaning up resources")
                force_terminate_background_processes()
            
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
        
        # Import the GAIA scoring functions
        from environments.smolagents_integration.gaia_scorer import (
            question_scorer, check_close_call, normalize_str, is_float
        )
        
        # First, use the advanced question_scorer for primary correctness check
        is_correct = question_scorer(result, task["true_answer"])
        
        # If not strictly correct, check for near-correct answers
        # Handle potential type issues - ensure we're working with strings for string operations
        try:
            # Need to handle numeric answers properly - if numerical, don't do near-correct check
            if is_float(task["true_answer"]):
                is_near_correct = False  # For numerical answers, we don't have "near correct"
            else:
                is_near_correct = check_close_call(str(result), str(task["true_answer"]), is_correct)
        except Exception as e:
            logger.warning(f"Error in near-correct check: {e}")
            is_near_correct = False
        
        # For code answers, also try additional normalization if still not correct
        # First check that we're not dealing with a numeric answer
        if not is_correct and not is_float(task["true_answer"]) and str(task["true_answer"]).strip().startswith(("def ", "class ", "function", "```")):
            import re
            
            # Extract function body from both expected and actual
            def normalize_code(code_str):
                # Remove whitespace and comments
                code_str = re.sub(r'\s+', '', code_str)
                code_str = re.sub(r'#.*', '', code_str)
                # Remove markdown code blocks
                code_str = re.sub(r'```\w*', '', code_str)
                code_str = re.sub(r'```', '', code_str)
                return code_str.lower()
            
            # Try to extract logical parts and compare
            expected_normalized = normalize_code(task["true_answer"])
            result_normalized = normalize_code(result)
            
            # Check if the normalized result contains the normalized expected answer
            is_code_correct = expected_normalized in result_normalized
            
            # Update correctness if code matching succeeded
            if is_code_correct:
                is_correct = True
                is_near_correct = True
        
        # Prepare results
        execution_result = {
            "task_id": task["task_id"],
            "question": task["question"],
            "true_answer": task["true_answer"],
            "prediction": result,
            "correct": is_correct,
            "near_correct": is_near_correct and not is_correct,  # Only true if near but not exact
            "agent_memory": agent_memory,
            "num_steps": len(agent.memory.steps) if hasattr(agent, 'memory') and hasattr(agent.memory, 'steps') else 0,
            "evaluation_details": {
                "evaluation_method": "question_scorer",
                "code_normalized": task["true_answer"].strip().startswith(("def ", "class ", "function", "```")),
                "is_float_comparison": is_float(task["true_answer"]) if 'is_float' in globals() else False,
                "is_list_comparison": any(char in task["true_answer"] for char in [",", ";"]),
            }
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
    else:
        # Keep logging minimal unless debug is enabled
        logging.getLogger().setLevel(logging.WARNING)
        logging.getLogger("environments.smolagents_integration").setLevel(logging.ERROR)
        logging.getLogger("httpx").setLevel(logging.ERROR)
        logging.getLogger("smolagents").setLevel(logging.WARNING)

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
    
    # Ensure we're using chat completion for models that require it
    if model.model_id and any(name in model.model_id for name in ["gpt-4", "gpt-3.5-turbo", "claude", "gemini"]):
        logger.info(f"Ensuring chat completion is enabled for {model.model_id}")
        model.use_chat_completion = True

    # Run the task
    result = await run_task(args, task, model)

    # Save the result
    output_path = os.path.join(args.output_dir, f"{args.task_id}_result.json")
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    # Print the result with clearer formatting
    print("\n" + "="*80)
    print(f"TASK RESULTS: {args.task_id}")
    print("="*80)
    
    # Determine status with 3-way classification
    if result['correct']:
        status = "✅ CORRECT"
        status_color = "\033[92m"  # Green
    elif result.get('near_correct', False):
        status = "🔶 NEAR CORRECT"
        status_color = "\033[93m"  # Yellow
    else:
        status = "❌ INCORRECT"
        status_color = "\033[91m"  # Red
    
    # Print status with color
    print(f"STATUS: {status_color}{status}\033[0m")
    print("-"*80)
    print("QUESTION:")
    print(f"{task['question']}")
    print("-"*80)
    print("EXPECTED ANSWER:")
    print(f"{task['true_answer']}")
    print("-"*80)
    print("AGENT'S ANSWER:")
    print(f"{result['prediction']}")
    print("-"*80)
    
    # Detailed explanation based on answer type
    print("EVALUATION DETAILS:")
    if result['correct']:
        print("✅ The agent's answer is correct!")
        
        # Explain what type of matching succeeded
        if result.get('evaluation_details', {}).get('is_float_comparison', False):
            print("   (Evaluated as a numerical answer)")
        elif result.get('evaluation_details', {}).get('is_list_comparison', False):
            print("   (Evaluated as a list of items)")
        elif result.get('evaluation_details', {}).get('code_normalized', False):
            print("   (Evaluated as code with normalization)")
        else:
            print("   (Evaluated as text with normalization)")
            
    elif result.get('near_correct', False):
        print("🔶 The agent's answer is nearly correct.")
        print("   The answer contains the key components but may have formatting differences.")
        
        # Add specific details based on answer type
        if result.get('evaluation_details', {}).get('is_float_comparison', False):
            print("   (Numerical answer is close but not exact)")
        elif result.get('evaluation_details', {}).get('is_list_comparison', False):
            print("   (List items are partially matched)")
        elif result.get('evaluation_details', {}).get('code_normalized', False):
            print("   (Code logic is similar but not exactly matching)")
        else:
            print("   (Text contains required elements in order but with differences)")
    else:
        print("❌ The agent's answer is incorrect.")
        
        # Try to explain why it might be wrong
        if result.get('evaluation_details', {}).get('is_float_comparison', False):
            print("   (Numerical values do not match)")
        elif result.get('evaluation_details', {}).get('is_list_comparison', False):
            print("   (List items do not match)")
        elif result.get('evaluation_details', {}).get('code_normalized', False):
            print("   (Code logic does not match the expected solution)")
        else:
            print("   (Text does not match the expected answer)")
    
    print("-"*80)
    print(f"Number of steps: {result['num_steps']}")
    print(f"Results saved to: {output_path}")
    print("="*80 + "\n")
    
    # Force cleanup of any lingering resources
    force_terminate_background_processes()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        # Final cleanup to prevent hanging
        force_terminate_background_processes()
        
        # Clean up AsyncBridge
        from atroposlib.utils.async_bridge import shutdown_bridge
        shutdown_bridge()
