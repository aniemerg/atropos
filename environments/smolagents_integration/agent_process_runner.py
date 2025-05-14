"""
Process-based runner for SmolaGents CodeAgent.
"""

import logging
import multiprocessing
import os
import time
import traceback
from typing import Any, Dict

from smolagents import CodeAgent
from smolagents.models import MessageRole

from environments.smolagents_integration.server_proxy import ServerProxy
from environments.smolagents_integration.smolagents_model import (
    ProcessSafeAtroposServerModel,
)

# Configure logging for the subprocess
logging.basicConfig(
    level=logging.INFO, format="Process-%(process)d: %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
# Prevent propagation to root logger to avoid duplicate logging
logger.propagate = False


def create_tools():
    """Create tools for the CodeAgent."""
    tools = []

    # Add file tools
    try:
        from environments.smolagents_integration.tools.file_tools import (
            append_to_file,
            read_file,
            write_file,
        )

        tools.extend([read_file, write_file, append_to_file])
        logger.info("Added file tools")
    except Exception as e:
        logger.error(f"Could not create file tools: {e}")

    # Add web search tool if TAVILY_API_KEY is available
    try:
        from environments.smolagents_integration.tools.tavily_tools import (
            TavilyExtractTool,
            TavilySearchTool,
        )

        if os.environ.get("TAVILY_API_KEY"):
            tavily_search = TavilySearchTool(api_key=os.environ.get("TAVILY_API_KEY"))
            tavily_extract = TavilyExtractTool(api_key=os.environ.get("TAVILY_API_KEY"))
            tools.extend([tavily_search, tavily_extract])
            logger.info("Added web search tools")
        else:
            logger.warning("TAVILY_API_KEY not set, web search disabled")
    except Exception as e:
        logger.error(f"Could not create web search tools: {e}")

    return tools


def run_agent_process(
    prompt: str,
    task_metadata: Dict[str, Any],
    server_proxy: ServerProxy,
    agent_config: Dict[str, Any],
    result_queue: multiprocessing.Queue,
):
    """
    Run the CodeAgent in a separate process.

    Args:
        prompt: The prompt to send to the agent
        task_metadata: Metadata about the task
        server_proxy: Proxy for communicating with the Atropos server
        agent_config: Configuration for the agent
        result_queue: Queue to put the result in
    """
    try:
        start_time = time.time()
        process_id = os.getpid()
        logger.info(
            f"Process {process_id} starting for task {task_metadata.get('task_id', 'unknown')}"
        )

        # Create a model using the server proxy
        model = ProcessSafeAtroposServerModel(
            server_proxy=server_proxy,
            use_chat_completion=agent_config.get("use_chat_completion", True),
            model_id=agent_config.get("model_name", "atropos-smolagents"),
        )

        # Create tools for the agent
        tools = create_tools()

        # Initialize the CodeAgent
        agent = CodeAgent(
            tools=tools,
            model=model,
            max_steps=agent_config.get("max_steps", 12),
            additional_authorized_imports=["*"],  # Allow all imports for flexibility
            verbosity_level=agent_config.get("verbosity", 2),
        )

        logger.info(
            f"Process {process_id}: Running agent on prompt with {len(prompt)} chars"
        )

        # Run the agent
        agent_response = agent.run(prompt)

        # Extract agent memory
        agent_memory = None
        try:
            if hasattr(agent, "write_memory_to_messages"):
                agent_memory = agent.write_memory_to_messages()
            elif hasattr(agent, "memory"):
                agent_memory = agent.memory
        except Exception as e:
            logger.error(f"Error extracting agent memory: {e}")

        # Calculate execution time
        execution_time = time.time() - start_time

        # Prepare result
        result = {
            "status": "success",
            "response": agent_response,
            "task_id": task_metadata.get("task_id"),
            "execution_time": execution_time,
            "agent_memory": agent_memory,
            "task_metadata": task_metadata,
        }

        logger.info(f"Process {process_id}: Agent completed in {execution_time:.2f}s")

        # Put result in queue
        result_queue.put(result)

    except Exception as e:
        # Log the exception
        logger.error(f"Process {os.getpid()}: Error in agent execution: {e}")
        logger.error(traceback.format_exc())

        # Put error result in queue
        result_queue.put(
            {
                "status": "error",
                "error_message": str(e),
                "error_traceback": traceback.format_exc(),
                "task_id": task_metadata.get("task_id"),
                "task_metadata": task_metadata,
            }
        )

    finally:
        # Clean up resources
        logger.info(f"Process {os.getpid()}: Cleanup complete")
