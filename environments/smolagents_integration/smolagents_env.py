"""
SmolagentsEnv - Environment for creating high-quality agent trajectories
for training language models using the SmolaGents agent framework.
"""

import asyncio
import json
import logging
import multiprocessing
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field
from smolagents import CodeAgent, Tool
from smolagents.tools import tool  # Import the tool decorator

import wandb
from atroposlib.envs.base import BaseEnv, BaseEnvConfig, ScoredDataGroup
from atroposlib.envs.server_handling.openai_server import OpenaiConfig, OpenAIServer
from atroposlib.envs.server_handling.server_manager import ServerManager
from environments.smolagents_integration.agent_process_runner import run_agent_process
from environments.smolagents_integration.server_proxy import ServerProxyManager
from environments.smolagents_integration.tools.file_tools import (
    append_to_file,
    read_file,
    write_file,
)


@dataclass
class Item:
    prompt: str
    metadata: Dict[str, Any]
    id: Optional[str] = None


# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Add a console handler to make sure logs are visible
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


class SmolagentsEnvConfig(BaseEnvConfig):
    """Configuration for SmolagentsEnv."""

    dataset_path: str = Field(default="data/gaia", description="Path to GAIA dataset")
    split: str = Field(
        default="validation", description="Dataset split to use (validation, test)"
    )
    use_chat_completion: bool = Field(
        default=True, description="Use chat completion API"
    )
    max_steps: int = Field(default=12, description="Maximum number of agent steps")
    agent_verbosity: int = Field(default=2, description="Agent verbosity level (0-3)")
    scoring_strategy: str = Field(
        default="combined",
        description="Scoring strategy: basic, correctness, or combined",
    )
    # Removed max_concurrent_agents as we only use process-based execution
    length_penalty_weight: float = Field(
        default=0.1, description="Weight for length penalty in scoring (0.0 to disable)"
    )
    save_full_traces: bool = Field(
        default=True, description="Save full agent execution traces in the output"
    )
    # Output path configured in __init__
    data_path_to_save_groups: Optional[str] = Field(
        default=None,
        description="Path to save JSONL output (defaults to timestamped file if None)",
    )
    # Process-based settings
    max_concurrent_processes: int = Field(
        default=5,
        description="Maximum number of concurrent processes for agent execution",
    )
    process_timeout: int = Field(
        default=240,  # 4 minutes by default
        description="Timeout for agent processes in seconds",
    )
    # Debugging options
    debug_scoring: bool = Field(
        default=False, description="Enable detailed score calculation logging"
    )


class SmolagentsEnv(BaseEnv):
    """
    Environment for generating high-quality agent trajectories using the SmolaGents framework.

    This environment:
    1. Loads tasks from the GAIA benchmark dataset
    2. Uses SmolaGents CodeAgent with appropriate tools
    3. Scores trajectories based on correctness and reasoning quality
    4. Integrates with Atropos SFT generation pipeline
    """

    name = "smolagents"
    env_config_cls = SmolagentsEnvConfig

    @classmethod
    def config_init(cls) -> Tuple[BaseEnvConfig, List[OpenaiConfig]]:
        """Initialize the config for CLI use."""
        env_config = SmolagentsEnvConfig(
            tokenizer_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
            group_size=8,
            use_wandb=True,
            rollout_server_url="http://localhost:8000",
            total_steps=1000,
            batch_size=32,
            steps_per_eval=100,
            max_token_length=4096,
            wandb_name="smolagents",
            include_messages=True,
            # Process-based settings
            max_concurrent_processes=8,
            process_timeout=240,
            # Common settings
            dataset_path="data/gaia",
            split="validation",  # GAIA only supports 'validation' and 'test' splits
            use_chat_completion=True,
            # Debugging options
            debug_scoring=False,  # Set to True to enable detailed score logging
            # Using default timestamped output path from the config definition
        )
        server_configs = [
            OpenaiConfig(
                model_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
                base_url="http://localhost:9001/v1",
                api_key="x",
                num_requests_for_eval=32,
            ),
        ]
        return env_config, server_configs

    def __init__(
        self,
        config: SmolagentsEnvConfig,
        server_configs: Union[List[OpenaiConfig], OpenaiConfig],
        slurm=False,
        testing=False,
    ):
        # Set a timestamped output file path if not provided
        if config.data_path_to_save_groups is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            config.data_path_to_save_groups = f"smolagents_output_{timestamp}.jsonl"
            print(
                f"Using auto-generated output path: {config.data_path_to_save_groups}"
            )

        # Initialize the base class
        super().__init__(config, server_configs, slurm, testing)

        # Initialize dataset variables
        self.examples = []
        self.current_index = 0
        self.iter = 0  # Add iter for checkpoint tracking

        # Initialize the server proxy manager for process-based execution
        self.server_proxy_manager = None  # Will be initialized in setup()

        # Save config for easier access
        self.max_steps = config.max_steps
        self.verbosity = config.agent_verbosity
        self.scoring_strategy = config.scoring_strategy
        self.debug_scoring = config.debug_scoring

        # Track agent execution times and metrics
        self.agent_execution_times = []
        self.percent_correct_buffer = []
        self.eval_metrics = []

    async def setup(self):
        """Set up the environment, load dataset, and initialize server components."""
        logger.info("Setting up SmolagentsEnv...")
        logger.info(f"Using dataset split: {self.config.split}")

        # Initialize the server proxy manager
        logger.info("Setting up process-based isolation for agent execution")
        self.server_proxy_manager = ServerProxyManager(
            server=self.server, max_workers=self.config.max_concurrent_processes
        )
        self.server_proxy_manager.start()
        logger.info(
            f"Started server proxy manager with max_workers={self.config.max_concurrent_processes}"
        )

        # Load the GAIA dataset
        try:
            import datasets

            logger.info(f"Loading GAIA dataset from {self.config.dataset_path}")
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
                        if example.get("file_name")
                        else ""
                    ),
                }
                for i, example in enumerate(self.dataset)
            ]

            logger.info(
                f"Loaded {len(self.examples)} examples from GAIA {self.config.split} set"
            )
        except Exception as e:
            logger.error(f"Error loading GAIA dataset: {e}")
            # Create empty list if dataset loading fails
            self.examples = []

        logger.info("SmolagentsEnv setup complete")

    def _create_tools(self) -> List[Tool]:
        """Create and return the tools for the CodeAgent based on config."""
        tools = []
        logger.info("Creating tools for CodeAgent")

        # Always add file tools (read, write, append)
        # These use the @tool decorator and are correctly formed SimpleTool instances
        tools.append(read_file)
        tools.append(write_file)
        tools.append(append_to_file)
        logger.info("Added file tools (read_file, write_file, append_to_file)")

        # Add web search tool if TAVILY_API_KEY is available
        try:
            from environments.smolagents_integration.tools.tavily_tools import (
                TavilyExtractTool,
                TavilySearchTool,
            )

            if os.environ.get("TAVILY_API_KEY"):
                tavily_search = TavilySearchTool(
                    api_key=os.environ.get("TAVILY_API_KEY")
                )
                tavily_extract = TavilyExtractTool(
                    api_key=os.environ.get("TAVILY_API_KEY")
                )
                tools.append(tavily_search)
                tools.append(tavily_extract)
                logger.info("Added web_search tool")
            else:
                logger.warning(
                    "TAVILY_API_KEY not set in environment, web search disabled"
                )
        except Exception as e:
            logger.warning(f"Could not create web search tool: {e}")

        # Log tool info for debugging
        tool_names = []
        for i, tool in enumerate(tools):
            if hasattr(tool, "name"):
                tool_names.append(tool.name)
                logger.info(f"Tool {i} name: {tool.name}")
            else:
                tool_names.append(str(tool))
                logger.warning(f"Tool {i} has no name attribute: {tool}")

        logger.info(f"Created {len(tools)} tools for CodeAgent: {tool_names}")
        return tools

    async def get_next_item(self) -> Item:
        """Get the next item from the GAIA dataset."""
        if not self.examples:
            logger.warning("No examples loaded in dataset")
            return None

        # Use iter to track position and support checkpointing
        example = self.examples[self.iter % len(self.examples)]
        self.iter += 1
        self.current_index = self.iter % len(self.examples)

        # Construct the prompt
        prompt = example["question"]

        # Add file information if available
        if example.get("file_name"):
            prompt += f"\n\nTo solve this task, you can use the file at: {example['file_name']}"

        # Create an Item object
        item = Item(
            prompt=prompt,
            metadata={
                "task_id": example["task_id"],
                "task": example["task"],
                "true_answer": example["true_answer"],
                "file_name": example.get("file_name", ""),
                "dataset_idx": self.current_index,
            },
        )

        return item

    async def collect_trajectories(self, items: Union[Item, List[Item]]) -> Tuple[
        Union[Optional[ScoredDataGroup], List[Optional[ScoredDataGroup]], List[Any]],
        List[Item],
    ]:
        """
        Collect trajectories for multiple items using process-based parallelism.
        """
        # Handle both single item and list of items
        if not isinstance(items, list):
            items = [items] * self.config.group_size

        return await self._collect_trajectories_process_based(items)

    async def _collect_trajectories_process_based(
        self, items: List[Item]
    ) -> Tuple[List[Any], List[Item]]:
        """
        Collect trajectories for multiple items using process-based parallelism.
        """
        logger.info(
            f"Collecting trajectories for {len(items)} items using process-based parallelism"
        )

        # Create a manager for shared objects
        manager = multiprocessing.Manager()
        result_queue = manager.Queue()

        # Create agent config dictionary
        agent_config = {
            "max_steps": self.max_steps,
            "verbosity": self.verbosity,
            "use_chat_completion": self.config.use_chat_completion,
            "model_name": getattr(self.server, "model_name", "unknown-model"),
        }

        # Start processes for each item
        processes = []
        proxies = []

        for item in items:
            # Create a server proxy for this process
            server_proxy, proxy_id = self.server_proxy_manager.create_server_proxy(
                model_name=agent_config["model_name"],
                timeout=self.config.process_timeout,
            )
            proxies.append(proxy_id)

            # Start a process for this item
            process = multiprocessing.Process(
                target=run_agent_process,
                args=(
                    item.prompt,
                    item.metadata,
                    server_proxy,
                    agent_config,
                    result_queue,
                ),
            )
            process.start()
            processes.append(process)

        logger.info(f"Started {len(processes)} agent processes")

        # Wait for all processes to complete or timeout
        for process in processes:
            process.join(timeout=self.config.process_timeout)

            # Check if process is still alive (timeout)
            if process.is_alive():
                logger.warning(f"Process {process.pid} timed out, terminating")
                process.terminate()
                process.join()

        # Clean up proxies
        for proxy_id in proxies:
            self.server_proxy_manager.remove_proxy(proxy_id)

        # Get all results from the queue
        results = []
        while not result_queue.empty():
            try:
                result = result_queue.get(block=False)
                results.append(result)
            except Exception as e:
                logger.error(f"Error getting result from queue: {e}")
                break

        logger.info(f"Collected {len(results)} results from processes")

        # Process results
        backlog = []
        to_postprocess = []

        for result in results:
            if result["status"] == "success":
                # Create scored data from successful result
                scored_data = {
                    "prompt": result["task_metadata"].get("prompt", ""),
                    "response": result["response"],
                    "task_id": result["task_id"],
                    "task": result["task_metadata"].get("task", ""),
                    "true_answer": result["task_metadata"].get("true_answer", ""),
                    "execution_time": result["execution_time"],
                }

                # Add agent memory if configured
                if self.config.save_full_traces and "agent_memory" in result:
                    scored_data["agent_memory"] = result["agent_memory"]

                # Score the trajectory
                score = self._score_trajectory(
                    scored_data["prompt"],
                    scored_data["response"],
                    scored_data["true_answer"],
                    scored_data.get("agent_memory"),
                    scored_data["execution_time"],
                )

                scored_data["score"] = score

                # Create ScoredDataGroup
                item_for_scoring = next(
                    (
                        i
                        for i in items
                        if i.metadata.get("task_id") == result["task_id"]
                    ),
                    None,
                )
                if item_for_scoring:
                    scored_group = self._create_scored_data_group(
                        item_for_scoring, scored_data
                    )
                    to_postprocess.append(scored_group)
                else:
                    logger.warning(
                        f"Could not find original item for task_id {result['task_id']}"
                    )
            else:
                # Handle error case - create fallback response
                logger.error(
                    f"Error in process for task {result['task_id']}: {result.get('error_message', 'Unknown error')}"
                )

                # Create fallback scored group
                item_for_scoring = next(
                    (
                        i
                        for i in items
                        if i.metadata.get("task_id") == result["task_id"]
                    ),
                    None,
                )
                if item_for_scoring:
                    fallback = self._create_fallback_response(
                        item_for_scoring, result.get("error_message", "Unknown error")
                    )
                    to_postprocess.append(fallback)
                else:
                    logger.warning(
                        f"Could not find original item for task_id {result['task_id']}"
                    )

        # Return processed results
        logger.info(f"Final to_postprocess: len={len(to_postprocess)}")
        return to_postprocess, backlog

    async def postprocess_histories(
        self, histories: Union[ScoredDataGroup, List[ScoredDataGroup]]
    ) -> ScoredDataGroup:
        """
        Post-process the agent histories.

        We need to merge multiple ScoredDataGroups into a single ScoredDataGroup.
        """
        logger.info(
            f"postprocess_histories called with: type={type(histories)}, is_none={histories is None}"
        )
        if isinstance(histories, list):
            logger.info(f"  List length: {len(histories)}")

        if not isinstance(histories, list):
            # If it's already a single ScoredDataGroup, return it with group_overrides
            logger.info(f"  Single history, returning directly: {type(histories)}")
            if (
                "group_overrides" not in histories
                or histories["group_overrides"] is None
            ):
                histories["group_overrides"] = {}
            return histories

        # If we have multiple ScoredDataGroups, merge them
        logger.info(f"  Merging {len(histories)} histories")
        merged = ScoredDataGroup(
            tokens=[],
            masks=[],
            scores=[],
            advantages=None,
            ref_logprobs=None,
            messages=[] if self.config.include_messages else None,
            group_overrides={},
            overrides=None,
        )

        # Merge all the fields
        for i, history in enumerate(histories):
            logger.info(
                f"  Processing history {i}: type={type(history)}, is_none={history is None}"
            )
            if history is not None:
                logger.info(f"    History {i} tokens: {len(history['tokens'])}")
                merged["tokens"].extend(history["tokens"])
                merged["masks"].extend(history["masks"])
                merged["scores"].extend(history["scores"])

                if merged["messages"] is not None and "messages" in history:
                    logger.info(f"    History {i} messages: {len(history['messages'])}")
                    merged["messages"].extend(history["messages"])

        logger.info(
            f"  Final merged data: tokens={len(merged['tokens'])}, scores={len(merged['scores'])}"
        )
        return merged

    def _score_trajectory(
        self,
        prompt: str,
        agent_response: str,
        true_answer: str,
        agent_memory: List[Dict] = None,
        execution_time: float = 0,
    ) -> float:
        """
        Score the agent trajectory based on the chosen scoring strategy.

        Supports multiple scoring methods:
        - basic: Simple correctness check
        - correctness: More sophisticated answer validation
        - combined: Blend of correctness, efficiency, and reasoning quality

        When debug_scoring is enabled, prints detailed breakdown of score calculations.
        """
        try:
            # Apply the appropriate scoring strategy
            if self.scoring_strategy == "basic":
                # Simple scoring: check if true answer appears in response
                has_correct_answer = true_answer.lower() in agent_response.lower()
                score = 1.0 if has_correct_answer else 0.0

                if self.debug_scoring:
                    logger.info("=== SCORE CALCULATION (basic) ===")
                    logger.info(f"True answer: '{true_answer}'")
                    logger.info(f"Found in response: {has_correct_answer}")
                    logger.info(f"Final score: {score:.3f}")

                return score

            elif self.scoring_strategy == "correctness":
                # Try to use gaia_scorer if available
                try:
                    from environments.smolagents_integration.gaia_scorer import (
                        score_answer,
                    )

                    score = score_answer(agent_response, true_answer)

                    if self.debug_scoring:
                        logger.info("=== SCORE CALCULATION (correctness) ===")
                        logger.info(f"True answer: '{true_answer}'")
                        logger.info(f"Using GAIA scorer, score: {score:.3f}")

                    return score
                except ImportError:
                    # Fall back to basic scoring
                    has_correct_answer = true_answer.lower() in agent_response.lower()
                    score = 1.0 if has_correct_answer else 0.0

                    if self.debug_scoring:
                        logger.info(
                            "=== SCORE CALCULATION (correctness - fallback) ==="
                        )
                        logger.info(f"True answer: '{true_answer}'")
                        logger.info(f"GAIA scorer not available, using fallback")
                        logger.info(f"Found in response: {has_correct_answer}")
                        logger.info(f"Final score: {score:.3f}")

                    return score

            elif self.scoring_strategy == "combined":
                # Combined scoring: correctness + efficiency + reasoning quality
                base_score = 0.0

                if self.debug_scoring:
                    logger.info("=== SCORE CALCULATION (combined) ===")
                    logger.info(f"True answer: '{true_answer}'")

                # 1. Check correctness (50% of score)
                try:
                    from environments.smolagents_integration.gaia_scorer import (
                        score_answer,
                    )

                    correctness_raw = score_answer(agent_response, true_answer)
                    correctness_score = correctness_raw * 0.5

                    if self.debug_scoring:
                        logger.info(f"1. Correctness component:")
                        logger.info(
                            f"   - Using GAIA scorer, raw score: {correctness_raw:.3f}"
                        )
                        logger.info(f"   - Weight: 0.5")
                        logger.info(f"   - Weighted score: {correctness_score:.3f}")
                except ImportError:
                    # Fall back to basic scoring
                    has_correct_answer = true_answer.lower() in agent_response.lower()
                    correctness_raw = 1.0 if has_correct_answer else 0.0
                    correctness_score = correctness_raw * 0.5

                    if self.debug_scoring:
                        logger.info(f"1. Correctness component:")
                        logger.info(f"   - GAIA scorer not available, using fallback")
                        logger.info(f"   - Found in response: {has_correct_answer}")
                        logger.info(f"   - Raw score: {correctness_raw:.3f}")
                        logger.info(f"   - Weight: 0.5")
                        logger.info(f"   - Weighted score: {correctness_score:.3f}")

                # 2. Check efficiency (25% of score)
                # Penalize long execution times and many steps
                steps_count = len(agent_memory) if agent_memory else 0
                efficiency_score = 0.25  # Start with full efficiency score
                step_penalty = 1.0
                time_penalty = 1.0

                # Penalty for excessive steps (above 75% of max)
                if steps_count > (self.max_steps * 0.75):
                    step_penalty = 0.7
                    efficiency_score *= step_penalty

                # Penalty for long execution time (if we have other executions to compare)
                if self.agent_execution_times and len(self.agent_execution_times) > 5:
                    avg_time = np.mean(self.agent_execution_times)
                    if execution_time > (avg_time * 1.5):
                        time_penalty = 0.8
                        efficiency_score *= time_penalty

                if self.debug_scoring:
                    logger.info(f"2. Efficiency component:")
                    logger.info(
                        f"   - Steps count: {steps_count} / max {self.max_steps}"
                    )
                    logger.info(f"   - Step penalty: {step_penalty:.2f}")
                    if (
                        self.agent_execution_times
                        and len(self.agent_execution_times) > 5
                    ):
                        logger.info(
                            f"   - Execution time: {execution_time:.2f}s (avg: {avg_time:.2f}s)"
                        )
                        logger.info(f"   - Time penalty: {time_penalty:.2f}")
                    else:
                        logger.info(
                            f"   - Execution time: {execution_time:.2f}s (not enough data for comparison)"
                        )
                    logger.info(f"   - Max possible score: 0.25")
                    logger.info(f"   - Final efficiency score: {efficiency_score:.3f}")

                # 3. Check reasoning quality (25% of score)
                reasoning_score = 0.0
                marker_count = 0

                # Look for step-by-step reasoning with clear intermediate steps
                # This is a simplified heuristic - could be much more sophisticated
                if agent_memory and len(agent_memory) > 0:
                    # Look for reasoning markers in the agent's work
                    reasoning_markers = [
                        "first",
                        "second",
                        "third",
                        "step",
                        "approach",
                        "reason",
                        "because",
                        "therefore",
                        "thus",
                        "hence",
                        "calculate",
                        "compute",
                        "solve",
                        "find",
                        "determine",
                    ]

                    # Count reasoning markers in memory content
                    marker_counts_by_type = {}
                    for message in agent_memory:
                        content = message.get("content", "")
                        if isinstance(content, str):
                            for marker in reasoning_markers:
                                if marker in content.lower():
                                    marker_counts_by_type[marker] = (
                                        marker_counts_by_type.get(marker, 0) + 1
                                    )
                                    marker_count += 1

                    # Normalize reasoning score
                    reasoning_score = min(0.25, (marker_count / 10) * 0.25)

                if self.debug_scoring:
                    logger.info(f"3. Reasoning quality component:")
                    logger.info(
                        f"   - Memory trace length: {len(agent_memory) if agent_memory else 0} messages"
                    )
                    logger.info(f"   - Total reasoning markers found: {marker_count}")
                    if marker_count > 0 and agent_memory:
                        logger.info(f"   - Marker breakdown:")
                        for marker, count in marker_counts_by_type.items():
                            logger.info(f"     - '{marker}': {count}")
                    logger.info(f"   - Max possible score: 0.25")
                    logger.info(f"   - Final reasoning score: {reasoning_score:.3f}")

                # Combine scores
                base_score = correctness_score + efficiency_score + reasoning_score

                # Apply length penalty if configured
                length_penalty = 0.0
                if self.config.length_penalty_weight > 0 and agent_response:
                    response_length = len(agent_response)
                    # Penalize very long responses
                    if response_length > 2000:
                        length_penalty = min(
                            0.3,
                            self.config.length_penalty_weight
                            * (response_length - 2000)
                            / 1000,
                        )
                        base_score = max(0.0, base_score - length_penalty)

                if self.debug_scoring:
                    logger.info(f"4. Length penalty:")
                    logger.info(
                        f"   - Response length: {len(agent_response)} characters"
                    )
                    logger.info(
                        f"   - Penalty weight: {self.config.length_penalty_weight}"
                    )
                    logger.info(f"   - Length penalty: {length_penalty:.3f}")
                    logger.info(f"5. Final score calculation:")
                    logger.info(f"   - Correctness: {correctness_score:.3f}")
                    logger.info(f"   - Efficiency: {efficiency_score:.3f}")
                    logger.info(f"   - Reasoning: {reasoning_score:.3f}")
                    logger.info(f"   - Length penalty: -{length_penalty:.3f}")
                    logger.info(f"   - FINAL SCORE: {base_score:.3f}")

                return base_score

            else:
                # Unknown scoring strategy
                logger.warning(
                    f"Unknown scoring strategy: {self.scoring_strategy}, using basic"
                )
                has_correct_answer = true_answer.lower() in agent_response.lower()
                score = 1.0 if has_correct_answer else 0.0

                if self.debug_scoring:
                    logger.info(
                        "=== SCORE CALCULATION (unknown strategy - fallback to basic) ==="
                    )
                    logger.info(f"True answer: '{true_answer}'")
                    logger.info(f"Found in response: {has_correct_answer}")
                    logger.info(f"Final score: {score:.3f}")

                return score

        except Exception as e:
            logger.error(f"Error in scoring: {e}")
            if self.debug_scoring:
                logger.error(f"Exception during score calculation: {e}")
            return 0.0

    def _create_scored_data_group(
        self, item: Item, scored_data: Dict
    ) -> ScoredDataGroup:
        """
        Create a ScoredDataGroup for the trainer API.

        Converts the agent trajectory into tokenized format for the trainer.
        """
        # Prepare the data in message format or token format
        if self.config.include_messages:
            # Create message format with agent memory if available
            messages = []

            # Add system message with task description
            messages.append(
                {
                    "role": "system",
                    "content": "You are an AI assistant solving a task with reasoning and problem-solving skills.",
                }
            )

            # Add user message with the prompt
            messages.append({"role": "user", "content": item.prompt})

            # For message format, extract agent memory if available
            if self.config.save_full_traces and "agent_memory" in scored_data:
                # Add intermediate reasoning steps
                for message in scored_data["agent_memory"]:
                    messages.append(message)
            else:
                # Just add the final response
                messages.append(
                    {"role": "assistant", "content": scored_data["response"]}
                )

            # Create the ScoredDataGroup
            scored_group = ScoredDataGroup(
                tokens=[self.tokenizer.encode(json.dumps(messages))],
                masks=[[1] * len(self.tokenizer.encode(json.dumps(messages)))],
                scores=[scored_data["score"]],
                messages=[messages],
            )

        else:
            # Create token format
            prefix = item.prompt
            completion = scored_data["response"]

            # Tokenize the prefix and completion
            prefix_tokens = self.tokenizer.encode(prefix)
            completion_tokens = self.tokenizer.encode(completion)

            # Create full token sequence and masks
            tokens = prefix_tokens + completion_tokens
            masks = [0] * len(prefix_tokens) + [1] * len(completion_tokens)

            # Create the ScoredDataGroup
            scored_group = ScoredDataGroup(
                tokens=[tokens],
                masks=[masks],
                scores=[scored_data["score"]],
                messages=None,
            )

        return scored_group

    def _create_fallback_response(
        self, item: Item, error_message: str
    ) -> ScoredDataGroup:
        """Create a fallback response for failed agent executions."""
        # Create a minimal valid response with low score
        if self.config.include_messages:
            messages = [
                {"role": "system", "content": "You are an AI assistant solving tasks."},
                {"role": "user", "content": item.prompt},
                {
                    "role": "assistant",
                    "content": f"I'm unable to solve this task. Error: {error_message}",
                },
            ]

            scored_group = ScoredDataGroup(
                tokens=[self.tokenizer.encode(json.dumps(messages))],
                masks=[[1] * len(self.tokenizer.encode(json.dumps(messages)))],
                scores=[0.1],  # Low score but not zero to allow some learning
                messages=[messages],
            )
        else:
            # Token format fallback
            prefix = item.prompt
            completion = f"I'm unable to solve this task. Error: {error_message}"

            prefix_tokens = self.tokenizer.encode(prefix)
            completion_tokens = self.tokenizer.encode(completion)

            tokens = prefix_tokens + completion_tokens
            masks = [0] * len(prefix_tokens) + [1] * len(completion_tokens)

            scored_group = ScoredDataGroup(
                tokens=[tokens],
                masks=[masks],
                scores=[0.1],  # Low score but not zero
                messages=None,
            )

        return scored_group

    async def evaluate(self, **kwargs):
        """
        Evaluate the agent on a subset of the GAIA benchmark.

        Provides metrics on:
        - Success rate
        - Average score
        - Execution time
        - Step efficiency
        """
        logger.info("Starting evaluation on GAIA benchmark")

        # Use a fixed subset of examples for evaluation
        # Start from a different point than training to avoid overlap
        eval_start = len(self.examples) // 2
        eval_count = min(
            10, len(self.examples) // 10
        )  # 10% of dataset or 10 examples max

        eval_examples = self.examples[eval_start : eval_start + eval_count]

        results = []
        correct_count = 0

        # Create items for evaluation
        eval_items = []
        for example in eval_examples:
            # Create an Item for this example
            item = Item(
                prompt=example["question"],
                metadata={
                    "task_id": example["task_id"],
                    "task": example["task"],
                    "true_answer": example["true_answer"],
                    "file_name": example.get("file_name", ""),
                },
            )
            eval_items.append(item)

        # Use the existing process-based trajectory collection
        scored_groups, _ = await self.collect_trajectories(eval_items)

        # Process the scored groups
        for scored_group in scored_groups:
            if isinstance(scored_group, ScoredDataGroup):
                score = (
                    scored_group["scores"][0]
                    if "scores" in scored_group and scored_group["scores"]
                    else 0
                )

                # Try to extract the task_id from metadata
                task_id = None
                if (
                    "group_overrides" in scored_group
                    and scored_group["group_overrides"]
                ):
                    task_id = scored_group["group_overrides"].get("task_id")

                results.append(
                    {
                        "task_id": task_id or "unknown",
                        "score": score,
                    }
                )

                if score > 0.5:  # Consider it correct if score > 0.5
                    correct_count += 1

                # Since we're using the process-based approach, the execution time
                # is stored in the server metrics which are already tracked

        # Calculate metrics
        if results:
            success_rate = correct_count / len(results)
            avg_score = sum(r["score"] for r in results) / len(results)

            # Calculate average time from agent_execution_times if available
            avg_time = 0
            if self.agent_execution_times:
                avg_time = sum(self.agent_execution_times) / len(
                    self.agent_execution_times
                )

            logger.info(f"Evaluation complete on {len(results)} examples:")
            logger.info(f"  Success rate: {success_rate:.2f}")
            logger.info(f"  Average score: {avg_score:.2f}")
            logger.info(f"  Average execution time: {avg_time:.2f}s")

            # Update wandb metrics
            if self.config.use_wandb:
                metrics = {
                    "eval/success_rate": success_rate,
                    "eval/avg_score": avg_score,
                    "eval/num_examples": len(results),
                    "eval/avg_execution_time": avg_time,
                }

                await self.wandb_log(metrics)

    def save_checkpoint(self, step, data=None):
        """Save environment state for checkpointing."""
        if data is None:
            data = {}
        # Save the iteration counter
        data["iter"] = self.iter
        # Save the current index in the dataset
        data["current_index"] = self.current_index
        # Call the parent class save_checkpoint
        super().save_checkpoint(step, data)

    def load_checkpoint(self):
        """Load environment state from checkpoint."""
        # Call parent method first
        super().load_checkpoint()
        # Check if we loaded iter and current_index
        if hasattr(self, "checkpoint_data"):
            if "iter" in self.checkpoint_data:
                self.iter = self.checkpoint_data["iter"]
            if "current_index" in self.checkpoint_data:
                self.current_index = self.checkpoint_data["current_index"]

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        """
        Log to wandb with comprehensive metrics.
        """
        if wandb_metrics is None:
            wandb_metrics = dict()

        # Try to calculate percent_correct, skip if there's a division by zero
        try:
            wandb_metrics["train/percent_correct"] = sum(
                self.percent_correct_buffer
            ) / len(self.percent_correct_buffer)
        except ZeroDivisionError:
            # Skip if buffer is empty
            pass

        # Log agent performance metrics
        if self.agent_execution_times and len(self.agent_execution_times) > 0:
            wandb_metrics["agent/avg_execution_time"] = sum(
                self.agent_execution_times
            ) / len(self.agent_execution_times)
            wandb_metrics["agent/max_execution_time"] = max(self.agent_execution_times)
            wandb_metrics["agent/min_execution_time"] = min(self.agent_execution_times)
            # Reset the buffer
            self.agent_execution_times = []

        # Add dataset iteration tracking
        wandb_metrics["train/dataset_iterations"] = self.iter
        wandb_metrics["train/current_dataset_index"] = self.current_index

        # Add custom evaluation metrics
        for item in self.eval_metrics:
            wandb_metrics[item[0]] = item[1]

        # Clear buffers after logging
        self.percent_correct_buffer = []
        self.eval_metrics = []

        # Call the parent method to handle the server metrics
        await super().wandb_log(wandb_metrics)

    async def cleanup(self):
        """Clean up resources when environment is closed."""
        logger.info("Cleaning up SmolagentsEnv resources")

        # Clean up the server proxy manager
        if self.server_proxy_manager:
            self.server_proxy_manager.stop()
            logger.info("Stopped server proxy manager")

        # Let the parent class do its cleanup
        await super().cleanup()


if __name__ == "__main__":
    SmolagentsEnv.cli()
