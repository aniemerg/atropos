"""
SmolagentsEnv - Environment for creating high-quality agent trajectories
for training language models using the SmolaGents agent framework.
"""

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import wandb
from pydantic import BaseModel, Field
from smolagents import CodeAgent, Tool
from smolagents.tools import tool  # Import the tool decorator

# Apply the patched AsyncBridge early to ensure all calls use the enhanced version
from environments.smolagents_integration.patched_async_bridge import patch_asyncbridge
patch_asyncbridge()  # Must be called before any imports that use AsyncBridge

from dataclasses import dataclass
from typing import Dict, Any, Optional

from atroposlib.envs.base import BaseEnv, BaseEnvConfig, ScoredDataGroup
from atroposlib.envs.server_handling.openai_server import OpenaiConfig, OpenAIServer
from atroposlib.envs.server_handling.server_manager import ServerManager
from environments.smolagents_integration.atropos_smolagents_integration import AtroposServerModel
from environments.smolagents_integration.tools.file_tools import read_file, write_file, append_to_file

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
formatter = logging.Formatter('%(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


class SmolagentsEnvConfig(BaseEnvConfig):
    """Configuration for SmolagentsEnv."""
    
    dataset_path: str = Field(
        default="data/gaia", description="Path to GAIA dataset"
    )
    split: str = Field(
        default="validation", description="Dataset split to use (validation, test)"
    )
    use_chat_completion: bool = Field(
        default=True, description="Use chat completion API"
    )
    max_steps: int = Field(
        default=12, description="Maximum number of agent steps"
    )
    tools_enabled: List[str] = Field(
        default=["file_reader", "file_writer", "web_search"], 
        description="Enabled tools (file_reader, file_writer, web_search)"
    )
    agent_verbosity: int = Field(
        default=2, description="Agent verbosity level (0-3)"
    )
    scoring_strategy: str = Field(
        default="combined", 
        description="Scoring strategy: basic, correctness, or combined"
    )
    bridge_timeout_factor: float = Field(
        default=1.5, 
        description="Safety factor for AsyncBridge timeouts"
    )
    max_concurrent_agents: int = Field(
        default=5, 
        description="Maximum concurrent agent executions"
    )
    length_penalty_weight: float = Field(
        default=0.1,
        description="Weight for length penalty in scoring (0.0 to disable)"
    )
    save_full_traces: bool = Field(
        default=True,
        description="Save full agent execution traces in the output"
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
            max_concurrent_agents=5,
            dataset_path="data/gaia",
            split="train",
            use_chat_completion=True,
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
        # Initialize the base class
        super().__init__(config, server_configs, slurm, testing)
        
        # Initialize dataset variables
        self.examples = []
        self.current_index = 0
        self.iter = 0  # Add iter for checkpoint tracking
        self.agent = None
        self.model = None
        self.tools = []
        self.agent_semaphore = None  # Will be initialized in setup()
        
        # Save config for easier access
        self.max_steps = config.max_steps
        self.tools_enabled = config.tools_enabled
        self.verbosity = config.agent_verbosity
        self.scoring_strategy = config.scoring_strategy
        
        # Track agent execution times and metrics
        self.agent_execution_times = []
        self.percent_correct_buffer = []
        self.eval_metrics = []
        
    async def setup(self):
        """Set up the environment, load dataset, and initialize agent components."""
        logger.info("Setting up SmolagentsEnv...")
        logger.info(f"Using dataset split: {self.config.split}")
        
        # Create a semaphore to limit concurrent agent executions
        self.agent_semaphore = asyncio.Semaphore(self.config.max_concurrent_agents)
        
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
            
            logger.info(f"Loaded {len(self.examples)} examples from GAIA {self.config.split} set")
        except Exception as e:
            logger.error(f"Error loading GAIA dataset: {e}")
            # Create empty list if dataset loading fails
            self.examples = []
        
        # Create AtroposServerModel wrapper
        self.model = AtroposServerModel(
            server=self.server,
            use_chat_completion=self.config.use_chat_completion,
            model_id="atropos-smolagents",
        )
        
        # Create tools for the agent
        self.tools = self._create_tools()
        
        # Initialize the CodeAgent with our tools and model
        self.agent = CodeAgent(
            tools=self.tools,
            model=self.model,
            max_steps=self.max_steps,
            additional_authorized_imports=["*"],  # Allow all imports for flexibility
            verbosity_level=self.verbosity,
        )
        
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
            from environments.smolagents_integration.tools.tavily_tools import TavilySearchTool, TavilyExtractTool
            if os.environ.get("TAVILY_API_KEY"):
                tavily_search = TavilySearchTool(api_key=os.environ.get("TAVILY_API_KEY"))
                tavily_extract =  TavilyExtractTool(api_key=os.environ.get("TAVILY_API_KEY"))
                tools.append(tavily_search)
                tools.append(tavily_extract)
                logger.info("Added web_search tool")
            else:
                logger.warning("TAVILY_API_KEY not set in environment, web search disabled")
        except Exception as e:
            logger.warning(f"Could not create web search tool: {e}")
        
        # Log tool info for debugging
        tool_names = []
        for i, tool in enumerate(tools):
            if hasattr(tool, 'name'):
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
    
    async def _collect_with_semaphore(self, semaphore, item: Item) -> Tuple[Any, List[Item]]:
        """Run agent with semaphore to limit concurrency."""
        async with semaphore:
            return await self.collect_trajectory(item)
    
    async def collect_trajectories(self, items: Union[Item, List[Item]]) -> Tuple[
        Union[Optional[ScoredDataGroup], List[Optional[ScoredDataGroup]], List[Any]],
        List[Item],
    ]:
        """
        Collect trajectories for multiple items with concurrency control.
        """
        # Handle both single item and list of items
        if not isinstance(items, list):
            items = [items] * self.config.group_size
        
        # Create tasks with semaphore to limit concurrency
        tasks = []
        for item in items:
            tasks.append(self._collect_with_semaphore(self.agent_semaphore, item))
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)
        
        # Process results
        backlog = []
        to_postprocess = []
        
        logger.info(f"Got {len(results)} results from agent executions")
        
        for i, result in enumerate(results):
            logger.info(f"Result {i}: type={type(result)}, is_none={result[0] is None}, backlog_len={len(result[1])}")
            if result[0] is not None:
                to_postprocess.append(result[0])
                logger.info(f"  Added result to to_postprocess: {type(result[0])}")
            else:
                logger.warning(f"  Skipping None result at index {i}")
            backlog.extend(result[1])
        
        logger.info(f"Final to_postprocess: type={type(to_postprocess)}, len={len(to_postprocess)}")
        logger.info(f"Final backlog: type={type(backlog)}, len={len(backlog)}")
        
        return to_postprocess, backlog
        
    async def postprocess_histories(
        self, histories: Union[ScoredDataGroup, List[ScoredDataGroup]]
    ) -> ScoredDataGroup:
        """
        Post-process the agent histories.
        
        We need to merge multiple ScoredDataGroups into a single ScoredDataGroup.
        """
        logger.info(f"postprocess_histories called with: type={type(histories)}, is_none={histories is None}")
        if isinstance(histories, list):
            logger.info(f"  List length: {len(histories)}")
            
        if not isinstance(histories, list):
            # If it's already a single ScoredDataGroup, return it with group_overrides
            logger.info(f"  Single history, returning directly: {type(histories)}")
            if "group_overrides" not in histories or histories["group_overrides"] is None:
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
            overrides=None
        )
        
        # Merge all the fields
        for i, history in enumerate(histories):
            logger.info(f"  Processing history {i}: type={type(history)}, is_none={history is None}")
            if history is not None:
                logger.info(f"    History {i} tokens: {len(history['tokens'])}")
                merged["tokens"].extend(history["tokens"])
                merged["masks"].extend(history["masks"])
                merged["scores"].extend(history["scores"])
                
                if merged["messages"] is not None and "messages" in history:
                    logger.info(f"    History {i} messages: {len(history['messages'])}")
                    merged["messages"].extend(history["messages"])
        
        logger.info(f"  Final merged data: tokens={len(merged['tokens'])}, scores={len(merged['scores'])}")
        return merged
    
    async def collect_trajectory(self, item: Item) -> Tuple[Any, List[Item]]:
        """
        Run the agent on a single problem and collect results.
        
        This method:
        1. Executes the CodeAgent with the prompt
        2. Times the execution for monitoring
        3. Extracts the agent memory and final answer
        4. Scores the trajectory against the reference solution
        5. Returns the scored data for training
        """
        logger.info(f"Running agent on task: {item.metadata['task_id']}")
        
        # Set up execution timing
        start_time = time.time()
        agent_result = None
        error = None
        
        try:
            # Execute the agent with timeout protection
            timeout = self.max_steps * 20  # Estimate: 20 seconds per step max
            agent_task = asyncio.create_task(self._run_agent(item.prompt))
            
            try:
                # Wait for the agent to complete with timeout
                agent_result = await asyncio.wait_for(agent_task, timeout=timeout)
                
                # Record execution time
                execution_time = time.time() - start_time
                self.agent_execution_times.append(execution_time)
                logger.info(f"Agent completed in {execution_time:.2f} seconds")
                
                # Extract agent memory and final answer
                agent_memory = self._extract_agent_memory()
                
                # Score the trajectory 
                score = self._score_trajectory(
                    item.prompt, 
                    agent_result, 
                    item.metadata["true_answer"],
                    agent_memory,
                    execution_time
                )
                
                # Create scored data for this trajectory
                scored_data = {
                    "prompt": item.prompt,
                    "response": agent_result,
                    "score": score,
                    "task_id": item.metadata["task_id"],
                    "task": item.metadata["task"],
                    "true_answer": item.metadata["true_answer"],
                    "execution_time": execution_time,
                }
                
                # Add agent memory if configured
                if self.config.save_full_traces:
                    scored_data["agent_memory"] = agent_memory
                
                # Generate ScoredDataGroup for the trainer
                scored_group = self._create_scored_data_group(item, scored_data)
                
                return scored_group, []
                
            except asyncio.TimeoutError:
                error = f"Agent execution timed out after {timeout} seconds"
                logger.error(f"{error} for task {item.metadata['task_id']}")
                # Create fallback response
                return self._create_fallback_response(item, error), []
                
        except Exception as e:
            execution_time = time.time() - start_time
            error = f"Error in agent execution: {str(e)}"
            logger.error(f"{error} for task {item.metadata['task_id']}")
            # Create fallback response for training
            return self._create_fallback_response(item, error), []
    
    async def _run_agent(self, prompt: str) -> str:
        """Run the agent with the given prompt."""
        # This is wrapped in its own method for easier error handling
        return self.agent.run(prompt)
    
    def _extract_agent_memory(self) -> List[Dict]:
        """Extract the agent's execution trace and memory."""
        try:
            # Extract chat history or other memory format from agent
            if hasattr(self.agent, "write_memory_to_messages"):
                return self.agent.write_memory_to_messages()
            elif hasattr(self.agent, "memory"):
                # Convert memory to message format if possible
                return self._format_agent_memory(self.agent.memory)
            else:
                logger.warning("Could not extract agent memory - agent has no memory attribute")
                return []
        except Exception as e:
            logger.error(f"Error extracting agent memory: {e}")
            return []
    
    def _format_agent_memory(self, memory) -> List[Dict]:
        """Format agent memory into a standardized message format."""
        messages = []
        
        # Handle different memory formats
        if isinstance(memory, list):
            for entry in memory:
                if isinstance(entry, dict) and "role" in entry and "content" in entry:
                    # Already in message format
                    messages.append(entry)
                elif hasattr(entry, "as_message"):
                    # Has conversion method
                    messages.append(entry.as_message())
                else:
                    # Try to infer format
                    role = "system"
                    if hasattr(entry, "role"):
                        role = entry.role
                    
                    content = str(entry)
                    if hasattr(entry, "content"):
                        content = entry.content
                        
                    messages.append({"role": role, "content": content})
        
        return messages
    
    def _score_trajectory(
        self, 
        prompt: str, 
        agent_response: str, 
        true_answer: str, 
        agent_memory: List[Dict] = None,
        execution_time: float = 0
    ) -> float:
        """
        Score the agent trajectory based on the chosen scoring strategy.
        
        Supports multiple scoring methods:
        - basic: Simple correctness check
        - correctness: More sophisticated answer validation
        - combined: Blend of correctness, efficiency, and reasoning quality
        """
        try:
            # Apply the appropriate scoring strategy
            if self.scoring_strategy == "basic":
                # Simple scoring: check if true answer appears in response
                if true_answer.lower() in agent_response.lower():
                    return 1.0
                return 0.0
                
            elif self.scoring_strategy == "correctness":
                # Try to use gaia_scorer if available
                try:
                    from environments.smolagents_integration.gaia_scorer import score_answer
                    return score_answer(agent_response, true_answer)
                except ImportError:
                    # Fall back to basic scoring
                    if true_answer.lower() in agent_response.lower():
                        return 1.0
                    return 0.0
                    
            elif self.scoring_strategy == "combined":
                # Combined scoring: correctness + efficiency + reasoning quality
                base_score = 0.0
                
                # 1. Check correctness (50% of score)
                try:
                    from environments.smolagents_integration.gaia_scorer import score_answer
                    correctness_score = score_answer(agent_response, true_answer) * 0.5
                except ImportError:
                    # Fall back to basic scoring
                    correctness_score = (1.0 if true_answer.lower() in agent_response.lower() else 0.0) * 0.5
                
                # 2. Check efficiency (25% of score)
                # Penalize long execution times and many steps
                steps_count = len(agent_memory) if agent_memory else 0
                efficiency_score = 0.25  # Start with full efficiency score
                
                # Penalty for excessive steps (above 75% of max)
                if steps_count > (self.max_steps * 0.75):
                    efficiency_score *= 0.7
                    
                # Penalty for long execution time (if we have other executions to compare)
                if self.agent_execution_times and len(self.agent_execution_times) > 5:
                    avg_time = np.mean(self.agent_execution_times)
                    if execution_time > (avg_time * 1.5):
                        efficiency_score *= 0.8
                
                # 3. Check reasoning quality (25% of score)
                reasoning_score = 0.0
                
                # Look for step-by-step reasoning with clear intermediate steps
                # This is a simplified heuristic - could be much more sophisticated
                if agent_memory and len(agent_memory) > 0:
                    # Look for reasoning markers in the agent's work
                    reasoning_markers = [
                        "first", "second", "third", "step", "approach",
                        "reason", "because", "therefore", "thus", "hence",
                        "calculate", "compute", "solve", "find", "determine"
                    ]
                    
                    # Count reasoning markers in memory content
                    marker_count = 0
                    for message in agent_memory:
                        content = message.get("content", "")
                        if isinstance(content, str):
                            marker_count += sum(1 for marker in reasoning_markers if marker in content.lower())
                    
                    # Normalize reasoning score
                    reasoning_score = min(0.25, (marker_count / 10) * 0.25)
                
                # Combine scores
                base_score = correctness_score + efficiency_score + reasoning_score
                
                # Apply length penalty if configured
                if self.config.length_penalty_weight > 0 and agent_response:
                    response_length = len(agent_response)
                    # Penalize very long responses
                    if response_length > 2000:
                        length_penalty = min(0.3, self.config.length_penalty_weight * (response_length - 2000) / 1000)
                        base_score = max(0.0, base_score - length_penalty)
                
                return base_score
            
            else:
                # Unknown scoring strategy
                logger.warning(f"Unknown scoring strategy: {self.scoring_strategy}, using basic")
                if true_answer.lower() in agent_response.lower():
                    return 1.0
                return 0.0
                
        except Exception as e:
            logger.error(f"Error in scoring: {e}")
            return 0.0
    
    def _create_scored_data_group(self, item: Item, scored_data: Dict) -> ScoredDataGroup:
        """
        Create a ScoredDataGroup for the trainer API.
        
        Converts the agent trajectory into tokenized format for the trainer.
        """
        # Prepare the data in message format or token format
        if self.config.include_messages:
            # Create message format with agent memory if available
            messages = []
            
            # Add system message with task description
            messages.append({
                "role": "system",
                "content": "You are an AI assistant solving a task with reasoning and problem-solving skills."
            })
            
            # Add user message with the prompt
            messages.append({
                "role": "user",
                "content": item.prompt
            })
            
            # For message format, extract agent memory if available
            if self.config.save_full_traces and "agent_memory" in scored_data:
                # Add intermediate reasoning steps
                for message in scored_data["agent_memory"]:
                    messages.append(message)
            else:
                # Just add the final response
                messages.append({
                    "role": "assistant",
                    "content": scored_data["response"]
                })
            
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
    
    def _create_fallback_response(self, item: Item, error_message: str) -> ScoredDataGroup:
        """Create a fallback response for failed agent executions."""
        # Create a minimal valid response with low score
        if self.config.include_messages:
            messages = [
                {"role": "system", "content": "You are an AI assistant solving tasks."},
                {"role": "user", "content": item.prompt},
                {"role": "assistant", "content": f"I'm unable to solve this task. Error: {error_message}"}
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
        eval_count = min(10, len(self.examples) // 10)  # 10% of dataset or 10 examples max
        
        eval_examples = self.examples[eval_start:eval_start + eval_count]
        
        results = []
        correct_count = 0
        total_time = 0
        
        for example in eval_examples:
            # Create an Item for this example
            item = Item(
                prompt=example["question"],
                metadata={
                    "task_id": example["task_id"],
                    "task": example["task"],
                    "true_answer": example["true_answer"],
                    "file_name": example.get("file_name", ""),
                }
            )
            
            # Use a separate worker for each evaluation
            worker = asyncio.create_task(self.collect_trajectory(item))
            self.eval_workers.add(worker)
            worker.add_done_callback(self.eval_workers.discard)
            
            # Wait for completion and collect results
            result, _ = await worker
            
            if result:
                # Extract response and score from scored group
                if isinstance(result, ScoredDataGroup):
                    score = result["scores"][0] if result["scores"] else 0
                    results.append({
                        "task_id": item.metadata["task_id"],
                        "score": score,
                    })
                    
                    if score > 0.5:  # Consider it correct if score > 0.5
                        correct_count += 1
                    
                    # Extract execution time if available
                    total_time += getattr(result, "execution_time", 0)
        
        # Calculate metrics
        if results:
            success_rate = correct_count / len(results)
            avg_score = sum(r["score"] for r in results) / len(results)
            avg_time = total_time / len(results) if total_time > 0 else 0
            
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
            wandb_metrics["agent/avg_execution_time"] = sum(self.agent_execution_times) / len(self.agent_execution_times)
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
        
        # Free AsyncBridge resources to prevent hanging
        try:
            from atroposlib.utils.async_bridge import shutdown_bridge
            shutdown_bridge()
            logger.info("Shutdown AsyncBridge successfully")
        except Exception as e:
            logger.error(f"Error shutting down AsyncBridge: {e}")
        
        # Let the parent class do its cleanup
        await super().cleanup()


if __name__ == "__main__":
    SmolagentsEnv.cli()