# Atropos-SmolaGents Integration

This integration enables the use of SmolaGents' agent capabilities with Atropos' server-based LLM architecture for both GAIA benchmark evaluation and high-quality training data generation.

## Overview

The integration consists of:

1. **AtroposServerModel**: A SmolaGents Model implementation that wraps Atropos servers.
2. **Tools**: File manipulation and web searching tools for the agent to use.
3. **GAIABenchmarkEnv**: An Atropos environment for running GAIA benchmark evaluations.
4. **SmolagentsEnv**: A full-fledged Atropos environment for generating high-quality agent trajectories.
5. **Scripts** for running the GAIA benchmark with the integration.

## Files

- `atropos_smolagents_integration.py`: Contains the `AtroposServerModel` class that acts as a bridge between SmolaGents and Atropos.
- `gaia_benchmark_env.py`: The Atropos environment for running GAIA benchmark evaluations.
- `smolagents_env.py`: The complete Atropos environment for generating training data.
- `patched_async_bridge.py`: Enhanced AsyncBridge with better error handling and debugging.
- `run_gaia_benchmark.py`: Script for running the GAIA benchmark with the integrated system.
- `run_gaia_single_task.py`: Script for running a single GAIA benchmark task with detailed output.
- `tools/`: Directory containing implementations of tools:
  - `file_tools.py`: Tools for reading, writing, and appending to files
  - `tavily_tools.py`: Web search and page extraction tools powered by Tavily

## Installation

1. First, make sure you have Atropos installed.
2. Install SmolaGents:
   ```
   pip install smolagents
   ```
3. Install the GAIA benchmark dependencies:
   ```
   pip install datasets pandas huggingface_hub
   ```
4. For web tools, install Tavily:
   ```
   pip install tavily-python
   ```

## Environment Variables

The integration uses the following environment variables:

- `OPENAI_API_KEY`: Required for OpenAI API access when using LiteLLM model (test mode) or when using OpenAI models with Atropos.
- `TAVILY_API_KEY`: Required for web search and page extraction tools. You can get a key from [Tavily](https://tavily.com/).

If you need to use a different API key per run, you can also provide them as command-line arguments:
```
--api-key your_api_key
```

## Using the Integration

### Using SmolagentsEnv for Training Data Generation

Generate SFT training data with the following command:

```bash
atropos-sft-gen output.jsonl --tokenizer NousResearch/DeepHermes-3-Llama-3-8B-Preview \
  --save-messages --env smolagents
```

For local testing without connecting to the API server:

```bash
python -m environments.smolagents_integration.smolagents_env process \
  --env_data_path_to_save_groups output.jsonl \
  --env_total_steps 10 \
  --env_group_size 2 \
  --env_include_messages true \
  --env_max_concurrent_agents 4 \
  --env_use_chat_completion true \
  --openai_model_name "gpt-4o" \
  --openai_base_url "https://api.openai.com/v1" \
  --openai_api_key "$OPENAI_API_KEY"
```

To serve the environment for a trainer:

```bash
python -m environments.smolagents_integration.smolagents_env serve \
  --env_rollout_server_url "http://localhost:8000" \
  --env_use_chat_completion true \
  --env_max_concurrent_agents 5 \
  --env_group_size 8 \
  --openai_model_name "gpt-4o" \
  --openai_base_url "https://api.openai.com/v1" \
  --openai_api_key "$OPENAI_API_KEY"
```

### Running a Single GAIA Task

To run a single task from the GAIA benchmark:

```bash
python -m environments.smolagents_integration.run_gaia_single_task --task-id <task_id> --model-name <model_name> --base-url <server_url>
```

Example:
```bash
python -m environments.smolagents_integration.run_gaia_single_task --task-id GAIA2023_P0003 --model-name llama-3-70b-instruct --base-url http://localhost:8000/v1
```

Options:
- `--use-chat-completion`: Use chat completion API instead of completion API.
- `--max-steps`: Maximum number of steps for the agent (default: 12).
- `--output-dir`: Directory to store results (default: "gaia_results").
- `--use-local-model`: Use LiteLLM instead of Atropos for testing without an Atropos server.

### Running the Full GAIA Benchmark

To run the full GAIA benchmark evaluation:

```bash
python -m environments.smolagents_integration.run_gaia_benchmark --model-name <model_name> --base-url <server_url>
```

Options:
- `--dataset-path`: Path to the GAIA benchmark data (default: "data/gaia").
- `--split`: Dataset split to use (default: "validation").
- `--batch-size`: Batch size for training (default: 1).
- `--use-wandb`: Enable wandb logging.

## How It Works

### SmolagentsEnv

The `SmolagentsEnv` class provides a complete environment for generating high-quality agent trajectories:

1. Loads tasks from the GAIA benchmark dataset
2. Creates an AtroposServerModel wrapping Atropos servers
3. Initializes a CodeAgent with configurable tools
4. Manages agent execution and trajectory collection
5. Scores trajectories based on correctness, efficiency, and reasoning quality
6. Integrates with Atropos SFT generation pipeline

Configuration options for `SmolagentsEnv`:

```python
class SmolagentsEnvConfig(BaseEnvConfig):
    dataset_path: str = Field(default="data/gaia", description="Path to GAIA dataset")
    split: str = Field(default="train", description="Dataset split to use")
    use_chat_completion: bool = Field(default=True, description="Use chat completion API")
    max_steps: int = Field(default=12, description="Maximum number of agent steps")
    tools_enabled: List[str] = Field(
        default=["python", "file_reader", "web_search"], 
        description="Enabled tools"
    )
    agent_verbosity: int = Field(default=2, description="Agent verbosity level")
    scoring_strategy: str = Field(
        default="combined", 
        description="Scoring strategy: basic, correctness, or combined"
    )
    bridge_timeout_factor: float = Field(default=1.5, description="Safety factor for AsyncBridge timeouts")
    max_concurrent_agents: int = Field(default=5, description="Maximum concurrent agent executions")
    length_penalty_weight: float = Field(default=0.1, description="Weight for length penalty in scoring")
    save_full_traces: bool = Field(default=True, description="Save full agent execution traces")
```

### AtroposServerModel

The `AtroposServerModel` class:

1. Wraps an Atropos server in SmolaGents' Model interface.
2. Converts between SmolaGents' message format and Atropos server parameters.
3. Bridges the async/sync boundary between the two frameworks.

```python
from environments.smolagents_integration.atropos_smolagents_integration import AtroposServerModel
from atroposlib.envs.server_handling.openai_server import OpenAIServer, OpenaiConfig

# Create an Atropos server
server_config = OpenaiConfig(
    api_key="x",
    base_url="http://localhost:8000/v1",
    model_name="llama-3-70b-instruct"
)
server = OpenAIServer(server_config)

# Wrap it in AtroposServerModel
model = AtroposServerModel(
    server=server,
    use_chat_completion=False
)
```

### GAIABenchmarkEnv

The `GAIABenchmarkEnv` class:

1. Creates an AtroposServerModel that wraps the Atropos server.
2. Initializes a CodeAgent with this model and appropriate tools.
3. Handles GAIA benchmark tasks, including file attachments.
4. Scores and evaluates agent performance.

## Creating Your Own Integrations

To create your own integration:

1. Create an instance of `AtroposServerModel` with your Atropos server.
2. Use it with any SmolaGents agent, such as CodeAgent or ToolCallingAgent.
3. Create custom tools appropriate for your use case, or use the provided tools.

Example:
```python
from environments.smolagents_integration.atropos_smolagents_integration import AtroposServerModel
from environments.smolagents_integration.tools.file_tools import read_file, write_file, append_to_file
from environments.smolagents_integration.tools.tavily_tools import TavilySearchTool, TavilyExtractTool
from smolagents import CodeAgent, Tool

# Assuming you have an Atropos server
model = AtroposServerModel(server=your_server)

# Create tools array with the tools you need
tools = []

# Add file tools
tools.extend([read_file, write_file, append_to_file])

# Add web tools if needed
tavily_api_key = os.environ.get("TAVILY_API_KEY")
if tavily_api_key:
    tools.extend([
        TavilySearchTool(api_key=tavily_api_key),
        TavilyExtractTool(api_key=tavily_api_key)
    ])

# Add your own custom tools
tools.append(
    Tool(
        name="your_tool",
        description="Description of your tool",
        inputs={"param": {"type": "string", "description": "Parameter description"}},
        function=your_function
    )
)

# Create agent - CodeAgent has Python execution built-in
agent = CodeAgent(
    tools=tools,
    model=model,
    max_steps=10,
    additional_authorized_imports=["os", "json", "re", "datetime", "requests"],
)

# Run the agent
result = agent.run("Your prompt here")
```

## Troubleshooting

- **Async/Sync issues**: If you encounter async-related errors, make sure you're correctly bridging the async/sync boundary. The `patched_async_bridge.py` provides enhanced error reporting and timeout handling.
- **Message format errors**: Check that message conversions between SmolaGents and Atropos formats are correct.
- **Missing GAIA data**: Make sure you've downloaded the GAIA benchmark data correctly.
- **Web tool errors**: If Tavily tools aren't working, make sure you have set the `TAVILY_API_KEY` environment variable and have installed the `tavily-python` package.
- **Tool import errors**: If you see errors about missing tool modules, ensure your working directory allows proper imports of the tools folder.
- **Permission errors with file tools**: Ensure your process has the correct permissions to read/write files in the directories being accessed.
- **Semaphore limitation**: If many agent runs are timing out, try lowering the `max_concurrent_agents` parameter in SmolagentsEnv.
- **Memory issues**: If you encounter memory leaks, ensure that AsyncBridge cleanup is being called during environment shutdown.

## Contributing

If you find bugs or have suggestions for improvements, please create an issue or submit a pull request.
