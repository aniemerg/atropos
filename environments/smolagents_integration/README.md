# Atropos-SmolaGents Integration

This integration enables the use of SmolaGents' agent capabilities with Atropos' server-based LLM architecture for benchmarking with the GAIA benchmark.

## Overview

The integration consists of:

1. **AtroposServerModel**: A SmolaGents Model implementation that wraps Atropos servers.
2. **Tools**: File manipulation and web searching tools for the agent to use.
3. **GAIABenchmarkEnv**: An Atropos environment that uses SmolaGents' CodeAgent with the Atropos server.
4. **Scripts** for running the GAIA benchmark with the integration.

## Files

- `atropos_smolagents_integration.py`: Contains the `AtroposServerModel` class that acts as a bridge between SmolaGents and Atropos.
- `gaia_benchmark_env.py`: The Atropos environment for running GAIA benchmark evaluations.
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

- **Async/Sync issues**: If you encounter async-related errors, make sure you're correctly bridging the async/sync boundary.
- **Message format errors**: Check that message conversions between SmolaGents and Atropos formats are correct.
- **Missing GAIA data**: Make sure you've downloaded the GAIA benchmark data correctly.
- **Web tool errors**: If Tavily tools aren't working, make sure you have set the `TAVILY_API_KEY` environment variable and have installed the `tavily-python` package.
- **Tool import errors**: If you see errors about missing tool modules, ensure your working directory allows proper imports of the tools folder.
- **Permission errors with file tools**: Ensure your process has the correct permissions to read/write files in the directories being accessed.

## Contributing

If you find bugs or have suggestions for improvements, please create an issue or submit a pull request.
