# Atropos-SmolaGents Integration

This integration enables the use of SmolaGents' agent capabilities with Atropos' server-based LLM architecture for benchmarking with the GAIA benchmark.

## Overview

The integration consists of:

1. **AtroposServerModel**: A SmolaGents Model implementation that wraps Atropos servers.
2. **GAIABenchmarkEnv**: An Atropos environment that uses SmolaGents' CodeAgent with the Atropos server.
3. **Scripts** for running the GAIA benchmark with the integration.

## Files

- `atropos_smolagents_integration.py`: Contains the `AtroposServerModel` class that acts as a bridge between SmolaGents and Atropos.
- `gaia_benchmark_env.py`: The Atropos environment for running GAIA benchmark evaluations.
- `run_gaia_benchmark.py`: Script for running the GAIA benchmark with the integrated system.
- `run_gaia_single_task.py`: Script for running a single GAIA benchmark task with detailed output.

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
3. Create custom tools appropriate for your use case.

Example:
```python
from environments.smolagents_integration.atropos_smolagents_integration import AtroposServerModel
from smolagents import CodeAgent, Tool

# Assuming you have an Atropos server
model = AtroposServerModel(server=your_server)

# Create tools
tools = [
    Tool(
        name="your_tool",
        description="Description of your tool",
        inputs={"param": {"type": "string", "description": "Parameter description"}},
        function=your_function
    )
]

# Create agent
agent = CodeAgent(
    tools=tools,
    model=model,
    max_steps=10
)

# Run the agent
result = agent.run("Your prompt here")
```

## Troubleshooting

- **Async/Sync issues**: If you encounter async-related errors, make sure you're correctly bridging the async/sync boundary.
- **Message format errors**: Check that message conversions between SmolaGents and Atropos formats are correct.
- **Missing GAIA data**: Make sure you've downloaded the GAIA benchmark data correctly.

## Contributing

If you find bugs or have suggestions for improvements, please create an issue or submit a pull request.
