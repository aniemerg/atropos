# AsyncBridge

AsyncBridge is a robust solution for executing asynchronous (async/await) code from synchronous code contexts, designed specifically for the Atropos framework. It provides a thread-safe and efficient way to bridge the gap between synchronous and asynchronous code without creating new event loops and threads for each call.

## Problem Statement

When working with libraries that use different programming paradigms, you often need to bridge between:

- **Synchronous code** (using standard function calls)
- **Asynchronous code** (using `async`/`await`)

The most common solution is to create a new thread and event loop for each async operation, but this approach has serious drawbacks:

- Creates excessive threads, leading to resource exhaustion
- Causes thread coordination issues and potentially deadlocks
- Makes error handling difficult across thread boundaries
- Leads to poor performance under load

## AsyncBridge Solution

AsyncBridge solves these problems by providing:

1. **Single Worker Thread**: Maintains one dedicated thread with a persistent event loop
2. **Thread-Safe Queue**: Manages task submission and coordination
3. **Proper Resource Management**: Handles cleanup and cancellation correctly
4. **Reliable Error Propagation**: Preserves exception context across threads
5. **Singleton Pattern**: Provides a central bridge instance throughout the application

## Usage Examples

### Basic Usage

```python
from atroposlib.utils.async_bridge import run_async

# Run an async function from synchronous code
async def fetch_data(url):
    # ... async implementation ...
    return result

# Call it from synchronous code
result = run_async(fetch_data, "https://example.com", timeout=30)

# Automatic cleanup at exit
from atroposlib.utils.async_bridge import shutdown_bridge
shutdown_bridge()
```

### With Atropos Server

```python
from atroposlib.envs.server_handling.openai_server import OpenAIServer, OpenaiConfig
from atroposlib.utils.async_bridge import run_async

# Create server configuration
server_config = OpenaiConfig(
    api_key="x",  # Use "x" for local servers
    base_url="http://localhost:8000/v1",
    model_name="gpt-3.5-turbo",
    timeout=30.0,
)

# Create server
server = OpenAIServer(server_config)

# Use AsyncBridge to call the async method
messages = [{"role": "user", "content": "Hello, world!"}]
response = run_async(
    server.chat_completion,
    messages=messages,
    max_tokens=50,
    temperature=0.7,
    timeout=30.0  # 30 second timeout
)

# Process response
content = response.choices[0].message.content
print(f"Response: {content}")
```

### In SmolaGents Integration

AsyncBridge is particularly useful in the Atropos-SmolaGents integration where SmolaGents' synchronous API needs to interact with Atropos' asynchronous servers. See the updated implementation in `environments/smolagents_integration/atropos_smolagents_integration.py`.

## API Reference

### Main Functions

- `run_async(coro_or_func, *args, timeout=None, **kwargs)`: Run an async function or coroutine from synchronous code
- `shutdown_bridge()`: Gracefully shut down the bridge and clean up resources

### Advanced Usage

- `get_bridge()`: Get or create the singleton AsyncBridge instance
- `AsyncBridge` class: For custom bridge management in advanced scenarios

## Advantages Over Direct Thread Creation

| Feature | AsyncBridge | Thread Per Call |
|---------|------------|-----------------|
| Resource Usage | Single worker thread | New thread per call |
| Memory Footprint | Low and stable | Grows with concurrent calls |
| Error Handling | Preserves context | Often loses context |
| Concurrency Control | Built-in queue management | Manual coordination required |
| Cancellation | Clean task cancellation | Often leaves dangling resources |
| Debugging | Centralized logging | Scattered across threads |

## Implementation Details

AsyncBridge uses a combination of:

- A worker thread with a dedicated event loop
- A thread-safe task queue for submissions
- Event-based notification for results
- Proper task cancellation and timeout handling
- Comprehensive resource management

For more details, see the implementation in `atroposlib/utils/async_bridge.py`.

## Testing and Performance

The AsyncBridge has been tested with:

- Unit tests covering core functionality (`test_async_bridge.py`)
- Concurrency tests with multiple simultaneous requests
- Timeout and error handling scenarios
- Integration with Atropos server components

Performance benchmarks show significant improvements over the thread-per-call approach, especially under high load conditions.