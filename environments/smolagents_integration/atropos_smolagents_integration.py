import asyncio
import os
from collections.abc import Generator
from typing import Any, Dict, List, Optional, Union

from smolagents.models import ChatMessage, ChatMessageStreamDelta, MessageRole, Model


class AtroposServerModel(Model):
    """
    A SmolaGents Model implementation that wraps Atropos servers.

    This class bridges the gap between SmolaGents' synchronous model interface and
    Atropos' asynchronous server architecture.

    Parameters:
        server: An Atropos server instance (OpenAIServer or ServerManager)
        use_chat_completion: Whether to use chat_completion or completion API
        model_id: Optional identifier for the model
        **kwargs: Additional parameters passed to the parent Model class
    """

    def __init__(
        self, server, use_chat_completion: bool = False, model_id: str = None, **kwargs
    ):
        self.server = server
        self.use_chat_completion = use_chat_completion
        
        # Automatically set chat completion for GPT models which require it
        if model_id and any(name in model_id for name in ["gpt-4", "gpt-3.5-turbo", "claude", "gemini"]):
            self.use_chat_completion = True
            
        super().__init__(model_id=model_id, **kwargs)

    def _prepare_completion_args(self, messages, stop_sequences=None, **kwargs):
        """
        Convert SmolaGents message format to Atropos server parameters.
        """
        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages, stop_sequences=stop_sequences, **kwargs
        )

        # Extract the user message (for completion API)
        prompt = self._extract_user_message(messages)

        # Populate the necessary arguments for the Atropos server
        server_args = {
            "prompt": prompt,
            "max_tokens": kwargs.get("max_tokens", 2048),
            "temperature": kwargs.get("temperature", 0.0),
            "stop": stop_sequences,
        }

        # For chat completion, we need to format messages differently
        if self.use_chat_completion:
            server_args["messages"] = self._format_chat_messages(messages)
            server_args.pop("prompt", None)  # Remove prompt for chat completion

        return server_args

    def _extract_user_message(self, messages):
        """Extract content from the last user message."""
        for msg in reversed(messages):
            if msg["role"] == "user":
                content = msg["content"]
                if isinstance(content, list):
                    # Handle list format [{"type": "text", "text": "content"}]
                    return "\n".join(
                        item["text"] for item in content if item["type"] == "text"
                    )
                return content
        raise ValueError("No user message found")

    def _format_chat_messages(self, messages):
        """Format messages for the chat completion API."""
        formatted_messages = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            # Extract text content if it's in the list format
            if isinstance(content, list):
                text_content = "\n".join(
                    item["text"] for item in content if item["type"] == "text"
                )
                formatted_messages.append({"role": role, "content": text_content})
            else:
                formatted_messages.append({"role": role, "content": content})

        return formatted_messages

    def generate(
        self,
        messages: list[dict[str, str | list[dict]]],
        stop_sequences: list[str] | None = None,
        grammar: str | None = None,
        tools_to_call_from: list | None = None,
        **kwargs,
    ) -> ChatMessage:
        """
        Process the input messages and return the model's response by calling Atropos server.

        Parameters:
            messages: A list of message dictionaries to be processed.
            stop_sequences: A list of strings that will stop the generation if encountered.
            grammar: The grammar or formatting structure to use (not used with Atropos).
            tools_to_call_from: List of tools (not used with Atropos).
            **kwargs: Additional keyword arguments for the server.

        Returns:
            ChatMessage: A chat message object containing the model's response.
        """
        # Special handling for CodeAgent stop sequences
        if stop_sequences is None:
            stop_sequences = ["Observation:", "<end_code>", "Calling tools:"]

        # Prepare the completion arguments
        completion_kwargs = self._prepare_completion_args(
            messages=messages, stop_sequences=stop_sequences, **kwargs
        )

        # Bridge the async/sync boundary and call Atropos server
        try:
            # Use a simple approach to handle the asyncio call properly
            # We need to respect Atropos's server handling
            
            # Check if we're already inside an event loop
            in_event_loop = False
            try:
                asyncio.get_running_loop()
                in_event_loop = True
            except RuntimeError:
                # We're not in an event loop
                pass
                
            if self.use_chat_completion:
                if in_event_loop:
                    # We're in an event loop, so create a new thread
                    import threading
                    import queue
                    
                    result_queue = queue.Queue()
                    
                    def thread_worker():
                        try:
                            # Create a new event loop for this thread
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            
                            # Call the async method in this thread's event loop
                            resp = loop.run_until_complete(
                                self.server.chat_completion(**completion_kwargs)
                            )
                            if resp and hasattr(resp, 'choices') and len(resp.choices) > 0:
                                content = resp.choices[0].message.content
                                result_queue.put((resp, content))
                            else:
                                result_queue.put((resp, "No response content"))
                        except Exception as e:
                            result_queue.put((None, f"Error during Atropos server call: {str(e)}"))
                            
                    # Start a thread for the async call
                    thread = threading.Thread(target=thread_worker)
                    thread.daemon = True
                    thread.start()
                    thread.join(timeout=120)  # Wait up to 2 minutes
                    
                    if thread.is_alive():
                        raise TimeoutError("Request timed out")
                        
                    # Get the result
                    response, content = result_queue.get()
                    if response is None:
                        raise ValueError(f"Error in chat completion: {content}")
                else:
                    # We're not in an event loop, so create one
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    response = loop.run_until_complete(
                        self.server.chat_completion(**completion_kwargs)
                    )
                    content = response.choices[0].message.content
            else:
                if in_event_loop:
                    # We're in an event loop, so create a new thread
                    import threading
                    import queue
                    
                    result_queue = queue.Queue()
                    
                    def thread_worker():
                        try:
                            # Create a new event loop for this thread
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            
                            # Call the async method in this thread's event loop
                            resp = loop.run_until_complete(
                                self.server.completion(**completion_kwargs)
                            )
                            if resp and hasattr(resp, 'choices') and len(resp.choices) > 0:
                                content = resp.choices[0].text
                                result_queue.put((resp, content))
                            else:
                                result_queue.put((resp, "No response content"))
                        except Exception as e:
                            result_queue.put((None, f"Error during Atropos server call: {str(e)}"))
                            
                    # Start a thread for the async call
                    thread = threading.Thread(target=thread_worker)
                    thread.daemon = True
                    thread.start()
                    thread.join(timeout=120)  # Wait up to 2 minutes
                    
                    if thread.is_alive():
                        raise TimeoutError("Request timed out")
                        
                    # Get the result
                    response, content = result_queue.get()
                    if response is None:
                        raise ValueError(f"Error in completion: {content}")
                else:
                    # We're not in an event loop, so create one
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    response = loop.run_until_complete(
                        self.server.completion(**completion_kwargs)
                    )
                    content = response.choices[0].text

            # Track token usage
            if hasattr(response, "usage"):
                self.last_input_token_count = response.usage.prompt_tokens
                self.last_output_token_count = response.usage.completion_tokens

            # Return in SmolaGents' format
            return ChatMessage(
                role=MessageRole.ASSISTANT, content=content, raw=response
            )
        except Exception as e:
            raise ValueError(f"Error during Atropos server call: {str(e)}")

    def generate_stream(
        self,
        messages: list[dict[str, str | list[dict]]],
        stop_sequences: list[str] | None = None,
        grammar: str | None = None,
        tools_to_call_from: list | None = None,
        **kwargs,
    ) -> Generator[ChatMessageStreamDelta]:
        """
        Stream the model's response by calling Atropos server.

        This streaming implementation is disabled to avoid issues with event loops.
        Instead, we get the full response at once and return it as a single chunk.

        Parameters:
            messages: A list of message dictionaries to be processed.
            stop_sequences: A list of strings that will stop the generation if encountered.
            grammar: The grammar or formatting structure to use (not used with Atropos).
            tools_to_call_from: List of tools (not used with Atropos).
            **kwargs: Additional keyword arguments for the server.

        Yields:
            ChatMessageStreamDelta: Stream of delta objects representing the model's response.
        """
        # Get the full response using the normal generate method
        try:
            # To avoid event loop issues, we use the non-streaming version and
            # deliver the entire content as a single delta
            import time
            
            # Add a small delay to prevent potential race conditions
            time.sleep(0.1)
            
            # Get the full response
            full_response = self.generate(
                messages=messages,
                stop_sequences=stop_sequences,
                grammar=grammar,
                tools_to_call_from=tools_to_call_from,
                **kwargs,
            )
            
            # Return the entire content as a single chunk
            content = full_response.content if full_response and hasattr(full_response, 'content') and full_response.content else ""
            yield ChatMessageStreamDelta(content=content)
            
        except Exception as e:
            # If there's an error, yield an error message with detailed information
            error_message = f"Error during streamed generation: {str(e)}"
            yield ChatMessageStreamDelta(content=error_message)
