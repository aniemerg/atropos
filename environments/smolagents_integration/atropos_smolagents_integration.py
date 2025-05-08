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
        self,
        server,
        use_chat_completion: bool = False,
        model_id: str = None,
        **kwargs
    ):
        self.server = server
        self.use_chat_completion = use_chat_completion
        self.event_loop = asyncio.get_event_loop()
        super().__init__(model_id=model_id, **kwargs)
    
    def _prepare_completion_args(self, messages, stop_sequences=None, **kwargs):
        """
        Convert SmolaGents message format to Atropos server parameters.
        """
        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages,
            stop_sequences=stop_sequences,
            **kwargs
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
                    return "\n".join(item["text"] for item in content if item["type"] == "text")
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
                text_content = "\n".join(item["text"] for item in content if item["type"] == "text")
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
        **kwargs
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
            messages=messages,
            stop_sequences=stop_sequences,
            **kwargs
        )
        
        # Bridge the async/sync boundary and call Atropos server
        try:
            if self.use_chat_completion:
                response = self.event_loop.run_until_complete(
                    self.server.chat_completion(**completion_kwargs)
                )
                content = response.choices[0].message.content
            else:
                response = self.event_loop.run_until_complete(
                    self.server.completion(**completion_kwargs)
                )
                content = response.choices[0].text
            
            # Track token usage
            if hasattr(response, "usage"):
                self.last_input_token_count = response.usage.prompt_tokens
                self.last_output_token_count = response.usage.completion_tokens
            
            # Return in SmolaGents' format
            return ChatMessage(
                role=MessageRole.ASSISTANT,
                content=content,
                raw=response
            )
        except Exception as e:
            raise ValueError(f"Error during Atropos server call: {str(e)}")
            
    def generate_stream(
        self,
        messages: list[dict[str, str | list[dict]]],
        stop_sequences: list[str] | None = None,
        grammar: str | None = None,
        tools_to_call_from: list | None = None,
        **kwargs
    ) -> Generator[ChatMessageStreamDelta]:
        """
        Stream the model's response by calling Atropos server with simulated streaming.
        
        Since Atropos doesn't natively support streaming, we simulate it by breaking
        the completed response into chunks.
        
        Parameters:
            messages: A list of message dictionaries to be processed.
            stop_sequences: A list of strings that will stop the generation if encountered.
            grammar: The grammar or formatting structure to use (not used with Atropos).
            tools_to_call_from: List of tools (not used with Atropos).
            **kwargs: Additional keyword arguments for the server.
        
        Yields:
            ChatMessageStreamDelta: Stream of delta objects representing the model's response.
        """
        # Special handling for CodeAgent stop sequences
        if stop_sequences is None:
            stop_sequences = ["Observation:", "<end_code>", "Calling tools:"]
        
        # First, get the full response using the normal generate method
        full_response = self.generate(
            messages=messages,
            stop_sequences=stop_sequences,
            grammar=grammar,
            tools_to_call_from=tools_to_call_from,
            **kwargs
        )
        
        # If there's no content, return an empty stream
        if not full_response.content:
            yield ChatMessageStreamDelta(content="")
            return
        
        # Simulate streaming by yielding characters from the content
        # For a more realistic simulation, we could chunk by tokens or words
        chunk_size = 4  # Adjust for more or less granular streaming
        content = full_response.content
        
        for i in range(0, len(content), chunk_size):
            chunk = content[i:i+chunk_size]
            yield ChatMessageStreamDelta(content=chunk)
            
            # Small delay for more realistic streaming simulation
            # Remove in production for maximum performance
            # import time
            # time.sleep(0.01)