import asyncio
import os
from collections.abc import Generator
from typing import Any, Dict, List, Optional, Union

from smolagents.models import ChatMessage, ChatMessageStreamDelta, MessageRole, Model

# Import our AsyncBridge utility
from atroposlib.utils.async_bridge import run_async


class AtroposServerModel(Model):
    """
    A SmolaGents Model implementation that wraps Atropos servers.

    This class bridges the gap between SmolaGents' synchronous model interface and
    Atropos' asynchronous server architecture using AsyncBridge.

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

        # Populate the server args based on completion type
        if self.use_chat_completion:
            # For chat completion, we format messages and don't use prompt
            server_args = {
                "messages": self._format_chat_messages(messages),
                "max_tokens": kwargs.get("max_tokens", 2048),
                "temperature": kwargs.get("temperature", 0.0),
                "stop": stop_sequences,
            }
        else:
            # Extract the user message for completion API
            prompt = self._extract_user_message(messages)
            server_args = {
                "prompt": prompt,
                "max_tokens": kwargs.get("max_tokens", 2048),
                "temperature": kwargs.get("temperature", 0.0),
                "stop": stop_sequences,
            }

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
        Process the input messages and return the model's response.
        Uses AsyncBridge to run the async method without creating new threads.

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
        
        # Prepare completion arguments
        completion_kwargs = self._prepare_completion_args(
            messages=messages, stop_sequences=stop_sequences, **kwargs
        )
        
        # Extract timeout from kwargs or use default
        timeout = kwargs.pop("timeout", 120)  # Default 2 minutes
        
        try:
            # Use AsyncBridge to call the async method
            if self.use_chat_completion:
                # Call chat_completion via AsyncBridge
                resp = run_async(
                    self.server.chat_completion,
                    **completion_kwargs,
                    timeout=timeout
                )
                
                # Process response
                if resp and hasattr(resp, 'choices') and len(resp.choices) > 0:
                    content = resp.choices[0].message.content
                else:
                    content = "No response content"
            else:
                # Call completion via AsyncBridge
                resp = run_async(
                    self.server.completion,
                    **completion_kwargs,
                    timeout=timeout
                )
                
                # Process response
                if resp and hasattr(resp, 'choices') and len(resp.choices) > 0:
                    content = resp.choices[0].text
                else:
                    content = "No response content"
            
            # Track token usage
            if hasattr(resp, "usage"):
                self.last_input_token_count = resp.usage.prompt_tokens
                self.last_output_token_count = resp.usage.completion_tokens
            
            # Return result in SmolaGents format
            return ChatMessage(
                role=MessageRole.ASSISTANT, content=content, raw=resp
            )
            
        except Exception as e:
            # Provide more detailed error information
            error_msg = f"Error during Atropos server call: {type(e).__name__}: {str(e)}"
            raise ValueError(error_msg)

    # We don't need streaming for the current integration
