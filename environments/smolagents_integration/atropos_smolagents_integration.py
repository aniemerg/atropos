import asyncio
import os
import logging
from collections.abc import Generator
from typing import Any, Dict, List, Optional, Union

from smolagents.models import ChatMessage, ChatMessageStreamDelta, MessageRole, Model

# Import our patched AsyncBridge utility
try:
    # First try to import the patched version if it exists
    from environments.smolagents_integration.patched_async_bridge import patched_run_async as run_async
    logging.getLogger(__name__).info("Using patched AsyncBridge")
except ImportError:
    # Fallback to the original
    from atroposlib.utils.async_bridge import run_async
    logging.getLogger(__name__).info("Using original AsyncBridge")

# Configure logger for the model class
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
        if model_id and any(name in model_id.lower() for name in ["gpt-4", "gpt-3.5-turbo", "claude", "gemini", "o", "llama"]):
            logger.info(f"Model {model_id} detected as a chat model. Forcing chat completion API.")
            self.use_chat_completion = True
            
        # Log the configuration
        logger.info(f"Initializing AtroposServerModel with model_id={model_id}, use_chat_completion={self.use_chat_completion}")
            
        super().__init__(model_id=model_id, **kwargs)

    def _prepare_completion_args(self, messages, stop_sequences=None, **kwargs):
        """
        Convert SmolaGents message format to Atropos server parameters.
        """
        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages, stop_sequences=stop_sequences, **kwargs
        )

        # Always use chat completion if configured that way
        # This is a critical fix to ensure consistency across all calls
        if self.use_chat_completion:
            # For chat completion, we format messages and don't use prompt
            server_args = {
                "messages": self._format_chat_messages(messages),
                "max_tokens": kwargs.get("max_tokens", 2048),
                "temperature": kwargs.get("temperature", 0.0),
                "stop": stop_sequences,
            }
            logger.debug(f"Prepared chat completion args: messages count={len(server_args['messages'])}")
            return server_args
        else:
            # Extract the user message for completion API
            prompt = self._extract_user_message(messages)
            server_args = {
                "prompt": prompt,
                "max_tokens": kwargs.get("max_tokens", 2048),
                "temperature": kwargs.get("temperature", 0.0),
                "stop": stop_sequences,
            }
            logger.debug(f"Prepared completion args: prompt length={len(prompt)}")
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
        # Debug information about the call
        import inspect
        current_frame = inspect.currentframe()
        caller_frame = inspect.getouterframes(current_frame, 2)
        caller_info = f"{caller_frame[1].filename}:{caller_frame[1].lineno} in {caller_frame[1].function}"
        logger.info(f"generate called from {caller_info}")
        
        # Special handling for CodeAgent stop sequences
        if stop_sequences is None:
            stop_sequences = ["Observation:", "<end_code>", "Calling tools:"]
        
        logger.info(f"Generate called with {len(messages)} messages, use_chat_completion={self.use_chat_completion}")
        
        # Debug information about messages - log at info for debugging
        if messages:
            for i, msg in enumerate(messages):
                if i < 3:  # Only log first 3 messages to avoid flooding
                    logger.info(f"Message {i}: role={msg.get('role')}, content_type={type(msg.get('content'))}")
        
        # Prepare completion arguments
        completion_kwargs = self._prepare_completion_args(
            messages=messages, stop_sequences=stop_sequences, **kwargs
        )
        
        # Debug the actual completion kwargs
        logger.info(f"completion_kwargs keys: {completion_kwargs.keys()}")
        if "messages" in completion_kwargs:
            logger.info(f"Will use chat completion API with {len(completion_kwargs['messages'])} messages")
        elif "prompt" in completion_kwargs:
            logger.info(f"Will use completion API with prompt length {len(completion_kwargs['prompt'])}")
        
        # Extract timeout from kwargs or use default - use a longer timeout for reliability
        timeout = kwargs.pop("timeout", 120)  # Default 2 minutes
        # Add a safety factor to ensure the AsyncBridge wait doesn't time out before the openai call
        bridge_timeout = timeout * 1.5  
        logger.info(f"Using timeout={timeout}s, bridge_timeout={bridge_timeout}s")
        
        # Always force chat completion for GPT-4o regardless of other settings
        force_chat = False
        if self.model_id and "gpt-4o" in self.model_id.lower():
            logger.info("GPT-4o detected, forcing chat completion API")
            force_chat = True
            self.use_chat_completion = True
        
        try:
            # For GPT-4o and similar models, always use chat_completion regardless
            if self.use_chat_completion or force_chat:
                logger.info("Using chat completion API")
                
                # We need to ensure we're sending the right format for chat completions
                if "prompt" in completion_kwargs and "messages" not in completion_kwargs:
                    # Convert prompt to messages format if needed
                    logger.info("Converting prompt to messages format")
                    completion_kwargs["messages"] = [{"role": "user", "content": completion_kwargs.pop("prompt")}]
                
                # Call chat_completion via AsyncBridge
                logger.info(f"Calling run_async(self.server.chat_completion) with timeout={bridge_timeout}")
                try:
                    resp = run_async(
                        self.server.chat_completion,
                        **completion_kwargs,
                        timeout=bridge_timeout
                    )
                    logger.info("run_async for chat_completion returned successfully")
                except Exception as async_e:
                    logger.error(f"AsyncBridge error during chat_completion: {type(async_e).__name__}: {str(async_e)}")
                    # Re-raise with more context
                    raise Exception(f"AsyncBridge error: {type(async_e).__name__}: {str(async_e)}")
                
                # Process response
                if resp and hasattr(resp, 'choices') and len(resp.choices) > 0:
                    content = resp.choices[0].message.content
                    logger.info(f"Got response with {len(content)} chars")
                else:
                    content = "No response content"
                    logger.warning("No content found in response")
            else:
                logger.info("Using completion API")
                # Call completion via AsyncBridge
                try:
                    resp = run_async(
                        self.server.completion,
                        **completion_kwargs,
                        timeout=bridge_timeout
                    )
                    logger.info("run_async for completion returned successfully")
                except Exception as async_e:
                    logger.error(f"AsyncBridge error during completion: {type(async_e).__name__}: {str(async_e)}")
                    # Re-raise with more context
                    raise Exception(f"AsyncBridge error: {type(async_e).__name__}: {str(async_e)}")
                
                # Process response
                if resp and hasattr(resp, 'choices') and len(resp.choices) > 0:
                    content = resp.choices[0].text
                    logger.info(f"Got response with {len(content)} chars")
                else:
                    content = "No response content"
                    logger.warning("No content found in response")
            
            # Track token usage
            if hasattr(resp, "usage"):
                self.last_input_token_count = resp.usage.prompt_tokens
                self.last_output_token_count = resp.usage.completion_tokens
                logger.info(f"Token usage: input={self.last_input_token_count}, output={self.last_output_token_count}")
            
            # Return result in SmolaGents format
            logger.info("Successfully returning ChatMessage")
            return ChatMessage(
                role=MessageRole.ASSISTANT, content=content, raw=resp
            )
            
        except Exception as e:
            # Provide more detailed error information
            error_msg = f"Error during Atropos server call: {type(e).__name__}: {str(e)}"
            logger.error(error_msg)
            
            # Print full stack trace for debugging
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            
            raise ValueError(error_msg)

    # Override all call methods to ensure they use the chat completion API if configured
    def __call__(self, *args, **kwargs):
        """
        Make the model directly callable. 
        Ensures proper API endpoint is used.
        """
        logger.debug("Model called directly")
        return super().__call__(*args, **kwargs)
        
    # Override parent method to ensure each call respects use_chat_completion
    def _prepare_completion_kwargs(self, *args, **kwargs):
        """
        Override parent method to ensure consistent settings.
        Marks the model as chat-only when using chat completion.
        """
        result = super()._prepare_completion_kwargs(*args, **kwargs)
        
        if self.use_chat_completion:
            # Force the is_chat_mode parameter for SmolaGents
            result["is_chat_mode"] = True
            
        return result
