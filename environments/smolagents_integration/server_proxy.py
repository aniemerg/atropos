"""
Proxy mechanism for communicating with Atropos server from child processes.
"""

import asyncio
import multiprocessing
import pickle
import time
import traceback
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

class ServerRequest:
    """Encapsulates a request to be sent to the server."""
    
    def __init__(self, method: str, kwargs: Dict[str, Any], request_id: str = None):
        self.method = method  # 'completion' or 'chat_completion'
        self.kwargs = kwargs
        self.request_id = request_id or str(uuid.uuid4())
        self.timestamp = time.time()

class ServerResponse:
    """Encapsulates a response from the server."""
    
    def __init__(self, 
                 request_id: str, 
                 result: Any = None, 
                 error: Optional[Exception] = None,
                 error_traceback: Optional[str] = None):
        self.request_id = request_id
        self.result = result
        self.error = error
        self.error_traceback = error_traceback
        self.timestamp = time.time()

class ServerProxy:
    """
    Proxy for communicating with the Atropos server from a child process.
    
    This class provides methods that mirror the server's API but communicate
    through multiprocessing queues instead of direct calls.
    """
    
    def __init__(self, 
                 request_queue: multiprocessing.Queue,
                 response_queue: multiprocessing.Queue,
                 model_name: str,
                 timeout: float = 120.0):
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.model_name = model_name
        self.timeout = timeout
        self.pending_requests = {}
    
    def completion(self, **kwargs):
        """Submit a completion request to the server."""
        # Create the request
        request = ServerRequest("completion", kwargs)
        
        # Send the request
        self.request_queue.put(request)
        
        # Wait for the response
        return self._wait_for_response(request.request_id)
    
    def chat_completion(self, **kwargs):
        """Submit a chat completion request to the server."""
        # Create the request
        request = ServerRequest("chat_completion", kwargs)
        
        # Send the request
        self.request_queue.put(request)
        
        # Wait for the response
        return self._wait_for_response(request.request_id)
    
    def _wait_for_response(self, request_id: str):
        """Wait for a response to a specific request."""
        start_time = time.time()
        
        while time.time() - start_time < self.timeout:
            # Check the response queue
            try:
                response = self.response_queue.get(timeout=0.1)
                
                # If this is the response we're waiting for, return it
                if response.request_id == request_id:
                    if response.error:
                        # Recreate the exception
                        raise type(response.error)(str(response.error))
                    return response.result
                
                # Otherwise, store it for later retrieval
                self.pending_requests[response.request_id] = response
                
                # Check if we already have the response we're waiting for
                if request_id in self.pending_requests:
                    response = self.pending_requests.pop(request_id)
                    if response.error:
                        raise type(response.error)(str(response.error))
                    return response.result
                
            except (multiprocessing.queues.Empty, EOFError):
                # Queue is empty, continue waiting
                pass
        
        # Timeout expired
        raise TimeoutError(f"Request {request_id} timed out after {self.timeout} seconds")

class ServerProxyManager:
    """
    Manager for creating server proxies for child processes.
    
    This class creates request/response queues and spawns a worker process
    that handles communication with the Atropos server.
    """
    
    def __init__(self, server, max_workers: int = 5):
        self.server = server
        self.max_workers = max_workers
        self.request_queue = multiprocessing.Queue()
        self.response_queues = {}
        self.worker_process = None
        self.running = False
    
    def start(self):
        """Start the server proxy manager."""
        if self.running:
            return
        
        self.running = True
        self.worker_process = multiprocessing.Process(
            target=self._server_worker,
            args=(self.request_queue, self.response_queues),
            daemon=True
        )
        self.worker_process.start()
    
    def stop(self):
        """Stop the server proxy manager."""
        if not self.running:
            return
        
        self.running = False
        
        # Signal the worker process to exit
        self.request_queue.put(None)
        
        # Wait for the worker process to exit
        if self.worker_process:
            self.worker_process.join(timeout=5.0)
            if self.worker_process.is_alive():
                self.worker_process.terminate()
            self.worker_process = None
    
    def create_server_proxy(self, model_name: str, timeout: float = 120.0) -> Tuple[ServerProxy, str]:
        """Create a server proxy for a child process."""
        response_queue = multiprocessing.Queue()
        proxy_id = str(uuid.uuid4())
        self.response_queues[proxy_id] = response_queue
        
        # Make sure the manager is running
        if not self.running:
            self.start()
        
        return ServerProxy(self.request_queue, response_queue, model_name, timeout), proxy_id
    
    def remove_proxy(self, proxy_id: str):
        """Remove a proxy when it's no longer needed."""
        if proxy_id in self.response_queues:
            del self.response_queues[proxy_id]
    
    def _server_worker(self, request_queue, response_queues):
        """Worker process that handles communication with the Atropos server."""
        # Set up the asyncio event loop for this process
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        async def handle_request(request):
            """Handle a server request."""
            if request is None:
                # Signal to exit
                return True
            
            try:
                # Call the appropriate server method
                if request.method == "completion":
                    result = await self.server.completion(**request.kwargs)
                elif request.method == "chat_completion":
                    result = await self.server.chat_completion(**request.kwargs)
                else:
                    raise ValueError(f"Unknown method: {request.method}")
                
                # Create the response
                response = ServerResponse(request.request_id, result=result)
                
                # Send the response to all queues (each proxy will filter by request_id)
                for proxy_id, queue in response_queues.items():
                    try:
                        queue.put(response)
                    except (BrokenPipeError, EOFError):
                        # The proxy might have been closed, ignore
                        pass
                
                return False
            
            except Exception as e:
                # Create an error response
                response = ServerResponse(
                    request.request_id,
                    error=e,
                    error_traceback=traceback.format_exc()
                )
                
                # Send the error response
                for proxy_id, queue in response_queues.items():
                    try:
                        queue.put(response)
                    except (BrokenPipeError, EOFError):
                        # The proxy might have been closed, ignore
                        pass
                
                return False
        
        async def main():
            """Main event loop for the server worker."""
            while True:
                try:
                    # Get a request from the queue
                    request = request_queue.get(timeout=0.1)
                    
                    # Handle the request
                    should_exit = await handle_request(request)
                    if should_exit:
                        break
                
                except (multiprocessing.queues.Empty, EOFError):
                    # Queue is empty, continue waiting
                    await asyncio.sleep(0.01)
                
                except Exception as e:
                    # Log the error and continue
                    print(f"Error in server worker: {e}")
                    traceback.print_exc()
        
        # Run the event loop
        loop.run_until_complete(main())
        loop.close()