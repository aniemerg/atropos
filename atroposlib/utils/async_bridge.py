"""
AsyncBridge - A thread-safe bridge between synchronous and asynchronous code.

This module provides a robust interface for executing asynchronous (async/await)
code from synchronous code contexts, without creating new event loops and threads
for each call. It maintains a single worker thread with a persistent event loop
that can handle multiple concurrent async operations efficiently.

Usage:
    from atroposlib.utils.async_bridge import run_async

    # Run an async function from synchronous code
    result = run_async(async_function, *args, timeout=30, **kwargs)

    # At application shutdown
    from atroposlib.utils.async_bridge import shutdown_bridge
    shutdown_bridge()  # Gracefully clean up resources
"""

import asyncio
import queue
import threading
import time
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar, Union, cast, Coroutine

# Type variables for generic typing
T = TypeVar('T')
Coro = TypeVar('Coro', bound=Coroutine)


class AsyncBridge:
    """
    A bridge between synchronous and asynchronous code.
    
    Maintains a single worker thread with an event loop to execute
    coroutines from synchronous code, avoiding the overhead of
    creating new threads and event loops for each async call.
    """
    
    def __init__(self):
        self._queue = queue.Queue()
        self._results: Dict[int, Tuple[Any, Optional[Exception]]] = {}
        self._result_events: Dict[int, threading.Event] = {}
        self._lock = threading.RLock()
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._running = False
        self._task_counter = 0
    
    def start(self):
        """Start the bridge with a dedicated thread and event loop."""
        with self._lock:
            if self._running:
                return
            self._running = True
            self._thread = threading.Thread(target=self._thread_worker, daemon=True)
            self._thread.start()
    
    def stop(self):
        """Gracefully stop the bridge."""
        with self._lock:
            if not self._running:
                return
            self._running = False
            # Signal worker thread to exit
            self._queue.put((None, None, None))
            if self._thread:
                self._thread.join(timeout=5.0)
                self._thread = None
            self._loop = None
    
    def _thread_worker(self):
        """Worker thread that runs the event loop."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        
        try:
            while self._running:
                try:
                    # Get next task from queue with timeout
                    task_id, coro, timeout = self._queue.get(timeout=0.1)
                    if task_id is None:  # Stop signal
                        break
                    
                    # Schedule the coroutine on the event loop and run it to completion
                    # This ensures that tasks actually complete before we continue
                    task = self._loop.create_task(
                        self._execute_task(task_id, coro, timeout)
                    )
                    
                    # Process events until this task is done
                    while not task.done():
                        self._loop.run_until_complete(asyncio.sleep(0.01))
                        
                except queue.Empty:
                    # No tasks in queue, continue
                    continue
                except Exception as e:
                    print(f"Error in AsyncBridge worker: {e}")
        finally:
            # Clean up resources
            self._cleanup_loop()
    
    def _cleanup_loop(self):
        """Clean up the event loop."""
        if not self._loop:
            return
            
        try:
            # Cancel all pending tasks
            pending = asyncio.all_tasks(self._loop)
            for task in pending:
                task.cancel()
                
            # Wait for tasks to complete/cancel
            if pending:
                self._loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
                
            # Shut down async generators
            self._loop.run_until_complete(self._loop.shutdown_asyncgens())
            
            # Close the loop
            self._loop.close()
        except Exception as e:
            print(f"Error cleaning up event loop: {e}")
    
    async def _execute_task(self, task_id, coro, timeout):
        """Execute a coroutine with a timeout and store the result."""
        try:
            if timeout:
                result = await asyncio.wait_for(coro, timeout=timeout)
            else:
                result = await coro
            
            with self._lock:
                self._results[task_id] = (result, None)
                if task_id in self._result_events:
                    self._result_events[task_id].set()
        except Exception as e:
            with self._lock:
                self._results[task_id] = (None, e)
                if task_id in self._result_events:
                    self._result_events[task_id].set()
    
    def run_coroutine(self, coro, timeout=None):
        """Run a coroutine on the bridge's event loop and return the result."""
        if not self._running:
            self.start()
        
        with self._lock:
            task_id = self._task_counter
            self._task_counter += 1
            event = threading.Event()
            self._result_events[task_id] = event
        
        # Submit task to worker thread
        self._queue.put((task_id, coro, timeout))
        
        # Wait for result with timeout
        wait_timeout = timeout or 3600  # Default 1 hour max wait
        if not event.wait(timeout=wait_timeout):
            with self._lock:
                self._result_events.pop(task_id, None)
            raise TimeoutError(f"Task {task_id} timed out after {wait_timeout} seconds")
        
        # Get result
        with self._lock:
            result, error = self._results.pop(task_id, (None, None))
            self._result_events.pop(task_id, None)
        
        if error:
            raise error
        return result


# Singleton instance
_BRIDGE = None

def get_bridge():
    """Get or create the singleton AsyncBridge instance."""
    global _BRIDGE
    if _BRIDGE is None:
        _BRIDGE = AsyncBridge()
        _BRIDGE.start()
    return _BRIDGE

def run_async(coro_or_func, *args, timeout=None, **kwargs):
    """
    Run an async function or coroutine from synchronous code.
    
    Args:
        coro_or_func: An async function or coroutine
        *args: Arguments for the function
        timeout: Optional timeout in seconds
        **kwargs: Keyword arguments for the function
        
    Returns:
        The result of the coroutine
    """
    bridge = get_bridge()
    
    if asyncio.iscoroutine(coro_or_func):
        # Already a coroutine
        return bridge.run_coroutine(coro_or_func, timeout=timeout)
    elif asyncio.iscoroutinefunction(coro_or_func):
        # Async function, create coroutine by calling it
        coro = coro_or_func(*args, **kwargs)
        return bridge.run_coroutine(coro, timeout=timeout)
    else:
        raise TypeError("Expected a coroutine or async function")

def shutdown_bridge():
    """Shut down the AsyncBridge."""
    global _BRIDGE
    if _BRIDGE is not None:
        _BRIDGE.stop()
        _BRIDGE = None