"""
Patched version of AsyncBridge with better debugging
"""

import asyncio
import logging
import queue
import threading
import time
import traceback
from typing import Any, Optional, Tuple, TypeVar, Coroutine

from atroposlib.utils.async_bridge import AsyncBridge as OriginalAsyncBridge
from atroposlib.utils.async_bridge import run_async as original_run_async
from atroposlib.utils.async_bridge import _BRIDGE, get_bridge as original_get_bridge

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)  # Reduced from INFO to WARNING

# Type variables for generic typing
T = TypeVar('T')
Coro = TypeVar('Coro', bound=Coroutine)


class PatchedAsyncBridge(OriginalAsyncBridge):
    """
    A modified version of AsyncBridge with better error reporting.
    """
    
    def __init__(self):
        super().__init__()
        logger.info("Created PatchedAsyncBridge")
    
    def start(self):
        """Start the bridge with a dedicated thread and event loop."""
        with self._lock:
            if self._running:
                logger.info("Bridge already running, not starting new thread")
                return
            self._running = True
            self._thread = threading.Thread(target=self._thread_worker, daemon=True, name="PatchedAsyncBridge-worker")
            logger.info("Starting AsyncBridge worker thread")
            self._thread.start()
    
    def _thread_worker(self):
        """Worker thread that runs the event loop."""
        logger.info("AsyncBridge worker thread started")
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        
        try:
            while self._running:
                try:
                    # Get next task from queue with timeout
                    task_id, coro, timeout = self._queue.get(timeout=0.1)
                    if task_id is None:  # Stop signal
                        logger.info("AsyncBridge worker received stop signal")
                        break
                    
                    logger.info(f"AsyncBridge processing task {task_id} with timeout {timeout}")
                    
                    # Schedule the coroutine on the event loop and run it to completion
                    task = self._loop.create_task(
                        self._execute_task(task_id, coro, timeout)
                    )
                    
                    # Process events until this task is done
                    polling_count = 0
                    start_time = time.time()
                    while not task.done():
                        self._loop.run_until_complete(asyncio.sleep(0.01))
                        polling_count += 1
                        if polling_count % 100 == 0:  # Log every ~1 second
                            elapsed = time.time() - start_time
                            logger.info(f"Still waiting for task {task_id}, elapsed time: {elapsed:.2f}s")
                        
                except queue.Empty:
                    # No tasks in queue, continue
                    continue
                except Exception as e:
                    logger.error(f"Error in AsyncBridge worker: {e}", exc_info=True)
        finally:
            logger.info("AsyncBridge worker thread exiting, cleaning up resources")
            self._cleanup_loop()
    
    async def _execute_task(self, task_id, coro, timeout):
        """Execute a coroutine with a timeout and store the result."""
        logger.info(f"Executing task {task_id} with timeout {timeout}")
        start_time = time.time()
        try:
            if timeout:
                logger.info(f"Using asyncio.wait_for with timeout {timeout}")
                result = await asyncio.wait_for(coro, timeout=timeout)
            else:
                logger.info("No timeout specified, running coroutine directly")
                result = await coro
            
            elapsed = time.time() - start_time
            logger.info(f"Task {task_id} completed successfully in {elapsed:.2f}s")
            
            with self._lock:
                self._results[task_id] = (result, None)
                if task_id in self._result_events:
                    logger.info(f"Setting result event for task {task_id}")
                    self._result_events[task_id].set()
        except asyncio.TimeoutError as e:
            elapsed = time.time() - start_time
            logger.error(f"Task {task_id} timed out after {elapsed:.2f}s")
            with self._lock:
                self._results[task_id] = (None, e)
                if task_id in self._result_events:
                    self._result_events[task_id].set()
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Task {task_id} failed after {elapsed:.2f}s: {type(e).__name__}: {str(e)}")
            logger.error(f"Exception traceback: {traceback.format_exc()}")
            with self._lock:
                self._results[task_id] = (None, e)
                if task_id in self._result_events:
                    self._result_events[task_id].set()
    
    def run_coroutine(self, coro, timeout=None):
        """Run a coroutine on the bridge's event loop and return the result."""
        if not self._running:
            logger.info("Bridge not running, starting it now")
            self.start()
        
        with self._lock:
            task_id = self._task_counter
            self._task_counter += 1
            event = threading.Event()
            self._result_events[task_id] = event
            logger.info(f"Created task {task_id} with timeout {timeout}")
        
        # Submit task to worker thread
        logger.info(f"Submitting task {task_id} to queue")
        self._queue.put((task_id, coro, timeout))
        
        # Wait for result with timeout
        wait_timeout = timeout * 1.5 if timeout else 3600  # Default 1 hour max wait, or 1.5x the task timeout
        logger.info(f"Waiting for event with timeout {wait_timeout}")
        
        wait_start = time.time()
        if not event.wait(timeout=wait_timeout):
            elapsed = time.time() - wait_start
            logger.error(f"Task {task_id} event wait timed out after {elapsed:.2f}s")
            with self._lock:
                self._result_events.pop(task_id, None)
            raise TimeoutError(f"Task {task_id} timed out after {wait_timeout} seconds")
        
        elapsed = time.time() - wait_start
        logger.info(f"Event for task {task_id} was set after {elapsed:.2f}s")
        
        # Get result
        with self._lock:
            result, error = self._results.pop(task_id, (None, None))
            self._result_events.pop(task_id, None)
        
        if error:
            logger.error(f"Task {task_id} had error: {type(error).__name__}: {str(error)}")
            raise error
            
        logger.info(f"Task {task_id} completed successfully, returning result")
        return result


# Create patched instance
_PATCHED_BRIDGE = None

def get_patched_bridge():
    """Get or create the singleton PatchedAsyncBridge instance."""
    global _PATCHED_BRIDGE
    if _PATCHED_BRIDGE is None:
        logger.info("Creating and starting patched AsyncBridge")
        _PATCHED_BRIDGE = PatchedAsyncBridge()
        _PATCHED_BRIDGE.start()
    return _PATCHED_BRIDGE

def patched_run_async(coro_or_func, *args, timeout=None, **kwargs):
    """
    Run an async function or coroutine from synchronous code using the patched bridge.
    
    Args:
        coro_or_func: An async function or coroutine
        *args: Arguments for the function
        timeout: Optional timeout in seconds
        **kwargs: Keyword arguments for the function
        
    Returns:
        The result of the coroutine
    """
    bridge = get_patched_bridge()
    
    if asyncio.iscoroutine(coro_or_func):
        # Already a coroutine
        logger.info(f"Running coroutine with patched bridge, timeout={timeout}")
        return bridge.run_coroutine(coro_or_func, timeout=timeout)
    elif asyncio.iscoroutinefunction(coro_or_func):
        # Async function, create coroutine by calling it
        logger.info(f"Creating and running coroutine from async function with patched bridge, timeout={timeout}")
        coro = coro_or_func(*args, **kwargs)
        return bridge.run_coroutine(coro, timeout=timeout)
    else:
        err_msg = f"Expected a coroutine or async function, got {type(coro_or_func)}"
        logger.error(err_msg)
        raise TypeError(err_msg)

def patch_asyncbridge():
    """Patch the global AsyncBridge with our enhanced version"""
    global _BRIDGE
    import atroposlib.utils.async_bridge
    
    # Replace the global bridge
    logger.info("Patching global AsyncBridge")
    if atroposlib.utils.async_bridge._BRIDGE is not None:
        logger.info("Shutting down existing AsyncBridge")
        atroposlib.utils.async_bridge._BRIDGE.stop()
    
    # Create our patched version
    bridge = get_patched_bridge()
    
    # Monkey patch the AsyncBridge module
    atroposlib.utils.async_bridge._BRIDGE = bridge
    atroposlib.utils.async_bridge.get_bridge = get_patched_bridge
    atroposlib.utils.async_bridge.run_async = patched_run_async
    
    logger.info("AsyncBridge successfully patched")
    return bridge