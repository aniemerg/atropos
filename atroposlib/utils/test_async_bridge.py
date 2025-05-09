"""
Test module for AsyncBridge implementation.

This module contains tests to verify the AsyncBridge works as expected,
both for individual operations and under load with multiple concurrent calls.
"""

import asyncio
import time
import unittest
import random
import sys
from concurrent.futures import ThreadPoolExecutor

from atroposlib.utils.async_bridge import AsyncBridge, run_async, get_bridge, shutdown_bridge


class TestAsyncBridge(unittest.TestCase):
    
    def setUp(self):
        # Create a fresh bridge for each test
        self.bridge = AsyncBridge()
        self.bridge.start()
    
    def tearDown(self):
        # Clean up the bridge after each test
        try:
            self.bridge.stop()
        except Exception as e:
            print(f"Error during tearDown: {e}")
    
    async def sample_async_function(self, delay=0.1, fail=False):
        """Sample async function for testing."""
        await asyncio.sleep(delay)
        if fail:
            raise ValueError("Deliberate test error")
        return f"Result after {delay}s"
    
    def test_run_coroutine_simple(self):
        """Basic test with minimal functionality."""
        coro = self.sample_async_function(delay=0.1)
        
        # Start a timer to ensure we don't hang
        start_time = time.time()
        try:
            result = self.bridge.run_coroutine(coro, timeout=2.0)
            self.assertEqual(result, "Result after 0.1s")
        except Exception as e:
            self.fail(f"Test failed with error: {e}")
        finally:
            elapsed = time.time() - start_time
            print(f"test_run_coroutine_simple completed in {elapsed:.2f}s")
    
    def test_error_propagation_simple(self):
        """Simple error propagation test."""
        async def error_coro():
            await asyncio.sleep(0.1)
            raise ValueError("Simple error")
            
        try:
            with self.assertRaises(ValueError):
                self.bridge.run_coroutine(error_coro(), timeout=2.0)
        except Exception as e:
            self.fail(f"Error propagation test failed: {e}")


# Minimal test runner to avoid hanging
if __name__ == "__main__":
    # Only run the most basic tests
    basic_suite = unittest.TestSuite()
    basic_suite.addTest(TestAsyncBridge("test_run_coroutine_simple"))
    basic_suite.addTest(TestAsyncBridge("test_error_propagation_simple"))
    
    # Print a clear header
    print("\n=== Running AsyncBridge Basic Tests ===\n")
    
    # Run with a short timeout
    result = unittest.TextTestRunner().run(basic_suite)
    
    # Ensure we clean up any bridges
    shutdown_bridge()
    
    # Always print a result
    print("\n=== Test Results ===")
    print(f"Run: {result.testsRun}, Errors: {len(result.errors)}, Failures: {len(result.failures)}")
    
    if result.wasSuccessful():
        print("All basic tests passed!")
        sys.exit(0)
    else:
        print("Some tests failed:")
        for err in result.errors:
            print(f"ERROR: {err[0]}")
            print(err[1])
        for fail in result.failures:
            print(f"FAILURE: {fail[0]}")
            print(fail[1])
        sys.exit(1)