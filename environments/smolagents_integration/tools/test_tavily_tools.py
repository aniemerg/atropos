#!/usr/bin/env python
"""
Test script for Tavily tools to display their output structure.
This helps with understanding what the tools return, which improves prompting.
"""

import os
import json
from dotenv import load_dotenv
from tavily_tools import TavilySearchTool, TavilyExtractTool

# Load environment variables
load_dotenv()

def pretty_print(obj):
    """Pretty print an object as JSON"""
    print(json.dumps(obj, indent=2, default=str))

def test_search_tool():
    """Test the search tool and display its output structure"""
    print("\n===== TESTING TAVILY SEARCH TOOL =====")
    search_tool = TavilySearchTool()
    
    # Perform a simple search
    query = "NVIDIA Omniverse technical specifications"
    print(f"\nPerforming search for: {query}")
    results = search_tool.forward(query, num_results=3)
    
    # Display the raw structure
    print("\nSearch results structure (first item):")
    if results:
        pretty_print(results[0])
        
        print(f"\nGot {len(results)} results. First three titles:")
        for i, result in enumerate(results[:3]):
            print(f"{i+1}. {result.get('title', 'No title')}")
            print(f"   URL: {result.get('url', 'No URL')}")
    else:
        print("No results found.")

def test_extract_tool():
    """Test the extract tool and display its output structure"""
    print("\n===== TESTING TAVILY EXTRACT TOOL =====")
    extract_tool = TavilyExtractTool()
    
    # Try to extract content from a known URL
    url = "https://www.nvidia.com/en-us/omniverse/"
    print(f"\nExtracting content from: {url}")
    result = extract_tool.forward(url)
    
    # Display the structure
    print("\nExtract result structure:")
    pretty_print({k: (v if k != "content" else f"{v[:300]}... [truncated]") for k, v in result.items()})
    
    print(f"\nTitle: {result.get('title', 'No title')}")
    print(f"Content length: {len(result.get('content', ''))}")
    
if __name__ == "__main__":
    print("Testing Tavily tools...")
    test_search_tool()
    test_extract_tool()
    print("\nTesting complete.")