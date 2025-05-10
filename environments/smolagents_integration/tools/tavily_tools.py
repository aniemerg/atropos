"""
Tavily integration tools for SmolAgents.
These tools replace the SerpAPI and SimpleTextBrowser based tools with Tavily's content extraction service.
"""

import os
from typing import Optional, List, Dict, Any
from tavily import TavilyClient
from smolagents import Tool


class TavilyExtractTool(Tool):
    name = "visit_page"
    description = """Visit a webpage at a given URL and extract its content.
    
    Returns an object containing:
    - url: The URL that was visited
    - title: The title of the webpage
    - content: The full text content of the webpage
    - success: Boolean indicating if the extraction was successful
    - error: Error message if extraction failed (null if successful)
    """
    inputs = {"url": {"type": "string", "description": "The URL to visit"}}
    output_type = "object"
    
    def __init__(self, api_key=None):
        super().__init__()
        self.api_key = api_key or os.environ.get("TAVILY_API_KEY")
        self.client = TavilyClient(api_key=self.api_key)
        
    def forward(self, url: str) -> Dict[str, Any]:
        """
        Visit a webpage and extract its content.
        
        Args:
            url: The URL to visit
            
        Returns:
            A dictionary containing:
            - url: The URL that was visited
            - title: The title of the webpage
            - content: The text content of the webpage
            - success: Boolean indicating if the extraction was successful
            - error: Error message if extraction failed
            
        Note: This function automatically prints the full page content (up to 25000 characters)
        when executed, regardless of whether you explicitly display the content in your code.
        """
        try:
            response = self.client.extract(
                urls=url,
                include_images=False,
                extract_depth="basic"
            )
            
            if not response["results"]:
                print(f"\n🌐 VISIT PAGE: {url}\nError: Failed to extract content")
                return {
                    "url": url,
                    "title": "Unknown",
                    "content": "",
                    "success": False,
                    "error": f"Failed to extract content from {url}"
                }
                
            content = response["results"][0]["raw_content"]
            title = response["results"][0].get("title", "Unknown title")
            
            # Print the entire content (up to 25000 chars) for the model to see
            # print(f"\n\n{'='*80}")
            # print(f"🌐 WEBPAGE CONTENT FROM: {url}")
            # print(f"TITLE: {title}")
            # print(f"{'='*80}\n")
            
            # Print content up to 25000 characters
            # display_content = content[:25000]
            # if len(content) > 25000:
            #     display_content += "... [Content truncated - full content available in return value]"
            # print(display_content)
            
            # print(f"\n{'='*80}")
            # print(f"END OF CONTENT - Length: {len(content)} characters")
            # print(f"{'='*80}\n")
            
            # Format the content as a structured object
            return {
                "url": url,
                "title": title,
                "content": content,
                "success": True,
                "error": None
            }
        except Exception as e:
            error_msg = f"Error extracting content from {url}: {str(e)}"
            print(f"\n🌐 VISIT PAGE: {url}\nError: {error_msg}")
            return {
                "url": url,
                "title": "Unknown",
                "content": "",
                "success": False,
                "error": error_msg
            }


class TavilySearchTool(Tool):
    name = "web_search"
    description = """Perform a web search query and return the search results.
    
    Returns an array of search result objects, each containing:
    - title: The title of the search result
    - url: The URL of the search result
    - content: The full text content from the search result
    - snippet: A text snippet from the content (same as content field)
    - date: The publication date if available (may be null)
    """
    inputs = {
        "query": {"type": "string", "description": "The web search query to perform."},
        "num_results": {"type": "integer", "description": "Number of results to return (default: 10)", "nullable": True},
        "filter_year": {"type": "string", "description": "Filter results to a specific year", "nullable": True}
    }
    output_type = "array"

    def __init__(self, api_key=None):
        super().__init__()
        self.api_key = api_key or os.environ.get("TAVILY_API_KEY")
        self.client = TavilyClient(api_key=self.api_key)
        
    def forward(self, query: str, num_results: Optional[int] = 10, filter_year: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Perform a web search.
        
        Args:
            query: The search query
            num_results: Number of results to return (default: 10)
            filter_year: Filter results to a specific year (optional)
            
        Returns:
            A list of search result objects, each containing:
            - title: The title of the search result
            - url: The URL of the search result
            - content: The content snippet from the search result
            - date: The date of the content if available
        """
        try:
            search_params = {
                "query": query,
                "search_depth": "advanced",
                "max_results": num_results or 10  # Default to 10 if None is provided
            }
            
            # Add year filter if provided
            if filter_year:
                search_params["query"] += f" {filter_year}"
                
            # Use Tavily's search API
            response = self.client.search(**search_params)
            
            if not response.get("results"):
                return []
            
            # Convert Tavily results to the expected format for the agent
            formatted_results = []
            for result in response["results"]:
                formatted_results.append({
                    'title': result.get("title", "No title"),
                    'url': result.get("url", ""),
                    'content': result.get("content", ""),
                    'snippet': result.get("content", ""),  # For compatibility with expected format
                    'date': result.get("published_date", None)
                })
            
            return formatted_results
        
        except Exception as e:
            print(f"Error searching for '{query}': {str(e)}")
            return []