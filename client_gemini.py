#!/usr/bin/env python3
"""
Simple MCP Client for Gemini
Connects to MCP servers and enables tool calling via Gemini API
"""

import asyncio
import json
import os
import sys
import re
from typing import Dict, Any, List, Optional
from contextlib import AsyncExitStack

# Third-party imports
import google.generativeai as genai
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import httpx

# Load environment variables
load_dotenv()


class GeminiMCPClient:
    """MCP Client that uses Gemini for natural language processing and tool calling"""
    
    def __init__(self):
        # Initialize Gemini
        api_key = "AIzaSyDh91qvAFN8X8vfuXylwWlE1OqxX0z3WTA"
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-1.5-flash")
        
        # MCP connection setup
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.tools: List[Any] = []
        
        # Location cache to avoid repeated lookups
        self.location_cache = {}
        
    async def connect_to_server(self, server_path: str):
        """Connect to an MCP server"""
        print(f"Connecting to MCP server: {server_path}")
        
        # Determine how to run the server
        if server_path.endswith(".py"):
            command = "python"
        elif server_path.endswith(".js"):
            command = "node"
        else:
            raise ValueError("Server must be a .py or .js file")
        
        # Set up server parameters
        server_params = StdioServerParameters(
            command=command,
            args=[server_path]
        )
        
        # Connect to server
        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        stdio, write = stdio_transport
        
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(stdio, write)
        )
        
        # Initialize the session
        await self.session.initialize()
        
        # List available tools
        response = await self.session.list_tools()
        self.tools = response.tools
        
        print(f"Connected! Available tools: {[tool.name for tool in self.tools]}")
        
    async def geocode_location(self, location: str) -> Optional[Dict[str, float]]:
        """Convert location name to coordinates using a free geocoding service"""
        # Check cache first
        if location in self.location_cache:
            return self.location_cache[location]
        
        try:
            print(f"🗺️ Looking up coordinates for: {location}")
            
            # Use OpenStreetMap Nominatim (free, no API key required)
            url = "https://nominatim.openstreetmap.org/search"
            params = {
                "q": location,
                "format": "json",
                "limit": 1
            }
            headers = {
                "User-Agent": "GeminiMCPClient/1.0"
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.get(url, params=params, headers=headers, timeout=10.0)
                response.raise_for_status()
                
                data = response.json()
                if data:
                    result = {
                        "latitude": float(data[0]["lat"]),
                        "longitude": float(data[0]["lon"])
                    }
                    
                    # Cache the result
                    self.location_cache[location] = result
                    print(f"📍 Found coordinates: {result['latitude']}, {result['longitude']}")
                    return result
                    
        except Exception as e:
            print(f"⚠️ Failed to geocode '{location}': {e}")
            
        return None
    
    def extract_locations(self, text: str) -> List[str]:
        """Extract potential location names from text"""
        # Common location patterns
        location_patterns = [
            # City, State patterns
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*),\s*([A-Z]{2})\b',
            # City State (without comma)
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+([A-Z]{2})\b',
            # Just city names (capitalized words)
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b'
        ]
        
        locations = []
        for pattern in location_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if isinstance(match, tuple):
                    locations.append(" ".join(match))
                else:
                    locations.append(match)
        
        # Filter out common non-location words
        exclude_words = {
            "Weather", "Forecast", "Alert", "Get", "Show", "What", "Where", 
            "Today", "Tomorrow", "Now", "Current", "The", "And", "Or", "For"
        }
        
        filtered_locations = []
        for loc in locations:
            if not any(word in loc for word in exclude_words):
                filtered_locations.append(loc)
        
        return list(set(filtered_locations))  # Remove duplicates
        
    def create_tools_prompt(self) -> str:
        """Create a prompt describing available tools"""
        if not self.tools:
            return "No tools available."
        
        tool_descriptions = []
        for tool in self.tools:
            desc = f"**{tool.name}**: {tool.description or 'No description'}"
            
            # Add parameter information if available
            if hasattr(tool, 'inputSchema') and tool.inputSchema:
                if 'properties' in tool.inputSchema:
                    params = []
                    properties = tool.inputSchema['properties']
                    required = tool.inputSchema.get('required', [])
                    
                    for param_name, param_info in properties.items():
                        param_type = param_info.get('type', 'unknown')
                        param_desc = param_info.get('description', '')
                        required_marker = " (required)" if param_name in required else ""
                        params.append(f"  - {param_name} ({param_type}){required_marker}: {param_desc}")
                    
                    if params:
                        desc += "\n" + "\n".join(params)
            
            tool_descriptions.append(desc)
        
        return "\n\n".join(tool_descriptions)
    
    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Execute an MCP tool with given arguments"""
        try:
            print(f"🔧 Executing {tool_name} with args: {arguments}")
            
            result = await self.session.call_tool(tool_name, arguments)
            
            # Extract content from result
            if hasattr(result, 'content'):
                if isinstance(result.content, list):
                    # Handle list of content items
                    content_parts = []
                    for item in result.content:
                        if hasattr(item, 'text'):
                            content_parts.append(item.text)
                        else:
                            content_parts.append(str(item))
                    return "\n".join(content_parts)
                else:
                    return str(result.content)
            else:
                return str(result)
                
        except Exception as e:
            error_msg = f"Error executing {tool_name}: {str(e)}"
            print(f"❌ {error_msg}")
            return error_msg
    
    def parse_tool_calls(self, text: str) -> List[Dict[str, Any]]:
        """Parse tool calls from Gemini's response"""
        tool_calls = []
        
        # Look for tool call patterns using regex
        # Pattern: CALL_TOOL: tool_name with arguments {json}
        pattern = r'CALL_TOOL:\s*(\w+)\s+with\s+arguments\s+({[^}]*})'
        matches = re.findall(pattern, text, re.IGNORECASE)
        
        for tool_name, args_str in matches:
            try:
                # Clean up the JSON string
                args_str = args_str.strip()
                arguments = json.loads(args_str)
                tool_calls.append({
                    "name": tool_name,
                    "arguments": arguments
                })
            except json.JSONDecodeError as e:
                print(f"⚠️ Failed to parse arguments for {tool_name}: {e}")
                # Try with empty arguments
                tool_calls.append({
                    "name": tool_name,
                    "arguments": {}
                })
        
        return tool_calls
    
    async def process_query(self, user_query: str) -> str:
        """Process a user query, potentially using tools"""
        
        # Extract locations from the query and try to geocode them
        locations = self.extract_locations(user_query)
        location_info = ""
        
        if locations:
            print(f"🔍 Found potential locations: {locations}")
            for location in locations:
                coords = await self.geocode_location(location)
                if coords:
                    location_info += f"\n- {location}: latitude {coords['latitude']}, longitude {coords['longitude']}"
        
        # Create system prompt with tool information and location context
        system_prompt = f"""You are a helpful weather assistant with access to these tools:

{self.create_tools_prompt()}

When you need to use a tool, use this EXACT format:
CALL_TOOL: tool_name with arguments {{"param": "value"}}

For example:
- CALL_TOOL: get_alerts with arguments {{"state": "CA"}}
- CALL_TOOL: get_forecast with arguments {{"latitude": 40.7128, "longitude": -74.0060}}

IMPORTANT LOCATION HANDLING:
- If the user mentions a location by name, I can help you find the coordinates
- For US weather alerts, you need the 2-letter state code (like CA, NY, TX, FL)
- For forecasts, you need latitude and longitude coordinates

{f"LOCATION INFO FOUND:{location_info}" if location_info else ""}

USER QUESTION: {user_query}

Please help the user with their weather request. If you need coordinates for a location, use the location information provided above, or ask me to look up coordinates for specific places.
"""
        
        max_iterations = 10
        conversation_history = []
        
        for iteration in range(max_iterations):
            try:
                # Generate response from Gemini
                if conversation_history:
                    # Continue conversation with history
                    prompt = "\n".join(conversation_history) + f"\n\nContinue helping with: {user_query}"
                else:
                    prompt = system_prompt
                
                response = self.model.generate_content(prompt)
                
                if not response.text:
                    return "I apologize, but I couldn't generate a response."
                
                response_text = response.text.strip()
                conversation_history.append(f"Assistant: {response_text}")
                
                # Check for tool calls
                tool_calls = self.parse_tool_calls(response_text)
                
                if not tool_calls:
                    # Check if the assistant is asking for location info
                    if any(phrase in response_text.lower() for phrase in 
                          ["coordinates", "latitude", "longitude", "location", "where is"]):
                        
                        # Try to extract new location from the response
                        new_locations = self.extract_locations(response_text)
                        for loc in new_locations:
                            if loc not in self.location_cache:
                                coords = await self.geocode_location(loc)
                                if coords:
                                    additional_info = f"\n\nI found coordinates for {loc}: latitude {coords['latitude']}, longitude {coords['longitude']}. You can now use these coordinates."
                                    response_text += additional_info
                    
                    return response_text
                
                # Execute tool calls
                tool_results = []
                for tool_call in tool_calls:
                    tool_name = tool_call["name"]
                    arguments = tool_call["arguments"]
                    
                    # Validate tool exists
                    if not any(tool.name == tool_name for tool in self.tools):
                        tool_results.append(f"Error: Tool '{tool_name}' not found")
                        continue
                    
                    result = await self.execute_tool(tool_name, arguments)
                    tool_results.append(f"Tool {tool_name} result: {result}")
                
                # Add tool results to conversation
                tool_results_text = "\n".join(tool_results)
                conversation_history.append(f"Tool Results: {tool_results_text}")
                
                # Update the query for next iteration
                user_query = f"Based on the tool results above, please provide a helpful response to the original question."
                
            except Exception as e:
                return f"Error processing query: {str(e)}"
        
        return "Maximum iterations reached. Please try a simpler question."
    
    async def chat_loop(self):
        """Interactive chat loop"""
        print("\n🤖 Gemini MCP Client Ready!")
        print("Type 'quit' or 'exit' to stop\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye! 👋")
                    break
                
                if not user_input:
                    continue
                
                # Process the query
                print("Thinking...")
                response = await self.process_query(user_input)
                print(f"\nAssistant: {response}\n")
                
            except KeyboardInterrupt:
                print("\nGoodbye! 👋")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    async def cleanup(self):
        """Clean up resources"""
        if self.exit_stack:
            await self.exit_stack.aclose()


async def main():
    """Main entry point"""
    if len(sys.argv) != 2:
        print("Usage: python gemini_mcp_client.py <server_script>")
        print("Example: python gemini_mcp_client.py weather.py")
        sys.exit(1)
    
    server_script = sys.argv[1]
    
    # Check if server file exists
    if not os.path.exists(server_script):
        print(f"Error: Server script '{server_script}' not found")
        sys.exit(1)
    
    client = GeminiMCPClient()
    
    try:
        await client.connect_to_server(server_script)
        await client.chat_loop()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())