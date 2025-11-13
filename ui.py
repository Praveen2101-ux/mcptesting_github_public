import streamlit as st
import asyncio
from typing import Dict, List, Any, Optional, Literal
from dataclasses import dataclass
import json
import os
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_openai import AzureChatOpenAI
import time
from datetime import datetime
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession
from mcp.client.sse import sse_client
import traceback
import base64
import io
from PIL import Image

# Configuration loading and saving functions
def load_mcp_servers_config():
    """Load MCP server configurations from JSON file"""
    config_file = "mcp_servers.json"
    if not os.path.exists(config_file):
        # Create default config if file doesn't exist
        default_config = {
            "mcp_servers": [
                {
                    "name": "Primary MCP Server",
                    "url": "http://10.142.26.34:8093/sse",
                    "description": "Main MCP server for primary tools",
                    "active": True
                }
            ]
        }
        with open(config_file, 'w') as f:
            json.dump(default_config, f, indent=2)
        return default_config
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        st.error(f"Error loading MCP servers configuration: {str(e)}")
        return {"mcp_servers": []}

def save_mcp_servers_config(config):
    """Save MCP server configurations to JSON file"""
    config_file = "mcp_servers.json"
    try:
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        return True
    except Exception as e:
        st.error(f"Error saving MCP servers configuration: {str(e)}")
        return False

def add_mcp_server(name, url, description, active=True):
    """Add a new MCP server to the configuration"""
    config = load_mcp_servers_config()
    
    # Check if server with same name or URL already exists
    for server in config.get("mcp_servers", []):
        if server["name"] == name:
            return False, f"Server with name '{name}' already exists"
        if server["url"] == url:
            return False, f"Server with URL '{url}' already exists"
    
    # Add new server
    new_server = {
        "name": name,
        "url": url,
        "description": description,
        "active": active
    }
    
    if "mcp_servers" not in config:
        config["mcp_servers"] = []
    
    config["mcp_servers"].append(new_server)
    
    if save_mcp_servers_config(config):
        return True, "Server added successfully"
    else:
        return False, "Failed to save configuration"

def remove_mcp_server(server_name):
    """Remove an MCP server from the configuration"""
    config = load_mcp_servers_config()
    
    original_count = len(config.get("mcp_servers", []))
    config["mcp_servers"] = [
        server for server in config.get("mcp_servers", [])
        if server["name"] != server_name
    ]
    
    if len(config["mcp_servers"]) < original_count:
        if save_mcp_servers_config(config):
            return True, "Server removed successfully"
        else:
            return False, "Failed to save configuration"
    else:
        return False, "Server not found"

def toggle_server_active(server_name):
    """Toggle the active status of an MCP server"""
    config = load_mcp_servers_config()
    
    for server in config.get("mcp_servers", []):
        if server["name"] == server_name:
            server["active"] = not server.get("active", True)
            if save_mcp_servers_config(config):
                return True, f"Server '{server_name}' {'activated' if server['active'] else 'deactivated'}"
            else:
                return False, "Failed to save configuration"
    
    return False, "Server not found"

# Import your existing classes (assuming they're in the same file or imported)
@dataclass
class AgentState:
    """State object for the agent workflow"""
    messages: List[Any] = None
    user_input: str = ""
    intent_analysis: Optional[Dict] = None  # Changed from simple intent string
    tool_calls: List[Dict] = None
    tool_results: List[Dict] = None
    llm_response: Optional[str] = None  # Added for general knowledge responses
    final_response: Optional[str] = None
    
    def __post_init__(self):
        if self.messages is None:
            self.messages = []
        if self.tool_calls is None:
            self.tool_calls = []
        if self.tool_results is None:
            self.tool_results = []
        if self.intent_analysis is None:
            self.intent_analysis = {}

class MCPClientWrapper:
    """Wrapper for MCP client using LangGraph MCP tool loader"""
    def __init__(self, server_info: dict):
        self.server_info = server_info
        self.server_url = server_info["url"]
        self.server_name = server_info["name"]

    async def list_tools(self) -> list:
        """Fetch available tools from live MCP server using langgraph."""
        url = self.server_url.strip()
        try:
            async with sse_client(url) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tools = await load_mcp_tools(session)
                    tool_list = []
                    for t in tools:
                        name = getattr(t, "name", type(t).__name__)
                        desc = getattr(t, "description", None) or getattr(t, "__doc__", "")
                        params = getattr(t, "args_schema", None)
                        param_dict = {}
                        if params:
                            for k, v in params.items():
                                if isinstance(v, dict):
                                    param_dict[k] = {
                                        "type": str(v.get("type", "")),
                                        "description": v.get("description", "")
                                    }
                                elif isinstance(v, list):
                                    param_dict[k] = {
                                        "type": "list",
                                        "description": ", ".join(str(item) for item in v)
                                    }
                                else:
                                    param_dict[k] = {
                                        "type": str(type(v)),
                                        "description": str(v)
                                    }
                        tool_list.append({
                            "name": name, 
                            "description": desc, 
                            "parameters": param_dict,
                            "server_name": self.server_name,
                            "server_url": self.server_url
                        })
                    return tool_list
        except Exception as e:
            st.warning(f"Failed to connect to {self.server_name} ({self.server_url}): {str(e)}")
            return []

    async def call_tool(self, tool_name: str, parameters: dict) -> any:
        """Call an MCP tool with given parameters using LangGraph MCP adapters"""
        url = self.server_url.strip()
        try:
            async with sse_client(url) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    
                    # Load all tools from the MCP server
                    tools = await load_mcp_tools(session)
                    
                    # Find the specific tool by name
                    target_tool = None
                    for tool in tools:
                        tool_name_attr = getattr(tool, "name", type(tool).__name__)
                        if tool_name_attr == tool_name:
                            target_tool = tool
                            break
                    
                    if target_tool is None:
                        raise Exception(f"Tool '{tool_name}' not found on server {self.server_name}")
                    
                    # Call the tool with parameters
                    if parameters:
                        result = await target_tool.ainvoke(parameters)
                    else:
                        result = await target_tool.ainvoke({})
                    
                    return result
                    
        except Exception as e:
            raise Exception(f"Failed to call tool '{tool_name}' on {self.server_name}: {str(e)}")

class HybridMCPAgent:
    """A hybrid agent that can handle both general conversations and MCP tool interactions"""
    
    def __init__(self, llm: BaseChatModel, mcp_client):
        self.llm = llm
        self.mcp_client = mcp_client
        self.available_tools = {}  # Initialize as empty, set after async fetch
        self.workflow = self._build_workflow()
        
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow with hybrid processing support"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("analyze_intent", self._analyze_intent)
        workflow.add_node("handle_general_only", self._handle_general_query)
        workflow.add_node("handle_hybrid", self._handle_hybrid_query)
        workflow.add_node("handle_tools_only", self._handle_tools_only)
        workflow.add_node("synthesize_response", self._synthesize_response)
        
        # Add conditional edge from analyze_intent
        workflow.add_conditional_edges(
            "analyze_intent",
            self._route_based_on_intent,
            {
                "general_only": "handle_general_only",
                "tools_only": "handle_tools_only", 
                "hybrid": "handle_hybrid"
            }
        )
        
        # Add regular edges
        workflow.add_edge("handle_general_only", END)
        workflow.add_edge("handle_tools_only", "synthesize_response")
        workflow.add_edge("handle_hybrid", "synthesize_response")
        workflow.add_edge("synthesize_response", END)
        
        # Set entry point
        workflow.set_entry_point("analyze_intent")
        
        return workflow.compile(checkpointer=MemorySaver())
    
    async def _analyze_intent(self, state: AgentState) -> AgentState:
        """Analyze query for multiple intents (tools, general knowledge, or hybrid)"""
        
        analysis_prompt = f"""
        You are an advanced intent analyzer. Analyze the user's input and determine what types of processing it requires.

        Available tools: {', '.join(self.available_tools.keys())}
        
        User input: {state.user_input}

        Analyze the query by considering:
        1. Does any part require real-time data, external services, or specific computations?
        2. Does any part require general knowledge, explanations, or reasoning?
        3. Can different parts be processed independently?
        
        Analyze the query and respond with a JSON object containing:
        {{
            "requires_tools": true/false,
            "requires_general_knowledge": true/false,
            "tool_parts": ["specific parts that need tools"],
            "knowledge_parts": ["parts that need general knowledge"],
            "processing_type": "general_only" | "tools_only" | "hybrid",
            "reasoning": "explanation of the analysis"
        }}
        """
        
        messages = [SystemMessage(content=analysis_prompt)]
        response = await self.llm.ainvoke(messages)
        
        try:
            intent_analysis = json.loads(response.content)
            state.intent_analysis = intent_analysis
        except json.JSONDecodeError:
            # Fallback analysis
            state.intent_analysis = {
                "requires_tools": True,
                "requires_general_knowledge": False,
                "tool_parts": [state.user_input],
                "knowledge_parts": [],
                "processing_type": "tools_only",
                "reasoning": "Fallback analysis due to JSON parsing error"
            }
        
        return state
    
    def _route_based_on_intent(self, state: AgentState) -> Literal["general_only", "tools_only", "hybrid"]:
        """Route to appropriate handler based on intent analysis"""
        return state.intent_analysis.get("processing_type", "general_only")
    
    async def _handle_hybrid_query(self, state: AgentState) -> AgentState:
        """Handle queries that need both tools and general knowledge"""
        
        # Step 1: Plan and execute tools for tool-requiring parts
        await self._plan_tool_usage(state)
        await self._execute_tools(state)
        
        # Step 2: Generate LLM response for knowledge-requiring parts
        knowledge_parts = state.intent_analysis.get("knowledge_parts", [])
        if knowledge_parts:
            knowledge_query = " ".join(knowledge_parts)
            
            system_prompt = f"""
            You are a helpful AI assistant. The user asked a complex question that has multiple parts.
            
            This part requires your general knowledge: {knowledge_query}
            
            Provide a comprehensive response using your knowledge. This will be combined with tool results later.
            """
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=knowledge_query)
            ]
            
            response = await self.llm.ainvoke(messages)
            state.llm_response = response.content
        
        return state
    
    async def _handle_tools_only(self, state: AgentState) -> AgentState:
        """Handle queries that only need tools"""
        await self._plan_tool_usage(state)
        await self._execute_tools(state)
        return state
    
    async def _handle_general_query(self, state: AgentState) -> AgentState:
        """Handle general conversation without tools"""
        
        system_prompt = """
        You are a helpful AI assistant. Respond to the user's question or engage in conversation naturally.
        You have access to various tools, but this particular query doesn't seem to require them.
        Provide a helpful, informative response based on your knowledge.
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=state.user_input)
        ]
        
        response = await self.llm.ainvoke(messages)
        state.final_response = response.content
        
        return state
    
    async def _plan_tool_usage(self, state: AgentState) -> AgentState:
        """Plan which tools to use and how"""
        
        planning_prompt = f"""
        Based on the user's request, plan which tools to use and in what order.
        
        Available tools:
        {json.dumps(self.available_tools, indent=2)}
        
        User request: {state.user_input}
        
        Respond with a JSON array of tool calls in this format:
        [
            {{
                "tool_name": "tool_name",
                "parameters": {{"param1": "value1", "param2": "value2"}},
                "reasoning": "Why this tool is needed"
            }}
        ]
        
        If no tools are needed, respond with an empty array: []
        """
        
        messages = [SystemMessage(content=planning_prompt)]
        response = await self.llm.ainvoke(messages)
        
        try:
            tool_calls = json.loads(response.content)
            state.tool_calls = tool_calls
        except json.JSONDecodeError:
            state.tool_calls = []
        
        return state
    
    async def _execute_tools(self, state: AgentState) -> AgentState:
        """Execute the planned tool calls"""
        
        results = []
        
        for tool_call in state.tool_calls:
            try:
                tool_name = tool_call.get("tool_name")
                parameters = tool_call.get("parameters", {})
                
                result = await self._call_mcp_tool(tool_name, parameters)
                
                results.append({
                    "tool_name": tool_name,
                    "parameters": parameters,
                    "result": result,
                    "success": True
                })
                
            except Exception as e:
                results.append({
                    "tool_name": tool_call.get("tool_name"),
                    "parameters": tool_call.get("parameters", {}),
                    "error": str(e),
                    "success": False
                })
        
        state.tool_results = results
        return state
    
    async def _call_mcp_tool(self, tool_name: str, parameters: Dict) -> Any:
        """Call an MCP tool with given parameters"""
        try:
            result = await self.mcp_client.call_tool(tool_name, parameters)
            return result
        except Exception as e:
            raise Exception(f"Tool execution failed: {str(e)}")
    
    async def _synthesize_response(self, state: AgentState) -> AgentState:
        """Synthesize final response from both tool results and LLM knowledge"""
        
        # Prepare synthesis prompt based on what data we have
        synthesis_components = [f"Original request: {state.user_input}"]
        
        if state.tool_results:
            synthesis_components.append(f"Tool execution results:\n{json.dumps(state.tool_results, indent=2)}")
        
        if state.llm_response:
            synthesis_components.append(f"General knowledge response:\n{state.llm_response}")
        
        synthesis_prompt = f"""
        Based on the user's original request and the available information, provide a comprehensive response.
        
        {chr(10).join(synthesis_components)}
        
        Your task:
        1. Address ALL parts of the user's original request
        2. Integrate tool results with general knowledge seamlessly
        3. Provide a natural, flowing response that doesn't feel fragmented
        4. If any tools failed, explain limitations gracefully
        5. Ensure the response is complete and helpful
        
        Create a unified response that feels like a knowledgeable assistant answering completely.
        """
        
        messages = [SystemMessage(content=synthesis_prompt)]
        response = await self.llm.ainvoke(messages)
        
        state.final_response = response.content
        return state
    
    async def process_input(self, user_input: str) -> tuple:
        """Main entry point for processing user input"""
        
        initial_state = AgentState(
            messages=[],
            user_input=user_input
        )
        
        config = {"configurable": {"thread_id": "default"}}
        final_state = await self.workflow.ainvoke(initial_state, config=config)
        
        # Handle both dict and AgentState responses
        if isinstance(final_state, dict):
            response = final_state.get("final_response", "No response generated")
            intent_analysis = final_state.get("intent_analysis", {})
            tool_calls = final_state.get("tool_calls", [])
        else:
            response = getattr(final_state, "final_response", "No response generated")
            intent_analysis = getattr(final_state, "intent_analysis", {})
            tool_calls = getattr(final_state, "tool_calls", [])
        
        # Determine call type based on processing type
        processing_type = intent_analysis.get("processing_type", "unknown")
        call_type_map = {
            "general_only": "💬 LLM Call",
            "tools_only": "🔧 Tool Call", 
            "hybrid": "🔄 Hybrid Call (LLM+tool)",
            "unknown": "❓ Unknown"
        }
        call_type = call_type_map.get(processing_type, "❓ Unknown")
        
        return response, call_type, tool_calls

def set_bg_image(image_path):
    """
    Sets a background image throughout the entire Streamlit app.
    """
    with open(image_path, "rb") as f:
        encoded_img = base64.b64encode(f.read()).decode()  # Correct encoding
    
    bg_css = f"""
    <style>
    .stApp {{
        background: url('data:image/jpeg;base64,{encoded_img}') no-repeat center center fixed;
        background-size: cover;
    }}
    </style>
    """
    st.markdown(bg_css, unsafe_allow_html=True)

# Streamlit UI
def main():
    st.set_page_config(
        page_title="MCP Client",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS based on edi_app.py styling
    st.markdown("""
    <style>
    /* Main container styles */
    .stApp {
        background-color: white;
    }      
    .header-container {
        display: flex;
        align-items: center;
        justify-content: center; /* Center the logo and text */
    }
   
    /* Fixed header styles */
    .fixed-header {
        position: fixed;
        margin-top: -100px;  /* More negative margin to move it higher */
        padding-bottom: 0px;
        z-index: 100;  /* Ensure it's above other elements */
        background-color: rgb(240, 246, 255);
        width: 100%;
        display: flex;
        justify-content: center;
        align-items: center;  
        left: 0; /* Ensure the header spans the full width */
        right: 0;      
    }
    .fixed-header h1 {
        margin-top: 0px;  /* Remove default margin */
        text-align: center;  /* Center the title instead of right-aligned */
        color: #003C7E;
        font-weight: bold;
        font-size: 28px;
    }
   
    /* Header content container */
    .header-content {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        width: 100%;
    }
   
    /* Logo container */
    .logo-container {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 0.5rem;
    }
   
    /* Title styling */
    .header-title {
        font-size: 24px;
        font-weight: bold;
        margin: 0;
        color: #000;
    }
   
    /* Chat container styles */
    .chat-container {
        margin-top: 10rem;  /* Increased space for banner and fixed header */
        margin-bottom: 5rem;  /* Space for fixed footer */
        padding: 0 2rem;
        overflow-y: auto;
    }
   
    /* Sidebar styles */
    section[data-testid="stSidebar"] {
        background-color: #2f3034 !important;
        color: #f7f8f8 !important;
    }
   
    section[data-testid="stSidebar"] .stFileUploader {
        background-color: #2f3034 !important;
        color: #f7f8f8 !important;
        border-radius: 6px;
        padding: 8px;
    }

    /* Style expander */
    section[data-testid="stSidebar"] details {
        background-color: #2f3034 !important;
        color: #f7f8f8 !important;
        border: 1px solid #888;
        border-radius: 6px;
        padding: 4px;
    }

    section[data-testid="stSidebar"] summary {
        color: #f7f8f8 !important;
    }

    /* Style buttons */
    section[data-testid="stSidebar"] button {
        background-color: #2f3034 !important;
        color: #f7f8f8 !important;
        border: 1px solid #888 !important;
        border-radius: 6px;
        width: 100%;
        padding: 8px 12px;
    }

    section[data-testid="stSidebar"] button:hover {
        background-color: #444 !important;
        color: #fff !important;
    }
    
    /* Apply dark background and light text to the file uploader dropzone */
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
        background-color: #2f3034 !important;
        color: #f7f8f8 !important;
        border: 1px dashed #888 !important;
        border-radius: 6px;
        padding: 10px;
    }

    /* Optional: darken text inside the uploader */
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] * {
        color: #f7f8f8 !important;
    }
    
    /* File uploader drag box and content */
    section[data-testid="stSidebar"] .stFileUploader > div:first-child {
        background-color: #2f3034 !important;
        color: #f7f8f8 !important;
        border: 1px dashed #888 !important;
        border-radius: 8px !important;
        padding: 8px;
    }

    /* Widget label styling */
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] {
        color: #f7f8f8 !important;
        font-weight: 500;
        font-size: 0.9rem;
        margin-bottom: 4px;
    }
    
    /* Set file name color */
    section[data-testid="stSidebar"] [data-testid="stFileUploaderFileName"] {
        color: #f7f8f8 !important;
    }

    /* Set file size (small tag) color */
    section[data-testid="stSidebar"] .stFileUploaderFileName + small {
        color: #f7f8f8 !important;
        font-size: 0.85rem;
    }

    /* Alert box styling (e.g. st.sidebar.warning/info) */
    section[data-testid="stSidebar"] [data-testid="stAlert"] {
        background-color: #ffffff !important;
        color: #000000 !important;
        border-radius: 6px;
    }

    /* Also apply to children inside alert */
    section[data-testid="stSidebar"] [data-testid="stAlert"] * {
        color: #000000 !important;
    }

    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #f7f8f8 !important;
        font-size: 0.8rem;
    }
    
    /* Spinner icon and label styling */
    section[data-testid="stSidebar"] [data-testid="stSpinner"] {
        color: #f7f8f8 !important;
    }

    /* Target the spinner SVG icon (rotating circle) */
    section[data-testid="stSidebar"] [data-testid="stSpinner"] svg {
        stroke: #f7f8f8 !important;
    }
    
    /* Hybrid mode indicator styling */
    .hybrid-mode-indicator {
        background: linear-gradient(135deg, #4CAF50 0%, #2196F3 100%);
        color: white;
        padding: 8px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        text-align: center;
        margin: 8px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    
    .mode-badge {
        display: inline-block;
        padding: 4px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 4px 0;
    }
    
    .rag-mode {
        background-color: #4CAF50;
        color: white;
    }
    
    .context-mode {
        background-color: #2196F3;
        color: white;
    }
    
    /* Server management styling */
    section[data-testid="stSidebar"] .stTextInput input {
        background-color: #404040 !important;
        color: #f7f8f8 !important;
        border: 1px solid #666 !important;
        border-radius: 6px;
    }
    
    section[data-testid="stSidebar"] .stTextArea textarea {
        background-color: #404040 !important;
        color: #f7f8f8 !important;
        border: 1px solid #666 !important;
        border-radius: 6px;
    }
    
    section[data-testid="stSidebar"] .stCheckbox {
        color: #f7f8f8 !important;
    }
    
    section[data-testid="stSidebar"] .stCheckbox label {
        color: #f7f8f8 !important;
    }
    
    /* Server status indicators */
    .server-status-active {
        color: #4CAF50;
        font-weight: bold;
    }
    
    .server-status-inactive {
        color: #f44336;
        font-weight: bold;
    }
   
    /* User info in sidebar */
    .user-info {
        position: fixed;
        bottom: 0;
        left: 0;
        padding: 1rem;
        background-color: #2f3034;
        width: 25rem;
        font-size: 1rem;
        color: #fff;
        border-top: 1px solid #fff;
        box-shadow: 0 -2px 8px rgba(0, 0, 0, 0.4);
    }
   
    /* Chat input container */
    .stChatInputContainer {
        position: fixed;
        bottom: 0;
        left: 16rem;
        right: 0;
        padding: 1rem 2rem;
        background: white;
        border-top: 1px solid #f0f0f0;
    }
   
    /* Ensure chat messages don't get hidden */
    .stChatMessageContent {
        max-width: 80%;
    }
    
    /* Latest message highlight */
    .latest-message {
        animation: highlight-pulse 2s ease-in-out;
        border-left: 4px solid #4CAF50;
        padding-left: 15px !important;
        background: rgba(76, 175, 80, 0.1) !important;
        border-radius: 8px;
        margin-bottom: 10px;
    }
    
    @keyframes highlight-pulse {
        0% { box-shadow: 0 0 0 0 rgba(76, 175, 80, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(76, 175, 80, 0); }
        100% { box-shadow: 0 0 0 0 rgba(76, 175, 80, 0); }
    }
    
    /* Message styling improvements */
    .stChatMessage {
        margin-bottom: 15px;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        border: 2px solid transparent;
    }
    
    .stChatMessage:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0,0,0,0.15);
    }
    
    /* User message styling - Blue theme with right alignment */
    .stChatMessage[data-testid*="user"] {
        background: linear-gradient(135deg, #4A90E2 0%, #357ABD 100%);
        color: white;
        margin-left: 25%;
        margin-right: 5%;
        border-left: 4px solid #2E5C8A;
        position: relative;
    }
    
    .stChatMessage[data-testid*="user"]::before {
        content: "👤";
        position: absolute;
        top: -5px;
        right: -5px;
        background: #2E5C8A;
        color: white;
        border-radius: 50%;
        width: 30px;
        height: 30px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 14px;
        border: 2px solid white;
    }
    
    /* Assistant message styling - Green theme with left alignment */
    .stChatMessage[data-testid*="assistant"] {
        background: linear-gradient(135deg, #50C878 0%, #3A9B5C 100%);
        color: white;
        margin-right: 25%;
        margin-left: 5%;
        border-left: 4px solid #2E7D40;
        position: relative;
    }
    
    .stChatMessage[data-testid*="assistant"]::before {
        content: "🤖";
        position: absolute;
        top: -5px;
        left: -5px;
        background: #2E7D40;
        color: white;
        border-radius: 50%;
        width: 30px;
        height: 30px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 14px;
        border: 2px solid white;
    }
    
    /* Enhanced text styling for better readability */
    .stChatMessage[data-testid*="user"] .stMarkdown {
        font-weight: 500;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    
    .stChatMessage[data-testid*="assistant"] .stMarkdown {
        font-weight: 400;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
    }
    
    /* Call type badges */
    .call-type-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: bold;
        margin-bottom: 10px;
        background: rgba(255,255,255,0.9);
        color: #333;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.3);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Welcome message styling */
    .welcome-message {
        text-align: center;
        padding: 50px;
        color: #666;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 15px;
        margin: 20px 0;
    }
    
    /* Hide default Streamlit elements we don't need */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)
    
    # Fixed header
    st.markdown("""
        <div class="fixed-header">
            <h1>🤖 MCP Playground</h1>
        </div>
    """, unsafe_allow_html=True)
    
    # Get the current directory where the script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load logo image for banner
    bg_image_path = os.path.join(current_dir, "C://Users//epv1cob//Desktop//LLM Course//MCP//MCP_Client_MCP_agent//bosch_logo_w340.png")
    if os.path.exists(bg_image_path):
        logo_img = Image.open(bg_image_path)
        buffered_logo = io.BytesIO()
        logo_img.save(buffered_logo, format="PNG")
        logo_base64 = base64.b64encode(buffered_logo.getvalue()).decode()
    else:
        logo_base64 = ""
    
    # === BANNER IMAGE (Full-width top bar) ===
    banner_path = os.path.join(current_dir, "C://Users//epv1cob//Desktop//LLM Course//MCP//MCP_Client_MCP_agent//Bosch Header Ribbon.png")  # your top banner image

    if os.path.exists(banner_path):
        banner_img = Image.open(banner_path)
        buffered_banner = io.BytesIO()
        banner_img.save(buffered_banner, format="PNG")
        banner_base64 = base64.b64encode(buffered_banner.getvalue()).decode()

        st.markdown(f"""
            <style>
            .banner-container {{
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                height: 75px;
                background-color: #fff;
                z-index: 9999;
                margin: 0;
                box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);
                display: flex;
                flex-direction: column;
                justify-content: center;
            }}

            .banner-line {{
                width: 100%;
                height: 10px;
                object-fit: cover;
                margin-bottom: 5px;
            }}

            .banner-content {{
                display: flex;
                align-items: center;
                justify-content: space-between;
                flex: 1;
            }}

            .banner-title {{
                flex: 1;
                text-align: center;
                font-size: 24px;
                font-weight: 600;
                color: #0F2A50;
                font-family: 'Segoe UI', sans-serif;
                margin: 0;
                padding-bottom: 10px;
            }}

            .logo-img {{
                height: 40px;
                width: auto;
                padding: 0 25px 10px 25px;
            }}
            </style>

            <div class="banner-container">
                <img src="data:image/png;base64,{banner_base64}" alt="Banner Line" class="banner-line" />
                <div class="banner-content">
                    <div class="banner-title">MCP Control Center</div>
                    <img src="data:image/png;base64,{logo_base64}" alt="Logo" class="logo-img" />
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    # Background image setup
    # Call the function with your image
    bg_image_path = os.path.join(current_dir, "ag.jpg")
    # Uncomment the line below to enable background image
    # set_bg_image(bg_image_path)  # Change filename if needed
    
    st.markdown("---")
    
    # Sidebar for configuration
    with st.sidebar:
        # Clear chat button at the top
        if st.button("🗑️ Clear Chat", type="secondary", use_container_width=True):
            st.session_state.messages = []
            # Reset server selection to default option
            st.session_state.selected_server_key = "Select MCP server"
            # Clear agent to force reinitialization when user selects a new server
            if "agent" in st.session_state:
                del st.session_state.agent
            if "current_server" in st.session_state:
                del st.session_state.current_server
            st.rerun()
        
        st.markdown("---")
        st.header("Configuration")
        
        # Load MCP servers configuration (inside sidebar to refresh on changes)
        servers_config = load_mcp_servers_config()
        active_servers = [server for server in servers_config.get("mcp_servers", []) if server.get("active", True)]
        
        # Create server options for selectbox
        default_option = "Select MCP server"
        if not active_servers:
            st.warning("⚠️ No active MCP servers found in configuration!")
            st.info("💡 Use the 'Add New MCP Server' section below to add a server.")
            server_options = {default_option: None}
        else:
            server_options = {default_option: None}  # Default option maps to None
            server_options.update({f"{server['name']}": server for server in active_servers})
        
        # MCP Server selection
        st.subheader("🌐 MCP Servers")
        
        # Initialize selected server key to default option
        if "selected_server_key" not in st.session_state or st.session_state.selected_server_key not in server_options:
            st.session_state.selected_server_key = default_option
        
        selected_server_display = st.selectbox(
            "Choose MCP Server",
            options=list(server_options.keys()),
            index=list(server_options.keys()).index(st.session_state.selected_server_key),
            help="Choose which MCP server to connect to",
            key="server_selectbox"
        )
        
        # Update session state when selection changes
        if selected_server_display != st.session_state.selected_server_key:
            st.session_state.selected_server_key = selected_server_display
            # Clear agent when server changes to force reinitialization
            if "agent" in st.session_state:
                del st.session_state.agent
            if "current_server" in st.session_state:
                del st.session_state.current_server
        
        selected_server = server_options[selected_server_display]
        
        # Server Management Section
        st.markdown("---")
        st.subheader("⚙️ Server Management")
        
        # Add new server form
        # Initialize expander state
        if "add_server_expanded" not in st.session_state:
            st.session_state.add_server_expanded = False
            
        with st.expander("➕ Add New MCP Server", expanded=st.session_state.add_server_expanded):
            with st.form("add_server_form"):
                # Use session state to control form values for clearing
                if "clear_add_form" not in st.session_state:
                    st.session_state.clear_add_form = False
                
                # Clear form values if flag is set
                name_value = "" if st.session_state.clear_add_form else None
                url_value = "" if st.session_state.clear_add_form else None
                desc_value = "" if st.session_state.clear_add_form else None
                active_value = True if st.session_state.clear_add_form else True
                
                new_server_name = st.text_input(
                    "Server Name", 
                    value=name_value,
                    placeholder="e.g., My Custom Server",
                    help="Give your server a unique name"
                )
                new_server_url = st.text_input(
                    "Server URL", 
                    value=url_value,
                    placeholder="e.g., http://localhost:8093/sse",
                    help="Full URL including protocol and path"
                )
                new_server_description = st.text_area(
                    "Description", 
                    value=desc_value,
                    placeholder="Brief description of what this server provides",
                    help="Optional description of the server's capabilities"
                )
                new_server_active = st.checkbox("Active", value=active_value, help="Whether this server should be available for selection")
                
                submitted = st.form_submit_button("Add Server", type="primary")
                
                if submitted:
                    if new_server_name and new_server_url:
                        # Validate URL format
                        if not (new_server_url.startswith("http://") or new_server_url.startswith("https://")):
                            st.error("❌ URL must start with http:// or https://")
                        else:
                            success, message = add_mcp_server(
                                new_server_name, 
                                new_server_url, 
                                new_server_description or "No description provided",
                                new_server_active
                            )
                            if success:
                                st.success(f"✅ {message}")
                                # Clear form and minimize expander
                                st.session_state.clear_add_form = True
                                st.session_state.add_server_expanded = False
                                st.rerun()  # Refresh to show new server
                            else:
                                st.error(f"❌ {message}")
                    else:
                        st.error("❌ Please provide both server name and URL")
                
                # Reset clear flag after processing
                if st.session_state.clear_add_form:
                    st.session_state.clear_add_form = False
        
        # Manage existing servers
        if len(active_servers) > 0:
            with st.expander("🔧 Manage Existing Servers"):
                st.write("**Current Servers:**")
                
                # Load all servers (including inactive ones) for management
                all_servers_config = load_mcp_servers_config()
                all_servers = all_servers_config.get("mcp_servers", [])
                
                for i, server in enumerate(all_servers):
                    col1_mgmt, col2_mgmt, col3_mgmt = st.columns([3, 1, 1])
                    
                    with col1_mgmt:
                        status_icon = "🟢" if server.get("active", True) else "🔴"
                        st.write(f"{status_icon} **{server['name']}**")
                        st.caption(f"URL: {server['url']}")
                        if server.get('description'):
                            st.caption(f"Description: {server['description']}")
                    
                    with col2_mgmt:
                        if st.button(
                            "🔄" if server.get("active", True) else "▶️", 
                            key=f"toggle_{i}",
                            help="Toggle active/inactive"
                        ):
                            success, message = toggle_server_active(server['name'])
                            if success:
                                st.success(message)
                                st.rerun()
                            else:
                                st.error(message)
                    
                    with col3_mgmt:
                        if st.button(
                            "🗑️", 
                            key=f"delete_{i}",
                            help="Delete server"
                        ):
                            success, message = remove_mcp_server(server['name'])
                            if success:
                                st.success(message)
                                # Reset selection if deleted server was selected
                                if selected_server and selected_server['name'] == server['name']:
                                    st.session_state.selected_server_key = default_option
                                    if "agent" in st.session_state:
                                        del st.session_state.agent
                                    if "current_server" in st.session_state:
                                        del st.session_state.current_server
                                st.rerun()
                            else:
                                st.error(message)
                    
                    st.markdown("---")
        
        # Only show server info and tools if a real server is selected
        if selected_server is not None:
            # Display selected server info
            st.markdown("---")
            st.subheader("🌐 Selected Server Info")
            st.info(f"""
            **Server:** {selected_server['name']}  
            **URL:** {selected_server['url']}  
            **Description:** {selected_server.get('description', 'No description available')}
            """)
            
            st.markdown("---")
            st.subheader("🔧 Available Tools")
        else:
            # Show message when default option is selected
            st.info("👆 Please select an MCP server from the dropdown above to view available tools and start chatting.")
            st.markdown("---")
            st.subheader("🔧 Available Tools")
            st.warning("🔍 No server selected. Please choose an MCP server to see available tools.")
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "agent" not in st.session_state:
        st.session_state.agent = None
    
    if "current_server" not in st.session_state:
        st.session_state.current_server = None
    
    if "processing_prompt" not in st.session_state:
        st.session_state.processing_prompt = None
    
    # Check if server has changed
    server_changed = (st.session_state.current_server != (selected_server['url'] if selected_server else None))
    
    # Initialize agent and fetch tools from selected MCP server (only if a real server is selected)
    if selected_server is not None and (st.session_state.agent is None or server_changed):
        try:
            with st.spinner(f"🚀 Initializing agent and loading tools from {selected_server['name']}..."):
                models = {"openai-gpt-4o-2024-05-13": "2024-02-15-preview"}
                model_name_azure = "openai-gpt-4o-2024-05-13"
                farmAccess = "askbosch-prod-farm-"
                azure_deployment = farmAccess + model_name_azure
                api_version = models[model_name_azure]
                llm = AzureChatOpenAI(
                        api_key="ef6eec30532d48a7a512fb077b837246",
                        azure_endpoint="https://aoai-farm.bosch-temp.com/api",
                        azure_deployment=azure_deployment,
                        api_version=api_version,
                        temperature=0
                    )
                mcp_client = MCPClientWrapper(selected_server)
                # Fetch tools asynchronously
                import nest_asyncio
                nest_asyncio.apply()
                try:
                    loop = asyncio.get_running_loop()
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(
                            asyncio.run,
                            mcp_client.list_tools()
                        )
                        tools_list = future.result(timeout=60)
                except RuntimeError:
                    tools_list = asyncio.run(mcp_client.list_tools())
                
                # Create agent and set available_tools
                agent = HybridMCPAgent(llm, mcp_client)
                agent.available_tools = {tool['name']: tool for tool in tools_list}
                st.session_state.agent = agent
                st.session_state.current_server = selected_server['url']
                
            if tools_list:
                st.success(f"✅ Agent initialized! Loaded {len(tools_list)} tools from {selected_server['name']}")
            else:
                st.warning(f"⚠️ Agent initialized but no tools were loaded from {selected_server['name']}")
                
        except Exception as e:
            tb = traceback.format_exc()
            # If the exception has 'exceptions' attribute (ExceptionGroup), print sub-exceptions
            sub_exceptions = getattr(e, 'exceptions', None)
            if sub_exceptions:
                st.error(f"❌ Error initializing agent or loading tools from {selected_server['name']}: {str(e)}\n\nSub-exceptions: {sub_exceptions}\n\nTraceback:\n{tb}")
            else:
                st.error(f"❌ Error initializing agent or loading tools from {selected_server['name']}: {str(e)}\n\nTraceback:\n{tb}")
    elif selected_server is None:
        # Clear agent if no server is selected
        if "agent" in st.session_state:
            del st.session_state.agent
        if "current_server" in st.session_state:
            del st.session_state.current_server
    
    # Display available tools in sidebar (only if a server is selected and agent is initialized)
    if selected_server is not None and st.session_state.agent and hasattr(st.session_state.agent, 'available_tools'):
        tools = st.session_state.agent.available_tools
        if tools:
            st.sidebar.markdown("### 🛠️ Tools from Selected Server")
            for tool_name, tool_info in tools.items():
                with st.sidebar.expander(f"🔧 {tool_name}"):
                    st.write(f"**Description:** {tool_info.get('description', 'N/A')}")
                    st.write(f"**Server:** {tool_info.get('server_name', 'Unknown')}")
                    if 'parameters' in tool_info and tool_info['parameters']:
                        st.write("**Parameters:**")
                        for param, details in tool_info['parameters'].items():
                            st.write(f"- {param}: {details.get('description', 'N/A')}")
                    else:
                        st.write("*No parameters required*")
        else:
            st.sidebar.warning("🔍 No tools available from the selected server")
    elif selected_server is not None and selected_server_display != "Select MCP server":
        st.sidebar.info("🔄 Please wait while tools are loading...")
    # If no server is selected, tools section is already handled in the sidebar logic above
    
    # Main chat interface
    col1, col2 = st.columns([3, 1])
    
    # Main chat interface with two sections
    col1, col2 = st.columns([3, 1])
    
    with col1:
        #st.subheader("Chat Interface")
        
        # Section 1: Chat messages area (scrollable)
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        # Display chat history
        if st.session_state.messages:
            for i, message in enumerate(st.session_state.messages):
                # Check if this is the latest message for highlighting
                is_latest = (i == len(st.session_state.messages) - 1)
                
                with st.chat_message(message["role"]):
                    # Add latest message styling
                    if is_latest:
                        st.markdown('<div class="latest-message">', unsafe_allow_html=True)
                    
                    # Show call type for assistant messages with badge styling
                    if message["role"] == "assistant" and "call_type" in message:
                        st.markdown(f'<span class="call-type-badge">{message["call_type"]}</span>', unsafe_allow_html=True)
                        
                        # Show tool information if it was a tool call
                        if message.get("tool_calls") and len(message["tool_calls"]) > 0:
                            with st.expander("🔍 Tool Details"):
                                for j, tool_call in enumerate(message["tool_calls"]):
                                    st.write(f"**Tool {j+1}:** {tool_call.get('tool_name', 'Unknown')}")
                                    if tool_call.get('parameters'):
                                        st.json(tool_call['parameters'])
                    
                    st.markdown(message["content"])
                    if "timestamp" in message:
                        st.caption(f"*{message['timestamp']}*")
                    
                    if is_latest:
                        st.markdown('</div>', unsafe_allow_html=True)
        else:
            # Welcome message when no chat history
            st.markdown("""
            <div class="welcome-message">
                <h3>🤖 Welcome to MCP Playground!</h3>
                <p>Start a conversation by typing your message in the chat box below.</p>
                <p>I can help with both general questions and tool-specific tasks.</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)  # Close chat-container
    
    # Section 2: Fixed chat input at bottom (outside of columns to span full width)
    # This creates the fixed chat input similar to edi_app.py
    
    # Check if we need to process a prompt from previous run
    if st.session_state.processing_prompt:
        prompt = st.session_state.processing_prompt
        st.session_state.processing_prompt = None
        
        # Process the input and get response with spinner
        with st.spinner("🔄 Agent Thinking..."):
            try:
                # Better async handling for Streamlit
                import nest_asyncio
                nest_asyncio.apply()
                
                # Check if there's already a running event loop
                try:
                    loop = asyncio.get_running_loop()
                    # If we're in a running loop, create a task
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(
                            asyncio.run,
                            st.session_state.agent.process_input(prompt)
                        )
                        response, call_type, tool_calls = future.result(timeout=60)
                except RuntimeError:
                    # No running loop, safe to use asyncio.run
                    response, call_type, tool_calls = asyncio.run(
                        st.session_state.agent.process_input(prompt)
                    )
                
                response_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Add assistant response to chat history with metadata
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "timestamp": response_timestamp,
                    "call_type": call_type,
                    "tool_calls": tool_calls
                })
                
            except Exception as e:
                error_msg = f"Error processing request: {str(e)}"
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "call_type": "❌ Error",
                    "tool_calls": []
                })
        
        # Rerun to show the response
        st.rerun()
    
    if prompt := st.chat_input("💬 Type your message here..."):
       
        if selected_server is None:
            st.error("❌ Please select an MCP server from the sidebar before starting a conversation.")
            st.stop()
       
        if not st.session_state.agent:
            st.error("Agent not initialized. Please wait for the agent to load or check your server connection.")
            st.stop()
        
        # Add user message to chat history
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
            "timestamp": timestamp
        })
        
        # Set the prompt to be processed in the next run
        st.session_state.processing_prompt = prompt
        
        # Immediately rerun to show the user message
        st.rerun()
    
    with col2:
        pass  # Column reserved for future use
    # Handle example prompt selection
    if hasattr(st.session_state, 'example_prompt'):
        st.rerun()

if __name__ == "__main__":
    main()
