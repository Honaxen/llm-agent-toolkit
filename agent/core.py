"""
core.py
-------
LLM Agent core — reasoning loop with tool use.

The agent:
1. Receives a user query
2. Decides which tool to use (if any)
3. Executes the tool
4. Generates a final response using Ollama
"""

import requests
from agent.tools.calculator import CalculatorTool
from agent.tools.memory import MemoryTool
from agent.tools.document import DocumentTool
from agent.tools.search import SearchTool


SYSTEM_PROMPT = """You are an AI agent with tools. You MUST use tools when appropriate.

Rules:
- For math questions: use calculator tool
- For storing/remembering information: use memory tool
- For document questions: use document tool
- For searching: use search tool

To use a tool, you MUST respond EXACTLY like this:
TOOL: tool_name: input

Examples:
- User: what is 2+2? -> TOOL: calculator: 2 + 2
- User: remember my name is Alex -> TOOL: memory: remember: name = Alex
- User: what is my name? -> TOOL: memory: recall: name

Only use ANSWER: if no tool is needed."""


class Agent:
    """
    LLM Agent with tools.
    Uses Ollama for reasoning and supports multiple tools.
    """

    def __init__(self, model: str = 'gemma3:12b',
                 ollama_url: str = 'http://127.0.0.1:11434'):
        self.model = model
        self.ollama_url = ollama_url
        self.history = []

        # Register tools
        self.tools = {
            'calculator': CalculatorTool(),
            'memory': MemoryTool(),
            'document': DocumentTool(),
            'search': SearchTool(),
        }

    def _call_model(self, messages: list) -> str:
        """Call Ollama model."""
        response = requests.post(
            f'{self.ollama_url}/api/chat',
            json={
                'model': self.model,
                'messages': messages,
                'stream': False
            },
            proxies={'http': None, 'https': None}
        )
        return response.json()['message']['content']

    def _parse_tool_call(self, response: str):
        """Parse tool call from model response."""
        response = response.strip()
        if response.upper().startswith('TOOL:'):
            parts = response[5:].strip().split(':', 1)
            if len(parts) == 2:
                tool_name = parts[0].strip().lower()
                tool_input = parts[1].strip()
                return tool_name, tool_input
        return None, None

    def run(self, query: str) -> dict:
        """
        Process a user query through the agent loop.

        Args:
            query: User input

        Returns:
            Dict with answer, tool_used, and tool_result
        """
        # Build messages
        messages = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
        ] + self.history + [
            {'role': 'user', 'content': query}
        ]

        # First call — decide if tool needed
        response = self._call_model(messages)

        tool_used = None
        tool_result = None
        final_answer = response

        # Check if tool was called
        tool_name, tool_input = self._parse_tool_call(response)

        if tool_name and tool_name in self.tools:
            tool_used = tool_name
            tool_result = self.tools[tool_name].run(tool_input)

            # Second call — generate final answer with tool result
            messages.append({'role': 'assistant', 'content': response})
            messages.append({
                'role': 'user',
                'content': f'Tool result: {tool_result}\nNow answer the original question.'
            })
            final_answer = self._call_model(messages)

        # Update history
        self.history.append({'role': 'user', 'content': query})
        self.history.append({'role': 'assistant', 'content': final_answer})

        return {
            'query': query,
            'answer': final_answer,
            'tool_used': tool_used,
            'tool_result': tool_result
        }

    def reset(self) -> None:
        """Clear conversation history."""
        self.history = []