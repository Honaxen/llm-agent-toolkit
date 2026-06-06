"""
api/main.py
-----------
FastAPI wrapper for the LLM Agent Toolkit.

Run with:
    uvicorn api.main:app --reload
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.core import Agent

app = FastAPI(
    title="LLM Agent Toolkit",
    description="A modular LLM agent with tools — calculator, memory, document, search.",
    version="1.0.0"
)

agent = Agent()


@app.get("/")
async def root():
    return {
        "name": "LLM Agent Toolkit",
        "version": "1.0.0",
        "tools": list(agent.tools.keys()),
        "docs": "/docs"
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model": agent.model,
        "tools": list(agent.tools.keys()),
        "history_length": len(agent.history)
    }


class QueryRequest(BaseModel):
    query: str


@app.post("/ask")
async def ask(request: QueryRequest):
    """
    Send a query to the agent.
    The agent will decide whether to use a tool or answer directly.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    result = agent.run(request.query)
    return result


@app.post("/reset")
async def reset():
    """Reset conversation history."""
    agent.reset()
    return {"message": "Conversation history cleared."}


@app.get("/tools")
async def list_tools():
    """List available tools and their descriptions."""
    return {
        name: tool.description
        for name, tool in agent.tools.items()
    }