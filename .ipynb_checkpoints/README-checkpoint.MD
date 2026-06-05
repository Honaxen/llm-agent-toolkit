# LLM Agent Toolkit

A modular toolkit for building LLM agents with tools.
Agents that can search, calculate, read documents, and remember context.

---

## Why This Project?

A model predicts. An agent acts.

The difference: tools.
An agent with tools can search the web, read files, run calculations,
and maintain memory across conversations.

This project builds that infrastructure from scratch.

---

## Agent Tools

| Tool | What It Does |
|------|-------------|
| Document Tool | Load and query documents using RAG |
| Calculator Tool | Evaluate mathematical expressions |
| Memory Tool | Store and retrieve conversation context |
| Search Tool | Query a local knowledge base |

---

## Architecture

```
User Query
    |
    v
Agent (reasoning loop)
    |
    v
Tool Selection
    |
    v
Tool Execution
    |
    v
Response Generation (Ollama)
    |
    v
Answer
```

---

## Project Structure

```
llm-agent-toolkit/
├── agent/
│   ├── core.py          — agent reasoning loop
│   ├── tools/
│   │   ├── document.py  — RAG-based document tool
│   │   ├── calculator.py
│   │   ├── memory.py
│   │   └── search.py
├── api/
│   └── main.py          — FastAPI wrapper
├── tests/
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Stack

Python · Ollama · FAISS · sentence-transformers · FastAPI

---

## What I Learned

TBD — will be updated after completion.

---

## Author

[Honaxen](https://github.com/Honaxen)