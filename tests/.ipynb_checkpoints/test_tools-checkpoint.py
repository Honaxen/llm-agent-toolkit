"""
test_tools.py
-------------
Tests for LLM Agent tools.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.tools.calculator import CalculatorTool
from agent.tools.memory import MemoryTool
from agent.tools.search import SearchTool


# --- Calculator Tests ---

def test_calculator_addition():
    tool = CalculatorTool()
    result = tool.run("2 + 2")
    assert "4" in result


def test_calculator_multiplication():
    tool = CalculatorTool()
    result = tool.run("3 * 7")
    assert "21" in result


def test_calculator_power():
    tool = CalculatorTool()
    result = tool.run("2 ** 8")
    assert "256" in result


def test_calculator_complex():
    tool = CalculatorTool()
    result = tool.run("(10 + 5) * 2")
    assert "30" in result


# --- Memory Tests ---

def test_memory_store_and_recall():
    tool = MemoryTool()
    tool.run("remember: name = David")
    result = tool.run("recall: name")
    assert "david" in result.lower()

def test_memory_list():
    tool = MemoryTool()
    tool.run("remember: project = agent")
    tool.run("remember: language = python")
    result = tool.run("list memories")
    assert "project" in result
    assert "language" in result


def test_memory_forget():
    tool = MemoryTool()
    tool.run("remember: temp = value")
    tool.run("forget: temp")
    result = tool.run("recall: temp")
    assert "No memory found" in result


def test_memory_empty_recall():
    tool = MemoryTool()
    result = tool.run("recall: nonexistent")
    assert "No memory found" in result


# --- Search Tests ---

def test_search_add_and_search():
    tool = SearchTool()
    tool.run("add: Python is a programming language")
    result = tool.run("search: programming")
    assert "Python" in result


def test_search_empty_knowledge_base():
    tool = SearchTool()
    result = tool.run("search: anything")
    assert "empty" in result.lower()


def test_search_size():
    tool = SearchTool()
    assert tool.size == 0
    tool.run("add: first document")
    assert tool.size == 1
    tool.run("add: second document")
    assert tool.size == 2