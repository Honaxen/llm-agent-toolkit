"""
memory.py
---------
Memory tool for the LLM agent.
Stores and retrieves key-value facts across conversation turns.
"""


class MemoryTool:
    """
    Simple key-value memory store for the agent.
    Allows the agent to remember facts across conversation turns.
    """

    name = "memory"
    description = (
        "Store and retrieve information. "
        "To store: 'remember: key = value'. "
        "To retrieve: 'recall: key'. "
        "To list all: 'list memories'."
    )

    def __init__(self):
        self._store = {}

    def run(self, command: str) -> str:
        """
        Execute a memory command.

        Args:
            command: Memory command string

        Returns:
            Result as string
        """
        command = command.strip().lower()

        if command.startswith("remember:"):
            content = command[len("remember:"):].strip()
            if "=" not in content:
                return "Format: 'remember: key = value'"
            key, value = content.split("=", 1)
            self._store[key.strip()] = value.strip()
            return f"Stored: {key.strip()} = {value.strip()}"

        elif command.startswith("recall:"):
            key = command[len("recall:"):].strip()
            if key in self._store:
                return f"{key} = {self._store[key]}"
            return f"No memory found for: {key}"

        elif command == "list memories":
            if not self._store:
                return "Memory is empty."
            return "\n".join(f"{k} = {v}" for k, v in self._store.items())

        elif command.startswith("forget:"):
            key = command[len("forget:"):].strip()
            if key in self._store:
                del self._store[key]
                return f"Forgotten: {key}"
            return f"No memory found for: {key}"

        else:
            return "Unknown command. Use: 'remember: key = value', 'recall: key', 'list memories', 'forget: key'"

    def clear(self) -> None:
        """Clear all stored memories."""
        self._store = {}

    @property
    def size(self) -> int:
        """Number of stored memories."""
        return len(self._store)


if __name__ == "__main__":
    tool = MemoryTool()
    print(tool.run("remember: user name = Hossein"))
    print(tool.run("remember: project = llm-agent-toolkit"))
    print(tool.run("list memories"))
    print(tool.run("recall: user name"))