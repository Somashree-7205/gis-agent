"""
02_tool_use_manual.py — Manual Tool Use Loop

Tools (function calling) let Claude take actions in the real world:
search the web, run code, read files, call APIs, etc.

Flow:
  1. You define tools (name + description + JSON schema for parameters)
  2. Claude responds with stop_reason="tool_use" and a tool_use block
  3. YOU run the actual function and return the result
  4. Claude uses the result to produce its final answer

This file shows the MANUAL loop — you control every step.
"""

import json
import math
import anthropic

client = anthropic.Anthropic()

# ---------------------------------------------------------------------------
# 1. Define your real Python functions
# ---------------------------------------------------------------------------

def calculator(operation: str, a: float, b: float) -> float:
    ops = {
        "add":      lambda x, y: x + y,
        "subtract": lambda x, y: x - y,
        "multiply": lambda x, y: x * y,
        "divide":   lambda x, y: x / y if y != 0 else "Error: division by zero",
        "power":    lambda x, y: x ** y,
        "sqrt":     lambda x, _: math.sqrt(x),
    }
    fn = ops.get(operation)
    if fn is None:
        return f"Unknown operation: {operation}"
    return fn(a, b)


def get_weather(city: str) -> dict:
    # Fake weather data — replace with a real API call
    fake_data = {
        "London":   {"temp": 15, "condition": "cloudy",  "humidity": 80},
        "New York": {"temp": 22, "condition": "sunny",   "humidity": 55},
        "Tokyo":    {"temp": 28, "condition": "humid",   "humidity": 90},
    }
    return fake_data.get(city, {"temp": 20, "condition": "unknown", "humidity": 60})


def search_database(query: str) -> list[dict]:
    # Fake DB — replace with real DB query
    records = [
        {"id": 1, "name": "Alice", "department": "Engineering", "salary": 95000},
        {"id": 2, "name": "Bob",   "department": "Marketing",   "salary": 72000},
        {"id": 3, "name": "Carol", "department": "Engineering", "salary": 105000},
    ]
    query_lower = query.lower()
    return [r for r in records if query_lower in r["name"].lower()
                                or query_lower in r["department"].lower()]


# ---------------------------------------------------------------------------
# 2. Describe the tools in the format Claude expects
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "calculator",
        "description": "Perform mathematical calculations.",
        "input_schema": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide", "power", "sqrt"],
                    "description": "The math operation to perform.",
                },
                "a": {"type": "number", "description": "First operand."},
                "b": {"type": "number", "description": "Second operand (ignored for sqrt)."},
            },
            "required": ["operation", "a", "b"],
        },
    },
    {
        "name": "get_weather",
        "description": "Get current weather for a city.",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name, e.g. 'London'"},
            },
            "required": ["city"],
        },
    },
    {
        "name": "search_database",
        "description": "Search employee records by name or department.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search term"},
            },
            "required": ["query"],
        },
    },
]

# ---------------------------------------------------------------------------
# 3. The dispatcher: maps tool names → Python functions
# ---------------------------------------------------------------------------

def execute_tool(name: str, inputs: dict):
    if name == "calculator":
        return calculator(**inputs)
    elif name == "get_weather":
        return get_weather(**inputs)
    elif name == "search_database":
        return search_database(**inputs)
    else:
        return f"Tool '{name}' not found"

# ---------------------------------------------------------------------------
# 4. The manual agentic loop
# ---------------------------------------------------------------------------

def run_agent(user_message: str) -> str:
    """Run the tool-use loop until Claude has a final text answer."""
    messages = [{"role": "user", "content": user_message}]

    print(f"\nUser: {user_message}")
    print("-" * 50)

    while True:
        response = client.messages.create(
            model="claude-opus-4-7",
            max_tokens=4096,
            tools=TOOLS,
            messages=messages,
        )

        # Add Claude's reply (could contain tool_use blocks) to history
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            # Claude is done — extract the final text
            for block in response.content:
                if block.type == "text":
                    return block.text

        elif response.stop_reason == "tool_use":
            # Claude wants to call one or more tools
            tool_results = []

            for block in response.content:
                if block.type == "tool_use":
                    print(f"  Tool call: {block.name}({json.dumps(block.input)})")
                    result = execute_tool(block.name, block.input)
                    print(f"  Result:    {result}")

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result),   # must be a string
                    })

            # Send all tool results back to Claude in one message
            messages.append({"role": "user", "content": tool_results})

        else:
            return f"Unexpected stop reason: {response.stop_reason}"


# ---------------------------------------------------------------------------
# 5. Run some examples
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("MANUAL TOOL USE — Agentic Loop")
    print("=" * 60)

    # Single tool call
    answer = run_agent("What is 1337 × 42?")
    print(f"\nFinal answer: {answer}")

    print("\n" + "=" * 60)

    # Multiple tools in one response (parallel tool calls)
    answer = run_agent(
        "What's the weather in London and Tokyo? "
        "Also, which city is warmer and by how many degrees?"
    )
    print(f"\nFinal answer: {answer}")

    print("\n" + "=" * 60)

    # Chained tool calls (Claude decides to use multiple tools in sequence)
    answer = run_agent(
        "Find all Engineering employees in the database and calculate "
        "the combined total of their salaries."
    )
    print(f"\nFinal answer: {answer}")
