"""
04_react_agent.py — ReAct Pattern (Reasoning + Acting)

ReAct = the agent explicitly reasons BEFORE it acts, then observes the
result and reasons again. This makes the agent's decision-making transparent
and easier to debug compared to a black-box tool loop.

Pattern per step:
  Thought:  → what should I do and why?
  Action:   → call a tool
  Observation: → result of the tool
  ... repeat until ...
  Final Answer: → done

We implement this with structured tool output that forces Claude to produce
visible reasoning at each step.
"""

import json
import datetime
import anthropic

client = anthropic.Anthropic()

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def web_search(query: str) -> str:
    """Simulated web search (returns fake but realistic snippets)."""
    results = {
        "python latest version": (
            "Python 3.13 was released in October 2024. Key new features: "
            "improved error messages, experimental free-threaded mode (no GIL), "
            "and a new JIT compiler."
        ),
        "anthropic claude": (
            "Anthropic is an AI safety company. Their latest models are Claude Opus 4.7, "
            "Claude Sonnet 4.6, and Claude Haiku 4.5. Claude Opus 4.7 features "
            "adaptive thinking and a 1M token context window."
        ),
        "default": "No relevant results found for this query.",
    }
    for key in results:
        if key in query.lower():
            return results[key]
    return results["default"]


def calculator(expression: str) -> str:
    """Safely evaluate a math expression like '100 * 0.15 + 200'."""
    try:
        # Only allow safe math — no builtins
        allowed = {k: v for k, v in vars(__import__("math")).items()
                   if not k.startswith("_")}
        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def get_current_date() -> str:
    """Return today's date and time."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def note_pad(action: str, note: str = "", key: str = "default") -> str:
    """
    A simple in-memory notepad.
    action='write': save note under key.
    action='read':  retrieve note by key.
    action='list':  list all keys.
    """
    if action == "write":
        note_pad._store[key] = note
        return f"Saved note under key '{key}'"
    elif action == "read":
        return note_pad._store.get(key, f"No note with key '{key}'")
    elif action == "list":
        return str(list(note_pad._store.keys()))
    return "Unknown action"

note_pad._store = {}

# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

TOOLS = {
    "web_search": web_search,
    "calculator": calculator,
    "get_current_date": get_current_date,
    "note_pad": note_pad,
}

TOOL_DEFINITIONS = [
    {
        "name": "web_search",
        "description": "Search the web for up-to-date information.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "calculator",
        "description": "Evaluate a math expression. E.g. '2 ** 10 + 5 * 3'",
        "input_schema": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
    {
        "name": "get_current_date",
        "description": "Get the current date and time.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "note_pad",
        "description": "Write or read notes. Use to remember intermediate results.",
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["write", "read", "list"]},
                "note":   {"type": "string", "description": "Content to save (for write)"},
                "key":    {"type": "string", "description": "Note identifier"},
            },
            "required": ["action"],
        },
    },
]

# ---------------------------------------------------------------------------
# ReAct system prompt — forces the Thought → Action → Observation structure
# ---------------------------------------------------------------------------

REACT_SYSTEM_PROMPT = """You are a ReAct agent. For every task you must follow this exact loop:

Thought: <reason about what to do next and why>
Action: <call a tool if needed>
Observation: <you will see the tool result here>

... repeat Thought/Action/Observation as many times as needed ...

Final Answer: <your complete, definitive answer to the user's question>

Rules:
- ALWAYS start with a Thought.
- Use tools whenever you need information or computation.
- Write intermediate results to note_pad so you don't lose them.
- Only write "Final Answer:" when you are fully done and certain.
"""

# ---------------------------------------------------------------------------
# The ReAct loop
# ---------------------------------------------------------------------------

def run_react_agent(task: str, max_steps: int = 10) -> str:
    messages = [{"role": "user", "content": task}]
    step = 0

    print(f"\nTask: {task}")
    print("=" * 60)

    while step < max_steps:
        step += 1
        response = client.messages.create(
            model="claude-opus-4-7",
            max_tokens=4096,
            system=REACT_SYSTEM_PROMPT,
            tools=TOOL_DEFINITIONS,
            messages=messages,
            thinking={"type": "adaptive"},
        )

        messages.append({"role": "assistant", "content": response.content})

        # Print visible text (Thought / Final Answer)
        for block in response.content:
            if block.type == "text" and block.text.strip():
                print(f"\n[Step {step}]\n{block.text.strip()}")

        if response.stop_reason == "end_turn":
            # Extract Final Answer
            for block in response.content:
                if block.type == "text" and "Final Answer:" in block.text:
                    return block.text.split("Final Answer:")[-1].strip()
            # Fallback: return all text
            texts = [b.text for b in response.content if b.type == "text"]
            return "\n".join(texts)

        elif response.stop_reason == "tool_use":
            tool_results = []

            for block in response.content:
                if block.type == "tool_use":
                    print(f"\n  → Tool: {block.name}({json.dumps(block.input)})")
                    fn = TOOLS.get(block.name)
                    if fn:
                        result = fn(**block.input)
                    else:
                        result = f"Tool '{block.name}' not found"
                    print(f"  ← Result: {result}")

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(result),
                    })

            messages.append({"role": "user", "content": tool_results})

    return "Max steps reached without a final answer."


# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Example 1: multi-step research + calculation
    result = run_react_agent(
        "What year was Python created? How many years ago was that from today? "
        "And what is that number of years squared?"
    )
    print(f"\n{'=' * 60}\nFINAL ANSWER: {result}\n")

    # Example 2: uses note_pad to track state across steps
    result = run_react_agent(
        "Search for info about Anthropic Claude, save the key facts to a note, "
        "then retrieve and summarize those facts."
    )
    print(f"\n{'=' * 60}\nFINAL ANSWER: {result}\n")
