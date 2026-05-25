"""
03_tool_runner.py — Automatic Tool Runner (Beta)

The beta tool runner eliminates the manual while-loop: you decorate your
Python functions with @beta_tool, then call client.beta.messages.run().
The SDK handles: calling Claude → detecting tool_use → executing your
function → feeding results back → looping until end_turn.

Much less code than the manual approach, same power.
"""

import math
import anthropic
from anthropic import beta_tool   # the decorator

client = anthropic.Anthropic()

# ---------------------------------------------------------------------------
# Decorate real functions — the decorator registers them AND generates the
# JSON schema from the type hints + docstring automatically.
# ---------------------------------------------------------------------------

@beta_tool
def calculator(operation: str, a: float, b: float) -> float:
    """Perform basic math. operation can be: add, subtract, multiply, divide, power, sqrt."""
    ops = {
        "add":      a + b,
        "subtract": a - b,
        "multiply": a * b,
        "divide":   a / b if b != 0 else float("nan"),
        "power":    a ** b,
        "sqrt":     math.sqrt(a),
    }
    return ops.get(operation, float("nan"))


@beta_tool
def get_stock_price(ticker: str) -> dict:
    """Get the latest stock price for a ticker symbol (fake data for demo)."""
    prices = {
        "AAPL": {"price": 189.50, "change": +1.2,  "currency": "USD"},
        "GOOG": {"price": 175.80, "change": -0.5,  "currency": "USD"},
        "MSFT": {"price": 420.30, "change": +2.1,  "currency": "USD"},
        "NVDA": {"price": 875.00, "change": +15.3, "currency": "USD"},
    }
    return prices.get(ticker.upper(), {"error": f"Ticker '{ticker}' not found"})


@beta_tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> dict:
    """Convert an amount from one currency to another (fake rates for demo)."""
    rates_to_usd = {"USD": 1.0, "EUR": 1.08, "GBP": 1.27, "JPY": 0.0067}
    if from_currency not in rates_to_usd or to_currency not in rates_to_usd:
        return {"error": "Unknown currency"}
    usd_amount = amount * rates_to_usd[from_currency]
    converted = usd_amount / rates_to_usd[to_currency]
    return {
        "original": f"{amount} {from_currency}",
        "converted": round(converted, 2),
        "to_currency": to_currency,
    }


@beta_tool
def list_files(directory: str) -> list[str]:
    """List Python files in a given directory path."""
    import os
    try:
        return [f for f in os.listdir(directory) if f.endswith(".py")]
    except FileNotFoundError:
        return [f"Directory '{directory}' not found"]


# ---------------------------------------------------------------------------
# Run with the automatic tool runner — just pass your decorated functions
# ---------------------------------------------------------------------------

def ask(question: str) -> str:
    print(f"\nUser: {question}")
    print("-" * 50)

    # client.beta.messages.run() handles the entire tool loop internally.
    # Pass the list of @beta_tool-decorated functions directly.
    result = client.beta.messages.run(
        model="claude-opus-4-7",
        max_tokens=4096,
        tools=[calculator, get_stock_price, convert_currency, list_files],
        messages=[{"role": "user", "content": question}],
    )

    # .get_final_message() waits for the loop to finish and returns the
    # final response object (as if you'd done client.messages.create())
    final = result.get_final_message()
    answer = final.content[0].text
    print(f"Claude: {answer}")
    return answer


if __name__ == "__main__":
    print("=" * 60)
    print("AUTOMATIC TOOL RUNNER (Beta)")
    print("=" * 60)

    ask("What is 2 to the power of 10, then multiply that by 3.14?")

    ask("Get me the price of NVDA and AAPL stocks, then tell me "
        "how much more expensive NVDA is in EUR (1 USD = 0.92 EUR).")

    ask(f"List the Python tutorial files in C:/Users/gissp/ai-agents-tutorial/")
