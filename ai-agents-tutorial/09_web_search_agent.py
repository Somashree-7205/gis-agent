"""
09_web_search_agent.py — Agent with Server-Side Web Search

Anthropic provides a web_search tool that runs entirely on Anthropic's
infrastructure. You declare it in tools[] and Claude searches the web
automatically — no API key for a search provider needed.

Tool name: "web_search_20260209"
Requires beta header: "web-search-2025-03-05"

Claude decides WHEN to search and WHAT to search for — you don't write
the query, it does.
"""

import anthropic

client = anthropic.Anthropic()

# The web search tool is declared server-side — no input_schema needed.
WEB_SEARCH_TOOL = {"type": "web_search_20260209"}

# ==========================================================================
# 1. Simple web search — single question
# ==========================================================================

def search_and_answer(question: str) -> str:
    """Ask Claude a question that requires current web information."""
    print(f"\nQuestion: {question}")
    print("-" * 50)

    response = client.beta.messages.create(
        model="claude-opus-4-7",
        max_tokens=4096,
        tools=[WEB_SEARCH_TOOL],
        messages=[{"role": "user", "content": question}],
        betas=["web-search-2025-03-05"],
    )

    # Web search results come through as tool_result blocks internally.
    # Claude's final text answer is in content blocks of type "text".
    answer = ""
    for block in response.content:
        if hasattr(block, "type") and block.type == "text":
            answer += block.text

    return answer


# ==========================================================================
# 2. Research agent — multi-step web research with synthesis
# ==========================================================================

def research_agent(topic: str) -> dict:
    """
    A research agent that:
    1. Searches for the topic
    2. Searches for recent news about it
    3. Synthesizes a structured report
    """
    print(f"\nResearching: {topic}")
    print("=" * 60)

    messages = [
        {
            "role": "user",
            "content": (
                f"Research the topic '{topic}'. Please:\n"
                "1. Search for background/overview information\n"
                "2. Search for the latest news and developments\n"
                "3. Search for key statistics or data points\n"
                "4. Write a comprehensive 3-section report: Overview, "
                "Recent Developments, Key Facts & Numbers"
            ),
        }
    ]

    # The agent can make multiple web searches in one session
    while True:
        response = client.beta.messages.create(
            model="claude-opus-4-7",
            max_tokens=8192,
            thinking={"type": "adaptive"},
            tools=[WEB_SEARCH_TOOL],
            messages=messages,
            betas=["web-search-2025-03-05"],
        )

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            # Extract text sections
            text_parts = [b.text for b in response.content if b.type == "text"]
            full_text = "\n".join(text_parts)
            return {"topic": topic, "report": full_text, "turns": len(messages)}

        # Web search tool use is handled by Anthropic servers automatically.
        # We still need to handle the tool_use/tool_result loop for the
        # client-side message history.
        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    print(f"  [Web search: {block.input.get('query', '?')}]")
                    # The server executes the search and returns results as tool_result.
                    # For server-side tools, the result content is pre-filled by the API.
                    # We append a placeholder — the actual result is in the assistant message.
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": "",  # Server already sent the result in the stream
                    })
            # If tool_results is empty, break to avoid infinite loop
            if not tool_results:
                text_parts = [b.text for b in response.content if b.type == "text"]
                return {"topic": topic, "report": "\n".join(text_parts)}
            messages.append({"role": "user", "content": tool_results})


# ==========================================================================
# 3. Fact-checking agent
# ==========================================================================

def fact_check(claim: str) -> dict:
    """
    Check whether a claim is true, false, or partially true
    by searching the web for evidence.
    """
    print(f"\nFact-checking: '{claim}'")

    response = client.beta.messages.create(
        model="claude-opus-4-7",
        max_tokens=4096,
        system=(
            "You are a professional fact-checker. When given a claim, "
            "search the web for evidence. Then rate the claim as: "
            "TRUE / FALSE / PARTIALLY TRUE / UNVERIFIABLE. "
            "Always cite your sources."
        ),
        tools=[WEB_SEARCH_TOOL],
        messages=[{
            "role": "user",
            "content": f"Fact-check this claim: '{claim}'"
        }],
        betas=["web-search-2025-03-05"],
    )

    text = ""
    for block in response.content:
        if block.type == "text":
            text += block.text

    return {"claim": claim, "verdict": text}


# ==========================================================================
# 4. Comparison agent — compare two things using web data
# ==========================================================================

def compare(thing_a: str, thing_b: str, aspect: str = "general capabilities") -> str:
    """Compare two things based on current web information."""
    question = (
        f"Compare {thing_a} and {thing_b} in terms of {aspect}. "
        f"Search the web for the most current information about each. "
        f"Present a structured comparison with pros, cons, and a recommendation."
    )

    response = client.beta.messages.create(
        model="claude-opus-4-7",
        max_tokens=4096,
        tools=[WEB_SEARCH_TOOL],
        messages=[{"role": "user", "content": question}],
        betas=["web-search-2025-03-05"],
    )

    return "\n".join(b.text for b in response.content if b.type == "text")


# ==========================================================================
# Main
# ==========================================================================

if __name__ == "__main__":

    # 1. Simple Q&A requiring live web data
    print("=" * 60)
    print("1. SIMPLE WEB SEARCH")
    print("=" * 60)

    answer = search_and_answer("What is the latest version of Python and when was it released?")
    print(f"\nAnswer:\n{answer}")

    # 2. Research report
    print("\n" + "=" * 60)
    print("2. RESEARCH AGENT")
    print("=" * 60)

    report = research_agent("Quantum computing in 2025")
    print(f"\nReport ({len(report['report'])} chars, {report['turns']} turns):")
    print(report["report"][:1000] + "..." if len(report["report"]) > 1000 else report["report"])

    # 3. Fact checking
    print("\n" + "=" * 60)
    print("3. FACT-CHECKING AGENT")
    print("=" * 60)

    fact = fact_check("Python is the most popular programming language in the world")
    print(f"\nVerdict:\n{fact['verdict']}")

    # 4. Comparison
    print("\n" + "=" * 60)
    print("4. COMPARISON AGENT")
    print("=" * 60)

    comparison = compare("PostgreSQL", "MongoDB", "performance for AI applications")
    print(f"\nComparison:\n{comparison[:800]}...")
