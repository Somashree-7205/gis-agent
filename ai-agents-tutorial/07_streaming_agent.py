"""
07_streaming_agent.py — Streaming Responses

By default, Claude returns the full response after it finishes generating.
Streaming sends tokens as soon as they're produced — critical for:
  - Long responses (avoid timeouts)
  - Good UX (user sees output immediately, not after a wait)
  - Real-time processing (process text as it arrives)

The SDK supports streaming with client.messages.stream() and a context manager.
"""

import time
import anthropic

client = anthropic.Anthropic()

# ==========================================================================
# 1. Basic Streaming — print tokens as they arrive
# ==========================================================================

def basic_stream(prompt: str):
    print(f"User: {prompt}")
    print("Claude: ", end="", flush=True)

    with client.messages.stream(
        model="claude-opus-4-7",
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for text_chunk in stream.text_stream:
            # Each chunk is a string (one or more tokens)
            print(text_chunk, end="", flush=True)

    print()  # newline after stream ends


# ==========================================================================
# 2. Stream Events — full control over every event type
# ==========================================================================

def stream_with_events(prompt: str):
    """
    Event types you'll encounter:
      message_start       — first event, contains model info
      content_block_start — beginning of a content block (text or tool_use)
      content_block_delta — the actual text/json delta
      content_block_stop  — block finished
      message_delta       — updates stop_reason / usage
      message_stop        — streaming done
    """
    print(f"\nUser: {prompt}")

    input_tokens = 0
    output_tokens = 0
    start = time.time()

    with client.messages.stream(
        model="claude-opus-4-7",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        # Iterate raw events
        for event in stream:
            event_type = event.type

            if event_type == "message_start":
                input_tokens = event.message.usage.input_tokens
                print(f"[Stream started — {input_tokens} input tokens]")

            elif event_type == "content_block_start":
                block = event.content_block
                print(f"[Block {event.index}: {block.type}]")
                if block.type == "text":
                    print("Claude: ", end="", flush=True)

            elif event_type == "content_block_delta":
                delta = event.delta
                if delta.type == "text_delta":
                    print(delta.text, end="", flush=True)

            elif event_type == "content_block_stop":
                print()  # newline after block

            elif event_type == "message_delta":
                output_tokens = event.usage.output_tokens

            elif event_type == "message_stop":
                elapsed = time.time() - start
                print(f"[Stream done — {output_tokens} output tokens, {elapsed:.2f}s]")


# ==========================================================================
# 3. Streaming with Tool Use — stream until tool needed, execute, continue
# ==========================================================================

import json

def stream_with_tools(user_message: str):
    """
    Stream works with tool use too. You get text deltas until Claude decides
    to call a tool, then you get the tool_use block, execute the tool,
    and restart streaming with the result.
    """

    # A simple tool: format current time
    TOOLS = [{
        "name": "get_time_and_timezone",
        "description": "Returns the current time in a given timezone.",
        "input_schema": {
            "type": "object",
            "properties": {
                "timezone": {"type": "string", "description": "e.g. 'UTC', 'US/Eastern'"}
            },
            "required": ["timezone"],
        },
    }]

    def get_time_and_timezone(timezone: str) -> str:
        import datetime, zoneinfo
        try:
            tz = zoneinfo.ZoneInfo(timezone)
            now = datetime.datetime.now(tz)
            return now.strftime(f"%Y-%m-%d %H:%M:%S {timezone}")
        except Exception:
            return f"Unknown timezone: {timezone}"

    messages = [{"role": "user", "content": user_message}]

    print(f"\nUser: {user_message}")

    while True:
        # .get_final_message() waits for the full streamed response, then returns
        # it as a complete Message object — easiest approach for tool-use streaming.
        with client.messages.stream(
            model="claude-opus-4-7",
            max_tokens=2048,
            tools=TOOLS,
            messages=messages,
        ) as stream:
            print("Claude: ", end="", flush=True)
            for chunk in stream.text_stream:
                print(chunk, end="", flush=True)

            final = stream.get_final_message()

        print()
        messages.append({"role": "assistant", "content": final.content})

        if final.stop_reason == "end_turn":
            break

        if final.stop_reason == "tool_use":
            tool_results = []
            for block in final.content:
                if block.type == "tool_use":
                    print(f"  [Tool call: {block.name}({block.input})]")
                    result = get_time_and_timezone(**block.input)
                    print(f"  [Result: {result}]")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })
            messages.append({"role": "user", "content": tool_results})


# ==========================================================================
# 4. Streaming with Adaptive Thinking — see thinking blocks as they arrive
# ==========================================================================

def stream_with_thinking(problem: str):
    """
    Adaptive thinking blocks appear in the stream before the final text.
    Setting display='summarized' makes the thinking visible (default is 'omitted').
    """
    print(f"\nProblem: {problem}")

    with client.messages.stream(
        model="claude-opus-4-7",
        max_tokens=8192,
        thinking={"type": "adaptive", "display": "summarized"},
        messages=[{"role": "user", "content": problem}],
    ) as stream:
        current_block_type = None

        for event in stream:
            if event.type == "content_block_start":
                current_block_type = event.content_block.type
                if current_block_type == "thinking":
                    print("\n[THINKING]", end="", flush=True)
                elif current_block_type == "text":
                    print("\n[ANSWER]", end="", flush=True)

            elif event.type == "content_block_delta":
                delta = event.delta
                if delta.type == "thinking_delta":
                    print(delta.thinking, end="", flush=True)
                elif delta.type == "text_delta":
                    print(delta.text, end="", flush=True)

            elif event.type == "content_block_stop":
                print()


# ==========================================================================
# 5. Collecting a stream into a string (for downstream processing)
# ==========================================================================

def stream_to_string(prompt: str) -> str:
    """Collect the entire streamed response into one string."""
    with client.messages.stream(
        model="claude-opus-4-7",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        return stream.get_final_text()   # blocks until done, returns full text


# ==========================================================================
# Main
# ==========================================================================

if __name__ == "__main__":

    print("=" * 60)
    print("1. BASIC STREAMING")
    print("=" * 60)
    basic_stream("Write a haiku about machine learning, then explain it.")

    print("\n" + "=" * 60)
    print("2. STREAM EVENTS (raw event inspection)")
    print("=" * 60)
    stream_with_events("What are 3 key differences between Python 2 and Python 3?")

    print("\n" + "=" * 60)
    print("3. STREAMING WITH TOOL USE")
    print("=" * 60)
    stream_with_tools("What time is it right now in Tokyo and New York?")

    print("\n" + "=" * 60)
    print("4. STREAMING WITH ADAPTIVE THINKING")
    print("=" * 60)
    stream_with_thinking(
        "A 3-digit number is such that when you reverse its digits you get a "
        "number 198 more than the original. Find all possible numbers."
    )

    print("\n" + "=" * 60)
    print("5. COLLECT STREAM TO STRING")
    print("=" * 60)
    text = stream_to_string("List 5 Python libraries for data science, one word each.")
    print(f"Collected: {text}")
    print(f"Word count: {len(text.split())}")
