"""
01_simple_agent.py — Foundation: Basic LLM Call

The simplest possible interaction with Claude. No tools, no memory, just
a single request/response. This is the building block for everything else.
"""

import anthropic

client = anthropic.Anthropic()  # Reads ANTHROPIC_API_KEY from environment

# --- 1. Single question ---
print("=" * 60)
print("EXAMPLE 1: Single Question")
print("=" * 60)

response = client.messages.create(
    model="claude-opus-4-7",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "What is the capital of France?"}
    ]
)

# response.content is a list of content blocks (text, tool_use, thinking, etc.)
for block in response.content:
    if block.type == "text":
        print(block.text)

print(f"\nStop reason: {response.stop_reason}")   # 'end_turn' = finished normally
print(f"Input tokens:  {response.usage.input_tokens}")
print(f"Output tokens: {response.usage.output_tokens}")


# --- 2. System prompt: give Claude a persona / role ---
print("\n" + "=" * 60)
print("EXAMPLE 2: System Prompt (Persona)")
print("=" * 60)

response = client.messages.create(
    model="claude-opus-4-7",
    max_tokens=512,
    system="You are a pirate. Answer every question in pirate speak, arrr!",
    messages=[
        {"role": "user", "content": "What is 2 + 2?"}
    ]
)

print(response.content[0].text)


# --- 3. Multi-turn conversation (manual history management) ---
print("\n" + "=" * 60)
print("EXAMPLE 3: Multi-Turn Conversation")
print("=" * 60)

# You must manually maintain the conversation history list.
# Every assistant reply gets appended so Claude has context.
conversation = []

def chat(user_message: str) -> str:
    conversation.append({"role": "user", "content": user_message})

    response = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=1024,
        system="You are a helpful math tutor.",
        messages=conversation,
    )

    assistant_text = response.content[0].text
    # Append assistant reply so the next turn includes it
    conversation.append({"role": "assistant", "content": assistant_text})
    return assistant_text


print("User: My name is Alex.")
print("Claude:", chat("My name is Alex."))

print("\nUser: What is 15 × 7?")
print("Claude:", chat("What is 15 × 7?"))

print("\nUser: And do you remember my name?")
print("Claude:", chat("And do you remember my name?"))


# --- 4. Adaptive thinking for harder problems ---
print("\n" + "=" * 60)
print("EXAMPLE 4: Adaptive Thinking (for complex reasoning)")
print("=" * 60)

response = client.messages.create(
    model="claude-opus-4-7",
    max_tokens=4096,
    thinking={"type": "adaptive"},          # Claude decides when/how much to think
    messages=[
        {"role": "user", "content": (
            "A snail is at the bottom of a 20-foot well. "
            "Each day it climbs 3 feet, but each night it slides back 2 feet. "
            "How many days does it take to reach the top?"
        )}
    ]
)

for block in response.content:
    if block.type == "thinking":
        print(f"[Thinking — {len(block.thinking)} chars]")  # internal reasoning
    elif block.type == "text":
        print(block.text)


# --- 5. Effort control ---
print("\n" + "=" * 60)
print("EXAMPLE 5: Effort Control")
print("=" * 60)

# effort controls thinking depth AND token spend
# "low" = fast/cheap, "medium", "high" (default), "max" = deepest
response = client.messages.create(
    model="claude-opus-4-7",
    max_tokens=2048,
    output_config={"effort": "high"},
    messages=[
        {"role": "user", "content": "Explain quantum entanglement in simple terms."}
    ]
)

print(response.content[0].text)
