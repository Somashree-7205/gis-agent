"""
05_memory_agent.py — Memory Systems for AI Agents

Agents need memory to be useful across time. There are three tiers:

1. In-Context Memory   — the messages[] list (works within one session,
                          lost when the session ends)
2. External Memory     — saved to disk / a database (persists across sessions)
3. Semantic Memory     — embeddings + vector search (find by meaning, not key)
                          [shown conceptually — requires sentence-transformers]

This file implements tier 1 and tier 2 fully, and tier 3 conceptually.
"""

import json
import os
from datetime import datetime
from pathlib import Path
import anthropic

client = anthropic.Anthropic()
MEMORY_FILE = Path("agent_memory.json")

# ==========================================================================
# TIER 1 — IN-CONTEXT MEMORY (conversation history as the "memory")
# ==========================================================================

class InContextMemoryAgent:
    """
    Maintains conversation history in a list.
    The list IS the memory — Claude can reference anything said earlier.
    Downside: limited to the context window (1M tokens for Opus 4.7).
    """

    def __init__(self, system_prompt: str = "You are a helpful assistant."):
        self.system = system_prompt
        self.history: list[dict] = []

    def chat(self, user_message: str) -> str:
        self.history.append({"role": "user", "content": user_message})

        response = client.messages.create(
            model="claude-opus-4-7",
            max_tokens=1024,
            system=self.system,
            messages=self.history,
        )

        assistant_reply = response.content[0].text
        self.history.append({"role": "assistant", "content": assistant_reply})
        return assistant_reply

    def clear(self):
        self.history.clear()

    def summarize_history(self) -> str:
        """Ask Claude to summarize the conversation so far — useful for compaction."""
        if not self.history:
            return "No conversation yet."
        summary_request = self.history + [{
            "role": "user",
            "content": "Please summarize our entire conversation in 3-5 bullet points."
        }]
        response = client.messages.create(
            model="claude-opus-4-7",
            max_tokens=512,
            system=self.system,
            messages=summary_request,
        )
        return response.content[0].text


# ==========================================================================
# TIER 2 — EXTERNAL MEMORY (persist facts to a JSON file)
# ==========================================================================

class ExternalMemoryAgent:
    """
    Stores key facts extracted from conversations to a JSON file.
    Can recall facts from previous sessions — true persistence.
    """

    def __init__(self):
        self.memory: dict = self._load()
        self.session_history: list[dict] = []

    def _load(self) -> dict:
        if MEMORY_FILE.exists():
            return json.loads(MEMORY_FILE.read_text())
        return {"user_facts": {}, "conversation_log": []}

    def _save(self):
        MEMORY_FILE.write_text(json.dumps(self.memory, indent=2))

    def remember(self, key: str, value: str):
        """Explicitly store a fact."""
        self.memory["user_facts"][key] = {
            "value": value,
            "updated": datetime.now().isoformat(),
        }
        self._save()
        print(f"  [Memory saved: {key} = {value}]")

    def recall(self, key: str) -> str | None:
        """Look up a fact by key."""
        entry = self.memory["user_facts"].get(key)
        return entry["value"] if entry else None

    def recall_all(self) -> str:
        """Format all stored facts for injection into a system prompt."""
        facts = self.memory["user_facts"]
        if not facts:
            return "No facts stored yet."
        lines = [f"- {k}: {v['value']}" for k, v in facts.items()]
        return "\n".join(lines)

    def _build_system(self) -> str:
        facts = self.recall_all()
        return (
            "You are a personal assistant with memory about the user.\n\n"
            f"KNOWN FACTS ABOUT THE USER:\n{facts}\n\n"
            "If the user shares a new fact, say 'I'll remember that.' "
            "and it will be saved for you."
        )

    def chat(self, user_message: str) -> str:
        self.session_history.append({"role": "user", "content": user_message})

        response = client.messages.create(
            model="claude-opus-4-7",
            max_tokens=1024,
            system=self._build_system(),
            messages=self.session_history,
        )

        reply = response.content[0].text
        self.session_history.append({"role": "assistant", "content": reply})

        # Log to persistent conversation log
        self.memory["conversation_log"].append({
            "timestamp": datetime.now().isoformat(),
            "user": user_message,
            "assistant": reply,
        })
        self._save()

        return reply


# ==========================================================================
# TIER 3 — SEMANTIC MEMORY (vector search, shown conceptually)
# ==========================================================================

class SemanticMemoryAgent:
    """
    Stores memories as text chunks and retrieves the most relevant ones
    for each query using cosine similarity.

    In production you'd use a vector DB (Chroma, Pinecone, Weaviate, etc.)
    and real embeddings. Here we use a simple TF-IDF-style bag-of-words
    just to illustrate the pattern without requiring extra dependencies.
    """

    def __init__(self):
        self.memories: list[dict] = []  # [{text, timestamp, relevance_score}]

    def store(self, text: str):
        self.memories.append({
            "text": text,
            "timestamp": datetime.now().isoformat(),
            "words": set(text.lower().split()),  # bag-of-words for simple search
        })

    def retrieve(self, query: str, top_k: int = 3) -> list[str]:
        """Return the top_k most relevant memories for this query."""
        if not self.memories:
            return []

        query_words = set(query.lower().split())

        # Score = Jaccard similarity
        scored = [
            (len(query_words & m["words"]) / len(query_words | m["words"]), m["text"])
            for m in self.memories
            if query_words | m["words"]
        ]
        scored.sort(reverse=True)
        return [text for _, text in scored[:top_k]]

    def chat(self, user_message: str) -> str:
        # 1. Retrieve relevant memories
        relevant = self.retrieve(user_message)
        memory_context = "\n".join(f"- {m}" for m in relevant) if relevant else "None"

        system = (
            "You are an assistant with semantic memory.\n\n"
            f"RELEVANT MEMORIES:\n{memory_context}\n\n"
            "Use these memories to inform your response."
        )

        response = client.messages.create(
            model="claude-opus-4-7",
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": user_message}],
        )

        reply = response.content[0].text

        # 2. Store this exchange as a new memory
        self.store(f"User asked: '{user_message}'. I replied: '{reply[:100]}...'")

        return reply


# ==========================================================================
# Demo
# ==========================================================================

if __name__ == "__main__":

    # -----------------------------------------------------------------------
    print("=" * 60)
    print("TIER 1: In-Context Memory Agent")
    print("=" * 60)

    agent1 = InContextMemoryAgent(
        system_prompt="You are a friendly assistant. Remember everything in this chat."
    )

    turns = [
        "Hi! My name is Priya and I'm a machine learning engineer.",
        "I love hiking and I'm planning a trip to Nepal next year.",
        "What's 144 / 12?",
        "Can you remind me what I said my job was and my travel plans?",
    ]

    for msg in turns:
        print(f"\nUser: {msg}")
        print(f"Claude: {agent1.chat(msg)}")

    print(f"\nSummary of conversation:\n{agent1.summarize_history()}")

    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TIER 2: External Memory Agent (persists to disk)")
    print("=" * 60)

    agent2 = ExternalMemoryAgent()

    # Seed some facts (in a real app, Claude would extract these automatically)
    agent2.remember("name", "Priya")
    agent2.remember("role", "machine learning engineer")
    agent2.remember("hobby", "hiking")
    agent2.remember("upcoming_trip", "Nepal")

    print(f"\nStored facts:\n{agent2.recall_all()}")

    print("\nUser: Do you remember anything about me from before?")
    print(f"Claude: {agent2.chat('Do you remember anything about me from before?')}")

    print("\nUser: Recommend a good AI book for someone with my background.")
    print(f"Claude: {agent2.chat('Recommend a good AI book for someone with my background.')}")

    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TIER 3: Semantic Memory Agent (retrieval by meaning)")
    print("=" * 60)

    agent3 = SemanticMemoryAgent()

    # Pre-seed some memories
    agent3.store("User loves Python and machine learning")
    agent3.store("User prefers concise code examples over long explanations")
    agent3.store("User is building an AI chatbot for customer support")
    agent3.store("User's tech stack: FastAPI, PostgreSQL, Redis, Docker")

    query = "What tech stack should I use for my AI project?"
    relevant = agent3.retrieve(query)
    print(f"\nQuery: {query}")
    print(f"Retrieved memories: {relevant}")

    print(f"\nClaude: {agent3.chat(query)}")

    # Cleanup the demo memory file
    if MEMORY_FILE.exists():
        os.remove(MEMORY_FILE)
        print(f"\n[Cleaned up {MEMORY_FILE}]")
