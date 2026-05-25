"""
06_multi_agent.py — Multi-Agent Orchestration

One agent is often not enough. Complex tasks benefit from specialization:
different agents with different system prompts, tools, and expertise.

Patterns covered:
  A. Orchestrator → Workers    (one boss, many specialists)
  B. Pipeline                  (each agent processes & passes to the next)
  C. Parallel Workers          (run agents concurrently, merge results)
  D. Critic / Debate           (agent A proposes, agent B critiques)
"""

import json
import asyncio
import anthropic

client = anthropic.Anthropic()

# ==========================================================================
# A. ORCHESTRATOR → WORKERS
# The orchestrator receives the user request, decides which workers to use,
# calls them, and synthesizes the final answer.
# ==========================================================================

class WorkerAgent:
    def __init__(self, name: str, system_prompt: str, tools: list = None):
        self.name = name
        self.system = system_prompt
        self.tools = tools or []

    def run(self, task: str) -> str:
        kwargs = {
            "model": "claude-opus-4-7",
            "max_tokens": 2048,
            "system": self.system,
            "messages": [{"role": "user", "content": task}],
        }
        if self.tools:
            kwargs["tools"] = self.tools

        response = client.messages.create(**kwargs)
        return response.content[0].text


# Specialized workers
researcher = WorkerAgent(
    name="Researcher",
    system_prompt=(
        "You are a research specialist. When given a topic, provide "
        "comprehensive, factual background information with key data points. "
        "Be thorough but concise. Use bullet points."
    ),
)

analyst = WorkerAgent(
    name="Analyst",
    system_prompt=(
        "You are a data analyst. Given research findings, identify trends, "
        "patterns, implications, and risks. Be analytical and precise."
    ),
)

writer = WorkerAgent(
    name="Writer",
    system_prompt=(
        "You are a technical writer. Given analysis and research, produce "
        "a clear, well-structured, engaging summary for a general audience."
    ),
)


class OrchestratorAgent:
    """Routes tasks to the right workers and synthesizes results."""

    ORCHESTRATION_TOOLS = [
        {
            "name": "delegate_to_researcher",
            "description": "Send a research task to the researcher worker.",
            "input_schema": {
                "type": "object",
                "properties": {"task": {"type": "string"}},
                "required": ["task"],
            },
        },
        {
            "name": "delegate_to_analyst",
            "description": "Send analysis task (with research context) to the analyst worker.",
            "input_schema": {
                "type": "object",
                "properties": {"task": {"type": "string"}},
                "required": ["task"],
            },
        },
        {
            "name": "delegate_to_writer",
            "description": "Send a writing task (with research + analysis) to the writer worker.",
            "input_schema": {
                "type": "object",
                "properties": {"task": {"type": "string"}},
                "required": ["task"],
            },
        },
    ]

    def run(self, user_request: str) -> str:
        messages = [{"role": "user", "content": user_request}]
        system = (
            "You are an orchestrator. Break the user's request into subtasks "
            "and delegate each one to the right specialist tool. "
            "Combine all results into a final answer."
        )

        # Accumulate worker outputs to pass to the next worker
        context: dict[str, str] = {}

        while True:
            response = client.messages.create(
                model="claude-opus-4-7",
                max_tokens=4096,
                system=system,
                tools=self.ORCHESTRATION_TOOLS,
                messages=messages,
            )

            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "end_turn":
                return response.content[0].text

            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue

                task_with_context = block.input["task"]
                if context:
                    ctx_str = "\n\n".join(f"### {k}\n{v}" for k, v in context.items())
                    task_with_context = f"{task_with_context}\n\nCONTEXT:\n{ctx_str}"

                print(f"\n  [Orchestrator → {block.name}]")
                print(f"  Task: {block.input['task'][:80]}...")

                if block.name == "delegate_to_researcher":
                    result = researcher.run(task_with_context)
                    context["Research"] = result
                elif block.name == "delegate_to_analyst":
                    result = analyst.run(task_with_context)
                    context["Analysis"] = result
                elif block.name == "delegate_to_writer":
                    result = writer.run(task_with_context)
                    context["Draft"] = result
                else:
                    result = "Unknown worker"

                print(f"  Result: {result[:100]}...")

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })

            messages.append({"role": "user", "content": tool_results})


# ==========================================================================
# B. PIPELINE — agents in sequence, each transforms the output
# ==========================================================================

def pipeline(input_text: str, agents: list[tuple[str, str]]) -> str:
    """
    Run a list of (name, system_prompt) agents in sequence.
    Each agent receives the previous agent's output.
    """
    current = input_text
    for name, system in agents:
        print(f"\n  [Pipeline stage: {name}]")
        response = client.messages.create(
            model="claude-opus-4-7",
            max_tokens=2048,
            system=system,
            messages=[{"role": "user", "content": current}],
        )
        current = response.content[0].text
        print(f"  Output: {current[:120]}...")
    return current


# ==========================================================================
# C. PARALLEL WORKERS with asyncio
# ==========================================================================

async def run_agent_async(name: str, system: str, task: str) -> tuple[str, str]:
    """Async wrapper so we can run agents concurrently."""
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,
        lambda: client.messages.create(
            model="claude-opus-4-7",
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": task}],
        )
    )
    return name, response.content[0].text


async def parallel_agents(topic: str) -> dict[str, str]:
    """Run multiple specialized agents in parallel on the same topic."""
    tasks = [
        run_agent_async(
            "Optimist",
            "You are an optimist. Focus only on benefits and opportunities.",
            f"Analyze this topic: {topic}",
        ),
        run_agent_async(
            "Pessimist",
            "You are a pessimist. Focus only on risks and downsides.",
            f"Analyze this topic: {topic}",
        ),
        run_agent_async(
            "Realist",
            "You are a realist. Give a balanced, evidence-based perspective.",
            f"Analyze this topic: {topic}",
        ),
    ]
    results = await asyncio.gather(*tasks)
    return dict(results)


# ==========================================================================
# D. CRITIC / DEBATE — agent proposes, critic improves it
# ==========================================================================

def critic_loop(task: str, rounds: int = 2) -> str:
    proposer_system = (
        "You are a creative thinker. Propose innovative solutions. "
        "Be bold and imaginative."
    )
    critic_system = (
        "You are a rigorous critic. Your job is to find flaws, risks, "
        "and improvements in proposals. Be constructive but thorough."
    )
    refiner_system = (
        "You are a refiner. Given a proposal and critique, produce an improved "
        "version that addresses all criticisms."
    )

    # Initial proposal
    response = client.messages.create(
        model="claude-opus-4-7", max_tokens=1024,
        system=proposer_system,
        messages=[{"role": "user", "content": task}],
    )
    proposal = response.content[0].text
    print(f"\n  [Proposer] {proposal[:150]}...")

    for round_num in range(rounds):
        # Critic
        response = client.messages.create(
            model="claude-opus-4-7", max_tokens=1024,
            system=critic_system,
            messages=[{"role": "user", "content": f"Critique this proposal:\n\n{proposal}"}],
        )
        critique = response.content[0].text
        print(f"\n  [Critic — round {round_num + 1}] {critique[:150]}...")

        # Refine
        response = client.messages.create(
            model="claude-opus-4-7", max_tokens=1024,
            system=refiner_system,
            messages=[{"role": "user", "content":
                f"Original proposal:\n{proposal}\n\nCritique:\n{critique}\n\nImprove it:"}],
        )
        proposal = response.content[0].text
        print(f"\n  [Refined] {proposal[:150]}...")

    return proposal


# ==========================================================================
# Main
# ==========================================================================

if __name__ == "__main__":

    # --- A. Orchestrator + Workers ---
    print("=" * 60)
    print("A. ORCHESTRATOR → WORKERS")
    print("=" * 60)

    orchestrator = OrchestratorAgent()
    result = orchestrator.run(
        "Research the impact of large language models on software engineering, "
        "analyze the key trends, and write a 3-paragraph summary."
    )
    print(f"\nFinal Report:\n{result}")

    # --- B. Pipeline ---
    print("\n" + "=" * 60)
    print("B. PIPELINE (sequential transformation)")
    print("=" * 60)

    pipeline_stages = [
        ("Brainstormer", "Generate 5 creative app ideas for the given topic. Be brief."),
        ("Evaluator",    "Rank these ideas by feasibility and market potential. Pick the best one."),
        ("Planner",      "Create a 5-step MVP plan for the winning idea."),
    ]

    final = pipeline("An app for people who want to learn to code", pipeline_stages)
    print(f"\nPipeline result:\n{final}")

    # --- C. Parallel Workers ---
    print("\n" + "=" * 60)
    print("C. PARALLEL WORKERS (async)")
    print("=" * 60)

    perspectives = asyncio.run(parallel_agents("AI replacing software developers"))
    for role, view in perspectives.items():
        print(f"\n{role}: {view[:200]}...")

    # Synthesize the parallel results
    combined = "\n\n".join(f"## {role}\n{view}" for role, view in perspectives.items())
    synthesis = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=1024,
        system="You are a synthesizer. Combine the following perspectives into one balanced view.",
        messages=[{"role": "user", "content": combined}],
    )
    print(f"\nSynthesis: {synthesis.content[0].text}")

    # --- D. Critic / Debate ---
    print("\n" + "=" * 60)
    print("D. CRITIC / DEBATE LOOP")
    print("=" * 60)

    final_proposal = critic_loop(
        "Propose a system for teaching programming to beginners using AI",
        rounds=2,
    )
    print(f"\nFinal refined proposal:\n{final_proposal}")
