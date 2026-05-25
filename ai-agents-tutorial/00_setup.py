"""
00_setup.py — Setup & Verification

Run this first to make sure everything is configured correctly.
"""

import subprocess
import sys
import os

print("=" * 60)
print("AI AGENTS TUTORIAL — Setup Checker")
print("=" * 60)

# 1. Check Python version
print(f"\n1. Python: {sys.version}")
assert sys.version_info >= (3, 10), "Python 3.10+ required"
print("   ✓ OK")

# 2. Install anthropic if needed
try:
    import anthropic
    print(f"\n2. anthropic SDK: {anthropic.__version__} ✓")
except ImportError:
    print("\n2. Installing anthropic SDK...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "anthropic", "-q"])
    import anthropic
    print(f"   ✓ Installed: {anthropic.__version__}")

# 3. Install pydantic (used in 08_structured_output.py)
try:
    import pydantic
    print(f"3. pydantic:   {pydantic.__version__} ✓")
except ImportError:
    print("3. Installing pydantic...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pydantic", "-q"])
    print("   ✓ Installed")

# 4. Check API key
api_key = os.environ.get("ANTHROPIC_API_KEY", "")
if not api_key:
    print("\n4. ✗ ANTHROPIC_API_KEY not set!")
    print("   Set it with:")
    print("   Windows PowerShell: $env:ANTHROPIC_API_KEY = 'sk-ant-...'")
    print("   Or add to your .env file and load it.")
    sys.exit(1)
else:
    print(f"\n4. API Key: sk-ant-...{api_key[-4:]} ✓")

# 5. Test a real API call
print("\n5. Testing API connection...")
client = anthropic.Anthropic()
response = client.messages.create(
    model="claude-opus-4-7",
    max_tokens=50,
    messages=[{"role": "user", "content": "Say 'Setup successful!' and nothing else."}],
)
print(f"   API response: {response.content[0].text}")
print("   ✓ API connection working")

print("\n" + "=" * 60)
print("ALL CHECKS PASSED — Ready to run the tutorials!")
print("=" * 60)
print("""
Run in this order:
  python 01_simple_agent.py          ← Foundation: basic LLM calls
  python 02_tool_use_manual.py       ← Manual tool loop
  python 03_tool_runner.py           ← Automatic tool runner (beta)
  python 04_react_agent.py           ← ReAct: Reasoning + Acting
  python 05_memory_agent.py          ← In-context + external memory
  python 06_multi_agent.py           ← Multi-agent orchestration
  python 07_streaming_agent.py       ← Streaming responses
  python 08_structured_output.py     ← Structured JSON output + Pydantic
  python 09_web_search_agent.py      ← Live web search (server tool)
  python 10_code_execution_agent.py  ← Server-side code execution
""")
