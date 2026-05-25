"""
10_code_execution_agent.py — Agent with Server-Side Code Execution

Claude can run Python code on Anthropic's servers. No sandbox setup
needed — just declare the tool and Claude writes + executes code automatically.

Tool name: "code_execution_20260120"
Requires beta header: "code-execution-2025-05-22"

Use cases:
  - Data analysis (pandas, numpy, matplotlib)
  - Math / statistics
  - File processing
  - Algorithm verification
  - Generating charts (returns base64 image data)
"""

import anthropic
import base64
import json
from pathlib import Path

client = anthropic.Anthropic()

CODE_EXEC_TOOL = {"type": "code_execution_20260120"}
BETA_HEADER = "code-execution-2025-05-22"

# ==========================================================================
# Helper: run a task and collect results including any file outputs
# ==========================================================================

def run_code_agent(task: str, files: list[dict] = None) -> dict:
    """
    Send a task to Claude. It will write and run code, then return results.

    files: optional list of {"name": "data.csv", "content": "...csv data..."}
    """
    print(f"\nTask: {task}")
    print("-" * 50)

    # Build file content blocks if any
    content_blocks = [{"type": "text", "text": task}]
    if files:
        for f in files:
            content_blocks.append({
                "type": "text",
                "text": f"File '{f['name']}':\n```\n{f['content']}\n```"
            })

    response = client.beta.messages.create(
        model="claude-opus-4-7",
        max_tokens=8192,
        tools=[CODE_EXEC_TOOL],
        messages=[{"role": "user", "content": content_blocks}],
        betas=[BETA_HEADER],
    )

    result = {
        "text": "",
        "code_executed": [],
        "output": [],
        "images": [],
    }

    for block in response.content:
        if block.type == "text":
            result["text"] += block.text
            print(f"Claude: {block.text[:200]}...")

        elif block.type == "tool_use" and block.name == "code_execution_20260120":
            code = block.input.get("code", "")
            result["code_executed"].append(code)
            print(f"\n[Code executed ({len(code)} chars)]")
            print("```python")
            print(code[:300] + ("..." if len(code) > 300 else ""))
            print("```")

        elif block.type == "tool_result":
            for item in (block.content if isinstance(block.content, list) else [block.content]):
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        result["output"].append(item["text"])
                        print(f"[Output]: {item['text'][:200]}")
                    elif item.get("type") == "image":
                        result["images"].append(item["source"]["data"])
                        print(f"[Image generated: {len(item['source']['data'])} bytes base64]")

    return result


# ==========================================================================
# 1. Math and Statistics
# ==========================================================================

def math_and_stats():
    result = run_code_agent(
        "Calculate descriptive statistics for these test scores: "
        "78, 92, 85, 67, 95, 88, 73, 91, 82, 79, 88, 94, 71, 86, 90. "
        "Find mean, median, mode, std deviation, variance, min, max. "
        "Also find the percentage of students who scored above 85."
    )
    return result["text"]


# ==========================================================================
# 2. Data Analysis with Pandas
# ==========================================================================

CSV_DATA = """name,department,salary,years,performance_score
Alice,Engineering,95000,5,4.2
Bob,Marketing,72000,3,3.8
Carol,Engineering,105000,8,4.7
Dave,HR,65000,2,3.5
Eve,Engineering,115000,10,4.9
Frank,Marketing,78000,4,4.0
Grace,Engineering,88000,4,4.1
Henry,HR,70000,6,3.9
Iris,Engineering,99000,7,4.4
Jack,Marketing,85000,5,4.3"""

def analyze_data():
    result = run_code_agent(
        "Analyze the employee data provided. Calculate:\n"
        "1. Average salary by department\n"
        "2. Correlation between years_of_experience and salary\n"
        "3. Top 3 highest-paid employees\n"
        "4. Department with highest average performance score\n"
        "5. Salary range (min/max) for Engineering\n"
        "Present results clearly.",
        files=[{"name": "employees.csv", "content": CSV_DATA}]
    )
    return result["text"]


# ==========================================================================
# 3. Algorithm Implementation and Testing
# ==========================================================================

def implement_and_test_algorithm():
    result = run_code_agent(
        "Implement these sorting algorithms from scratch and compare their "
        "performance on a list of 1000 random integers:\n"
        "1. Bubble Sort\n"
        "2. Merge Sort\n"
        "3. Quick Sort\n"
        "4. Python's built-in sorted()\n\n"
        "Measure execution time for each, verify they all produce the same result, "
        "and explain the time complexity of each."
    )
    return result["text"]


# ==========================================================================
# 4. Chart Generation (returns base64 PNG)
# ==========================================================================

def generate_chart():
    result = run_code_agent(
        "Using matplotlib, create a bar chart showing these monthly sales figures:\n"
        "Jan: 45000, Feb: 52000, Mar: 48000, Apr: 61000, May: 58000, Jun: 71000\n"
        "Add a trend line, proper labels, title 'Monthly Sales 2025', "
        "and use a professional color scheme. Save to 'sales_chart.png'."
    )

    # Save any generated images
    if result["images"]:
        for i, img_data in enumerate(result["images"]):
            img_path = Path(f"generated_chart_{i}.png")
            img_path.write_bytes(base64.b64decode(img_data))
            print(f"\nChart saved to: {img_path.absolute()}")

    return result["text"]


# ==========================================================================
# 5. Text Processing / NLP with code
# ==========================================================================

def text_processing():
    text = (
        "Artificial intelligence is transforming industries worldwide. "
        "Machine learning models are being deployed in healthcare, finance, "
        "and education. Natural language processing enables chatbots to "
        "understand human language. Computer vision systems can detect "
        "objects in images with superhuman accuracy. Deep learning algorithms "
        "power recommendation systems used by millions of users daily. "
        "Reinforcement learning agents have mastered complex games and "
        "are being applied to robotics and autonomous vehicles."
    )

    result = run_code_agent(
        f"Analyze this text using Python (no external NLP libraries, just string ops):\n\n"
        f"'{text}'\n\n"
        "1. Word frequency (top 10 words, excluding stopwords)\n"
        "2. Average sentence length in words\n"
        "3. Percentage of sentences with AI-related keywords\n"
        "4. Count of unique words\n"
        "5. Readability score (Flesch-Kincaid grade level formula)"
    )
    return result["text"]


# ==========================================================================
# 6. File upload + code execution (binary files)
# ==========================================================================

def process_json_data():
    import json
    data = {
        "products": [
            {"id": 1, "name": "Laptop", "price": 999.99, "stock": 45, "category": "Electronics"},
            {"id": 2, "name": "Headphones", "price": 199.99, "stock": 120, "category": "Electronics"},
            {"id": 3, "name": "Desk Chair", "price": 349.99, "stock": 30, "category": "Furniture"},
            {"id": 4, "name": "Monitor", "price": 599.99, "stock": 60, "category": "Electronics"},
            {"id": 5, "name": "Keyboard", "price": 149.99, "stock": 200, "category": "Electronics"},
            {"id": 6, "name": "Bookshelf", "price": 199.99, "stock": 25, "category": "Furniture"},
        ]
    }

    result = run_code_agent(
        "Analyze this product inventory data:\n"
        "1. Total inventory value per category\n"
        "2. Product with highest inventory value (price × stock)\n"
        "3. Average price per category\n"
        "4. Low stock alert: products with fewer than 35 units\n"
        "5. Sort products by value descending",
        files=[{"name": "inventory.json", "content": json.dumps(data, indent=2)}]
    )
    return result["text"]


# ==========================================================================
# Main
# ==========================================================================

if __name__ == "__main__":

    print("=" * 60)
    print("1. MATH & STATISTICS")
    print("=" * 60)
    print(math_and_stats())

    print("\n" + "=" * 60)
    print("2. DATA ANALYSIS (CSV)")
    print("=" * 60)
    print(analyze_data())

    print("\n" + "=" * 60)
    print("3. ALGORITHM COMPARISON")
    print("=" * 60)
    print(implement_and_test_algorithm())

    print("\n" + "=" * 60)
    print("4. CHART GENERATION")
    print("=" * 60)
    print(generate_chart())

    print("\n" + "=" * 60)
    print("5. TEXT PROCESSING")
    print("=" * 60)
    print(text_processing())

    print("\n" + "=" * 60)
    print("6. JSON DATA ANALYSIS")
    print("=" * 60)
    print(process_json_data())
