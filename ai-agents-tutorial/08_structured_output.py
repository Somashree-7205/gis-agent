"""
08_structured_output.py — Structured Outputs

By default Claude returns free-form text. Structured outputs force Claude
to return valid JSON matching a schema you define — perfect for:
  - Extracting data from text (NER, classification, parsing)
  - Building typed, machine-readable responses
  - Eliminating post-processing / regex hacks

Two approaches:
  A. output_config with inline JSON schema  (messages.create)
  B. Pydantic models with messages.parse()  (automatic validation + Python objects)
"""

import json
from typing import Optional
from pydantic import BaseModel, Field
import anthropic

client = anthropic.Anthropic()

# ==========================================================================
# A. Inline JSON Schema via output_config
# ==========================================================================

def extract_entities(text: str) -> dict:
    """Extract named entities using a JSON schema constraint."""

    response = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=1024,
        output_config={
            "format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "EntityExtraction",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "people":        {"type": "array", "items": {"type": "string"}},
                            "organizations": {"type": "array", "items": {"type": "string"}},
                            "locations":     {"type": "array", "items": {"type": "string"}},
                            "dates":         {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["people", "organizations", "locations", "dates"],
                    },
                    "strict": True,
                },
            }
        },
        messages=[{
            "role": "user",
            "content": f"Extract all named entities from this text:\n\n{text}"
        }],
    )

    return json.loads(response.content[0].text)


def classify_sentiment(texts: list[str]) -> list[dict]:
    """Classify sentiment for a batch of texts."""

    batch_text = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))

    response = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=2048,
        output_config={
            "format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "SentimentBatch",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "results": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "index":      {"type": "integer"},
                                        "sentiment":  {"type": "string", "enum": ["positive", "negative", "neutral", "mixed"]},
                                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                                        "reason":     {"type": "string"},
                                    },
                                    "required": ["index", "sentiment", "confidence", "reason"],
                                },
                            }
                        },
                        "required": ["results"],
                    },
                    "strict": True,
                },
            }
        },
        messages=[{
            "role": "user",
            "content": f"Classify the sentiment of each numbered text:\n\n{batch_text}"
        }],
    )

    return json.loads(response.content[0].text)["results"]


# ==========================================================================
# B. Pydantic Models with messages.parse()
# messages.parse() validates the response against your Pydantic model and
# returns a typed Python object — no json.loads() needed.
# ==========================================================================

class ResearchPaper(BaseModel):
    title: str
    authors: list[str]
    year: int
    abstract_summary: str = Field(..., description="2-3 sentence summary")
    key_findings: list[str] = Field(..., description="Top 3-5 findings")
    methodology: str
    field: str
    citations_estimate: Optional[int] = None


class JobPosting(BaseModel):
    job_title: str
    company: str
    location: str
    remote: bool
    salary_min: Optional[int] = None
    salary_max: Optional[int] = None
    required_skills: list[str]
    preferred_skills: list[str]
    experience_years_min: int
    seniority: str = Field(..., description="junior/mid/senior/lead/principal")
    responsibilities: list[str]


class CodeReview(BaseModel):
    class Issue(BaseModel):
        line: Optional[int]
        severity: str = Field(..., description="critical/high/medium/low")
        category: str = Field(..., description="bug/performance/security/style/maintainability")
        description: str
        suggestion: str

    overall_rating: int = Field(..., ge=1, le=10, description="1-10 quality score")
    summary: str
    issues: list[Issue]
    positive_aspects: list[str]
    test_coverage_comment: str


def parse_paper(raw_text: str) -> ResearchPaper:
    result = client.messages.parse(
        model="claude-opus-4-7",
        max_tokens=2048,
        response_format=ResearchPaper,
        messages=[{
            "role": "user",
            "content": f"Parse this research paper description into structured data:\n\n{raw_text}"
        }],
    )
    return result.parsed


def parse_job_posting(raw_text: str) -> JobPosting:
    result = client.messages.parse(
        model="claude-opus-4-7",
        max_tokens=2048,
        response_format=JobPosting,
        messages=[{
            "role": "user",
            "content": f"Parse this job posting into structured data:\n\n{raw_text}"
        }],
    )
    return result.parsed


def review_code(code: str, language: str = "python") -> CodeReview:
    result = client.messages.parse(
        model="claude-opus-4-7",
        max_tokens=4096,
        thinking={"type": "adaptive"},  # Deeper analysis with thinking
        response_format=CodeReview,
        messages=[{
            "role": "user",
            "content": f"Review this {language} code:\n\n```{language}\n{code}\n```"
        }],
    )
    return result.parsed


# ==========================================================================
# Main
# ==========================================================================

if __name__ == "__main__":

    # --- A1. Entity Extraction ---
    print("=" * 60)
    print("A1. Entity Extraction (inline schema)")
    print("=" * 60)

    news = (
        "On March 15 2024, Elon Musk's company SpaceX announced a partnership "
        "with NASA in Houston, Texas. The deal, worth $2.9 billion, was signed "
        "at the Kennedy Space Center in Florida by NASA administrator Bill Nelson."
    )

    entities = extract_entities(news)
    print(json.dumps(entities, indent=2))

    # --- A2. Sentiment Batch ---
    print("\n" + "=" * 60)
    print("A2. Sentiment Classification (batch)")
    print("=" * 60)

    reviews = [
        "This product is absolutely amazing! Best purchase I've ever made.",
        "Terrible quality. Broke after 2 days. Complete waste of money.",
        "It's okay. Does what it says but nothing special.",
        "Love the design but the battery life is disappointing.",
    ]

    results = classify_sentiment(reviews)
    for r in results:
        print(f"[{r['sentiment'].upper():8s} {r['confidence']:.0%}] {reviews[r['index']-1][:60]}")
        print(f"                   Reason: {r['reason']}")

    # --- B1. Research Paper Parsing ---
    print("\n" + "=" * 60)
    print("B1. Research Paper Parsing (Pydantic)")
    print("=" * 60)

    paper_text = """
    'Attention Is All You Need' by Ashish Vaswani, Noam Shazeer, Niki Parmar et al. (2017).
    This landmark paper introduces the Transformer architecture, dispensing with recurrence
    and convolutions entirely in favor of self-attention mechanisms. The authors show that
    Transformers achieve state-of-the-art results on translation tasks (28.4 BLEU on WMT
    2014 English-to-German, 41.0 BLEU on English-to-French), while being significantly
    more parallelizable and requiring less training time. The paper spawned BERT, GPT, and
    nearly every modern language model.
    """

    paper = parse_paper(paper_text)
    print(f"Title: {paper.title}")
    print(f"Authors: {', '.join(paper.authors[:3])} et al.")
    print(f"Year: {paper.year}")
    print(f"Field: {paper.field}")
    print(f"Summary: {paper.abstract_summary}")
    print("Key findings:")
    for finding in paper.key_findings:
        print(f"  • {finding}")

    # --- B2. Code Review ---
    print("\n" + "=" * 60)
    print("B2. Code Review (Pydantic)")
    print("=" * 60)

    bad_code = '''
def process_users(db_conn, user_input):
    query = "SELECT * FROM users WHERE name = '" + user_input + "'"
    result = db_conn.execute(query)
    data = []
    for row in result:
        data.append(row)
    password = "admin123"
    for i in range(len(data)):
        print(data[i])
    return data
'''

    review = review_code(bad_code)
    print(f"Rating: {review.overall_rating}/10")
    print(f"Summary: {review.summary}")
    print(f"\nIssues ({len(review.issues)}):")
    for issue in review.issues:
        line_str = f"L{issue.line}" if issue.line else "N/A"
        print(f"  [{issue.severity.upper():8s}] {line_str:4s} [{issue.category}] {issue.description}")
        print(f"            → {issue.suggestion}")
    print(f"\nPositives:")
    for p in review.positive_aspects:
        print(f"  ✓ {p}")
