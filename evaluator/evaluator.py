"""
Invoice Insights Evaluator

Flow:
1. Gemini multimodal reads the invoice image -> extracted text
2. Judges compare extracted text with generated insights

Prompt design follows Arize Phoenix patterns:
- Explanation-first: reasoning before label/score (reduces variance)
- Structured data blocks with [BEGIN DATA] / [END DATA]
- Per-item classification over aggregate numeric scores
- Explicit rubrics with defined levels
- Clear term definitions before evaluation task
"""

import os
import sys
import json
import mimetypes
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# Configure Gemini
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY", ""))
MODEL = "gemini-3-flash-preview"


def extract_invoice_text(image_path: str) -> str:
    """Use Gemini multimodal to read and extract all text/data from invoice image"""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img_data = path.read_bytes()
    mime_type = mimetypes.guess_type(str(path))[0] or "image/png"

    response = client.models.generate_content(
        model=MODEL,
        contents=[
            types.Part.from_bytes(data=img_data, mime_type=mime_type),
            "Extract ALL text and data from this invoice. Include vendor name, dates, amounts, line items, totals - everything visible.",
        ],
    )
    return response.text


def judge_factual_completeness(extracted_text: str, insights: list[str]) -> dict:
    """
    Evaluate insights for completeness and factual accuracy against invoice data.

    Grading: Per-insight classification (factual / hallucinated / partial) +
    per-field coverage check (covered / missing). Explanation-first.
    """
    insights_formatted = "\n".join(f"  Insight {i+1}: {ins}" for i, ins in enumerate(insights))

    prompt = f"""You are an expert invoice auditor. Your task is to evaluate generated insights against the source invoice data for two criteria: COMPLETENESS and FACTUAL ACCURACY.

Definitions:
- "Completeness" means whether the insights, taken together, reference all key data points present in the invoice (vendor name, invoice number, dates, line items, quantities, unit prices, subtotal, taxes, total, payment terms, etc.).
- "Factual accuracy" means whether each insight's claims (numbers, dates, names, calculations) exactly match the source invoice data. A "hallucination" is any claim not supported by or contradicting the invoice data.

[BEGIN DATA]
[Invoice Data]
{extracted_text}

[Generated Insights]
{insights_formatted}
[END DATA]

Evaluate step by step:

1. First, list every key data point present in the invoice data.
2. For each data point, classify whether it is "covered" (referenced in at least one insight) or "missing" (not mentioned in any insight).
3. For each insight, classify it as:
   - "factual" — all claims match the invoice data exactly
   - "hallucinated" — contains information not in the invoice or contradicts it
   - "partial" — some claims are correct but others are wrong or unsupported
   For any non-factual insight, quote the specific problematic claim.

After your analysis, produce your final verdict as JSON:
```json
{{
  "explanation": "<your step-by-step reasoning>",
  "data_points_found": ["<list of all key data points in invoice>"],
  "covered": ["<data points referenced in insights>"],
  "missing": ["<data points NOT referenced in any insight>"],
  "per_insight": [
    {{"insight": 1, "label": "factual|hallucinated|partial", "issue": "<null or description of problem>"}}
  ],
  "completeness_score": <0-100, percentage of data points covered>,
  "accuracy_score": <0-100, percentage of insights that are fully factual>,
  "score": <0-100, average of completeness_score and accuracy_score>
}}
```"""

    response = client.models.generate_content(model=MODEL, contents=prompt)
    return _parse_json(response.text)


def judge_quality(insights: list[str]) -> dict:
    """
    Evaluate insight quality using a rubric-based multi-class classification.

    Grading: Each sub-criterion classified into a defined level (excellent/good/
    fair/poor) with explicit rubric descriptions. Explanation-first.
    """
    insights_formatted = "\n".join(f"  {i+1}. {ins}" for i, ins in enumerate(insights))

    prompt = f"""You are an expert content evaluator. Your task is to assess the quality of generated invoice insights using the rubric below.

[BEGIN DATA]
[Generated Insights]
{insights_formatted}
[END DATA]

Evaluate each criterion using the following rubric:

CLARITY (Is each insight easy to understand?)
- excellent: Every insight is immediately clear with no ambiguity
- good: Most insights are clear, minor ambiguity in one or two
- fair: Several insights require re-reading to understand
- poor: Most insights are confusing or poorly worded

SPECIFICITY (Does each insight reference concrete data from the invoice?)
- excellent: Every insight cites specific numbers, dates, or names from the invoice
- good: Most insights reference specific data, a few are generic
- fair: Insights are mostly generic with occasional specific references
- poor: Insights are vague and could apply to any invoice

DIVERSITY (Are the insights non-repetitive and cover different aspects?)
- excellent: Each insight covers a distinct aspect with no overlap
- good: Minor thematic overlap between one or two insights
- fair: Several insights cover the same aspect or repeat information
- poor: Most insights are redundant or repetitive

ACTIONABILITY (Does each insight provide useful, non-obvious information?)
- excellent: Insights surface patterns, anomalies, or actionable observations
- good: Most insights go beyond restating data to add value
- fair: Insights mostly restate invoice data without adding interpretation
- poor: Insights are trivial restatements that add no value

First, explain your reasoning for each criterion. Then classify each criterion into one of the four levels.

Produce your final verdict as JSON:
```json
{{
  "explanation": "<your reasoning for each criterion>",
  "clarity": {{"label": "excellent|good|fair|poor", "score": "<4=excellent, 3=good, 2=fair, 1=poor>"}},
  "specificity": {{"label": "excellent|good|fair|poor", "score": "<4|3|2|1>"}},
  "diversity": {{"label": "excellent|good|fair|poor", "score": "<4|3|2|1>"}},
  "actionability": {{"label": "excellent|good|fair|poor", "score": "<4|3|2|1>"}},
  "score": "<average of the four scores, rounded to 1 decimal>"
}}
```"""

    response = client.models.generate_content(model=MODEL, contents=prompt)
    return _parse_json(response.text)


def judge_parsing_consistency(extracted_text: str, parser_raw_text: str) -> dict:
    """
    Compare Gemini-extracted text vs OCR raw_text for data agreement.

    Grading: Per-field binary classification (match / mismatch) with values
    from each source quoted. Explanation-first.
    """
    prompt = f"""You are an expert data reconciliation analyst. Your task is to compare two independent extractions of the same invoice and determine whether they agree on all key data points.

Definitions:
- "Match" means both sources report the same value for a data point (minor formatting differences like "$1,000" vs "1000.00" are acceptable matches).
- "Mismatch" means the sources report different values, or one source includes a data point the other is missing entirely.

[BEGIN DATA]
[Source A: Multimodal Vision Extraction]
{extracted_text}

[Source B: OCR Extraction]
{parser_raw_text}
[END DATA]

Evaluate step by step:

1. Identify every key data point present in either source (vendor name, invoice number, dates, line items, amounts, tax, total, payment terms, addresses, etc.).
2. For each data point, compare the values from both sources.
3. Classify each data point as "match" or "mismatch". For mismatches, quote the value from each source.

After your analysis, produce your final verdict as JSON:
```json
{{
  "explanation": "<your step-by-step comparison reasoning>",
  "fields_compared": ["<list of all data points checked>"],
  "matches": [
    {{"field": "<name>", "value": "<agreed value>"}}
  ],
  "mismatches": [
    {{"field": "<name>", "source_a": "<value from vision>", "source_b": "<value from OCR>"}}
  ],
  "score": <0-100, percentage of fields that match>
}}
```"""

    response = client.models.generate_content(model=MODEL, contents=prompt)
    return _parse_json(response.text)


def evaluate(image_path: str, insights: list[str], parser_raw_text: str = "") -> dict:
    """Run full evaluation pipeline"""
    # Step 1: Extract text from image
    try:
        extracted_text = extract_invoice_text(image_path)
    except FileNotFoundError:
        return {"error": f"Image not found: {image_path}"}
    except Exception as e:
        return {"error": f"Failed to extract invoice text: {e}"}

    # Step 2: Run all judges with error handling
    try:
        factual_completeness = judge_factual_completeness(extracted_text, insights)
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        factual_completeness = {"error": f"Judge failed: {e}", "score": 0}
    except Exception as e:
        factual_completeness = {"error": f"Judge failed: {e}", "score": 0}

    try:
        quality = judge_quality(insights)
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        quality = {"error": f"Judge failed: {e}", "score": 0}
    except Exception as e:
        quality = {"error": f"Judge failed: {e}", "score": 0}

    try:
        parsing_consistency = judge_parsing_consistency(extracted_text, parser_raw_text) if parser_raw_text else {"score": 0, "skipped": "No parser raw text provided"}
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        parsing_consistency = {"error": f"Judge failed: {e}", "score": 0}
    except Exception as e:
        parsing_consistency = {"error": f"Judge failed: {e}", "score": 0}

    # Overall score (weighted across 3 judges)
    # Quality score is on 1-4 scale, normalize to 0-100
    quality_normalized = quality.get("score", 0)
    if isinstance(quality_normalized, (int, float)) and quality_normalized <= 4:
        quality_normalized = quality_normalized * 25

    overall = (
        factual_completeness.get("score", 0) * 0.4 +
        quality_normalized * 0.3 +
        parsing_consistency.get("score", 0) * 0.3
    )

    return {
        "extracted_text": extracted_text,
        "factual_completeness": factual_completeness,
        "quality": quality,
        "parsing_consistency": parsing_consistency,
        "overall_score": round(overall, 2),
    }


def _parse_json(text: str) -> dict:
    """Extract JSON from response text, handling markdown code fences"""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        # Remove markdown code fence
        first_newline = cleaned.index("\n")
        cleaned = cleaned[first_newline + 1:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        start, end = text.find("{"), text.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end])
            except (json.JSONDecodeError, ValueError):
                return {"error": "Parse failed", "raw": text}
        return {"error": "Parse failed", "raw": text}


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python evaluator.py <image_path> '<insights_json>'")
        sys.exit(1)

    results = evaluate(sys.argv[1], json.loads(sys.argv[2]))
    print(json.dumps(results, indent=2))
