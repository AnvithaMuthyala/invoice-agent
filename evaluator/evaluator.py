"""
Invoice Insights Evaluator

Flow:
1. Gemini multimodal reads the invoice image -> extracted text
2. Judges compare extracted text with generated insights
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
    """Check completeness and factual accuracy of insights against extracted invoice data"""
    prompt = f"""INVOICE DATA:
{extracted_text}

GENERATED INSIGHTS:
{chr(10).join(f'{i+1}. {ins}' for i, ins in enumerate(insights))}

Task: Evaluate the insights on TWO criteria:
1. COMPLETENESS: What percentage of key invoice data points (vendor, dates, amounts, line items, totals, etc.) are covered in the insights?
2. FACTUAL ACCURACY: Are all numbers, dates, names, and claims in the insights correct? Flag any hallucinations or errors.

Return JSON only:
{{"score": <0-100 combined score>, "covered": [<list of covered data points>], "missing": [<list of missing data points>], "issues": [<list of factual errors or hallucinations, empty if none>]}}"""

    response = client.models.generate_content(model=MODEL, contents=prompt)
    return _parse_json(response.text)


def judge_quality(insights: list[str]) -> dict:
    """Score insight quality: clarity, actionability, specificity, uniqueness"""
    prompt = f"""INSIGHTS:
{chr(10).join(f'{i+1}. {ins}' for i, ins in enumerate(insights))}

Rate each 1-10:
- Clarity: Easy to understand?
- Actionability: Provides useful info?
- Specificity: References real data?
- Uniqueness: Non-repetitive?

Return JSON only:
{{"score": <overall 1-10>, "clarity": <1-10>, "actionability": <1-10>, "specificity": <1-10>, "uniqueness": <1-10>, "reasoning": "<brief>"}}"""

    response = client.models.generate_content(model=MODEL, contents=prompt)
    return _parse_json(response.text)


def judge_parsing_consistency(extracted_text: str, parser_raw_text: str) -> dict:
    """Compare Gemini-extracted text vs Docling OCR raw_text for consistency"""
    prompt = f"""SOURCE A (Gemini extraction):
{extracted_text}

SOURCE B (OCR extraction):
{parser_raw_text}

Task: Compare these two extractions of the same invoice. Check if they agree on key data points: vendor name, amounts, dates, line items, totals.

Return JSON only:
{{"score": <0-100 consistency>, "matches": [<list of data points that agree>], "mismatches": [<list of data points that differ, with values from each source>]}}"""

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
    overall = (
        factual_completeness.get("score", 0) * 0.4 +
        quality.get("score", 0) * 10 * 0.3 +
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
    """Extract JSON from response"""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        start, end = text.find("{"), text.rfind("}") + 1
        if start != -1 and end > start:
            return json.loads(text[start:end])
        return {"error": "Parse failed", "raw": text}


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python evaluator.py <image_path> '<insights_json>'")
        sys.exit(1)

    results = evaluate(sys.argv[1], json.loads(sys.argv[2]))
    print(json.dumps(results, indent=2))
