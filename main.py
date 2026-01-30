"""
Invoice Insights Agent - Main Entry Point

Flow: Image → LangGraph Workflow → Evaluator → Final Output
"""

import sys
from pathlib import Path

from app.workflow import create_workflow
from evaluator.evaluator import evaluate


def run(image_path: str) -> dict:
    """
    Run the complete pipeline:
    1. LangGraph workflow parses image and generates insights
    2. Evaluator scores the generated insights
    3. Returns combined output
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Step 1: Run LangGraph workflow
    print("Running workflow...")
    workflow = create_workflow()
    workflow_result = workflow.invoke({"image_path": str(path)})

    if workflow_result.get("error"):
        return {"error": workflow_result["error"]}

    insights = workflow_result.get("insights", [])
    parsed_invoice = workflow_result.get("parsed_invoice")
    raw_text = parsed_invoice.raw_text if parsed_invoice else ""
    print(f"Generated {len(insights)} insights")

    # Step 2: Run evaluator
    print("Running evaluator...")
    eval_result = evaluate(str(path), insights, parser_raw_text=raw_text)

    # Step 3: Combine outputs
    return {
        "workflow": {
            "parser_used": workflow_result.get("parser_used"),
            "insights": insights,
        },
        "evaluation": eval_result,
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <image_path>")
        sys.exit(1)

    result = run(sys.argv[1])

    # Print results
    print("\n" + "=" * 60)
    print("INSIGHTS")
    print("=" * 60)
    for i, insight in enumerate(result.get("workflow", {}).get("insights", []), 1):
        print(f"{i}. {insight}")

    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)
    eval_data = result.get("evaluation", {})
    fc = eval_data.get("factual_completeness", {})
    q = eval_data.get("quality", {})
    pc = eval_data.get("parsing_consistency", {})
    print(f"Factual Completeness: {fc.get('score', 'N/A')}% (accuracy: {fc.get('accuracy_score', 'N/A')}%, coverage: {fc.get('completeness_score', 'N/A')}%)")
    print(f"Quality: {q.get('score', 'N/A')}/4 (clarity: {q.get('clarity', {}).get('label', 'N/A')}, specificity: {q.get('specificity', {}).get('label', 'N/A')}, diversity: {q.get('diversity', {}).get('label', 'N/A')}, actionability: {q.get('actionability', {}).get('label', 'N/A')})")
    print(f"Parsing Consistency: {pc.get('score', 'N/A')}%")
    print(f"Overall: {eval_data.get('overall_score', 'N/A')}/100")


if __name__ == "__main__":
    main()
