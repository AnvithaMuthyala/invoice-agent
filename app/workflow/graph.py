from langgraph.graph import StateGraph, START, END

from app.parsers import OCRParser
from app.generation import InsightGenerator
from .state import WorkflowState


def parse_invoice(state: WorkflowState) -> dict:
    """
    Parse the invoice image using configured parser.

    Returns dict with parsed_invoice and parser_used, or error if failed.
    """
    try:
        parser = OCRParser()
        parsed = parser.parse(state["image_path"])

        return {
            "parsed_invoice": parsed,
            "parser_used": "ocr",
        }
    except Exception as e:
        return {"error": f"Parsing failed: {str(e)}"}


def generate_insights(state: WorkflowState) -> dict:
    """
    Generate insights from parsed invoice.
    LLM decides how many insights to generate.

    Returns dict with insights list, or error if failed.
    """
    if state.get("error"):
        return {}

    parsed_invoice = state.get("parsed_invoice")
    if not parsed_invoice:
        return {"error": "No parsed invoice data available"}

    try:
        generator = InsightGenerator(provider="groq")
        insights = generator.generate(parsed_invoice)

        return {"insights": insights}
    except Exception as e:
        return {"error": f"Generation failed: {str(e)}"}


def create_workflow():
    """
    Create the LangGraph workflow for invoice processing.

    Flow: START -> parse -> generate -> END
    """
    workflow = StateGraph(WorkflowState)

    # Add nodes
    workflow.add_node("parse", parse_invoice)
    workflow.add_node("generate", generate_insights)

    # Define edges
    workflow.add_edge(START, "parse")
    workflow.add_edge("parse", "generate")
    workflow.add_edge("generate", END)

    # Compile and return
    return workflow.compile()
