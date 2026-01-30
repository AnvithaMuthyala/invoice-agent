from typing_extensions import TypedDict

from app.parsers.ocr_parser import ParsedInvoice


class WorkflowState(TypedDict, total=False):
    """
    State for the invoice processing workflow.

    Following LangGraph best practices:
    - Store raw data, not formatted text
    - Only include data that needs to persist across steps
    """
    # Input
    image_path: str

    # After parsing
    parsed_invoice: ParsedInvoice

    # After generation
    insights: list[str]

    # Metadata
    parser_used: str
    error: str
