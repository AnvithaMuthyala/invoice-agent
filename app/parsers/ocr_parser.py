from pathlib import Path
from pydantic import BaseModel
from docling.document_converter import DocumentConverter, InputFormat, ImageFormatOption


class ParsedInvoice(BaseModel):
    """Structured output from invoice parsing"""
    raw_text: str


class OCRParser:
    """Parser using Docling OCR"""

    def __init__(self):
        """Initialize Docling converter with image support"""
        self.converter = DocumentConverter(
            allowed_formats=[InputFormat.IMAGE],
            format_options={
                InputFormat.IMAGE: ImageFormatOption(),
            },
        )

    def parse(self, image_path: str | Path) -> ParsedInvoice:
        """Parse invoice using Docling OCR"""
        image_path = Path(image_path)

        # Convert document
        result = self.converter.convert(str(image_path))

        # Extract markdown text
        raw_text = result.document.export_to_markdown()

        return ParsedInvoice(raw_text=raw_text)
