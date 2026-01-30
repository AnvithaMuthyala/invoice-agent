
import sys
sys.path.append(".")
from app.parsers.ocr_parser import OCRParser

parser = OCRParser()
result = parser.parse("tests\\sample_invoices\\X51007433809.jpg")
print(result.raw_text)

