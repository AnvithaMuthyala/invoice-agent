# invoice-agent

## File Structure

```
invoice-agent/
|-- app/
|   |-- config.py                    # API keys and model settings
|   |-- parsers/
|   |   |-- ocr_parser.py            # Docling OCR parser + ParsedInvoice model
|   |-- generation/
|   |   |-- insight_generator.py     # Groq/Gemini insight generation
|   |-- workflow/
|       |-- state.py                 # WorkflowState TypedDict
|       |-- graph.py                 # LangGraph workflow (parse -> generate)
|
|-- evaluator/
|   |-- evaluator.py                 # 3 judges + evaluate() entry point
|
|-- tests/
|   |-- sample_invoices/             # Test invoice images
|   |-- run_all.py                   # Batch runner, saves results to results/
|
|-- main.py                          # CLI entry point (workflow + evaluator)
|-- streamlit_app.py                 # Web UI
|-- requirements.txt
|-- .env                             # GROQ_API_KEY, GOOGLE_API_KEY
```

## Running

```bash
# CLI - single invoice
python main.py tests/sample_invoices/batch1-0419.jpg

# Batch test - all sample invoices, results saved to results/
python tests/run_all.py

# Web UI
streamlit run streamlit_app.py
```
