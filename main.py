from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pdfplumber
import pytesseract
from PIL import Image
import io
import re
import json
import time
from typing import Optional
import tempfile
import os

app = FastAPI(
    title="DocParser API",
    description="Document Parsing Pipeline for PDFs and Scanned Statements",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def extract_text_from_pdf(file_bytes: bytes) -> dict:
    """Extract text, tables, and metadata from a PDF."""
    result = {
        "pages": [],
        "tables": [],
        "metadata": {},
        "word_count": 0,
        "char_count": 0,
    }

    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        result["metadata"] = {
            "num_pages": len(pdf.pages),
            "info": {k: str(v) for k, v in (pdf.metadata or {}).items()},
        }

        all_text = []
        for i, page in enumerate(pdf.pages):
            page_text = page.extract_text() or ""
            all_text.append(page_text)

            tables = page.extract_tables()
            page_tables = []
            for table in tables:
                if table:
                    page_tables.append(
                        {
                            "headers": table[0] if table else [],
                            "rows": table[1:] if len(table) > 1 else [],
                        }
                    )
                    result["tables"].append(
                        {
                            "page": i + 1,
                            "headers": table[0] if table else [],
                            "rows": table[1:] if len(table) > 1 else [],
                        }
                    )

            result["pages"].append(
                {
                    "page_number": i + 1,
                    "text": page_text,
                    "char_count": len(page_text),
                    "tables": page_tables,
                    "width": page.width,
                    "height": page.height,
                }
            )

        full_text = "\n".join(all_text)
        result["word_count"] = len(full_text.split())
        result["char_count"] = len(full_text)
        result["full_text"] = full_text

    return result


def extract_text_from_image(file_bytes: bytes) -> dict:
    """OCR extraction from scanned image."""
    image = Image.open(io.BytesIO(file_bytes))
    text = pytesseract.image_to_string(image)
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

    words = [w for w in data["text"] if w.strip()]
    confidences = [
        c for c, w in zip(data["conf"], data["text"]) if w.strip() and c != -1
    ]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0

    return {
        "full_text": text,
        "word_count": len(text.split()),
        "char_count": len(text),
        "ocr_confidence": round(avg_confidence, 2),
        "image_size": {"width": image.width, "height": image.height},
        "pages": [
            {
                "page_number": 1,
                "text": text,
                "char_count": len(text),
                "tables": [],
                "width": image.width,
                "height": image.height,
            }
        ],
        "tables": [],
        "metadata": {
            "num_pages": 1,
            "info": {"format": image.format or "Unknown", "mode": image.mode},
        },
    }


def detect_financial_entities(text: str) -> dict:
    """Detect financial amounts, dates, account numbers from text."""
    amounts = re.findall(r"[\$₹€£]?\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})?", text)
    dates = re.findall(
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b",
        text,
        re.IGNORECASE,
    )
    account_numbers = re.findall(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b", text)
    emails = re.findall(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", text)

    return {
        "amounts": list(set(amounts[:20])),
        "dates": list(set(dates[:20])),
        "account_numbers": list(set(account_numbers[:10])),
        "emails": list(set(emails[:10])),
    }


@app.get("/")
def root():
    return {"message": "DocParser API is running", "version": "1.0.0"}


@app.get("/health")
def health():
    return {"status": "healthy", "timestamp": time.time()}


@app.post("/parse")
async def parse_document(file: UploadFile = File(...)):
    """
    Parse an uploaded document (PDF or image).
    Returns extracted text, tables, metadata, and detected entities.
    """
    start_time = time.time()

    allowed_types = [
        "application/pdf",
        "image/png",
        "image/jpeg",
        "image/jpg",
        "image/tiff",
        "image/bmp",
    ]

    content_type = file.content_type or ""
    filename = file.filename or ""

    if not any(
        t in content_type for t in ["pdf", "image"]
    ) and not filename.lower().endswith((".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp")):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Please upload a PDF or image file.",
        )

    file_bytes = await file.read()
    file_size = len(file_bytes)

    if file_size > 50 * 1024 * 1024:  # 50MB limit
        raise HTTPException(status_code=413, detail="File too large. Maximum size is 50MB.")

    try:
        if "pdf" in content_type or filename.lower().endswith(".pdf"):
            doc_type = "pdf"
            extracted = extract_text_from_pdf(file_bytes)
        else:
            doc_type = "image"
            extracted = extract_text_from_image(file_bytes)

        entities = detect_financial_entities(extracted.get("full_text", ""))

        processing_time = round(time.time() - start_time, 3)

        return JSONResponse(
            content={
                "success": True,
                "filename": filename,
                "file_size": file_size,
                "document_type": doc_type,
                "processing_time_seconds": processing_time,
                "extraction": extracted,
                "entities": entities,
                "summary": {
                    "total_pages": extracted["metadata"].get("num_pages", 1),
                    "total_words": extracted["word_count"],
                    "total_chars": extracted["char_count"],
                    "tables_found": len(extracted["tables"]),
                    "entities_found": sum(len(v) for v in entities.values()),
                },
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Parsing failed: {str(e)}")


@app.post("/parse/batch")
async def parse_batch(files: list[UploadFile] = File(...)):
    """Parse multiple documents in batch."""
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 files per batch.")

    results = []
    for file in files:
        file_bytes = await file.read()
        filename = file.filename or ""
        content_type = file.content_type or ""

        try:
            if "pdf" in content_type or filename.lower().endswith(".pdf"):
                extracted = extract_text_from_pdf(file_bytes)
                doc_type = "pdf"
            else:
                extracted = extract_text_from_image(file_bytes)
                doc_type = "image"

            entities = detect_financial_entities(extracted.get("full_text", ""))
            results.append(
                {
                    "filename": filename,
                    "success": True,
                    "document_type": doc_type,
                    "summary": {
                        "total_pages": extracted["metadata"].get("num_pages", 1),
                        "total_words": extracted["word_count"],
                        "tables_found": len(extracted["tables"]),
                        "entities_found": sum(len(v) for v in entities.values()),
                    },
                }
            )
        except Exception as e:
            results.append({"filename": filename, "success": False, "error": str(e)})

    return {"batch_results": results, "total_processed": len(results)}


@app.get("/parse/formats")
def supported_formats():
    """List all supported file formats and their capabilities."""
    return {
        "formats": [
            {"extension": ".pdf", "mime": "application/pdf", "ocr": False, "tables": True, "metadata": True},
            {"extension": ".png", "mime": "image/png", "ocr": True, "tables": False, "metadata": False},
            {"extension": ".jpg", "mime": "image/jpeg", "ocr": True, "tables": False, "metadata": False},
            {"extension": ".tiff", "mime": "image/tiff", "ocr": True, "tables": False, "metadata": False},
            {"extension": ".bmp", "mime": "image/bmp", "ocr": True, "tables": False, "metadata": False},
        ]
    }
