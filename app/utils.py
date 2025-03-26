import os
import logging
from typing import List, Union

import fitz  # PyMuPDF
from pdf2image import convert_from_path
import pytesseract
from openai import OpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]  # In production, consider secure file logging with rotation
)

# Custom exception for PDF processing errors
class PDFProcessingError(Exception):
    pass

def analyze_pdf_type(pdf_path: str, text_threshold: float = 0.7, min_text_length: int = 100) -> str:
    """
    Analyze a PDF and determine if it is predominantly text or image-based.
    
    Args:
        pdf_path (str): The file path to the PDF document.
        text_threshold (float): The fraction of pages that must be text to classify as a text PDF.
        min_text_length (int): Minimum character count to consider a page as text.
        
    Returns:
        str: "text" if predominantly text, "image" otherwise.
    """
    try:
        text_pages = 0
        with fitz.open(pdf_path) as doc:
            total_pages = len(doc)
            for page in doc:
                page_text = page.get_text().strip()
                if len(page_text) > min_text_length:
                    text_pages += 1
        if total_pages == 0:
            raise PDFProcessingError("PDF has no pages.")
        ratio = text_pages / total_pages
        logging.info(f"Text page ratio: {ratio:.2f}")
        return "text" if ratio > text_threshold else "image"
    except Exception as e:
        logging.error("Error analyzing PDF type", exc_info=True)
        raise PDFProcessingError("Failed to analyze PDF type") from e

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text content from all pages of a PDF.
    
    Args:
        pdf_path (str): The file path to the PDF document.
        
    Returns:
        str: The concatenated text from the PDF pages.
    """
    try:
        texts = []
        with fitz.open(pdf_path) as doc:
            for page in doc:
                texts.append(page.get_text())
        return "\n".join(texts)
    except Exception as e:
        logging.error("Error extracting text from PDF", exc_info=True)
        raise PDFProcessingError("Failed to extract text from PDF") from e

def extract_images_from_pdf(pdf_path: str, dpi: int = 300) -> List:
    """
    Convert each page of the PDF into an image.
    
    Args:
        pdf_path (str): The file path to the PDF document.
        dpi (int): Resolution for image conversion.
        
    Returns:
        List: List of images corresponding to PDF pages.
    """
    try:
        images = convert_from_path(pdf_path, dpi=dpi)
        return images
    except Exception as e:
        logging.error("Error extracting images from PDF", exc_info=True)
        raise PDFProcessingError("Failed to extract images from PDF") from e

def ocr_images_to_text(images: List) -> str:
    """
    Use OCR to extract text from a list of images.
    
    Args:
        images (List): List of image objects.
        
    Returns:
        str: The OCR-extracted text.
    """
    try:
        ocr_texts = [pytesseract.image_to_string(img) for img in images]
        return "\n\n".join(ocr_texts)
    except Exception as e:
        logging.error("Error during OCR processing", exc_info=True)
        raise PDFProcessingError("Failed to extract text via OCR") from e

def format_with_openai(text: str) -> str:
    """
    Format text using OpenAI's chat model into structured markdown.
    
    Args:
        text (str): Input text to be formatted.
        
    Returns:
        str: The formatted markdown text.
    """
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Missing OPENAI_API_KEY environment variable")
        
        client = OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert document formatter specializing in financial documents. "
                        "Your task is to convert the provided text into a highly structured Markdown document. "
                        "Please ensure the following: \n"
                        "- Preserve all original tables, headings, subheadings, bullet points, and lists accurately. \n"
                        "- Maintain all numerical data, currency symbols, percentages, dates, and financial details without alteration. \n"
                        "- Correct minor OCR errors or misalignments without changing the intended financial information; annotate ambiguous sections with '[Review]' for manual verification. \n"
                        "- Ensure a clear hierarchical structure (sections, subsections, etc.) is maintained. \n"
                        "- Do not remove or mask sensitive data; focus solely on formatting improvements to enhance clarity and readability. \n"
                        "- Handle all possible formatting inconsistencies robustly to produce a clean, professional Markdown document."
                        "NOTICE: BE CAREFUL ABOUT THE STRUCTURE OF THE DOCUMENT. DO NOT ALTER THE MEANING OF THE TEXT."
                    )
                },
                {
                    "role": "user",
                    "content": text
                }
            ]
        )
        formatted_text = response.choices[0].message.content
        return formatted_text
    except Exception as e:
        logging.error("Error formatting text with OpenAI", exc_info=True)
        raise PDFProcessingError("Failed to format text with OpenAI") from e

# Example main processing function
def process_pdf(pdf_path: str) -> Union[str, None]:
    """
    Process the PDF file by determining its type, extracting text (or OCR from images),
    and formatting it with OpenAI.
    
    Args:
        pdf_path (str): The file path to the PDF document.
    
    Returns:
        str: The final formatted text.
    """
    try:
        pdf_type = analyze_pdf_type(pdf_path)
        logging.info(f"Detected PDF type: {pdf_type}")

        if pdf_type == "text":
            extracted_text = extract_text_from_pdf(pdf_path)
        else:
            images = extract_images_from_pdf(pdf_path)
            extracted_text = ocr_images_to_text(images)
        
        if not extracted_text.strip():
            logging.warning("No text extracted from PDF.")
            return None
        
        formatted_text = format_with_openai(extracted_text)
        return formatted_text
    except PDFProcessingError as e:
        logging.error("PDF processing failed", exc_info=True)
        return None
