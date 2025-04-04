import os, io, logging, json, re
from typing import List, Union

import fitz  # PyMuPDF
from pdf2image import convert_from_path
from PIL import Image
from openai import OpenAI
from mistralai import Mistral

import pytesseract
import google.generativeai as genai

from spire.pdf.common import *
from spire.pdf import *

api_key = "mkH3CjEEYwTKErWuIMg1UYqlYnh88OJW"

client = Mistral(api_key=api_key)



# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    # In production, consider secure file logging with rotation
    handlers=[logging.StreamHandler()]
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


def ocr_one_image_to_text(image) -> str:
    """
    Use OCR to extract text from a list of images.

    Args:
        images (List): List of image objects.

    Returns:
        str: The OCR-extracted text.
    """
    try:
        ocr_texts = pytesseract.image_to_string(image)
        logging.info(ocr_texts)
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
                        "You are an expert document formatter specializing in financial documents. The keys should be: Date of birth, First Name, Middle Name, Surname, Date of Expiry, Sex, Country of Citizenship, FCN/National ID. NOTE: FOR THE Date of birth and Date of Expiry IGNORE DATES WITH AMHARIC OR UNKNOWN KEYWORDS and DONT add additional fields other than the ones specified"
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


def format_with_gemini(text: str, mistral_text: str) -> str:
    """
    Format text using OpenAI's chat model into structured markdown.

    Args:
        text (str): Input text to be formatted.
        mistral_text (str): text found from mistral ai

    Returns:
        str: The formatted markdown text.
    """
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Missing OPENAI_API_KEY environment variable")

        genai.configure(api_key=api_key)

        # client = genai.Client(api_key=gemini_api_key)

        prompt = [
            "You are an expert document formatter specializing in financial documents. "
            "You also will recieve identification documents and extract the key value pairs to a json document accordingly",
            "Your task is to convert the provided text into a highly structured JSON Format. ",
            "You will be provided with two ocr results, one from tesseract ocr and another from mistral ocr. Always take the best response from the two and try your best to understand the document using both of the ocr results."
            "Please ensure the following: \n"
            "- Preserve all original tables, headings, subheadings, bullet points, and lists accurately. \n"
            "- Maintain all numerical data, currency symbols, percentages, dates, and financial details without alteration. \n"
            "- Correct minor OCR errors or misalignments without changing the intended financial information; annotate ambiguous sections with '[Review]' for manual verification. \n"
            "- Ensure a clear hierarchical structure (sections, subsections, etc.) is maintained. \n"
            "- Do not remove or mask sensitive data; focus solely on formatting improvements to enhance clarity and readability. \n"
            "- Handle all possible formatting inconsistencies robustly to produce a clean, professional json format."
            "- For the keys of the json unless a specific schema is provided try to pick a standard way of representing the variables in their specific domain, so use 'Total Assets' instead of 'total_assets'"
            "NOTICE: BE CAREFUL ABOUT THE STRUCTURE OF THE DOCUMENT. DO NOT ALTER THE MEANING OF THE TEXT.",
            "Respond ONLY with a valid JSON that can be parsed using Python's json.loads()",
            f"tesseract ocr: {text}",
            f"mistral ocr: {mistral_text} "
        ]
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(
            '\n'.join(prompt)
        )
        
        response_text = response.text

        response_json = repair_json_response(response_text)

        return response_json
    except Exception as e:
        logging.error("Error formatting text with Gemini", exc_info=True)
        raise PDFProcessingError("Failed to format text with Gemini") from e

# Example main processing function

def repair_json_response(response_text):
    """
    Ultra-robust JSON repair function with specific handling for string interpolation
    """
    # Remove Markdown code block markers if present
    response_text = re.sub(r'^```json\s*|\s*```$', '',
                           response_text, flags=re.MULTILINE).strip()

    # Try to find the JSON part (between first { and last })
    json_start = response_text.find('{')
    json_end = response_text.rfind('}')

    if json_start == -1 or json_end == -1:
        return None

    json_str = response_text[json_start:json_end+1]

    # First try to parse as-is
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    # If that fails, try more aggressive repair
    try:
        # Extract generated_code content with more flexible matching
        code_match = re.search(
            r'"generated_code"\s*:\s*"(.*?)"(?=\s*[},])', json_str, re.DOTALL)
        if not code_match:
            return None

        # Get the code content
        code_content = code_match.group(1)

        # First escape all backslashes
        code_content = code_content.replace('\\', '\\\\')

        # Then escape all quotes that aren't part of string interpolation
        # This is the key change - we only escape quotes not preceded by $
        code_content = re.sub(r'(?<!\$)(?<!\\)"', r'\"', code_content)

        # Reconstruct the JSON with properly escaped content
        fixed_json = json_str[:code_match.start(
            1)] + code_content + json_str[code_match.end(1):]

        # Try parsing again
        return json.loads(fixed_json)
    except:
        return None

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

        formatted_text = format_with_gemini(extracted_text)
        return formatted_text
    except PDFProcessingError as e:
        logging.error("PDF processing failed", exc_info=True)
        return None

def image_to_pdf(image_path: str, save_path: str):
    doc = PdfDocument()
    
    # remote the page marigns
    doc.PageSettings.SetMargins(0.0)

    # load the image
    image = PdfImage.FromFile(image_path)

    # Get the image width and height
    width = image.PhysicalDimension.Width
    height = image.PhysicalDimension.Height

    # Add a page with the same width and height to the PDF
    page = doc.Pages.Add(SizeF(width, height))   

    # Draw the image on the newly added page
    page.Canvas.DrawImage(image, 0.0, 0.0, width, height)

    doc.SaveToFile(save_path)
    doc.Close()

def extract_with_mistral(pdf_path: str):
    # upload the image to mistral
    uploaded_pdf = client.files.upload(
        file={
            "file_name": "test.pdf",
            "content": open(pdf_path, "rb"),
        },
        purpose="ocr"
    )
    
    # logging.info(f'Coverted image to pdf path: {resulting_pdf}')
        
    signed_url = client.files.get_signed_url(file_id=uploaded_pdf.id)

    # get the result
    ocr_response = client.ocr.process(
        model="mistral-ocr-latest",
        document={
            "type": "document_url",
            "document_url": signed_url.url,
        }
    )

    result = json.loads(ocr_response.model_dump_json())
    logging.info('Before formating: ', result)
    final = ''
    for page in result['pages']:
        final += page["markdown"]

    logging.info("mistral result: " + final, )

    return final
