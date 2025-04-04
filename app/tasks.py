from celery import Celery
from utils import (
    analyze_pdf_type,
    extract_text_from_pdf,
    extract_images_from_pdf,
    extract_with_mistral,
    ocr_images_to_text,
    ocr_one_image_to_text,
    format_with_gemini
)
import os
from PIL import Image

celery = Celery(__name__)
celery.conf.broker_url = "redis://redis:6379/0"
celery.conf.result_backend = "redis://redis:6379/1"

@celery.task(bind=True, max_retries=3)
def process_pdf_task(self, pdf_path):
    try:
        # Determine PDF type
        pdf_type = analyze_pdf_type(pdf_path)
        
        # Extract text
        if pdf_type == "text":
            raw_text = extract_text_from_pdf(pdf_path)
            mistral_text = extract_with_mistral(pdf_path)
        else:
            images = extract_images_from_pdf(pdf_path)
            raw_text = ocr_images_to_text(images)
            mistral_text = extract_with_mistral(pdf_path)
        
        # Format with OpenAI
        formatted_md = format_with_gemini(raw_text, mistral_text)
        
        return {
            "document_type": pdf_type,
            "parsed_document": formatted_md
        }
        
    except Exception as e:

        self.retry(exc=e)
    finally:
        print("COMPLETED ")


@celery.task(bind=True, max_retries=3)
def process_image_task(self, image_path):
    try:
        # Validate image format
        with Image.open(image_path) as img:
            img.verify()

        # Extract text using OCR
        raw_text = ocr_one_image_to_text(image_path)

        # Extract text using mistral
        mistral_text = extract_with_mistral(image_path, image = True)


        # Format extracted text
        formatted_md = format_with_gemini(raw_text, mistral_text)

        return {
            "document_type": "image",
            "parsed_document": formatted_md
        }

    except Exception as e:
        self.retry(exc=e)
    finally:
        print("COMPLETED")