from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from tasks import process_pdf_task, process_image_task
import uuid
import os
from PIL import Image
from io import BytesIO

app = FastAPI()

@app.post("/process-pdf")
async def process_pdf(file: UploadFile = File(...)):
    # Save file temporarily
    file_id = str(uuid.uuid4())
    temp_path = f"uploads/{file_id}.pdf"  
    # ensure the path exists
    os.makedirs(os.path.dirname(temp_path), exist_ok=True)
    
    print(f"Saving file to {temp_path}")  

    with open(temp_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # Start processing
    task = process_pdf_task.delay(temp_path)
    return JSONResponse({"task_id": task.id})


IMAGE_MIME_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/gif", "image/webp"}

@app.post("/process-img")
async def process_image(file: UploadFile = File(...)):
    # Validate MIME type
    if file.content_type not in IMAGE_MIME_TYPES:
        raise HTTPException(status_code=400, detail="Invalid file type. Only image files are allowed.")
    
    # Generate unique file name
    file_id = str(uuid.uuid4())
    extension = file.filename.split('.')[-1]
    temp_path = f"uploads/{file_id}.{extension}"

    # Ensure upload directory exists
    os.makedirs(os.path.dirname(temp_path), exist_ok=True)

    # Read and validate image
    content = await file.read()
    try:
        image = Image.open(BytesIO(content))
        image.verify()  # Verify if it's a valid image
    except Exception:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")

    # Save image
    with open(temp_path, "wb") as f:
        f.write(content)

    # Start processing (assuming process_image_task is defined)
    task = process_image_task.delay(temp_path)
    return JSONResponse({"task_id": task.id, "saved_path": temp_path})

@app.get("/tasks/{task_id}")
def get_task_status(task_id: str):
    task = process_pdf_task.AsyncResult(task_id)
    return {
        "task_id": task_id,
        "status": task.status,
        "result": task.result if task.ready() else None
    }