from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tasks import process_pdf_task
import uuid
import os

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

@app.get("/tasks/{task_id}")
def get_task_status(task_id: str):
    task = process_pdf_task.AsyncResult(task_id)
    return {
        "task_id": task_id,
        "status": task.status,
        "result": task.result if task.ready() else None
    }