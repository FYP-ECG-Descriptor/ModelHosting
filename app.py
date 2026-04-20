import os
import modal
from fastapi import FastAPI, UploadFile, File, Form
import uvicorn
from typing import Optional

# Initialize FastAPI
app = FastAPI()

# Connect to your already deployed Modal App
modal_app_name = "pulse-ecg-analyzer"

# =======================================================
# 1. ORIGINAL ENDPOINT 
# =======================================================
@app.post("/v1/analyze-llava")
async def analyze_ecg(file: UploadFile = File(...)):
    try:
        # 1. Read the uploaded image bytes
        image_bytes = await file.read()
        
        # 2. Lookup the deployed Modal Class/Method
        # 🚀 FIX: Updated from .lookup() to .from_name()
        PulseModel = modal.Cls.from_name(modal_app_name, "PulseECGModel")
        model_instance = PulseModel()
        
        print(f"Forwarding {file.filename} to Modal GPU...")
        result = model_instance.analyze.remote(image_bytes)
        
        return {"status": "success", "analysis": result}
    
    except Exception as e:
        return {"status": "error", "message": str(e)}

# =======================================================
# 2. NEW DYNAMIC ENDPOINT 
# =======================================================
@app.post("/v1/analyze-dynamic-llava")
async def analyze_dynamic_llava(
    prompt: str = Form(...), 
    file: Optional[UploadFile] = File(None)
):
    try:
        image_bytes = None
        if file:
            image_bytes = await file.read()
            print("Forwarding Text + Image to Modal GPU...")
        else:
            print("Forwarding Text-Only to Modal GPU...")
        
        PulseModel = modal.Cls.from_name(modal_app_name, "PulseECGModel")
        model_instance = PulseModel()
        
        # Call the new dynamic method on Modal
        result = model_instance.analyze_dynamic.remote(prompt_text=prompt, image_bytes=image_bytes)
        
        return {"status": "success", "analysis": result}
    
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    # Runs on your Mac at port 8007
    uvicorn.run(app, host="0.0.0.0", port=8004)