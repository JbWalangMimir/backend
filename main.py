from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from model import EfficientNetModel
import os

app = FastAPI(title="orkidAsIyey Image Classifier")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://orkidasiyey.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
MODEL_PATH = "efficientnet_model1.keras"
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model file not found at {MODEL_PATH}")

model = EfficientNetModel(MODEL_PATH)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await file.read()
    try:
        predictions = model.predict_top5(contents)
        return {"predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "EfficientNet Classifier API is running"}

# ✅ Add this route to test CORS from the frontend
@app.get("/test-cors")
def test_cors():
    return {"message": "CORS is working"}
