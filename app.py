import json
import base64
import tempfile
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from deepface import DeepFace
from fastapi.responses import JSONResponse

app = FastAPI()

# 定義輸入數據模型，使用 Pydantic
class FaceVerificationRequest(BaseModel):
    image_1: str  # Base64 編碼的圖片 1
    image_2: str  # Base64 編碼的圖片 2

@app.get("/")
async def root():
    return {"message": "success"}

@app.post("/face_verification")
async def face_verification(request: FaceVerificationRequest):
    try:
        # 讀取 Base64 編碼的圖片數據
        data_1 = request.image_1
        data_2 = request.image_2

        # 去掉 Base64 前綴
        if data_1.startswith('data:image'):
            data_1 = data_1.split(',')[1]
        if data_2.startswith('data:image'):
            data_2 = data_2.split(',')[1]

        # 解碼 Base64 圖像數據
        img_data_1 = base64.b64decode(data_1)
        img_data_2 = base64.b64decode(data_2)

        # 儲存為臨時文件
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file_1:
            tmp_file_1.write(img_data_1)
            img_1_path = tmp_file_1.name
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file_2:
            tmp_file_2.write(img_data_2)
            img_2_path = tmp_file_2.name
        
        # 使用 DeepFace 進行人臉驗證
        result = DeepFace.verify(img_1_path, img_2_path, model_name="VGG-Face", detector_backend="opencv", 
                                 distance_metric="cosine", enforce_detection=True, align=True, normalization="base")

        # 準備回應數據
        response_data = {
            "verification_distance": result['distance'],
            "verification_threshold": result['threshold'],
            "verification_result": bool(result['verified'])
        }

        # 刪除臨時文件
        os.remove(img_1_path)
        os.remove(img_2_path)

        return JSONResponse(content=response_data)

    except Exception as e:
        # 當發生錯誤時返回 400 錯誤
        raise HTTPException(status_code=400, detail=f"Error: {str(e)}")

