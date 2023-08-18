from fastapi import FastAPI, Request

import pickle
import numpy as np
import cv2
import base64
from app.hog import getHog_descriptors
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

def base642img(str_img):
   en_data = str_img.split(',')[1]
   nparr = np.fromstring(base64.b64decode(en_data), np.uint8)
   img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
   return img
  
@app.get("/")
def root():
        return {"message": "This is my api"}

@app.get("/api/hog")
async def read_str(data : Request):
    json = await data.json()
    item_str = json['img']
    img = base642img(item_str)
    hog = getHog_descriptors(img)
    return {'hog':hog.tolist()}