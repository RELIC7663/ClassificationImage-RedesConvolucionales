from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware  # Importar CORSMiddleware
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
from io import BytesIO

# Cargar el modelo entrenado
model = load_model('model_Mnist_LeNet.h5')

# Crear una instancia de FastAPI
app = FastAPI()

# Agregar CORS a la aplicación FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permitir todos los orígenes, o puedes especificar dominios como ["http://localhost:3000"]
    allow_credentials=False,
    allow_methods=["*"],  # Permitir todos los métodos HTTP
    allow_headers=["*"],  # Permitir todos los encabezados
)

# Ruta para la predicción
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Leer la imagen subida
    image_bytes = await file.read()
    img = Image.open(BytesIO(image_bytes)).convert('L')  # Convertir a escala de grises

    # Redimensionar la imagen para que coincida con el tamaño esperado por el modelo
    img = img.resize((28, 28))

    # Convertir la imagen a un array de numpy
    img_array = np.array(img, dtype=np.float32)

    # Normalizar la imagen
    img_array /= 255.0

    # Cambiar la forma de la imagen para que sea compatible con el modelo
    img_array = img_array.reshape(1, 28, 28, 1)

    # Hacer la predicción
    prediction = model.predict(img_array)

    # Devolver el número predicho
    predicted_number = np.argmax(prediction)

    return {"predicted_number": int(predicted_number)}