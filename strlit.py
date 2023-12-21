import streamlit as st
from torchvision import transforms
import torch
import torch.nn as nn
from PIL import Image
import os
from torchvision.models import resnet50
import json  

# Definir funciones para cargar modelos y realizar predicciones
def cargar_modelo(modelo_path, num_clases):
    # Obtener la ruta completa al modelo
    ruta_completa_modelo = os.path.join("modelos", modelo_path)
    
    # Cargar y configurar el modelo según sea necesario
    pretrain_model = resnet50(pretrained=True)
    in_features = pretrain_model.fc.in_features
    pretrain_model.fc = nn.Linear(in_features, num_clases)
    modelo_cargado = pretrain_model
    modelo_cargado.load_state_dict(torch.load(ruta_completa_modelo, map_location=torch.device('cpu')))
    modelo_cargado.eval()
    return modelo_cargado

def predecir_imagen(modelo, imagen):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    imagen_transformada = transform(imagen).unsqueeze(0)
    outputs = modelo(imagen_transformada)
    _, predicted = torch.max(outputs, 1)
    return predicted.item()

# Configuración de la aplicación Streamlit
st.title("Diagnóstico de Enfermedades")

# Menú de opciones para seleccionar la enfermedad
enfermedades = {
    "Enfermedad A": "modelos/pneumonia/info.json",
    "Enfermedad B": "modelos/tumor_cerebral/info.json",
    # Agregar más enfermedades según sea necesario
}

enfermedad_seleccionada = st.selectbox("Selecciona la enfermedad a diagnosticar:", list(enfermedades.keys()))


# Cargar la información de la enfermedad
ruta_carpeta_enfermedad = enfermedades[enfermedad_seleccionada]
ruta_info_enfermedad = os.path.join("modelos", ruta_carpeta_enfermedad, "info.json")
with open(ruta_info_enfermedad, 'r') as json_file:
    info_enfermedad = json.load(json_file)

# Cargar el modelo seleccionado
    
modelo_seleccionado = cargar_modelo(os.path.join(ruta_carpeta_enfermedad, "modelo_entrenado.pth"), info_enfermedad['num_clases'])

# Subir una imagen
uploaded_file = st.file_uploader("Elige una imagen...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convertir la imagen a formato RGB

    imagen = Image.open(uploaded_file).convert('RGB')

    # Realizar la predicción
    clase_predicha = predecir_imagen(modelo_seleccionado, imagen)

    # Mostrar la imagen y resultados
    st.image(imagen, caption='Imagen cargada', use_column_width=True)
    st.write("Resultado:")
    st.write(f"Clase predicha para {info_enfermedad['nombre']}: {clase_predicha}")
