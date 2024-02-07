import streamlit as st
from torchvision import transforms
import torch
import torch.nn as nn
from PIL import Image
import os
from torchvision.models import resnet50
import json
import pydicom
import numpy as np
import io

# Configuración para manejar archivos DICOM con datos de longitud incorrecta
pydicom.config.convert_wrong_length_to_UN = True

def cargar_modelo(modelo_path, num_clases):
    pretrain_model = resnet50(pretrained=True)
    in_features = pretrain_model.fc.in_features
    pretrain_model.fc = nn.Linear(in_features, num_clases)
    modelo_cargado = pretrain_model
    modelo_cargado.load_state_dict(torch.load(modelo_path, map_location=torch.device('cpu')))
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

def convertir_dicom_a_pil(dicom_data):
    if 'PixelData' not in dicom_data:
        raise ValueError("El archivo DICOM no contiene datos de imagen.")
    
    pixel_array = dicom_data.pixel_array
    if pixel_array.dtype != np.uint8:
        pixel_array = ((pixel_array - pixel_array.min()) / pixel_array.ptp()) * 255
    pixel_array = np.uint8(pixel_array)
    if len(pixel_array.shape) == 2:  # Escala de grises
        pixel_array = np.stack((pixel_array,)*3, axis=-1)  # Convertir a RGB
    imagen_pil = Image.fromarray(pixel_array)
    return imagen_pil

st.title("Diagnóstico de Enfermedades")

enfermedades = {
    "Pneumonia": "modelos/pneumonia/",
    "Tumor Cerebral": "modelos/tumor_cerebral/",
}

enfermedad_seleccionada = st.selectbox("Selecciona la enfermedad a diagnosticar:", list(enfermedades.keys()))

ruta_carpeta_enfermedad = enfermedades[enfermedad_seleccionada]
ruta_info_enfermedad = os.path.join(ruta_carpeta_enfermedad, "info.json")

try:
    with open(ruta_info_enfermedad, 'r') as json_file:
        info_enfermedad = json.load(json_file)
except Exception as e:
    st.error(f"Error al leer el archivo {ruta_info_enfermedad}: {str(e)}")
    raise e

ruta_completa_modelo = os.path.join(ruta_carpeta_enfermedad, "modelo_entrenado.pth")
if not os.path.isfile(ruta_completa_modelo):
    st.error(f"No se pudo encontrar el archivo del modelo: {ruta_completa_modelo}.")
    st.stop()

modelo_seleccionado = cargar_modelo(ruta_completa_modelo, info_enfermedad['num_clases'])

uploaded_file_or_folder = st.file_uploader("Elige una imagen o carpeta...", type=["*"], accept_multiple_files=True)

if uploaded_file_or_folder is not None:
    resultados = []
    imagenes_distintas_de_sano_list = []

    for uploaded_item in uploaded_file_or_folder:
        contenido = uploaded_item.read()
        try:
            # Intenta tratar el archivo como DICOM
            dicom_data = pydicom.dcmread(io.BytesIO(contenido), force=True)
            imagen_pil = convertir_dicom_a_pil(dicom_data)
        except Exception as e:
            # Si falla, intenta como imagen regular
            try:
                imagen_pil = Image.open(io.BytesIO(contenido)).convert('RGB')
            except Exception as image_error:
                st.error(f"Error al procesar el archivo {uploaded_item.name}: {image_error}")
                continue

        clase_predicha = predecir_imagen(modelo_seleccionado, imagen_pil)
        nombre_clase_predicha = info_enfermedad['clases'][str(clase_predicha)]
        resultados.append({"imagen": uploaded_item.name, "clase_predicha": nombre_clase_predicha})

        if nombre_clase_predicha != "Sano":
            imagenes_distintas_de_sano_list.append((imagen_pil, nombre_clase_predicha))

    if resultados:
        st.write("Resultados:")
        for resultado in resultados:
            st.write(f"Imagen: {resultado['imagen']}, Clase Predicha: {resultado['clase_predicha']}")

    if imagenes_distintas_de_sano_list:
        st.write("Imágenes distintas a 'Sano':")
        for imagen, clase_predicha in imagenes_distintas_de_sano_list:
            st.image(imagen, caption=f"Clase predicha: {clase_predicha}", use_column_width=True)

