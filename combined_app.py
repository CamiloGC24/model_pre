import streamlit as st
import os
import shutil
from zipfile import ZipFile
import tempfile
from torchvision import transforms
import torch
import torch.nn as nn
from PIL import Image
import pydicom
import numpy as np
import io
import json
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
from torchvision.models import resnet50
from streamlit_lottie import st_lottie
import requests

# Configuración para manejar archivos DICOM con datos de longitud incorrecta
pydicom.config.convert_wrong_length_to_UN = True

# Crear carpetas para guardar las imágenes procesadas
os.makedirs("output", exist_ok=True)

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Función de la primera aplicación
def add_dcm_extension_and_zip(input_dir, output_zip):
    with ZipFile(output_zip, 'w') as zipf:
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                new_file_name = file + ".dcm"
                new_file_path = os.path.join(root, new_file_name)
                zipf.write(os.path.join(root, file), arcname=new_file_path)

def dcm_app():
    st.header("Añadir extensión .dcm a archivos")
    st.markdown("### Sube una carpeta en formato zip que contenga tus archivos para añadirles la extensión `.dcm`:")

    uploaded_files = st.file_uploader("Subir carpeta en formato zip", accept_multiple_files=False, type='zip')

    if uploaded_files is not None:
        with tempfile.TemporaryDirectory() as temp_dir:
            zip_path = os.path.join(temp_dir, "uploaded.zip")
            with open(zip_path, "wb") as f:
                f.write(uploaded_files.getvalue())
            with ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            output_zip_path = os.path.join(temp_dir, "modified.zip")
            add_dcm_extension_and_zip(temp_dir, output_zip_path)
            with open(output_zip_path, "rb") as f:
                st.download_button(label="Descargar carpeta con archivos .dcm", data=f, file_name="modified.zip", mime="application/zip")

# Funciones y código de la segunda aplicación
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

def mostrar_grafico(resultados):
    df = pd.DataFrame(resultados)
    fig = px.bar(df, x='imagen', y='clase_predicha', title='Resultados del Diagnóstico')
    st.plotly_chart(fig)

# Nueva función para la sección de síntomas
def disease_symptoms_app():
    st.header("Diagnóstico por Síntomas")
    st.markdown("### Introduce tus síntomas y obtén posibles enfermedades y pruebas necesarias:")

    # Diccionario de síntomas, posibles enfermedades y pruebas
    symptoms_data = {
        "fiebre": {
            "enfermedades": ["Gripe", "COVID-19", "Infección Bacteriana"],
            "pruebas": ["Prueba de PCR", "Análisis de Sangre"]
        },
        "dolor de cabeza": {
            "enfermedades": ["Migraña", "Tensión", "Infección Sinusal"],
            "pruebas": ["Escáner CT", "MRI"]
        },
        "tos": {
            "enfermedades": ["Bronquitis", "COVID-19", "Neumonía"],
            "pruebas": ["Radiografía de Tórax", "Prueba de PCR"]
        }
    }

    # Entrada de síntomas
    sintomas_usuario = st.text_input("Introduce tus síntomas separados por comas (e.g., fiebre, tos)")

    if st.button("Diagnosticar"):
        if sintomas_usuario:
            sintomas_lista = [sintoma.strip().lower() for sintoma in sintomas_usuario.split(",")]
            posibles_enfermedades = set()
            pruebas_necesarias = set()

            for sintoma in sintomas_lista:
                if sintoma in symptoms_data:
                    posibles_enfermedades.update(symptoms_data[sintoma]["enfermedades"])
                    pruebas_necesarias.update(symptoms_data[sintoma]["pruebas"])

            if posibles_enfermedades and pruebas_necesarias:
                st.write("### Posibles Enfermedades:")
                for enfermedad in posibles_enfermedades:
                    st.write(f"- {enfermedad}")

                st.write("### Pruebas Necesarias:")
                for prueba in pruebas_necesarias:
                    st.write(f"- {prueba}")
            else:
                st.write("No se encontraron coincidencias para los síntomas ingresados.")
        else:
            st.write("Por favor, introduce algunos síntomas.")

# Aplicación principal con pestañas
def main():
    st.sidebar.title("Praeventio")
    st.sidebar.markdown("### Navegación")
    tabs = st.sidebar.radio("Ir a", ["Añadir extensión .dcm", "Diagnóstico de Enfermedades", "Diagnóstico por Síntomas"])

    st.sidebar.image("logo.png", use_column_width=True)  # Reemplaza con la ruta correcta del logo

    lottie_animation = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_7Ghxdy.json")
    if lottie_animation:
        st.sidebar.markdown("### Animación")
        st_lottie(lottie_animation, height=200)

    if tabs == "Añadir extensión .dcm":
        dcm_app()
    elif tabs == "Diagnóstico de Enfermedades":
        disease_diagnosis_app()
    elif tabs == "Diagnóstico por Síntomas":
        disease_symptoms_app()

# Estilo personalizado para la app
st.markdown(
    """
    <style>
    .stApp {
        background-color: #242424;
        color: #FFFFFF;
        font-family: 'Helvetica', sans-serif;
    }
    .stSidebar {
        background-color: #2C3E50;
        color: white;
    }
    .stHeader {
        background-color: #2C3E50;
        color: white;
        padding: 10px;
        border-radius: 5px;
    }
    .stButton>button {
        background-color: #007bff;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #0056b3;
    }
    </style>
    """,
    unsafe_allow_html=True
)

if __name__ == "__main__":
    main()
