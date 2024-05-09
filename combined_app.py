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
from torchvision.models import resnet50


# Configuración para manejar archivos DICOM con datos de longitud incorrecta
pydicom.config.convert_wrong_length_to_UN = True

# Crear carpetas para guardar las imágenes procesadas
os.makedirs("output", exist_ok=True)

# Función de la primera aplicación
def add_dcm_extension_and_zip(input_dir, output_zip):
    with ZipFile(output_zip, 'w') as zipf:
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                new_file_name = file + ".dcm"
                new_file_path = os.path.join(root, new_file_name)
                zipf.write(os.path.join(root, file), arcname=new_file_path)

def dcm_app():
    st.title("Añadir extensión .dcm a archivos")

    uploaded_files = st.file_uploader("Sube una carpeta en formato zip que contenga tus archivos", accept_multiple_files=False, type='zip')

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

def disease_diagnosis_app():
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

    uploaded_file_or_folder = st.file_uploader("Elige una imagen o carpeta...", type=["jpg", "jpeg", "png", "dcm"], accept_multiple_files=True)

    if uploaded_file_or_folder is not None:
        resultados = []
        imagenes_distintas_de_sano_list = []

        for uploaded_item in uploaded_file_or_folder:
            contenido = uploaded_item.read()
            file_name = uploaded_item.name.lower()
            file_extension = file_name.split('.')[-1]

            if file_extension not in ['png', 'jpg', 'jpeg']:
                try:
                    dicom_data = pydicom.dcmread(io.BytesIO(contenido), force=True)
                    if 'PixelData' in dicom_data:
                        imagen_pil = convertir_dicom_a_pil(dicom_data)
                    else:
                        raise ValueError("El archivo DICOM no contiene datos de imagen.")
                except Exception as e:
                    st.error(f"No se pudo procesar el archivo {uploaded_item.name}: {e}")
                    continue
            else:
                try:
                    imagen_pil = Image.open(io.BytesIO(contenido)).convert('RGB')
                except IOError as e:
                    st.error(f"No se pudo procesar el archivo de imagen {uploaded_item.name}: {e}")
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

# Aplicación principal con pestañas
def main():
    st.sidebar.title("Navegación")
    tabs = st.sidebar.radio("Ir a", ["Añadir extensión .dcm", "Diagnóstico de Enfermedades"])

    if tabs == "Añadir extensión .dcm":
        dcm_app()
    elif tabs == "Diagnóstico de Enfermedades":
        disease_diagnosis_app()

if __name__ == "__main__":
    main()
