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

# Configuración para manejar archivos DICOM con datos de longitud incorrecta
pydicom.config.convert_wrong_length_to_UN = True

# Definir funciones para cargar modelos y realizar predicciones
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

# Configuración de la aplicación Streamlit
st.title("Diagnóstico de Enfermedades")

# Menú de opciones para seleccionar la enfermedad
enfermedades = {
    "Pneumonia": "modelos/pneumonia/",
    "Tumor Cerebral": "modelos/tumor_cerebral/",
    # Agregar más enfermedades según sea necesario
}

enfermedad_seleccionada = st.selectbox("Selecciona la enfermedad a diagnosticar:", list(enfermedades.keys()))

# Cargar la información de la enfermedad
ruta_carpeta_enfermedad = enfermedades[enfermedad_seleccionada]
ruta_info_enfermedad = os.path.join(ruta_carpeta_enfermedad, "info.json")

try:
    with open(ruta_info_enfermedad, 'r') as json_file:
        info_enfermedad = json.load(json_file)
except FileNotFoundError:
    st.error(f"No se pudo encontrar el archivo {ruta_info_enfermedad}. Verifica la ruta y asegúrate de que el archivo exista.")
    raise FileNotFoundError(f"No se pudo encontrar el archivo {ruta_info_enfermedad}.")
except Exception as e:
    st.error(f"Error al leer el archivo {ruta_info_enfermedad}: {str(e)}")
    raise e  # Para mostrar el error completo en el registro

# Cargar el modelo seleccionado
ruta_completa_modelo = os.path.join(ruta_carpeta_enfermedad, "modelo_entrenado.pth")
if not os.path.isfile(ruta_completa_modelo):
    st.error(f"No se pudo encontrar el archivo del modelo: {ruta_completa_modelo}. Asegúrate de que el archivo exista.")
    st.stop()

modelo_seleccionado = cargar_modelo(ruta_completa_modelo, info_enfermedad['num_clases'])

# Subir una imagen o carpeta
uploaded_file_or_folder = st.file_uploader("Elige una imagen o carpeta...", type=["jpg", "jpeg", "png", "dcm"], accept_multiple_files=True)

if uploaded_file_or_folder is not None:
    # Inicializar variables
    resultados = []
    imagenes_distintas_de_sano_list = []

    # Recorrer archivos en caso de que se haya subido una carpeta
    for uploaded_item in uploaded_file_or_folder:
        if uploaded_item.type == "application/octet-stream":  # Streamlit puede usar este tipo MIME para archivos DICOM
            try:
                dicom_data = pydicom.dcmread(uploaded_item, force=True)
                imagen_array = dicom_data.pixel_array
                if np.amax(imagen_array) > 255:  # Normalizar si es necesario
                    imagen_array = (imagen_array / np.amax(imagen_array)) * 255
                imagen_array = imagen_array.astype(np.uint8)
                if len(imagen_array.shape) == 2:  # Imagen en escala de grises
                    imagen_pil = Image.fromarray(imagen_array).convert('RGB')
                else:  # Imagen ya es RGB
                    imagen_pil = Image.fromarray(imagen_array)
            except Exception as e:
                st.error(f"Error al procesar el archivo DICOM {uploaded_item.name}: {e}")
                continue
        else:  # Para otros tipos de archivos como JPG, JPEG, PNG
            imagen_pil = Image.open(uploaded_item).convert('RGB')

        # Realizar la predicción
        clase_predicha = predecir_imagen(modelo_seleccionado, imagen_pil)

        # Obtener el nombre de la clase a partir del mapeo en info.json
        nombre_clase_predicha = info_enfermedad['clases'][str(clase_predicha)]

        # Guardar resultados
        resultados.append({"imagen": uploaded_item.name, "clase_predicha": nombre_clase_predicha})

        # Verificar si la clase predicha es distinta a "Sano"
        if nombre_clase_predicha != "Sano":
            imagenes_distintas_de_sano_list.append((imagen_pil, nombre_clase_predicha))

    # Mostrar resultados
    if resultados:
        st.write("Resultados:")
        for resultado in resultados:
            st.write(f"Imagen: {resultado['imagen']}, Clase Predicha: {resultado['clase_predicha']}")

    # Mostrar imágenes distintas a "Sano" en una galería
    if imagenes_distintas_de_sano_list:
        st.write("Imágenes distintas a 'Sano':")
        for imagen, clase_predicha in imagenes_distintas_de_sano_list:
            st.image(imagen, caption=f"Clase predicha: {clase_predicha}", use_column_width=True)


