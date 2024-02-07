import streamlit as st
import os
import shutil
from zipfile import ZipFile
import tempfile

# Función para añadir la extensión .dcm a todos los archivos de un directorio, incluyendo subdirectorios
def add_dcm_extension_and_zip(input_dir, output_zip):
    with ZipFile(output_zip, 'w') as zipf:
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                # Define el nuevo nombre del archivo con la extensión .dcm
                new_file_name = file + ".dcm"
                # Crea el path completo del nuevo archivo
                new_file_path = os.path.join(root, new_file_name)
                # Añade el archivo al zip con el nuevo nombre
                zipf.write(os.path.join(root, file), arcname=new_file_path)

# Streamlit UI
st.title("Añadir extensión .dcm a archivos")

uploaded_files = st.file_uploader("Sube una carpeta en formato zip que contenga tus archivos", accept_multiple_files=False, type='zip')

if uploaded_files is not None:
    # Crea un directorio temporal para extraer el contenido del zip
    with tempfile.TemporaryDirectory() as temp_dir:
        # Path del archivo zip temporal
        zip_path = os.path.join(temp_dir, "uploaded.zip")
        # Guarda el archivo subido en el directorio temporal
        with open(zip_path, "wb") as f:
            f.write(uploaded_files.getvalue())
        # Extrae el contenido del zip
        with ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        # Prepara el archivo zip de salida
        output_zip_path = os.path.join(temp_dir, "modified.zip")
        # Añade la extensión .dcm a todos los archivos y crea un nuevo zip
        add_dcm_extension_and_zip(temp_dir, output_zip_path)
        # Permite al usuario descargar el nuevo archivo zip
        with open(output_zip_path, "rb") as f:
            st.download_button(label="Descargar carpeta con archivos .dcm", data=f, file_name="modified.zip", mime="application/zip")
