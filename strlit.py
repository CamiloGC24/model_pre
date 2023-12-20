import streamlit as st
from torchvision import transforms
import torch
import torch.nn as nn
from PIL import Image
from torchvision.models import resnet50

# Cargar el modelo preentrenado
pretrain_model = resnet50(pretrained=True)
in_features = pretrain_model.fc.in_features
pretrain_model.fc = nn.Linear(in_features, 4)
modelo_guardado = pretrain_model
modelo_guardado.load_state_dict(torch.load('modelo_entrenado_20%2.0.pth', map_location=torch.device('cpu')))
modelo_guardado.eval()

# Mover el modelo a GPU si está disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modelo_guardado = modelo_guardado.to(device)

# Transformaciones de la imagen
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Función para realizar la predicción
def predecir_imagen(imagen):
    imagen_transformada = transform(imagen).unsqueeze(0)
    imagen_transformada = imagen_transformada.to(device)

    with torch.no_grad():
        outputs = modelo_guardado(imagen_transformada)
        _, predicted = torch.max(outputs, 1)

    return predicted.item()

# Configuración de la aplicación Streamlit
st.title("Predicción de Clases de Imágenes")
st.write("Sube una imagen y predice su clase.")

uploaded_file = st.file_uploader("Elige una imagen...", type="jpg")

if uploaded_file is not None:
    # Cargar la imagen
    imagen = Image.open(uploaded_file).convert('RGB')
    st.image(imagen, caption='Imagen cargada', use_column_width=True)

    # Realizar la predicción
    clase_predicha = predecir_imagen(imagen)

    # Mostrar resultados
    st.write("Resultado:")
    st.write(f"Clase predicha: {clase_predicha}")
