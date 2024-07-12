import streamlit as st
import numpy as np
import os
from PIL import Image, ImageFile
from descriptor import glcm, bitdesc
from distances import retrieve_similar_image
import tempfile

# Asegurarse de que se pueden cargar imágenes incompletas
ImageFile.LOAD_TRUNCATED_IMAGES = True

def load_features(descriptor):
    features_path = f'features_{descriptor}.npy'
    if os.path.exists(features_path):
        return np.load(features_path, allow_pickle=True)
    else:
        st.error(f"Feature file not found: {features_path}")
        return None

# Función para obtener la ruta del archivo temporal
def get_temporary_file_path(uploaded_file):
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_file.getvalue())
    temp_file.close()
    return temp_file.name

# Configuracion de la pagina web
st.set_page_config(layout="wide")

# Título de la aplicación
st.title("Recherche d’images similaires")

# Variables para la lógica de búsqueda
descriptor = None
distance_metric = None
num_results = None
image_chercher = None
similar = []

# Botones de radio para seleccionar el tipo de descriptor
descriptor = st.selectbox("Selectionnez le Descripteur", ["glcm", "bitdesc"])

# Lista desplegable para seleccionar el tipo de distancia
distance_metric = st.selectbox("Selectionnez le type de Distance", ["manhattan", "euclidean", "chebyshev", "canberra"])

# Caja de texto para escribir un número
num_results = st.slider("Number of Results to Display", 1, 10)

st.header("Téléchargement d'images et affichage des résultats")

# Cargar una imagen
uploaded_file = st.file_uploader("Sélectionnez l'image à rechercher", type=["png", "jpg", "jpeg", "bmp", "tiff"])

if uploaded_file is not None:
    # Obtener la ruta del archivo temporal
    image_path = get_temporary_file_path(uploaded_file)
    st.image(image_path, caption='Image pour la recherche', width=300)

# Botón para iniciar la búsqueda
if st.button('Chercher') and uploaded_file is not None:
    st.write("Extraire les caractéristiques de l’image chargée...")
    if descriptor == "glcm":
        uploaded_features = glcm(image_path)
    else:
        uploaded_features = bitdesc(image_path)
    
    st.write(f"Caractéristiques extraites: {uploaded_features}")

    # Cargar la base de datos de descriptores
    features_db = load_features(descriptor)
    if features_db is not None:
        st.write(f"Total de caractéristiques: {len(features_db)}")
        similar = retrieve_similar_image(features_db, uploaded_features, distance_metric, num_results)
        st.write(f"Similitudes calculadas: {similar}")
    else:
        st.error("Impossible de charger les caractéristiques.")

# Mostrar resultados
if similar:
    st.header('Résultats de recherche')
    num_columns = 3
    cols = st.columns(num_columns)
    for idx, img_info in enumerate(similar):
        image_path = img_info[0]
        image_similarity_score = img_info[2]
        if os.path.isfile(image_path):
            with cols[idx % num_columns]:
                image = Image.open(image_path)
                st.image(image, caption=f'Image trouvée: {img_info[2]}', width=150)
        else:
            st.error(f"Image not found: {image_path}")

# Ejecutar la aplicación Streamlit
# streamlit run app.py
