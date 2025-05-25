import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import re
from nltk.corpus import stopwords
from wordcloud import WordCloud
from collections import Counter
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import nltk

nltk.download('stopwords')

# Configuración de la página
st.set_page_config(page_title="analisis de Sentimientos", layout="wide")

# Cargar datos (asegúrate de tener el CSV en la carpeta 'data')
try:
    df = pd.read_csv("data/opiniones_clientes.csv")
except FileNotFoundError:
    st.error("Archivo no encontrado: Crea una carpeta 'data' y coloca allí 'opiniones_clientes.csv'")
    st.stop()

# Cargar modelo de análisis de sentimientos
@st.cache_resource
def cargar_modelo():
    modelo = "nlptown/bert-base-multilingual-uncased-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(modelo)
    model = AutoModelForSequenceClassification.from_pretrained(modelo)
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

clasificador = cargar_modelo()

def interpretar(label):
    estrellas = int(label[0])
    return "Positivo" if estrellas >= 4 else "Neutro" if estrellas == 3 else "Negativo"

# Procesar opiniones
opiniones = df['Opinion'].astype(str).tolist()
resultados = clasificador(opiniones)
df['Sentimiento'] = [interpretar(r['label']) for r in resultados]

# Interfaz de usuario
st.title("📊 Análisis de Opiniones de Clientes")
st.dataframe(df[['Opinion', 'Sentimiento']].head(20), use_container_width=True)

# Gráficos y análisis (igual que en el código anterior)
# ... (incluye aquí el resto del código de gráficos y el clasificador de nuevos comentarios)
