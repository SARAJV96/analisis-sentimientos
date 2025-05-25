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
st.set_page_config(page_title="Análisis de Sentimientos", layout="wide")

# Cargar datos CON MANEJO DE COMAS INTERNAS
try:
    df = pd.read_csv(
        "data/opiniones_clientes.csv",
        quotechar='"',        # Especificar que usamos comillas dobles
        engine='python',      # Usar parser más flexible
        on_bad_lines='skip',  # Omitir líneas mal formateadas
        delimiter='\n',       # Leer todo como una sola columna
        names=['Opinion']     # Nombrar la columna
    )
    # Eliminar filas vacías si las hay
    df = df.dropna()
except Exception as e:
    st.error(f"Error al leer el archivo: {str(e)}")
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

# ... (el resto de tu código para gráficos y análisis)