import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import re
from nltk.corpus import stopwords
from wordcloud import WordCloud
from collections import Counter
from transformers import pipeline
import nltk
import requests
from io import StringIO
import os

# Configuración inicial
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('spanish'))

# --- Configuración para Render ---
PORT = int(os.environ.get("PORT", 8501))
st.set_page_config(page_title="📊 Analizador", layout="wide")

# --- Carga desde GitHub ---
@st.cache_data(ttl=3600)  # Cache por 1 hora
def cargar_datos():
    try:
        # URL raw de GitHub (asegúrate de que sea el enlace directo al archivo raw)
        github_url = "https://github.com/SARAJV96/analisis-sentimientos/blob/main/opiniones_clientes.csv"
        
        response = requests.get(github_url)
        response.raise_for_status()  # Verifica errores HTTP
        
        # Leer CSV directamente desde la respuesta
        df = pd.read_csv(
            StringIO(response.text),
            usecols=['Opinion'],
            encoding='utf-8',
            on_bad_lines='skip'
        ).dropna()
        
        return df.sample(min(50, len(df)))  # Limitar a 50 registros para pruebas
    
    except Exception as e:
        st.error(f"❌ Error al cargar datos: {str(e)}")
        st.info("ℹ️ Asegúrate que la URL del CSV en GitHub sea correcta y pública")
        st.stop()

# --- Modelo ultra ligero ---
@st.cache_resource
def cargar_modelo():
    try:
        return pipeline(
            "text-classification",
            model="finiteautomata/bertweet-base-sentiment-analysis",
            device=-1,
            truncation=True,
            max_length=64  # Más corto para ahorrar memoria
        )
    except Exception as e:
        st.error(f"❌ Error al cargar modelo: {str(e)}")
        st.stop()

# --- Visualizaciones optimizadas ---
def mostrar_graficos(df):
    # 1. Distribución de sentimientos
    with st.expander("📈 Distribución de Sentimientos", expanded=True):
        fig, ax = plt.subplots()
        df['Sentimiento'].value_counts().plot(kind='bar', color=['#4CAF50','#2196F3','#F44336'])
        st.pyplot(fig)
        plt.close(fig)
    
    # 2. WordCloud
    with st.expander("☁️ Nube de Palabras", expanded=False):
        text = ' '.join(df['Opinion'].astype(str))
        text = re.sub(r'[^\w\s]','', text.lower())
        text = ' '.join([w for w in text.split() if w not in stop_words])
        
        wc = WordCloud(width=800, height=400, background_color='white').generate(text)
        fig, ax = plt.subplots()
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
        plt.close(fig)

# --- Interfaz principal ---
def main():
    st.title("🔍 Analizador de Opiniones (GitHub + Render)")
    
    # Precarga el modelo primero (evita timeout)
    with st.spinner("⚙️ Cargando modelo (puede tomar 1-2 minutos)..."):
        modelo = cargar_modelo()
    
    # Carga de datos
    with st.spinner("📥 Descargando datos desde GitHub..."):
        df = cargar_datos()
    
    # Análisis
    with st.spinner("🧠 Analizando sentimientos..."):
        df['Sentimiento'] = df['Opinion'].apply(
            lambda x: modelo(str(x)[:64])[0]['label']
        )
        df['Sentimiento'] = df['Sentimiento'].map({
            'POS': '⭐ Positivo', 
            'NEU': '🔄 Neutro', 
            'NEG': '⚠️ Negativo'
        })
    
    # Resultados
    st.dataframe(df, use_container_width=True)
    mostrar_graficos(df)
    
    if st.button("🔄 Actualizar"):
        st.cache_data.clear()
        st.rerun()

if __name__ == "__main__":
    os.system(f"streamlit run app.py --server.port={PORT} --server.address=0.0.0.0 --server.headless=true")
