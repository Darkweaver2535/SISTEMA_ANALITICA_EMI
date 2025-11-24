# SISTEMA_ANALITICA_EMI

Sistema de Analítica de Datos utilizando técnicas Open Source Intelligence (OSINT) para el Vicerrectorado de Grado de la Escuela Militar de Ingeniería.

## 📋 Descripción

El proyecto desarrolla un sistema integral de analítica de datos que combina técnicas OSINT con Inteligencia Artificial para automatizar la recolección, procesamiento y análisis de información proveniente de fuentes abiertas. El sistema permite al Vicerrectorado de Grado optimizar la toma de decisiones académicas mediante el análisis automatizado de patrones y tendencias.

## 🎯 Objetivos

### Objetivo General
Desarrollar un sistema de analítica de datos utilizando técnicas de Open Source Intelligence que permita la identificación de patrones para reducir tiempo en el flujo de información y la toma de decisiones en el Vicerrectorado de Grado de la Escuela Militar de Ingeniería.

### Objetivos Específicos
- Analizar datos provenientes de fuentes abiertas utilizando técnicas OSINT
- Diseñar un módulo de visualización mediante dashboard interactivo
- Aplicar modelos de IA, Machine Learning y NLP para análisis de datos
- Evaluar el funcionamiento mediante pruebas de efectividad

## 🏗️ Arquitectura del Sistema

El sistema se estructura en varios módulos integrados:

### 1. Módulo de Recolección de Datos (OSINT)

```python
# Ejemplo de recolección de datos desde fuentes abiertas
import requests
from bs4 import BeautifulSoup
import pandas as pd

class OSINTCollector:
    def __init__(self, sources):
        self.sources = sources
        self.data = []
    
    def collect_social_media_data(self, platform_url):
        """
        Recolecta datos de plataformas de redes sociales
        usando APIs oficiales o web scraping ético
        """
        try:
            response = requests.get(platform_url)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extracción de datos relevantes
            posts = soup.find_all('div', class_='post-content')
            
            for post in posts:
                self.data.append({
                    'content': post.text,
                    'timestamp': post.find('time')['datetime'],
                    'source': platform_url
                })
            
            return self.data
        except Exception as e:
            print(f"Error en recolección: {e}")
            return None
    
    def save_to_database(self, db_connection):
        """
        Almacena datos recolectados en base de datos
        """
        df = pd.DataFrame(self.data)
        df.to_sql('raw_data', db_connection, if_exists='append')
```

### 2. Módulo de Procesamiento con IA

```python
# Análisis de sentimientos con NLP
from transformers import pipeline
import numpy as np

class NLPAnalyzer:
    def __init__(self):
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="nlptown/bert-base-multilingual-uncased-sentiment"
        )
    
    def analyze_sentiment(self, texts):
        """
        Analiza el sentimiento de textos recolectados
        """
        results = []
        for text in texts:
            sentiment = self.sentiment_analyzer(text[:512])[0]
            results.append({
                'text': text,
                'label': sentiment['label'],
                'score': sentiment['score']
            })
        return results
    
    def detect_patterns(self, data):
        """
        Identifica patrones mediante Machine Learning
        """
        from sklearn.cluster import KMeans
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # Vectorización de textos
        vectorizer = TfidfVectorizer(max_features=100)
        X = vectorizer.fit_transform(data['content'])
        
        # Clustering para identificar temas
        kmeans = KMeans(n_clusters=5, random_state=42)
        clusters = kmeans.fit_predict(X)
        
        return clusters
```

### 3. Base de Datos

```sql
-- Estructura de base de datos PostgreSQL

-- Tabla de datos brutos recolectados
CREATE TABLE raw_data (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    source VARCHAR(255),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    platform VARCHAR(100),
    metadata JSONB
);

-- Tabla de análisis procesados
CREATE TABLE processed_analysis (
    id SERIAL PRIMARY KEY,
    raw_data_id INTEGER REFERENCES raw_data(id),
    sentiment_score DECIMAL(3,2),
    sentiment_label VARCHAR(50),
    topic_cluster INTEGER,
    keywords TEXT[],
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tabla de alertas y patrones detectados
CREATE TABLE detected_patterns (
    id SERIAL PRIMARY KEY,
    pattern_type VARCHAR(100),
    description TEXT,
    severity_level VARCHAR(20),
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(50) DEFAULT 'active',
    related_data_ids INTEGER[]
);

-- Vista para dashboard de métricas
CREATE VIEW dashboard_metrics AS
SELECT 
    DATE(timestamp) as date,
    platform,
    COUNT(*) as total_mentions,
    AVG(pa.sentiment_score) as avg_sentiment,
    COUNT(DISTINCT dp.id) as patterns_detected
FROM raw_data rd
LEFT JOIN processed_analysis pa ON rd.id = pa.raw_data_id
LEFT JOIN detected_patterns dp ON rd.id = ANY(dp.related_data_ids)
GROUP BY DATE(timestamp), platform;

-- Índices para optimización
CREATE INDEX idx_raw_data_timestamp ON raw_data(timestamp);
CREATE INDEX idx_raw_data_platform ON raw_data(platform);
CREATE INDEX idx_processed_analysis_sentiment ON processed_analysis(sentiment_label);
```

### 4. Dashboard de Visualización

```python
# Dashboard interactivo con Streamlit
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

class AnalyticsDashboard:
    def __init__(self, db_connection):
        self.db = db_connection
    
    def render_main_dashboard(self):
        st.title("📊 Sistema de Analítica OSINT - Vicerrectorado de Grado")
        
        # KPIs principales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_data = self.get_total_records()
            st.metric("Total Datos Recolectados", total_data)
        
        with col2:
            avg_sentiment = self.get_average_sentiment()
            st.metric("Sentimiento Promedio", f"{avg_sentiment:.2f}")
        
        with col3:
            patterns = self.get_active_patterns()
            st.metric("Patrones Detectados", patterns)
        
        with col4:
            alerts = self.get_pending_alerts()
            st.metric("Alertas Pendientes", alerts, delta="-2")
        
        # Gráficos de tendencias
        self.render_sentiment_timeline()
        self.render_topic_distribution()
        self.render_pattern_alerts()
    
    def render_sentiment_timeline(self):
        """
        Visualiza evolución del sentimiento en el tiempo
        """
        query = """
            SELECT DATE(rd.timestamp) as date,
                   AVG(pa.sentiment_score) as avg_score
            FROM raw_data rd
            JOIN processed_analysis pa ON rd.id = pa.raw_data_id
            WHERE rd.timestamp >= NOW() - INTERVAL '30 days'
            GROUP BY DATE(rd.timestamp)
            ORDER BY date
        """
        
        df = pd.read_sql(query, self.db)
        
        fig = px.line(df, x='date', y='avg_score',
                     title='Evolución del Sentimiento - Últimos 30 días',
                     labels={'avg_score': 'Puntuación Promedio'})
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_topic_distribution(self):
        """
        Muestra distribución de temas identificados
        """
        query = """
            SELECT topic_cluster, COUNT(*) as count
            FROM processed_analysis
            WHERE topic_cluster IS NOT NULL
            GROUP BY topic_cluster
        """
        
        df = pd.read_sql(query, self.db)
        
        fig = px.pie(df, values='count', names='topic_cluster',
                    title='Distribución de Temas Identificados')
        
        st.plotly_chart(fig, use_container_width=True)
```

### 5. Sistema de Alertas

```python
# Sistema de detección y notificación de alertas
class AlertSystem:
    def __init__(self, db_connection):
        self.db = db_connection
        self.thresholds = {
            'sentiment_negative': -0.5,
            'mention_spike': 50,
            'critical_keywords': ['crisis', 'problema', 'urgente']
        }
    
    def detect_sentiment_alerts(self):
        """
        Detecta caídas significativas en sentimiento
        """
        query = """
            SELECT AVG(sentiment_score) as avg_score
            FROM processed_analysis
            WHERE processed_at >= NOW() - INTERVAL '24 hours'
        """
        
        result = pd.read_sql(query, self.db)
        avg_score = result['avg_score'][0]
        
        if avg_score < self.thresholds['sentiment_negative']:
            self.create_alert(
                pattern_type='sentiment_drop',
                description=f'Caída de sentimiento detectada: {avg_score:.2f}',
                severity='high'
            )
    
    def create_alert(self, pattern_type, description, severity):
        """
        Registra nueva alerta en base de datos
        """
        query = """
            INSERT INTO detected_patterns 
            (pattern_type, description, severity_level)
            VALUES (%s, %s, %s)
        """
        
        cursor = self.db.cursor()
        cursor.execute(query, (pattern_type, description, severity))
        self.db.commit()
        
        # Notificación por email
        self.send_notification(pattern_type, description, severity)
```

## 🛠️ Tecnologías Utilizadas

- **Backend**: Python 3.9+
- **Base de Datos**: PostgreSQL
- **IA/ML**: scikit-learn, transformers, spaCy
- **Visualización**: Streamlit, Plotly
- **OSINT**: BeautifulSoup, Selenium, APIs oficiales
- **NLP**: BERT multilingual, análisis de sentimientos

## 📦 Instalación

```bash
# Clonar repositorio
git clone https://github.com/Darkweaver2535/SISTEMA_ANALITICA_EMI.git
cd SISTEMA_ANALITICA_EMI

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Configurar base de datos
psql -U postgres -f database/schema.sql
```

## 🚀 Uso

```bash
# Iniciar recolección de datos
python src/collectors/osint_collector.py

# Procesar datos con IA
python src/processors/nlp_analyzer.py

# Lanzar dashboard
streamlit run src/dashboard/app.py
```

## 📊 Resultados Esperados

- Reducción del 70% en tiempo de análisis manual
- Identificación automática de patrones en tiempo real
- Dashboard interactivo con métricas clave
- Sistema de alertas tempranas

## 🎓 Contexto Institucional

### Escuela Militar de Ingeniería

La Escuela Militar de Ingeniería "Mariscal Antonio José de Sucre" fue creada mediante Decreto Supremo No 2229 del 29 de Octubre de 1950, con la misión de formar profesionales de excelencia en ingeniería.

#### Misión
Formar y especializar profesionales de excelencia, con principios, valores ético-morales y cívicos, caracterizados por su responsabilidad social, espíritu emprendedor, liderazgo y disciplina; promoviendo la internacionalización, Interacción Social y desarrollo de la Ciencia, Tecnología e Innovación, para contribuir al desarrollo del Estado.

#### Visión
Ser la Universidad líder en la formación de profesionales en Ingeniería y de especialización, caracterizada por el estudio, aplicación e innovación tecnológica, con responsabilidad social y reconocida a nivel nacional e internacional.

## 🔍 Fundamentos Teóricos

### Open Source Intelligence (OSINT)
OSINT es la práctica de recolectar y analizar información obtenida exclusivamente de fuentes abiertas y disponibles públicamente para apoyar actividades de inteligencia. El sistema implementa técnicas OSINT para:

- Recolección automatizada de datos de redes sociales
- Análisis de fuentes públicas institucionales
- Monitoreo de tendencias y percepciones
- Identificación de patrones de comportamiento

### Inteligencia Artificial
El sistema utiliza IA para el procesamiento automático de datos no estructurados mediante:

- **Procesamiento de Lenguaje Natural (NLP)**: Análisis de sentimientos y extracción de entidades
- **Machine Learning**: Clustering y clasificación de datos
- **Deep Learning**: Redes neuronales para reconocimiento de patrones

## 📈 Alcances

### Alcance Temático
- **Área General**: Ingeniería de Sistemas
- **Área de Investigación**: Gestión del conocimiento y nuevas tecnologías
- **Línea de Investigación**: Nuevas Tecnologías

### Alcance Geográfico
Escuela Militar de Ingeniería - Unidad Académica La Paz (UALP), Bolivia

### Alcance Temporal
Gestión II/2025 - I/2026

## ⚠️ Límites del Sistema

- Monitoreo de máximo 3 plataformas digitales principales
- Análisis enfocado en percepción institucional y tendencias generales
- Procesamiento de información de los últimos 6 meses

## 📚 Metodología

**Tipo de Investigación**: Investigación Aplicada  
**Enfoque**: Descriptivo con elementos cualitativos y cuantitativos  
**Método**: Observación y descripción de fenómenos para identificar patrones

## 👥 Autor

**Alvaro Encinas**  
Estudiante de Ingeniería de Sistemas  
Escuela Militar de Ingeniería - UALP  
Gestión II/2025 - I/2026

## 📄 Licencia

Este proyecto es desarrollado como Trabajo de Grado para la Escuela Militar de Ingeniería.

## 🙏 Agradecimientos

Al Vicerrectorado de Grado de la Escuela Militar de Ingeniería por el apoyo y colaboración en el desarrollo de este proyecto.

---

**Escuela Militar de Ingeniería "Mcal. Antonio José de Sucre"**  
*Formando profesionales de excelencia con valores ético-morales*
