# gemini_explainer.py - Módulo para utils
import streamlit as st
import google.generativeai as genai
from typing import Dict, Any, Optional

def generate_dataset_explanation(dataset, api_key=None):
    """
    Generate a dataset explanation using Gemini AI
    
    Args:
        dataset (pd.DataFrame): DataFrame to explain
        api_key (str, optional): Gemini API key
    
    Returns:
        str: Explanation of the dataset
    """
    try:
        # Prepare dataset information
        dataset_info = {
            'rows': len(dataset),
            'columns': len(dataset.columns),
            'column_names': list(dataset.columns),
            'data_types': str(dataset.dtypes),
            'first_rows': dataset.head().to_string(),
            'basic_stats': dataset.describe().to_string()
        }
        
        # Initialize Gemini Explainer
        explainer = GeminiExplainer(api_key)
        
        # Generate explanation
        explanation = explainer.generate_dataset_explanation(dataset_info)
        
        return explanation
    
    except Exception as e:
        return f"Error generating dataset explanation: {str(e)}"
    
def generate_model_explanation(self, model_info: Dict[str, Any]) -> str:
        """
        Generar una explicación detallada de un modelo de machine learning
        
        Args:
            model_info (dict): Información del modelo
        
        Returns:
            str: Explicación generada por Gemini
        """
        prompt = f"""Proporciona una explicación detallada del modelo de machine learning:

        Información del Modelo:
        - Nombre del Modelo: {model_info.get('name', 'N/A')}
        - Tipo de Problema: {model_info.get('problem_type', 'N/A')}
        - Hiperparámetros: {model_info.get('hyperparameters', 'N/A')}
        - Métricas de Rendimiento:
            * Accuracy/R²: {model_info.get('performance_metric', 'N/A')}
            * Otras métricas: {model_info.get('additional_metrics', 'N/A')}

        En tu explicación, incluye:
        1. Descripción del algoritmo
        2. Funcionamiento interno del modelo
        3. Interpretación de los hiperparámetros
        4. Análisis de las métricas de rendimiento
        5. Fortalezas y limitaciones del modelo
        6. Recomendaciones para posibles mejoras"""

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error al generar explicación: {str(e)}"
        
class GeminiExplainer:
    def __init__(self, api_key: Optional[str] = None):
        """
        Inicializar el explicador de Gemini
        
        Args:
            api_key (str, opcional): API key de Google Generative AI
        """
        self.api_key = api_key or st.session_state.get('gemini_api_key')
        
        if not self.api_key:
            raise ValueError("No se ha proporcionado una API key de Gemini")
        
        # Configurar la API de Gemini
        genai.configure(api_key=self.api_key)
        
        # Seleccionar modelo
        self.model = genai.GenerativeModel('gemini-1.5-flash')

    def generate_dataset_explanation(self, dataset_info: Dict[str, Any]) -> str:
        """
        Generar una explicación detallada del dataset
        
        Args:
            dataset_info (dict): Información del dataset
        
        Returns:
            str: Explicación generada por Gemini
        """
        prompt = f"""Analiza este dataset y proporciona una explicación clara y concisa de su estructura y contenido:

        Información del Dataset:
        - Dimensiones: {dataset_info.get('rows', 'N/A')} filas × {dataset_info.get('columns', 'N/A')} columnas
        - Columnas: {', '.join(dataset_info.get('column_names', []))}
        - Tipos de datos: {dataset_info.get('data_types', 'N/A')}
        - Primeras filas: {dataset_info.get('first_rows', 'N/A')}
        - Estadísticas básicas: {dataset_info.get('basic_stats', 'N/A')}

        En tu explicación, incluye:
        1. Descripción general del dataset
        2. Tipos de variables presentes
        3. Posibles desafíos o consideraciones para el análisis
        4. Sugerencias iniciales de preprocesamiento
        5. Potenciales insights o patrones preliminares"""

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error al generar explicación: {str(e)}"

    def generate_model_explanation(self, model_info: Dict[str, Any]) -> str:
        """
        Generar una explicación detallada de un modelo de machine learning
        
        Args:
            model_info (dict): Información del modelo
        
        Returns:
            str: Explicación generada por Gemini
        """
        prompt = f"""Proporciona una explicación detallada del modelo de machine learning:

        Información del Modelo:
        - Nombre del Modelo: {model_info.get('name', 'N/A')}
        - Tipo de Problema: {model_info.get('problem_type', 'N/A')}
        - Hiperparámetros: {model_info.get('hyperparameters', 'N/A')}
        - Métricas de Rendimiento:
            * Accuracy/R²: {model_info.get('performance_metric', 'N/A')}
            * Otras métricas: {model_info.get('additional_metrics', 'N/A')}

        En tu explicación, incluye:
        1. Descripción del algoritmo
        2. Funcionamiento interno del modelo
        3. Interpretación de los hiperparámetros
        4. Análisis de las métricas de rendimiento
        5. Fortalezas y limitaciones del modelo
        6. Recomendaciones para posibles mejoras"""

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error al generar explicación: {str(e)}"

    def generate_clustering_explanation(self, clustering_info: Dict[str, Any]) -> str:
        """
        Generar una explicación de resultados de clustering
        
        Args:
            clustering_info (dict): Información del clustering
        
        Returns:
            str: Explicación generada por Gemini
        """
        prompt = f"""Analiza los resultados del método de clustering:

        Información del Clustering:
        - Método: {clustering_info.get('method', 'N/A')}
        - Número de Clusters: {clustering_info.get('n_clusters', 'N/A')}
        - Parámetros: {clustering_info.get('parameters', 'N/A')}
        - Métricas:
            * Silhouette Score: {clustering_info.get('silhouette_score', 'N/A')}
            * Calinski-Harabasz: {clustering_info.get('calinski_score', 'N/A')}
            * Davies-Bouldin: {clustering_info.get('davies_bouldin', 'N/A')}

        En tu explicación, incluye:
        1. Descripción del método de clustering
        2. Interpretación de los parámetros utilizados
        3. Significado de las métricas de evaluación
        4. Análisis de la calidad de los clusters
        5. Posibles insights o patrones detectados
        6. Recomendaciones para ajustar el clustering"""

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error al generar explicación: {str(e)}"

    def generate_feature_importance_explanation(self, feature_importance_info: Dict[str, Any]) -> str:
        """
        Generar una explicación de la importancia de características
        
        Args:
            feature_importance_info (dict): Información de importancia de características
        
        Returns:
            str: Explicación generada por Gemini
        """
        prompt = f"""Analiza la importancia de las características en el modelo:

        Información de Importancia de Características:
        - Método de Evaluación: {feature_importance_info.get('method', 'N/A')}
        - Características: {feature_importance_info.get('features', 'N/A')}
        - Valores de Importancia: {feature_importance_info.get('importance_values', 'N/A')}

        En tu explicación, incluye:
        1. Descripción del método de evaluación de importancia
        2. Análisis de las características más importantes
        3. Interpretación de los valores de importancia
        4. Posibles implicaciones para el modelado
        5. Recomendaciones para selección de características"""

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error al generar explicación: {str(e)}"

def initialize_gemini_explainer():
    """
    Función de utilidad para inicializar el explicador de Gemini en Streamlit
    
    Returns:
        GeminiExplainer: Instancia del explicador de Gemini
    """
    try:
        explainer = GeminiExplainer()
        return explainer
    except ValueError as e:
        st.error(str(e))
        return None

# Ejemplo de uso en Streamlit
def main():
    st.title("Explicaciones con Gemini")

    # Verificar configuración de API key
    if 'gemini_api_key' not in st.session_state:
        st.warning("Configura tu API key de Gemini")
        return

    explainer = initialize_gemini_explainer()
    
    if explainer:
        # Ejemplo de uso de métodos de explicación
        st.subheader("Explicación de Dataset")
        dataset_info = {
            'rows': 100,
            'columns': 5,
            'column_names': ['age', 'income', 'education', 'credit_score', 'loan_approved'],
            'data_types': 'Mixed (numeric and categorical)',
            'first_rows': 'Sample data preview',
            'basic_stats': 'Mean, median, standard deviation'
        }
        
        if st.button("Explicar Dataset"):
            explanation = explainer.generate_dataset_explanation(dataset_info)
            st.markdown(explanation)

        st.subheader("Explicación de Modelo")
        model_info = {
            'name': 'Random Forest Classifier',
            'problem_type': 'Clasificación binaria',
            'hyperparameters': {
                'n_estimators': 100,
                'max_depth': 5,
                'learning_rate': 0.1
            },
            'performance_metric': 0.85,
            'additional_metrics': {
                'precision': 0.82,
                'recall': 0.88,
                'f1_score': 0.85
            }
        }

        if st.button("Explicar Modelo"):
            explanation = explainer.generate_model_explanation(model_info)
            st.markdown(explanation)

        st.subheader("Explicación de Clustering")
        clustering_info = {
            'method': 'K-Means',
            'n_clusters': 3,
            'parameters': {
                'eps': 0.5,
                'min_samples': 5
            },
            'silhouette_score': 0.7,
            'calinski_score': 150.5,
            'davies_bouldin': 0.4
        }

        if st.button("Explicar Clustering"):
            explanation = explainer.generate_clustering_explanation(clustering_info)
            st.markdown(explanation)

        st.subheader("Explicación de Importancia de Características")
        feature_importance_info = {
            'method': 'SHAP Values',
            'features': ['age', 'income', 'education', 'credit_score'],
            'importance_values': {
                'age': 0.35,
                'income': 0.25,
                'education': 0.2,
                'credit_score': 0.2
            }
        }

        if st.button("Explicar Importancia de Características"):
            explanation = explainer.generate_feature_importance_explanation(feature_importance_info)
            st.markdown(explanation)

# Función para manejar errores de API key
def validate_gemini_api_key(api_key: str) -> bool:
    """
    Validar la API key de Gemini
    
    Args:
        api_key (str): API key a validar
    
    Returns:
        bool: True si la API key es válida, False en caso contrario
    """
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        # Intentar generar una respuesta simple
        response = model.generate_content("Hola, ¿estás funcionando?")
        return True
    except Exception as e:
        st.error(f"Error de validación de API key: {str(e)}")
        return False

# Función de configuración de API key en Streamlit
def setup_gemini_api_key():
    """
    Configurar y validar la API key de Gemini en Streamlit
    """
    st.sidebar.header("🔑 Configuración de Gemini API")
    
    # Input para la API key
    api_key = st.sidebar.text_input(
        "Ingresa tu Gemini API Key", 
        type="password",
        help="Puedes obtener tu API key en Google AI Studio"
    )
    
    # Botón de validación
    if st.sidebar.button("Validar API Key"):
        if api_key:
            if validate_gemini_api_key(api_key):
                st.session_state.gemini_api_key = api_key
                st.sidebar.success("✅ API Key validada correctamente")
            else:
                st.sidebar.error("❌ API Key inválida")
        else:
            st.sidebar.warning("Por favor, ingresa una API Key")
    
    # Mostrar estado actual
    if 'gemini_api_key' in st.session_state:
        st.sidebar.info("API Key configurada")

# Configuraciones adicionales y documentación
def get_gemini_documentation():
    """
    Generar documentación sobre el uso de Gemini en el proyecto
    
    Returns:
        str: Documentación en formato markdown
    """
    documentation = """
    ## 🤖 Explicaciones con Gemini AI

    ### Características
    - Generación de explicaciones detalladas para:
      * Datasets
      * Modelos de Machine Learning
      * Resultados de Clustering
      * Importancia de Características

    ### Requisitos
    - API Key de Google AI Studio
    - Conexión a internet
    - Biblioteca `google-generativeai`

    ### Configuración
    1. Obtén tu API Key en [Google AI Studio](https://makersuite.google.com/app/apikey)
    2. Configura la API Key en la barra lateral
    3. Valida la conexión con el botón "Validar API Key"

    ### Limitaciones
    - Depende de la disponibilidad del servicio
    - Consumo de tokens de API
    - Explicaciones generadas por IA pueden no ser 100% precisas

    ### Mejores Prácticas
    - Usar como complemento, no como única fuente de verdad
    - Verificar siempre las explicaciones generadas
    - Tener contexto del problema al interpretar resultados
    """
    return documentation

# Punto de entrada principal
if __name__ == "__main__":
    main()