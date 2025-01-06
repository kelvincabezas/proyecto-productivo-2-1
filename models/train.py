# train.py - Módulo para models
import time
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample
from sklearn.metrics import (
    mean_squared_error, r2_score, 
    accuracy_score, classification_report, confusion_matrix
)
import os
import sys
import pickle
import io
import h2o

# Importaciones
from utils.model_utils import (
    ModelTrainer,  # Importar la clase
    get_model_options,
    train_model_pipeline,
    process_classification_data,
    create_class_distribution_plot
)
from utils.gemini_explainer import initialize_gemini_explainer
from utils.gemini_explainer import generate_model_explanation
from utils.shap_explainer import create_shap_analysis_dashboard

def safe_init_h2o(url=None, **kwargs):
    """
    Safely initialize H2O cluster if not already running.
    
    Args:
        url (str, optional): H2O cluster URL. Defaults to None (local instance).
        **kwargs: Additional arguments to pass to h2o.init()
    
    Returns:
        h2o._backend.H2OConnection: The H2O connection object
    """
    # Get current H2O instance if exists
    current = h2o.connection()
    
    # Check if H2O is already running
    if current and current.cluster:
        print("H2O is already running at", current.base_url)
        return current
    
    # Initialize new H2O instance
    print("Starting new H2O instance...")
    return h2o.init(url=url, **kwargs)

# Obtener la ruta del directorio raíz del proyecto
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

def validate_data_preparation(train):
    """
    Validar que los datos estén preparados correctamente
    
    Args:
        train (pd.DataFrame): Datos de entrenamiento
    
    Returns:
        bool: Indica si los datos están listos para entrenamiento
    """
    if train is None or train.empty:
        st.warning("⚠️ No hay datos preparados en la sesión.")
        return False
    return True

def select_features_and_target(train):
    """
    Permitir al usuario seleccionar características y variable objetivo
    
    Args:
        train (pd.DataFrame): Datos de entrenamiento
    
    Returns:
        tuple: Variables predictoras (X) y variable objetivo (y)
    """
    numeric_cols = train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Mantener las selecciones en session_state
    if 'feature_cols' not in st.session_state:
        st.session_state.feature_cols = []
        
    feature_cols = st.multiselect(
        "Selecciona las variables predictoras (X):",
        numeric_cols,
        default=st.session_state.feature_cols
    )
    st.session_state.feature_cols = feature_cols
    
    # Obtener TODAS las columnas disponibles para target
    all_cols = train.columns.tolist()
    available_targets = [col for col in all_cols if col not in feature_cols]
    
    if not available_targets:
        st.warning("Por favor, deselecciona algunas variables predictoras para poder seleccionar la variable objetivo.")
        return None, None
        
    if ('target_col' not in st.session_state or 
        st.session_state.target_col not in available_targets):
        st.session_state.target_col = available_targets[0]
    
    target_col = st.selectbox(
        "Selecciona la variable objetivo (y):",
        available_targets,
        index=available_targets.index(st.session_state.target_col)
    )
    st.session_state.target_col = target_col
    
    if not (feature_cols and target_col):
        st.warning("Por favor selecciona variables predictoras y objetivo.")
        return None, None
        
    X = train[feature_cols]
    y = train[target_col]
    
    return X, y

def determine_problem_type(y):
    """
    Determinar el tipo de problema de machine learning
    
    Args:
        y (pd.Series): Variable objetivo
    
    Returns:
        str: Tipo de problema ('classification' o 'regression')
    """
    is_categorical = y.dtype == 'object' or (y.dtype.name.startswith(('int', 'float')) and y.nunique() <= 10)
    problem_type = 'classification' if is_categorical else 'regression'
    st.write(f"Tipo de problema identificado: **{problem_type}**")
    return problem_type

def handle_data_balancing(X, y, random_state):
    """
    Manejar el desbalanceo de clases
    
    Args:
        X (pd.DataFrame): Variables predictoras
        y (pd.Series): Variable objetivo
        random_state (int): Semilla aleatoria
    
    Returns:
        tuple: Variables predictoras y objetivo balanceadas
    """
    if y.value_counts().min() / y.value_counts().max() < 0.5:
        st.write("⚠️ Se detectó desbalanceo en las clases")
        balance_method = st.selectbox(
            "Técnica de balanceo:",
            ["Ninguno", "Submuestreo", "Sobremuestreo", "SMOTE"]
        )
        
        if balance_method != "Ninguno":
            with st.spinner("Aplicando técnica de balanceo..."):
                if balance_method == "Submuestreo":
                    min_class_size = y.value_counts().min()
                    X, y = resample(X, y, n_samples=min_class_size*2, stratify=y)
                elif balance_method == "Sobremuestreo":
                    max_class_size = y.value_counts().max()
                    X, y = resample(X, y, n_samples=max_class_size*2, stratify=y)
                else:  # SMOTE
                    smote = SMOTE(random_state=random_state)
                    X, y = smote.fit_resample(X, y)
            st.success("Balanceo completado!")
    
    return X, y

def show_train():
    """
    Función principal para mostrar la interfaz de entrenamiento de modelos
    """
    st.title("Desarrollo de Modelos")
    
    # Verificar preparación de datos
    if 'prepared_data' not in st.session_state:
        st.warning("⚠️ No hay datos preparados en la sesión. Por favor, carga y prepara los datos primero.")
        return
        
    if st.session_state.prepared_data is None:
        st.warning("⚠️ Los datos preparados están vacíos. Por favor, verifica la preparación de datos.")
        return
    
    # Inicializar 'trained_models' si no existe
    if 'trained_models' not in st.session_state:
        st.session_state.trained_models = {}
        
    train = st.session_state.prepared_data
    
    try:
        # Seleccionar características y objetivo
        X, y = select_features_and_target(train)
        if X is None or y is None:
            return
        
        # Verificar valores nulos
        if X.isnull().sum().sum() > 0 or y.isnull().sum() > 0:
            st.error("Hay valores nulos en los datos. Por favor, vuelve a la página de preparación y maneja los valores faltantes.")
            return
        
        # Determinar tipo de problema
        problem_type = determine_problem_type(y)
        
        # Configuraciones de entrenamiento
        col1, col2, col3 = st.columns(3)
        with col1:
            test_size = st.slider("Tamaño del conjunto de prueba:", 0.1, 0.5, 0.2)
        with col2:
            random_state = st.number_input("Random State:", min_value=0, value=42)
        with col3:
            n_folds = st.number_input("Número de folds para validación cruzada:", min_value=2, max_value=10, value=5)
            st.session_state.n_folds = n_folds
        
        # Preprocesamiento de datos para clasificación
        if problem_type == 'classification':
            y_original = y
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y))
            st.session_state.label_encoder = le
            st.write("Mapeo de clases:", dict(enumerate(le.classes_)))
            
            # Visualizar distribución de clases
            fig = create_class_distribution_plot(y_original)
            st.plotly_chart(fig)
            
            # Manejar desbalanceo de clases
            X, y = handle_data_balancing(X, y, random_state)
        
        # Obtener opciones de modelos
        model_options = get_model_options(problem_type)
        # Gestionar modelos seleccionados
        if 'selected_models' not in st.session_state:
            st.session_state.selected_models = []
    
        selected_models = st.multiselect(
            "Selecciona los modelos a entrenar:",
            list(model_options.keys()),
            default=st.session_state.selected_models
        )
        st.session_state.selected_models = selected_models
    
        if not selected_models:
            st.warning("Por favor selecciona al menos un modelo para entrenar.")
            return
    
        # Configurar re-entrenamiento
        if st.button("Reentrenar Modelos"):
            st.session_state.retrain_models = True
        else:
            # Solo establecer a False si no está ya en sesión
            if 'retrain_models' not in st.session_state:
                st.session_state.retrain_models = False
    
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state,
            stratify=y if problem_type == 'classification' else None
        )
    
        # Crear columnas para mostrar resultados de modelos
        cols = st.columns(len(selected_models))
        
        # Entrenar y mostrar resultados de cada modelo
        for i, model_name in enumerate(selected_models):
            with cols[i]:
                st.write(f"### {model_name}")
                
                # Verificar si el modelo ya está entrenado y si no se solicita reentrenamiento
                if (model_name not in st.session_state.trained_models) or st.session_state.retrain_models:
                    # Entrenar modelo
                    trained_model = train_model_pipeline(
                        X_train=X_train,
                        y_train=y_train,
                        model_config=model_options[model_name],
                        X_test=X_test,
                        y_test=y_test,
                        cv=st.session_state.n_folds,
                        scoring=None,
                        random_state=random_state,  # Pasar random_state
                        n_jobs=-1,     # Para usar todos los núcleos disponibles
                        verbose=1
                    )
                    
                    # Almacenar el modelo entrenado en session_state
                    if 'trained_models' not in st.session_state:
                        st.session_state.trained_models = {}
                    st.session_state.trained_models[model_name] = trained_model
                else:
                    # Reutilizar el modelo ya entrenado
                    trained_model = st.session_state.trained_models[model_name]
                
                # Mostrar resultados del modelo
                show_model_results(
                    model_name, 
                    problem_type, 
                    y_test, 
                    cols[i],
                    trained_model
                )
    
    except Exception as e:
        st.error(f"Error inesperado: {str(e)}")


def show_model_results(model_name, problem_type, y_test, col, trained_model):
    """
    Mostrar resultados detallados de un modelo entrenado
    
    Args:
        model_name (str): Nombre del modelo
        problem_type (str): Tipo de problema
        y_test (pd.Series): Datos de prueba
        col (streamlit.delta_generator.DeltaGenerator): Columna de Streamlit
        trained_model (dict): Resultados del entrenamiento
    """
    with col:
        # Verificar si el modelo está en la sesión de modelos entrenados
        if model_name in st.session_state.trained_models:
            results = st.session_state.trained_models[model_name]
            
            # Verificar si hubo un error durante el entrenamiento
            if 'error' in results:
                st.error(results['error'])
                return
            
            # Mostrar métricas de rendimiento
            if 'training_time' in results:
                st.success(f"¡Entrenamiento completado en {results['training_time']:.2f} segundos!")
            st.write("Mejores parámetros:", results['best_params'])
            
            # Métricas específicas según el tipo de problema
            if problem_type == 'classification':
                st.write("Accuracy:", results.get('test_accuracy', 'N/A'))
                st.text("Reporte de clasificación:")
                st.text(pd.DataFrame(results.get('classification_report', {})).transpose().to_string())
            else:
                st.write("R² Score:", results.get('test_r2', 'N/A'))
                st.write("MSE:", results.get('test_mse', 'N/A'))
                st.write("RMSE:", results.get('test_rmse', 'N/A'))
                st.write("MAE:", results.get('test_mae', 'N/A'))
    
            # Sección de explicación de parámetros con Gemini
            st.write("---")
            st.write("### Explicación de Parámetros")
            
            # Verificar disponibilidad de API key de Gemini
            has_api_key = 'gemini_api_key' in st.session_state and st.session_state.gemini_api_key
            
            if not has_api_key:
                st.warning("Configure su API key de Gemini en la sección superior izquierda para usar la explicación automática de los parámetros.")
            
            # Inicializar el explainer si no lo has hecho ya
            if 'explainer' not in st.session_state:
                st.session_state.explainer = initialize_gemini_explainer()
            
            explainer = st.session_state.explainer
            
            # Inicializar explicaciones en el estado de la sesión
            if 'model_explanations' not in st.session_state:
                st.session_state.model_explanations = {}
            
            # Botón para generar explicación
            explain_button = st.button(
                "Explicar Parámetros",
                disabled=not has_api_key,
                key=f"explain_{model_name}"
            )
            
            # Mostrar explicación existente si está disponible
            if model_name in st.session_state.model_explanations:
                st.markdown(st.session_state.model_explanations[model_name])
            
        # Inicializar el explainer solo cuando se necesite
        if explain_button and has_api_key:
            explainer = initialize_gemini_explainer()
            if explainer:  # Verificar que el explainer se inicializó correctamente
                try:
                    with st.spinner("Generando explicación..."):
                        model_info = {
                            'name': model_name,
                            'problem_type': problem_type,
                            'hyperparameters': results['best_params'],
                            'performance_metric': results.get('test_accuracy', results.get('test_r2', 'N/A')),
                            'additional_metrics': results.get('additional_metrics', 'N/A')
                        }

                        explanation = explainer.generate_model_explanation(model_info)
                        
                        # Almacenar explicación
                        st.session_state.model_explanations[model_name] = explanation
                        
                        # Mostrar explicación
                        st.markdown(explanation)
                except Exception as e:
                    st.error(f"Error al generar la explicación: {str(e)}")
            else:
                st.error("No se pudo inicializar el explicador de Gemini")

        # Sección de análisis SHAP
        st.write("---")
        st.write("### Análisis SHAP")
        
        if st.button("Mostrar Análisis SHAP", key=f"shap_button_{model_name}"):
            try:
                # Obtener datos preparados
                X = st.session_state.prepared_data[st.session_state.feature_cols]
                
                # Crear dashboard de análisis SHAP
                create_shap_analysis_dashboard(
                    results['best_model'],  # Usar el mejor modelo
                    X, 
                    problem_type
                )
            except Exception as e:
                st.error(f"Error en el análisis SHAP: {str(e)}")

        # Sección de descarga del modelo
        st.write("---")
        st.write("### Descarga del modelo")
        
        # Generar nombre de archivo
        model_file_key = f"model_file_{model_name}"
        if model_file_key not in st.session_state:
            st.session_state[model_file_key] = f"{model_name.lower().replace(' ', '_')}_{int(time.time())}.pkl"
        
        # Input para nombre de archivo
        model_name_input = st.text_input(
            "Nombre del archivo:",
            value=st.session_state[model_file_key],
            key=f"name_input_{model_name}"
        )
        
        # Botón de descarga
        model_buffer = io.BytesIO()
        pickle.dump(results['best_model'], model_buffer)  # Guardar el mejor modelo
        model_buffer.seek(0)
        
        download_key = f"download_{model_name}"
        st.download_button(
            label="Descargar Modelo",
            data=model_buffer,
            file_name=model_name_input,
            mime="application/octet-stream",
            key=download_key
        )
