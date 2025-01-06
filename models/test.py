# test.py - Módulo para models
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error, 
    r2_score, 
    mean_absolute_percentage_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.preprocessing import LabelEncoder
import io

class ModelTester:
    def __init__(self, model, X, y, problem_type):
        self.model = model
        self.X = X
        self.y = y
        self.problem_type = problem_type
        self.label_encoder = None

    def _prepare_data(self, test_size=0.2, random_state=42):
        """Preparar datos para prueba"""
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, 
            test_size=test_size, 
            random_state=random_state,
            stratify=self.y if self.problem_type == 'classification' else None
        )
        return X_train, X_test, y_train, y_test

    def _encode_target(self, y):
        """Codificar variable objetivo para clasificación"""
        if self.problem_type == 'classification':
            self.label_encoder = LabelEncoder()
            return self.label_encoder.fit_transform(y)
        return y

    def evaluate_regression(self, X_test, y_test):
        """Evaluar modelo de regresión"""
        y_pred = self.model.predict(X_test)
        
        metrics = {
            'MSE': mean_squared_error(y_test, y_pred),
            'R² Score': r2_score(y_test, y_pred),
            'MAPE': mean_absolute_percentage_error(y_test, y_pred) * 100
        }
        
        return metrics, y_pred

    def evaluate_classification(self, X_test, y_test):
        """Evaluar modelo de clasificación"""
        y_test_encoded = self._encode_target(y_test)
        y_pred = self.model.predict(X_test)
        
        metrics = {
            'Accuracy': accuracy_score(y_test_encoded, y_pred),
            'Precision': precision_score(y_test_encoded, y_pred, average='weighted'),
            'Recall': recall_score(y_test_encoded, y_pred, average='weighted'),
            'F1 Score': f1_score(y_test_encoded, y_pred, average='weighted')
        }
        
        return metrics, y_pred

    def plot_regression_results(self, y_test, y_pred):
        """Crear gráfico de resultados de regresión"""
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=y_test, y=y_pred, 
            mode='markers', 
            name='Predicciones vs Valores Reales'
        ))
        fig.add_trace(go.Scatter(
            x=[y_test.min(), y_test.max()], 
            y=[y_test.min(), y_test.max()], 
            mode='lines', 
            name='Línea Perfecta',
            line=dict(color='red', dash='dash')
        ))
        fig.update_layout(
            title='Predicciones vs Valores Reales',
            xaxis_title='Valores Reales',
            yaxis_title='Predicciones'
        )
        return fig

    def plot_classification_results(self, y_test, y_pred):
        """Crear matriz de confusión para clasificación"""
        cm = confusion_matrix(
            self._encode_target(y_test), 
            y_pred
        )
        
        fig = px.imshow(
            cm, 
            labels=dict(x="Predicción", y="Real"),
            x=[str(c) for c in self.label_encoder.classes_] if self.label_encoder else None,
            y=[str(c) for c in self.label_encoder.classes_] if self.label_encoder else None,
            title="Matriz de Confusión"
        )
        return fig

def load_model(uploaded_file):
    """Cargar modelo desde archivo pickle"""
    try:
        with uploaded_file as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None
    
def get_model_features(model):
    """Extract feature names from the model if available."""
    if hasattr(model, 'feature_names_in_'):
        return list(model.feature_names_in_)
    return None

def align_features(X, model_features):
    """Align input features with model's expected features."""
    if model_features is None:
        return X
    
    # Create a new DataFrame with the correct features in the correct order
    missing_cols = set(model_features) - set(X.columns)
    extra_cols = set(X.columns) - set(model_features)
    
    if missing_cols:
        st.warning(f"Missing features: {missing_cols}. These will need to be provided.")
        return None
    
    if extra_cols:
        st.warning(f"Extra features detected: {extra_cols}. These will be ignored.")
    
    return X[model_features]

def determine_problem_type(model):
    """Determine if the model is for classification or regression."""
    class_methods = ['predict_proba', 'classes_']
    return 'classification' if any(hasattr(model, method) for method in class_methods) else 'regression'

def generate_model_explanation(model, metrics, problem_type):
    """Generar explicación del modelo usando Gemini"""
    try:
        genai.configure(api_key=st.session_state.get('gemini_api_key'))
        model_ai = genai.GenerativeModel('gemini-1.5-flash')

        metrics_text = "\n".join([f"{k}: {v}" for k, v in metrics.items()])
        
        prompt = f"""Analiza los siguientes resultados de un modelo de {problem_type}:

        Métricas de Rendimiento:
        {metrics_text}

        Proporciona:
        1. Interpretación de las métricas
        2. Fortalezas y debilidades del modelo
        3. Posibles mejoras o alternativas
        4. Contexto práctico de estos resultados
        """

        response = model_ai.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Error generando explicación: {e}")
        return "No se pudo generar la explicación."

def show_test():
    st.title("Prueba de Modelo")

    # Cargar modelo
    uploaded_model = st.file_uploader(
        "Cargar modelo entrenado", 
        type=['pkl']
    )

    if not uploaded_model:
        st.warning("Por favor, cargue un modelo entrenado")
        return

    # Cargar datos preparados
    if 'prepared_data' not in st.session_state:
        st.warning("No hay datos preparados. Por favor, prepare los datos primero.")
        return

    data = st.session_state.prepared_data

    # Selección de características y objetivo
    st.subheader("Configuración de Prueba")
    
    # Columnas numéricas
    model_features = get_model_features(uploaded_model)
    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    if model_features:
        # Pre-select features that match the model's expected features
        default_features = [col for col in model_features if col in numeric_cols]
        feature_cols = st.multiselect(
            "Seleccionar variables predictoras (X):",
            numeric_cols,
            default=default_features
        )
    else:
        feature_cols = st.multiselect(
            "Seleccionar variables predictoras (X):",
            numeric_cols,
            default=st.session_state.get('feature_cols', [])
        )
    
    available_targets = [col for col in data.columns if col not in feature_cols]
    target_col = st.selectbox(
        "Seleccionar variable objetivo (y):",
        available_targets,
        index=available_targets.index(st.session_state.get('target_col', available_targets[0]))
        if st.session_state.get('target_col') in available_targets else 0
    )

    if not feature_cols or not target_col:
        st.warning("Seleccione variables predictoras y objetivo")
        return

    # Cargar modelo
    model = load_model(uploaded_model)
    if not model:
        return

    # Preparar datos
    X = data[feature_cols]
    y = data[target_col]

    # Determinar tipo de problema
    problem_type = 'classification' if y.dtype == 'object' or y.nunique() <= 10 else 'regression'
    st.write(f"Tipo de problema detectado: {problem_type}")

    # Opciones de prueba
    test_size = st.slider(
        "Tamaño del conjunto de prueba", 
        0.1, 0.5, 0.2
    )

    # Probar modelo
    if 'model_evaluated' not in st.session_state:
        st.session_state.model_evaluated = False

    if st.button("Evaluar Modelo"):
        # Crear tester
        model_tester = ModelTester(model, X, y, problem_type)
        
        # Preparar datos
        X_train, X_test, y_train, y_test = model_tester._prepare_data(test_size)

        # Evaluar modelo según el tipo de problema
        if problem_type == 'regression':
            metrics, y_pred = model_tester.evaluate_regression(X_test, y_test)
            
            # Métricas de rendimiento
            st.subheader("Métricas de Rendimiento")
            col1, col2, col3 = st.columns(3)
            col1.metric("MSE", f"{metrics['MSE']:.4f}")
            col2.metric("R² Score", f"{metrics['R² Score']:.4f}")
            col3.metric("MAPE", f"{metrics['MAPE']:.2f}%")

            # Visualización de resultados
            st.subheader("Visualización de Resultados")
            fig = model_tester.plot_regression_results(y_test, y_pred)
            st.plotly_chart(fig, use_container_width=True)

        else:  # Clasificación
            metrics, y_pred = model_tester.evaluate_classification(X_test, y_test)
            
            # Métricas de rendimiento
            st.subheader("Métricas de Rendimiento")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
            col2.metric("Precision", f"{metrics['Precision']:.4f}")
            col3.metric("Recall", f"{metrics['Recall']:.4f}")
            col4.metric("F1 Score", f"{metrics['F1 Score']:.4f}")

            # Matriz de confusión
            st.subheader("Matriz de Confusión")
            fig = model_tester.plot_classification_results(y_test, y_pred)
            st.plotly_chart(fig)

            # Reporte de clasificación
            st.subheader("Reporte de Clasificación")
            st.text(classification_report(
                model_tester._encode_target(y_test), 
                y_pred
            ))
        
        # Guardar métricas en session state
        st.session_state.metrics = metrics
        st.session_state.model_evaluated = True

    # Explicación del modelo con Gemini (fuera del if anterior)
    st.subheader("Análisis de Resultados")
    if st.session_state.get('gemini_api_key'):
        if st.button("Generar Explicación Detallada", disabled=not st.session_state.model_evaluated, help="Evalúa el modelo primero"):
            with st.spinner("Generando explicación..."):
                explanation = generate_model_explanation(
                    model, st.session_state.metrics, problem_type
                )
                st.markdown(explanation)
    else:
        st.warning("Configure la API key de Gemini para obtener explicaciones detalladas")

        # Predicciones de ejemplo
        st.subheader("Predicciones de Ejemplo")
        num_samples = st.slider(
            "Número de muestras a mostrar", 
            5, 50, 10
        )

        # Seleccionar muestras aleatorias
        sample_indices = np.random.choice(
            len(X_test), 
            min(num_samples, len(X_test)), 
            replace=False
        )
        sample_X = X_test.iloc[sample_indices]
        sample_y_true = y_test.iloc[sample_indices]
        sample_y_pred = model.predict(sample_X)

        # Crear DataFrame de comparación
        comparison_df = pd.DataFrame({
            'Características': [
                ', '.join([f"{col}: {val}" for col, val in row.items()]) 
                for _, row in sample_X.iterrows()
            ],
            'Valor Real': sample_y_true,
            'Predicción': sample_y_pred,
            'Error Absoluto' if problem_type == 'regression' 
            else 'Predicción Correcta': 
                np.abs(sample_y_true - sample_y_pred) if problem_type == 'regression' 
                else (sample_y_true == sample_y_pred)
        })
        
        st.dataframe(comparison_df)

        # Opciones de descarga
        st.subheader("Descargar Resultados")
        
        # Guardar métricas
        metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Valor'])
        
        # Selector de formato
        download_format = st.selectbox(
            "Seleccionar formato de descarga", 
            ["CSV", "Excel"]
        )

        if download_format == "CSV":
            csv_data = metrics_df.to_csv().encode('utf-8')
            st.download_button(
                label="Descargar Métricas (CSV)", 
                data=csv_data, 
                file_name="model_metrics.csv", 
                mime="text/csv"
            )
        else:
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                metrics_df.to_excel(writer, index=True, sheet_name='Métricas')
                comparison_df.to_excel(writer, index=False, sheet_name='Predicciones')
            excel_buffer.seek(0)
            st.download_button(
                label="Descargar Resultados (Excel)", 
                data=excel_buffer, 
                file_name="model_results.xlsx", 
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

def main():
    """Función principal para ejecutar la página de prueba de modelos"""
    show_test()

if __name__ == "__main__":
    main()
