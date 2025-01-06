# shap_explainer.py - Módulo para utils
import streamlit as st
import pandas as pd
import numpy as np
import shap
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, Union, Optional
from sklearn.pipeline import Pipeline

class SHAPExplainer:
    """
    Clase para realizar explicaciones de modelos usando SHAP (SHapley Additive exPlanations)
    """
    def __init__(self, model, X: pd.DataFrame, problem_type: str = 'classification', explanation_method: str = 'auto'):
        """
        Inicializar el explicador SHAP

        Args:
            model: Modelo de machine learning entrenado
            X (pd.DataFrame): Datos de entrada para el modelo
            problem_type (str): Tipo de problema ('classification' o 'regression')
            explanation_method (str): Método de explicación ('auto', 'tree', 'linear', 'kernel')
        """
        self.model = model
        self.X = X
        self.problem_type = problem_type
        self.explanation_method = explanation_method
        self.explainer = self._create_explainer()
        self.X_sample = None  # Inicializar X_sample

    def _create_explainer(self):
        """
        Crear el explicador SHAP apropiado según el tipo de modelo y método seleccionado,
        manejando correctamente los Pipelines.

        Returns:
            Explainer de SHAP
        """
        try:
            # Si el modelo es un Pipeline, extraer el estimador final
            if isinstance(self.model, Pipeline):
                model = self.model.steps[-1][1]
            else:
                model = self.model

            # Crear el explicador usando el método seleccionado
            if self.explanation_method.lower() == 'tree':
                return shap.TreeExplainer(model)
            elif self.explanation_method.lower() == 'linear':
                return shap.LinearExplainer(model, self.X, feature_dependence="independent")
            elif self.explanation_method.lower() == 'kernel':
                return shap.KernelExplainer(model.predict, shap.sample(self.X, 100))
            else:
                # 'auto' o cualquier otro valor: usar shap.Explainer que selecciona automáticamente
                return shap.Explainer(model, self.X)

        except Exception as e:
            st.error(f"Error al crear explicador SHAP: {str(e)}")
            return None

    def compute_shap_values(self, X_sample: Optional[pd.DataFrame] = None, max_samples: int = 100):
        """
        Calcular valores SHAP

        Args:
            X_sample (pd.DataFrame, opcional): Muestra de datos para calcular SHAP
            max_samples (int): Número máximo de muestras a procesar

        Returns:
            Valores SHAP
        """
        try:
            # Usar muestra si no se proporciona
            if X_sample is None:
                X_sample = self.X.sample(n=min(max_samples, len(self.X)), random_state=42)

            # Almacenar el subconjunto de datos utilizado
            self.X_sample = X_sample

            # Asegurarse de que X_sample es 2D
            if X_sample.ndim != 2:
                raise ValueError(f"Debe pasar una entrada 2D a SHAP. La forma actual es {X_sample.shape}")

            # Calcular valores SHAP usando el explicador
            shap_values = self.explainer(X_sample)

            # Asegurarse de que shap_values es 2D
            if isinstance(shap_values.values, list):
                # Para clasificación multi-clase, promediar las contribuciones
                shap_values = np.mean([np.abs(s.values) for s in shap_values.values], axis=0)
            else:
                shap_values = np.abs(shap_values.values)

            return shap_values

        except Exception as e:
            st.error(f"Error al calcular valores SHAP: {str(e)}")
            return None

    def plot_summary(self, shap_values, title: str = "SHAP Summary Plot"):
        """
        Generar gráfico de resumen de valores SHAP

        Args:
            shap_values: Valores SHAP calculados
            title (str): Título del gráfico

        Returns:
            Figura de Plotly
        """
        try:
            feature_names = self.X.columns.tolist()

            # Calcular importancia de características
            feature_importance = np.mean(shap_values, axis=0)
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': feature_importance
            }).sort_values('importance', ascending=False)

            # Gráfico de barras de importancia
            fig = px.bar(
                importance_df, 
                x='importance', 
                y='feature', 
                orientation='h',
                title=title,
                labels={'importance': 'Importancia SHAP', 'feature': 'Características'}
            )

            return fig

        except Exception as e:
            st.error(f"Error al generar gráfico de resumen: {str(e)}")
            return None

    def plot_dependence(self, shap_values, feature_name: str):
        """
        Generar gráfico de dependencia para una característica

        Args:
            shap_values: Valores SHAP calculados
            feature_name (str): Nombre de la característica

        Returns:
            Figura de Plotly
        """
        try:
            feature_idx = self.X.columns.get_loc(feature_name)

            # Preparar datos usando el mismo subconjunto de datos utilizado para SHAP
            if self.X_sample is not None:
                x = self.X_sample.iloc[:, feature_idx]
            else:
                x = self.X.iloc[:, feature_idx]

            y = shap_values[:, feature_idx]

            # Verificar que las longitudes coincidan
            if len(x) != len(y):
                raise ValueError(f"Longitud de 'x' ({len(x)}) y 'y' ({len(y)}) no coinciden.")

            # Crear scatter plot
            fig = px.scatter(
                x=x, 
                y=y, 
                title=f'SHAP Dependence Plot - {feature_name}',
                labels={'x': feature_name, 'y': 'SHAP Value'}
            )

            return fig

        except Exception as e:
            st.error(f"Error al generar gráfico de dependencia: {str(e)}")
            return None

    def generate_feature_importance_report(self, shap_values) -> Dict[str, Any]:
        """
        Generar un informe detallado de importancia de características

        Args:
            shap_values: Valores SHAP calculados

        Returns:
            Diccionario con información de importancia de características
        """
        try:
            # Calcular importancia
            feature_importance = np.mean(shap_values, axis=0)

            # Crear DataFrame de importancia
            importance_df = pd.DataFrame({
                'feature': self.X.columns,
                'importance': feature_importance
            }).sort_values('importance', ascending=False)

            # Generar informe
            report = {
                'top_features': importance_df.head(5).to_dict('records'),
                'bottom_features': importance_df.tail(5).to_dict('records'),
                'total_features': len(importance_df),
                'max_importance': importance_df['importance'].max(),
                'min_importance': importance_df['importance'].min()
            }

            return report

        except Exception as e:
            st.error(f"Error al generar informe de importancia: {str(e)}")
            return {}

def create_shap_analysis_dashboard(model, X: pd.DataFrame, problem_type: str = 'classification'):
    """
    Crear un dashboard de análisis SHAP en Streamlit

    Args:
        model: Modelo de machine learning
        X (pd.DataFrame): Datos de entrada
        problem_type (str): Tipo de problema
    """
    st.title("🔍 Análisis de Explicabilidad SHAP")

    # Inicializar los valores SHAP en session_state si no existen
    if 'shap_explainer' not in st.session_state:
        # Parámetros por defecto
        explanation_method = 'auto'
        max_samples = 100

        # Crear y almacenar el explicador SHAP
        st.session_state.shap_explainer = SHAPExplainer(
            model=model,
            X=X,
            problem_type=problem_type,
            explanation_method=explanation_method
        )

        # Calcular y almacenar los valores SHAP
        st.session_state.shap_values = st.session_state.shap_explainer.compute_shap_values(
            max_samples=max_samples
        )

    shap_explainer = st.session_state.shap_explainer
    shap_values = st.session_state.shap_values

    if shap_values is None:
        st.error("No se pudieron calcular los valores SHAP")
        return

    # Pestañas para diferentes visualizaciones
    tab1, tab2, tab3, tab4 = st.tabs([
        "Resumen de Importancia", 
        "Dependencia de Características", 
        "Informe Detallado", 
        "Configuración Avanzada"
    ])

    with tab1:
        st.header("Resumen de Importancia de Características")

        # Gráfico de resumen
        summary_fig = shap_explainer.plot_summary(shap_values)
        if summary_fig:
            st.plotly_chart(summary_fig, use_container_width=True)

        # Selector de características para análisis detallado
        selected_feature = st.selectbox(
            "Seleccionar característica para análisis detallado",
            X.columns.tolist()
        )

        # Gráfico de dependencia para la característica seleccionada
        dependence_fig = shap_explainer.plot_dependence(shap_values, selected_feature)
        if dependence_fig:
            st.plotly_chart(dependence_fig, use_container_width=True)

    with tab2:
        st.header("Análisis de Dependencia de Características")

        # Matriz de correlación de valores SHAP
        shap_correlation = pd.DataFrame(shap_values).corr()

        # Heatmap de correlación de valores SHAP
        fig_corr = px.imshow(
            shap_correlation, 
            title="Correlación entre Valores SHAP de Características",
            labels=dict(x="Características", y="Características", color="Correlación")
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    with tab3:
        st.header("Informe Detallado de Importancia")

        # Generar informe de importancia de características
        importance_report = shap_explainer.generate_feature_importance_report(shap_values)

        # Mostrar características más importantes
        st.subheader("Top 5 Características Más Importantes")
        top_features_df = pd.DataFrame(importance_report.get('top_features', []))
        st.dataframe(top_features_df)

        # Visualización de características más importantes
        fig_top_features = px.bar(
            top_features_df, 
            x='importance', 
            y='feature',
            orientation='h',
            title="Top 5 Características por Importancia SHAP"
        )
        st.plotly_chart(fig_top_features, use_container_width=True)

        # Métricas de resumen
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total de Características", importance_report.get('total_features', 'N/A'))
        with col2:
            st.metric("Máxima Importancia", f"{importance_report.get('max_importance', 'N/A'):.4f}")
        with col3:
            st.metric("Mínima Importancia", f"{importance_report.get('min_importance', 'N/A'):.4f}")

    with tab4:
        st.header("Configuración Avanzada")

        # Controles de configuración
        st.subheader("Parámetros de Explicación")

        # Selector de método de explicación
        explanation_method = st.selectbox(
            "Método de Explicación",
            ["auto", "tree", "linear", "kernel"]
        )

        # Número de muestras para cálculo
        num_samples = st.slider(
            "Número de Muestras para Análisis",
            min_value=10,
            max_value=min(1000, len(X)),
            value=min(100, len(X))
        )

        # Botón para recalcular con nuevos parámetros
        if st.button("Recalcular SHAP"):
            with st.spinner("Recalculando valores SHAP..."):
                try:
                    # Crear y actualizar el explicador SHAP con nuevos parámetros
                    shap_explainer = SHAPExplainer(
                        model=model,
                        X=X,
                        problem_type=problem_type,
                        explanation_method=explanation_method
                    )
                    st.session_state.shap_explainer = shap_explainer

                    # Calcular y actualizar los valores SHAP
                    shap_values = shap_explainer.compute_shap_values(
                        max_samples=num_samples
                    )
                    st.session_state.shap_values = shap_values

                    st.success("Valores SHAP recalculados correctamente.")

                except Exception as e:
                    st.error(f"Error al recalcular SHAP: {str(e)}")

def validate_shap_compatibility(model):
    """
    Validar si un modelo es compatible con SHAP
    
    Args:
        model: Modelo de machine learning
    
    Returns:
        bool: True si es compatible, False en caso contrario
    """
    compatible_types = [
        'RandomForestClassifier',
        'RandomForestRegressor',
        'GradientBoostingClassifier',
        'GradientBoostingRegressor',
        'XGBClassifier',
        'XGBRegressor',
        'DecisionTreeClassifier',
        'DecisionTreeRegressor',
        'LogisticRegression',
        'LinearRegression'
    ]
    
    return any(
        comp_type in str(type(model).__name__) 
        for comp_type in compatible_types
    )

def generate_shap_documentation():
    """
    Generar documentación sobre el uso de SHAP
    
    Returns:
        str: Documentación en formato markdown
    """
    documentation = """
    ## 🔍 Explicabilidad de Modelos con SHAP

    ### ¿Qué es SHAP?
    SHAP (SHapley Additive exPlanations) es una metodología para explicar las predicciones 
    de modelos de machine learning basada en la teoría de juegos.

    ### Características Principales
    - Interpretación global y local de modelos
    - Calcula la contribución de cada característica a la predicción
    - Funciona con diferentes tipos de modelos

    ### Tipos de Visualizaciones
    1. **Summary Plot**: Importancia general de características
    2. **Dependence Plot**: Relación entre características y predicciones
    3. **Force Plot**: Contribución individual de características

    ### Limitaciones
    - Computacionalmente intensivo para grandes datasets
    - Puede ser lento con modelos complejos
    - Requiere comprensión estadística para interpretación precisa

    ### Mejores Prácticas
    - Usar como complemento, no como única fuente de verdad
    - Combinar con otras técnicas de explicabilidad
    - Interpretar en contexto del problema de negocio
    """
    return documentation

# Punto de entrada principal para pruebas
def main():
    import streamlit as st
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import load_iris
    
    # Cargar datos de ejemplo
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target
    
    # Entrenar modelo de ejemplo
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Crear dashboard de análisis SHAP
    create_shap_analysis_dashboard(model, X)

if __name__ == "__main__":
    main()