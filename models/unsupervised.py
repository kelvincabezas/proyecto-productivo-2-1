# unsupervised.py - Módulo para models
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import google.generativeai as genai
import umap

class UnsupervisedAnalyzer:
    def __init__(self, data):
        self.data = data
        self.scaler = StandardScaler()

    def preprocess_data(self, feature_cols):
        """Escalar datos seleccionados"""
        X = self.data[feature_cols]
        return self.scaler.fit_transform(X)

    def perform_kmeans(self, X_scaled, n_clusters):
        """Realizar clustering K-Means"""
        kmeans = KMeans(
            n_clusters=n_clusters, 
            random_state=42, 
            n_init=10
        )
        clusters = kmeans.fit_predict(X_scaled)

        # Calcular métricas
        metrics = {
            'Silhouette Score': silhouette_score(X_scaled, clusters),
            'Calinski-Harabasz Score': calinski_harabasz_score(X_scaled, clusters),
            'Davies-Bouldin Score': davies_bouldin_score(X_scaled, clusters)
        }

        return {
            'clusters': clusters,
            'metrics': metrics,
            'centroids': kmeans.cluster_centers_
        }

    def perform_dbscan(self, X_scaled, eps, min_samples):
        """Realizar clustering DBSCAN"""
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(X_scaled)

        # Calcular métricas
        unique_clusters = np.setdiff1d(np.unique(clusters), [-1])
        metrics = {
            'Noise Points': np.sum(clusters == -1),
            'Number of Clusters': len(unique_clusters)
        }

        # Solo calcular métricas si hay clusters válidos
        if len(unique_clusters) > 0:
            non_noise_mask = clusters != -1
            metrics.update({
                'Silhouette Score': silhouette_score(X_scaled[non_noise_mask], clusters[non_noise_mask]),
                'Calinski-Harabasz Score': calinski_harabasz_score(X_scaled[non_noise_mask], clusters[non_noise_mask]),
                'Davies-Bouldin Score': davies_bouldin_score(X_scaled[non_noise_mask], clusters[non_noise_mask])
            })
        else:
            metrics.update({
                'Silhouette Score': None,
                'Calinski-Harabasz Score': None,
                'Davies-Bouldin Score': None
            })

        return {
            'clusters': clusters,
            'metrics': metrics
        }

    def perform_hierarchical_clustering(self, X_scaled, n_clusters):
        """Realizar clustering jerárquico"""
        hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
        clusters = hierarchical.fit_predict(X_scaled)

        # Calcular métricas
        metrics = {
            'Silhouette Score': silhouette_score(X_scaled, clusters),
            'Calinski-Harabasz Score': calinski_harabasz_score(X_scaled, clusters),
            'Davies-Bouldin Score': davies_bouldin_score(X_scaled, clusters)
        }

        return {
            'clusters': clusters,
            'metrics': metrics
        }

    def perform_dimensionality_reduction(self, X_scaled, method='PCA', n_components=2):
        """Realizar reducción de dimensionalidad"""
        if method == 'PCA':
            reducer = PCA(n_components=n_components)
            reduced_data = reducer.fit_transform(X_scaled)
            return {
                'reduced_data': reduced_data,
                'explained_variance': reducer.explained_variance_ratio_
            }
        elif method == 't-SNE':
            reducer = TSNE(n_components=n_components, random_state=42)
            reduced_data = reducer.fit_transform(X_scaled)
            return {
                'reduced_data': reduced_data
            }
        elif method == 'UMAP':
            reducer = umap.UMAP(n_components=n_components, random_state=42)
            reduced_data = reducer.fit_transform(X_scaled)
            return {
                'reduced_data': reduced_data
            }

def generate_method_explanation(method, parameters, metrics):
    """Generar explicación del método usando Gemini"""
    try:
        genai.configure(api_key=st.session_state.gemini_api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')

        # Preparar prompt basado en el método
        prompt = f"""Explica detalladamente el método de análisis no supervisado: {method}

        Parámetros:
        {', '.join([f"{k}: {v}" for k, v in parameters.items()])}

        Métricas:
        {', '.join([f"{k}: {v}" for k, v in metrics.items()])}

        En tu explicación, incluye:
        1. Objetivo principal del método
        2. Cómo funciona el algoritmo
        3. Interpretación de los parámetros
        4. Significado de las métricas
        5. Casos de uso recomendados"""

        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error al generar explicación: {str(e)}"

def visualize_clustering(X_scaled, clusters, method_name, n_components=2):
    """Visualización de clustering"""
    reducer = PCA(n_components=n_components)
    X_reduced = reducer.fit_transform(X_scaled)
    
    if n_components == 2:
        fig = px.scatter(
            x=X_reduced[:, 0], 
            y=X_reduced[:, 1],
            color=clusters.astype(str),
            title=f'Clustering {method_name} - Visualización PCA',
            labels={'x': 'PCA Componente 1', 'y': 'PCA Componente 2'}
        )
    else:
        fig = go.Figure(data=[
            go.Scatter3d(
                x=X_reduced[:, 0],
                y=X_reduced[:, 1],
                z=X_reduced[:, 2],
                mode='markers',
                marker=dict(
                    size=5,
                    color=clusters,
                    colorscale='Viridis',
                    opacity=0.8
                )
            )
        ])
        fig.update_layout(
            title=f'Clustering {method_name} - Visualización 3D',
            scene=dict(
                xaxis_title='PCA 1',
                yaxis_title='PCA 2', 
                zaxis_title='PCA 3'
            )
        )
    
    return fig

def show_unsupervised_analysis():
    st.title("Análisis No Supervisado")

    # Verificar datos preparados
    if 'prepared_data' not in st.session_state or st.session_state.prepared_data is None:
        st.warning("Por favor, carga y prepara tus datos primero")
        return

    # Obtener datos
    data = st.session_state.prepared_data

    # Seleccionar columnas numéricas
    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()

    if not numeric_cols:
        st.error("No hay variables numéricas para análisis no supervisado")
        return

    # Selección de características
    feature_cols = st.multiselect(
        "Seleccionar Variables para Análisis", 
        numeric_cols, 
        default=numeric_cols[:min(5, len(numeric_cols))]
    )

    if not feature_cols:
        st.warning("Selecciona al menos una variable")
        return

    # Inicializar analizador
    analyzer = UnsupervisedAnalyzer(data)
    X_scaled = analyzer.preprocess_data(feature_cols)

    # Selección de métodos
    methods = st.multiselect(
        "Seleccionar Métodos de Análisis",
        [
            "K-Means", 
            "DBSCAN", 
            "Clustering Jerárquico", 
            "PCA", 
            "t-SNE", 
            "UMAP"
        ]
    )

    # Contenedor para resultados
    results = {}

    # Columnas para visualización
    if methods:
        cols = st.columns(len(methods))

        for i, method in enumerate(methods):
            with cols[i]:
                st.subheader(method)

                # Parámetros específicos por método
                if method == "K-Means":
                    n_clusters = st.slider(
                        "Número de Clusters", 
                        min_value=2, 
                        max_value=10, 
                        value=3, 
                        key=f"kmeans_clusters_{i}"
                    )
                    result = analyzer.perform_kmeans(X_scaled, n_clusters)
                    results['K-Means'] = result

                    # Visualización
                    fig = visualize_clustering(X_scaled, result['clusters'], method)
                    st.plotly_chart(fig)

                    # Métricas
                    st.write("Métricas:")
                    for metric, value in result['metrics'].items():
                        st.metric(metric, f"{value:.4f}")

                    # Explicación con Gemini
                    if st.session_state.get('gemini_api_key'):
                        explanation = generate_method_explanation(
                            method, 
                            {'Número de Clusters': n_clusters}, 
                            result['metrics']
                        )
                        with st.expander("Explicación del Método"):
                            st.markdown(explanation)

                elif method == "DBSCAN":
                    eps = st.slider(
                        "Epsilon", 
                        min_value=0.1, 
                        max_value=2.0, 
                        value=0.5, 
                        key=f"dbscan_eps_{i}"
                    )
                    min_samples = st.slider(
                        "Mínimo de Muestras", 
                        min_value=2, 
                        max_value=20, 
                        value=5, 
                        key=f"dbscan_min_samples_{i}"
                    )
                    result = analyzer.perform_dbscan(X_scaled, eps, min_samples)
                    results['DBSCAN'] = result

                    # Visualización
                    fig = visualize_clustering(X_scaled, result['clusters'], method)
                    st.plotly_chart(fig)

                    # Métricas
                    st.write("Métricas:")
                    for metric, value in result['metrics'].items():
                        st.metric(metric, str(value))

                    # Explicación con Gemini
                    if st.session_state.get('gemini_api_key'):
                        explanation = generate_method_explanation(
                            method, 
                            {
                                'Epsilon': eps, 
                                'Mínimo de Muestras': min_samples
                            }, 
                            result['metrics']
                        )
                        with st.expander("Explicación del Método"):
                            st.markdown(explanation)

                elif method == "Clustering Jerárquico":
                    n_clusters = st.slider(
                        "Número de Clusters", 
                        min_value=2, 
                        max_value=10, 
                        value=3, 
                        key=f"hierarchical_clusters_{i}"
                    )
                    result = analyzer.perform_hierarchical_clustering(X_scaled, n_clusters)
                    results['Clustering Jerárquico'] = result

                    # Visualización
                    fig = visualize_clustering(X_scaled, result['clusters'], method)
                    st.plotly_chart(fig)

                    # Métricas
                    st.write("Métricas:")
                    for metric, value in result['metrics'].items():
                        st.metric(metric, f"{value:.4f}")

                    # Explicación con Gemini
                    if st.session_state.get('gemini_api_key'):
                        explanation = generate_method_explanation(
                            method, 
                            {'Número de Clusters': n_clusters}, 
                            result['metrics']
                        )
                        with st.expander("Explicación del Método"):
                            st.markdown(explanation)

                elif method in ["PCA", "t-SNE", "UMAP"]:
                    n_components = st.slider(
                        "Número de Componentes", 
                        min_value=2, 
                        max_value=3, 
                        value=2, 
                        key=f"{method}_components_{i}"
                    )
                    result = analyzer.perform_dimensionality_reduction(
                        X_scaled, 
                        method=method, 
                        n_components=n_components
                    )
                    results[method] = result

                    # Visualización de reducción de dimensionalidad
                    fig = px.scatter(
                        x=result['reduced_data'][:, 0],
                        y=result['reduced_data'][:, 1],
                        title=f'Reducción de Dimensionalidad - {method}'
                    )
                    st.plotly_chart(fig)

                    # Varianza explicada para PCA
                    if method == 'PCA':
                        st.write("Varianza Explicada:")
                        varianza_df = pd.DataFrame({
                            'Componente': range(1, len(result['explained_variance']) + 1),
                            'Varianza Explicada (%)': result['explained_variance'] * 100,
                            'Varianza Acumulada (%)': np.cumsum(result['explained_variance']) * 100
                        })
                        st.dataframe(varianza_df)

                    # Explicación con Gemini
                    if st.session_state.get('gemini_api_key'):
                        explanation = generate_method_explanation(
                            method, 
                            {'Número de Componentes': n_components}, 
                            {}
                        )
                        with st.expander("Explicación del Método"):
                            st.markdown(explanation)

        # Exportar resultados
        if st.button("Exportar Resultados"):
            export_data = []
            for method, result in results.items():
                method_data = {
                    'Método': method,
                    'Variables': ', '.join(feature_cols)
                }
                
                # Agregar métricas si están disponibles
                if 'metrics' in result:
                    method_data.update(result['metrics'])
                
                export_data.append(method_data)
            
            export_df = pd.DataFrame(export_data)
            csv = export_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Descargar Resultados",
                data=csv,
                file_name="unsupervised_analysis_results.csv",
                mime="text/csv",
                key="download_unsupervised_results"
            )

def show_unsupervised():
    """Función principal para mostrar la página de análisis no supervisado"""
    st.title("🔍 Análisis No Supervisado")

    # Verificar datos preparados
    if 'prepared_data' not in st.session_state or st.session_state.prepared_data is None:
        st.warning("Por favor, carga y prepara tus datos primero en la página de Preparación.")
        return

    # Obtener datos preparados
    data = st.session_state.prepared_data

    # Sección de selección de variables
    st.header("Configuración del Análisis")

    # Seleccionar columnas numéricas
    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()

    if not numeric_cols:
        st.error("No hay variables numéricas disponibles para realizar análisis no supervisado.")
        return

    # Selección de características
    st.subheader("Selección de Variables")
    feature_cols = st.multiselect(
        "Selecciona las variables para el análisis", 
        numeric_cols, 
        default=numeric_cols[:min(5, len(numeric_cols))]
    )

    if not feature_cols:
        st.warning("Por favor, selecciona al menos una variable.")
        return

    # Inicializar analizador
    analyzer = UnsupervisedAnalyzer(data)
    X_scaled = analyzer.preprocess_data(feature_cols)

    # Sección de métodos de análisis
    st.header("Métodos de Análisis")

    # Selección de métodos
    metodos = st.multiselect(
        "Elige los métodos de análisis no supervisado",
        [
            "K-Means", 
            "DBSCAN", 
            "Clustering Jerárquico", 
            "Análisis de Componentes Principales (PCA)", 
            "t-SNE", 
            "UMAP"
        ]
    )

    # Contenedor de resultados
    resultados = {}

    # Procesamiento de métodos seleccionados
    if metodos:
        # Crear columnas dinámicamente
        cols = st.columns(len(metodos))

        for i, metodo in enumerate(metodos):
            with cols[i]:
                st.subheader(metodo)

                # Parámetros específicos por método
                if metodo == "K-Means":
                    n_clusters = st.slider(
                        "Número de Clusters", 
                        min_value=2, 
                        max_value=10, 
                        value=3, 
                        key=f"kmeans_clusters_{i}"
                    )
                    
                    # Realizar K-Means
                    resultado = analyzer.perform_kmeans(X_scaled, n_clusters)
                    resultados[metodo] = resultado

                    # Visualización
                    fig = visualize_clustering(X_scaled, resultado['clusters'], metodo)
                    st.plotly_chart(fig)

                    # Mostrar métricas
                    st.subheader("Métricas")
                    for metrica, valor in resultado['metrics'].items():
                        st.metric(metrica, f"{valor:.4f}")

                    # Explicación con Gemini
                    if st.session_state.get('gemini_api_key'):
                        explicacion = generate_method_explanation(
                            metodo, 
                            {'Número de Clusters': n_clusters}, 
                            resultado['metrics']
                        )
                        with st.expander("Explicación del Método"):
                            st.markdown(explicacion)

                elif metodo == "DBSCAN":
                    eps = st.slider(
                        "Epsilon", 
                        min_value=0.1, 
                        max_value=2.0, 
                        value=0.5, 
                        key=f"dbscan_eps_{i}"
                    )
                    min_samples = st.slider(
                        "Mínimo de Muestras", 
                        min_value=2, 
                        max_value=20, 
                        value=5, 
                        key=f"dbscan_min_samples_{i}"
                    )

                    # Realizar DBSCAN
                    resultado = analyzer.perform_dbscan(X_scaled, eps, min_samples)
                    resultados[metodo] = resultado

                    # Visualización
                    fig = visualize_clustering(X_scaled, resultado['clusters'], metodo)
                    st.plotly_chart(fig)

                    # Mostrar métricas
                    st.subheader("Métricas")
                    for metrica, valor in resultado['metrics'].items():
                        st.metric(metrica, str(valor))

                    # Explicación con Gemini
                    if st.session_state.get('gemini_api_key'):
                        explicacion = generate_method_explanation(
                            metodo, 
                            {
                                'Epsilon': eps, 
                                'Mínimo de Muestras': min_samples
                            }, 
                            resultado['metrics']
                        )
                        with st.expander("Explicación del Método"):
                            st.markdown(explicacion)

                # Continuar con los demás métodos de manera similar...

        # Sección de exportación de resultados
        if st.button("Exportar Resultados del Análisis"):
            # Crear DataFrame con resultados
            datos_exportacion = []
            for metodo, resultado in resultados.items():
                datos_metodo = {
                    'Método': metodo,
                    'Variables': ', '.join(feature_cols)
                }
                
                # Agregar métricas si están disponibles
                if 'metrics' in resultado:
                    datos_metodo.update(resultado['metrics'])
                
                datos_exportacion.append(datos_metodo)
            
            df_exportacion = pd.DataFrame(datos_exportacion)
            
            # Descargar CSV
            csv = df_exportacion.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Descargar Resultados",
                data=csv,
                file_name="analisis_no_supervisado.csv",
                mime="text/csv"
            )

# Función principal para ejecutar el análisis no supervisado
def main():
    show_unsupervised()

if __name__ == "__main__":
    main()