# upload.py - Módulo para datos
import streamlit as st
import pandas as pd
import numpy as np
import io
import requests
from typing import Optional, Dict, List
import importlib.util
import os
import plotly.express as px
from supabase import create_client
import re
from pygwalker.api.streamlit import StreamlitRenderer

# Importaciones específicas del proyecto
from utils.gemini_explainer import generate_dataset_explanation

# Ejemplos de datasets
DATASET_OPTIONS = {
    "Iris": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv",
    "Titanic": "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
    "Boston Housing": "https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/sklearn/datasets/data/boston_house_prices.csv",
    "Wine Quality": "https://raw.githubusercontent.com/uiuc-cse/data-fa14/master/data/wine.csv",
    "Diabetes": "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
}

def check_package(package_name: str) -> bool:
    """Verifica si un paquete está instalado"""
    return importlib.util.find_spec(package_name) is not None

def get_supported_formats() -> Dict[str, List[str]]:
    """Retorna un diccionario con los formatos soportados basado en las dependencias instaladas"""
    formats = {
        'CSV': ['csv'],
        'Excel': ['xls', 'xlsx', 'xlsm', 'xlsb', 'odf', 'ods', 'odt'],
        'JSON': ['json']
    }
    
    # Verificar soporte para parquet
    if check_package('pyarrow') or check_package('fastparquet'):
        formats['Parquet'] = ['parquet']
    
    # Verificar soporte para feather
    if check_package('pyarrow'):
        formats['Feather'] = ['feather']
    
    # Verificar soporte para HDF5
    if check_package('tables'):
        formats['HDF5'] = ['h5', 'hdf5']
    
    # Verificar soporte para SQLite
    if check_package('sqlite3'):
        formats['SQLite'] = ['db', 'sqlite', 'sqlite3']
    
    # Verificar soporte para Pickle
    formats['Pickle'] = ['pkl', 'pickle']
    
    # Verificar soporte para STATA
    if check_package('pandas.io.stata'):
        formats['STATA'] = ['dta']
    
    # Verificar soporte para SAS
    if check_package('pandas.io.sas'):
        formats['SAS'] = ['sas7bdat']
    
    return formats

def load_file(file_obj: io.BytesIO, file_format: str) -> Optional[pd.DataFrame]:
    """Carga un archivo en un DataFrame basado en su formato"""
    try:
        if file_format in ['csv']:
            return pd.read_csv(file_obj)
        elif file_format in ['xls', 'xlsx', 'xlsm', 'xlsb', 'odf', 'ods', 'odt']:
            return pd.read_excel(file_obj)
        elif file_format in ['json']:
            return pd.read_json(file_obj)
        elif file_format in ['parquet'] and (check_package('pyarrow') or check_package('fastparquet')):
            return pd.read_parquet(file_obj)
        elif file_format in ['feather'] and check_package('pyarrow'):
            return pd.read_feather(file_obj)
        elif file_format in ['h5', 'hdf5'] and check_package('tables'):
            return pd.read_hdf(file_obj)
        elif file_format in ['pkl', 'pickle']:
            return pd.read_pickle(file_obj)
        elif file_format in ['dta'] and check_package('pandas.io.stata'):
            return pd.read_stata(file_obj)
        elif file_format in ['sas7bdat'] and check_package('pandas.io.sas'):
            return pd.read_sas(file_obj)
        elif file_format in ['db', 'sqlite', 'sqlite3'] and check_package('sqlite3'):
            import sqlite3
            conn = sqlite3.connect(file_obj)
            tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
            if len(tables) > 0:
                table_name = st.selectbox("Selecciona una tabla:", tables['name'].tolist())
                return pd.read_sql_query(f"SELECT * FROM {table_name};", conn)
            else:
                st.error("No se encontraron tablas en la base de datos")
                return None
    except Exception as e:
        st.error(f"Error al cargar el archivo: {str(e)}")
        return None

def load_gsheet(sharing_link: str) -> pd.DataFrame:
    """Carga un Google Sheet como DataFrame usando su link de compartir"""
    sheet_export = sharing_link.replace("/edit?usp=sharing", "/export?format=csv")
    return pd.read_csv(sheet_export)

def convert_to_raw_github_url(url: str) -> str:
    """Convierte una URL de GitHub en su versión 'raw'"""
    # Patrón para URLs de GitHub
    github_pattern = r'https://github\.com/([^/]+/[^/]+)/blob/([^/]+/.*)'
    
    if match := re.match(github_pattern, url):
        # Construir la URL raw
        return f'https://raw.githubusercontent.com/{match.group(1)}/{match.group(2)}'
    return url

def load_url_file(url: str) -> Optional[pd.DataFrame]:
    """Carga un archivo desde una URL detectando automáticamente el formato"""
    try:
        # Convertir a URL raw si es una URL de GitHub
        raw_url = convert_to_raw_github_url(url)
        
        # Configurar headers para simular un navegador
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(raw_url, headers=headers, verify=True)
        if response.status_code != 200:
            raise Exception(f"Error al descargar el archivo (Status code: {response.status_code})")
        
        content = io.BytesIO(response.content)
        
        # Detectar formato basado en la extensión de la URL
        extension = url.split('.')[-1].lower()
        
        # Validar la extensión antes de procesar
        supported_formats = ['csv', 'xls', 'xlsx', 'xlsm', 'xlsb', 'odf', 'ods', 'odt', 
                           'json', 'parquet', 'feather', 'h5', 'hdf5', 'pkl', 'pickle',
                           'dta', 'sas7bdat', 'db', 'sqlite', 'sqlite3']
        
        if extension not in supported_formats:
            st.error(f"Formato de archivo no soportado: {extension}")
            return None
        
        # Verificar que el contenido descargado no esté vacío
        if len(response.content) == 0:
            raise Exception("El archivo descargado está vacío")
        
        # Para archivos Excel, usar directamente openpyxl
        if extension in ['xlsx', 'xlsm', 'xlsb']:
            return pd.read_excel(content, engine='openpyxl')
        
        return load_file(content, extension)
        
    except requests.exceptions.SSLError:
        st.error("Error de seguridad SSL al descargar el archivo. Intente con una URL diferente.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Error en la solicitud HTTP: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Error al cargar la URL: {str(e)}")
        return None
    
def show_supabase_setup_info():
    """Muestra información de configuración para Supabase"""
    
    setup_sql = """
create or replace function get_tables()
returns table (table_name text)
language sql
as $$
    select table_name::text
    from information_schema.tables
    where table_schema = 'public'
    and table_type = 'BASE TABLE';
$$;
"""
    
    with st.expander("ℹ️ Configuración de Supabase", expanded=False):
        st.markdown("""
        ### Pasos para configurar Supabase

        1. **Crear función RPC en Supabase:**
            - Ve al Editor SQL de Supabase
            - Copia y ejecuta el siguiente código:
        """)
        
        # Mostrar el SQL con botón de copiado
        st.code(setup_sql, language='sql')
        
        st.markdown("""
        2. **Verificar credenciales:**
            - URL del proyecto: `Settings -> API -> Project URL`
            - API Key: `Settings -> API -> Project API keys -> anon/public`
            
        3. **Permisos necesarios:**
            - La función necesita acceso a `information_schema.tables`
            - El usuario debe tener permisos para ejecutar la función RPC
            
        4. **Solución de problemas:**
            - Asegúrate de que existan tablas en el esquema público
            - Verifica que la base de datos esté activa
            - Confirma que las políticas de seguridad permitan el acceso
        """)

def get_supabase_tables(supabase_url: str, supabase_key: str) -> Optional[List[str]]:
    """Obtiene la lista de tablas disponibles en Supabase"""
    try:
        from supabase import create_client, Client
        
        # Crear cliente de Supabase
        supabase: Client = create_client(supabase_url, supabase_key)
        
        try:
            # Intenta primero usando RPC
            result = supabase.rpc('get_tables').execute()
            
            if hasattr(result, 'data') and result.data:
                tables = [table['table_name'] for table in result.data]
                if tables:
                    return sorted(tables)  # Ordenar las tablas alfabéticamente
        except Exception as rpc_error:
            st.warning(f"Método RPC falló: {str(rpc_error)}")
            
            try:
                # Si RPC falla, intenta con una consulta SQL directa
                result = supabase.from_('information_schema.tables')\
                    .select('table_name')\
                    .eq('table_schema', 'public')\
                    .eq('table_type', 'BASE TABLE')\
                    .execute()
                
                if hasattr(result, 'data') and result.data:
                    return sorted([table['table_name'] for table in result.data])
            except Exception as sql_error:
                st.warning(f"Consulta SQL directa falló: {str(sql_error)}")
                
                # Último intento usando postgREST
                try:
                    result = supabase.table('tables').select('*').execute()
                    if hasattr(result, 'data') and result.data:
                        return sorted([table['name'] for table in result.data])
                except Exception as postgrest_error:
                    st.error(f"Todos los métodos de consulta fallaron: {str(postgrest_error)}")
        
        st.warning("No se encontraron tablas en el esquema público")
        # Mostrar ayuda de configuración
        show_supabase_setup_info()
        return None
                
    except Exception as e:
        st.error(f"Error al conectar con Supabase: {str(e)}")
        st.write("Detalles del error:", str(e))
        # Mostrar ayuda de configuración
        show_supabase_setup_info()
        return None

def load_supabase_table(supabase_url: str, supabase_key: str, table_name: str) -> Optional[pd.DataFrame]:
    """Carga una tabla de Supabase como DataFrame"""
    try:
        from supabase import create_client, Client
        
        # Crear cliente de Supabase
        supabase: Client = create_client(supabase_url, supabase_key)
        
        # Realizar la consulta a la tabla
        response = supabase.table(table_name).select("*").execute()
        
        if hasattr(response, 'data'):
            df = pd.DataFrame(response.data)
            if not df.empty:
                return df
            else:
                st.warning(f"La tabla '{table_name}' está vacía")
                return None
        else:
            st.error("No se pudieron obtener datos de la tabla")
            return None
            
    except Exception as e:
        st.error(f"Error al cargar la tabla de Supabase: {str(e)}")
        st.write("Detalles del error:", str(e))
        return None
        
def show_upload():
    """Función principal para cargar y analizar datos"""
    st.subheader('Aprenda con sus datos')
    
    # Inicializar la variable de estado
    if 'er_data' not in st.session_state:
        st.session_state.er_data = None
    
    # Obtener formatos soportados
    SUPPORTED_FORMATS = get_supported_formats()
    accepted_extensions = [ext for formats in SUPPORTED_FORMATS.values() for ext in formats]
    
    # Mostrar formatos disponibles
    with st.expander("Ver formatos soportados"):
        for format_type, extensions in SUPPORTED_FORMATS.items():
            st.write(f"**{format_type}**: {', '.join(extensions)}")
    
    # Sección de Ejemplos Predeterminados
    st.markdown("#### 0. Ejemplos Predeterminados")
    selected_example = st.selectbox(
        "Selecciona un dataset de ejemplo", 
        list(DATASET_OPTIONS.keys()) + ["Ninguno"],
        index=len(DATASET_OPTIONS)  # Seleccionar "Ninguno" por defecto
    )
    
    if selected_example != "Ninguno":
        example_url = DATASET_OPTIONS[selected_example]
        if st.button(f"Cargar Dataset de {selected_example}"):
            try:
                with st.spinner(f"Cargando dataset {selected_example}..."):
                    df = load_url_file(example_url)
                    if df is not None:
                        st.session_state.er_data = df
                        st.success(f"Dataset {selected_example} cargado exitosamente")
            except Exception as e:
                st.error(f"Error al cargar el dataset de ejemplo: {str(e)}")
    
    # Secciones de carga de datos
    st.markdown("#### 1. Subir Archivo Local")
    data_file = st.file_uploader("Arrastra o selecciona tu archivo", type=accepted_extensions)

    if data_file:
        extension = data_file.name.split('.')[-1].lower()
        df = load_file(data_file, extension)
        if df is not None:
            st.session_state.er_data = df
            st.success(f"Archivo local cargado: {data_file.name}")

    # Carga desde Google Sheet
    st.markdown("#### 2. Cargar desde Google Sheet")
    sharing_link = st.text_input(
        "Link de Google Sheet:",
        placeholder="https://docs.google.com/spreadsheets/d/SHEET-ID/edit?usp=sharing"
    )
    if sharing_link and st.button("Cargar Sheet"):
        try:
            st.session_state.er_data = load_gsheet(sharing_link)
            st.success("Google Sheet cargado exitosamente")
        except Exception as e:
            st.error(f"Error al cargar el Google Sheet: {str(e)}")

    # Carga desde URL
    st.markdown("#### 3. Cargar desde URL")
    url = st.text_input(
        'URL del archivo:',
        placeholder='Ejemplo: https://ejemplo.com/datos.csv'
    )
    if url and st.button('Cargar URL'):
        df = load_url_file(url)
        if df is not None:
            st.session_state.er_data = df

    # Carga desde Supabase
    st.markdown("#### 4. Carga desde Supabase")
    
    # Verificar credenciales
    has_credentials = (
        'supabase_url' in st.session_state and 
        'supabase_key' in st.session_state and 
        st.session_state.supabase_url.strip() and 
        st.session_state.supabase_key.strip()
    )

    # Inicializar variables de estado
    if 'supabase_tables' not in st.session_state:
        st.session_state.supabase_tables = None
    if 'supabase_connected' not in st.session_state:
        st.session_state.supabase_connected = False

    status_container = st.empty()

    if not has_credentials:
        status_container.warning("👉 Configura tus credenciales de Supabase en la sección superior izquierda antes de continuar.")
    else:
        col1, col2 = st.columns([1, 4])
        
        with col1:
            if st.button(
                "Conectar" if not st.session_state.supabase_connected else "Reconectar",
                key="connect_supabase",
                help="Conectar a Supabase y listar tablas disponibles"
            ):
                with st.spinner("Conectando a Supabase..."):
                    tables = get_supabase_tables(
                        st.session_state.supabase_url,
                        st.session_state.supabase_key
                    )
                    
                    if tables:
                        st.session_state.supabase_tables = tables
                        st.session_state.supabase_connected = True
                        status_container.success("✅ Conexión exitosa a Supabase")
                    else:
                        st.session_state.supabase_connected = False
                        status_container.error("❌ No se pudieron obtener las tablas. Verifica tus credenciales.")

        if st.session_state.supabase_connected and st.session_state.supabase_tables:
            table_container = st.container()
            
            with table_container:
                selected_table = st.selectbox(
                    "Selecciona una tabla:",
                    st.session_state.supabase_tables,
                    key="supabase_table_selector"
                )
                
                if st.button("Cargar Tabla", key="load_supabase_table"):
                    try:
                        with st.spinner("Cargando datos..."):
                            df = load_supabase_table(
                                st.session_state.supabase_url,
                                st.session_state.supabase_key,
                                selected_table
                            )
                            if df is not None:
                                st.session_state.er_data = df
                                st.success(f"✅ Tabla '{selected_table}' cargada exitosamente")
                            else:
                                st.error(f"❌ No se pudo cargar la tabla '{selected_table}'. La tabla puede estar vacía.")
                    except Exception as e:
                        st.error(f"❌ Error al cargar la tabla: {str(e)}")
                        st.write("Detalles del error:", str(e))

    # Análisis de datos
    if st.session_state.er_data is not None:
        analyze_dataset(st.session_state.er_data)
    
    return st.session_state.er_data

def analyze_dataset(data):
    """Analizar el dataset cargado"""
    # Generar explicación automática con Gemini
    if 'dataset_explanation' not in st.session_state:
        st.session_state.dataset_explanation = None

    has_api_key = 'gemini_api_key' in st.session_state and st.session_state.gemini_api_key

    if st.button(
        "Explicar Dataset",
        key="explain_dataset_button",
        disabled=not has_api_key,
        help="Requiere API key de Gemini para funcionar"
    ):
        st.session_state.dataset_explanation = generate_dataset_explanation(
            data, 
            st.session_state.gemini_api_key
        )

    # Mostrar explicación si existe
    if st.session_state.dataset_explanation:
        st.markdown("### Explicación del Dataset")
        st.write(st.session_state.dataset_explanation)

        # Botón para limpiar explicación
        if st.button("Limpiar Explicación", key="clear_explanation"):
            st.session_state.dataset_explanation = None
            st.rerun()
    
    # Mostrar datos si se han cargado
    # st.markdown("### Dataset Cargado")
    # st.dataframe(data.head())
    renderer = StreamlitRenderer(data)
    renderer.explorer()
    st.info(f"📊 Dimensiones: {data.shape[0]} filas × {data.shape[1]} columnas")
    # Mostrar tipos de datos en columnas múltiples
    with st.expander("📊 Ver tipos de datos por columna", expanded=False):
        # Slider para número de columnas
        num_columns = st.slider(
            "Número de columnas para mostrar tipos de datos", 
            min_value=1, 
            max_value=10, 
            value=5,
            help="Desliza para ajustar el número de columnas en la visualización de tipos de datos",
            key="num_columns_slider"
        )
        
        # Obtener tipos de datos de cada columna
        data_types = data.dtypes.reset_index()
        data_types.columns = ["Columna", "Tipo de dato"]
        
        st.write("**Tipos de datos por columna:**")
        
        # Calcular elementos por columna
        items_per_column = len(data_types) // num_columns + (1 if len(data_types) % num_columns != 0 else 0)
        
        # Crear columnas en Streamlit
        cols = st.columns(num_columns)
        
        # Distribuir tipos de datos entre columnas
        for col_idx in range(num_columns):
            start_idx = col_idx * items_per_column
            end_idx = min(start_idx + items_per_column, len(data_types))
            
            if start_idx < len(data_types):
                with cols[col_idx]:
                    for idx in range(start_idx, end_idx):
                        st.write(f"**{data_types.iloc[idx]['Columna']}**: {data_types.iloc[idx]['Tipo de dato']}")
        
        # Mostrar resumen de tipos de datos
        st.markdown("---")
        st.write("**Resumen de tipos de datos:**")
        type_summary = data.dtypes.value_counts()
        summary_cols = st.columns(len(type_summary))
        for i, (dtype, count) in enumerate(type_summary.items()):
            with summary_cols[i]:
                st.metric(f"Tipo: {dtype}", f"{count} columnas")

    # Análisis de Variables por Tipo
    st.markdown("### Análisis de Variables por Tipo")
    
    # Crear columnas para mostrar variables numéricas y categóricas
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Variables Numéricas")
        lista_var_numericas = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        if lista_var_numericas:
            df_numericas = pd.DataFrame({
                'Variable': lista_var_numericas,
                'Tipo': [str(data[col].dtype) for col in lista_var_numericas]
            })
            st.dataframe(df_numericas, hide_index=True)
            
            if st.checkbox("Ver estadísticas básicas de variables numéricas", key="show_numeric_stats"):
                st.write(data[lista_var_numericas].describe())
            
            selected_num_vars = st.multiselect(
                "Seleccionar variables numéricas para análisis",
                lista_var_numericas,
                default=lista_var_numericas[0] if lista_var_numericas else None,
                key="numeric_vars_select"
            )
            
            if selected_num_vars:
                st.write("**Histograma de variables seleccionadas:**")
                for var in selected_num_vars:
                    fig_hist = px.histogram(
                        data, 
                        x=var, 
                        title=f'Histograma de {var}'
                    )
                    st.plotly_chart(fig_hist)
        else:
            st.info("No se encontraron variables numéricas en el dataset")
    
    with col2:
        st.markdown("#### Variables Categóricas")
        lista_var_object = data.select_dtypes(include=['object']).columns.tolist()
        
        if lista_var_object:
            df_categoricas = pd.DataFrame({
                'Variable': lista_var_object,
                'Tipo': [str(data[col].dtype) for col in lista_var_object]
            })
            st.dataframe(df_categoricas, hide_index=True)
            
            if st.checkbox("Ver valores únicos de variables categóricas", key="show_categorical_stats"):
                selected_cat_var = st.selectbox(
                    "Seleccionar variable categórica",
                    lista_var_object,
                    key="categorical_var_select"
                )
                if selected_cat_var:
                    unique_values = data[selected_cat_var].value_counts()
                    st.write("Valores únicos en {selected_cat_var}")
                             
                    # Gráfico de barras de valores únicos
                    fig_bar = px.bar(
                        x=unique_values.index, 
                        y=unique_values.values, 
                        title=f'Distribución de {selected_cat_var}'
                    )
                    st.plotly_chart(fig_bar)

                    # Tabla de frecuencia
                    freq_df = pd.DataFrame({
                        'Valor': unique_values.index, 
                        'Frecuencia': unique_values.values, 
                        'Porcentaje': (unique_values.values / len(data) * 100).round(2)
                    })
                    st.dataframe(freq_df)
        else:
            st.info("No se encontraron variables categóricas en el dataset")

    # Matriz de Correlación para Variables Numéricas
    st.markdown("### Matriz de Correlación")
    numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns.tolist()

    if len(numeric_columns) > 1:
        # Selección de variables para correlación
        corr_variables = st.multiselect(
            "Selecciona las variables para la matriz de correlación", 
            options=numeric_columns, 
            default=numeric_columns[:min(5, len(numeric_columns))]
        )

        if corr_variables:
            # Calcular matriz de correlación
            corr_matrix = data[corr_variables].corr(method='pearson')

            # Mapa de calor de correlación
            fig_corr = px.imshow(
                corr_matrix, 
                text_auto=True, 
                aspect="auto", 
                color_continuous_scale='RdBu_r', 
                title='Matriz de Correlación de Pearson'
            )
            st.plotly_chart(fig_corr, use_container_width=True)

            # Análisis de correlaciones significativas
            st.write("### Análisis de Correlaciones Significativas")
            threshold = st.slider(
                "Umbral mínimo de correlación", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.5, 
                step=0.05
            )

            # Obtener correlaciones significativas
            corr_pairs = corr_matrix.unstack()
            significant_corr = corr_pairs[
                (abs(corr_pairs) >= threshold) & 
                (abs(corr_pairs) < 1)
            ].sort_values(ascending=False)

            if not significant_corr.empty:
                st.write(f"Correlaciones significativas (|correlación| ≥ {threshold}):")
                for (var1, var2), corr_value in significant_corr.items():
                    st.write(f"- **{var1}** y **{var2}**: correlación de **{corr_value:.2f}**")
            else:
                st.write("No se encontraron correlaciones significativas con el umbral seleccionado.")

    else:
        st.warning("No hay suficientes variables numéricas para calcular correlaciones.")

    # Detección de Outliers
    st.markdown("### Detección de Outliers")
    numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns.tolist()

    if numeric_columns:
        outlier_vars = st.multiselect(
            "Selecciona variables para análisis de outliers", 
            options=numeric_columns
        )

        if outlier_vars:
            for var in outlier_vars:
                # Cálculo de Q1, Q3 e IQR
                Q1 = data[var].quantile(0.25)
                Q3 = data[var].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                # Identificación de outliers
                outliers = data[(data[var] < lower_bound) | (data[var] > upper_bound)][var]
                num_outliers = outliers.shape[0]

                st.write(f"### Análisis de Outliers para {var}")
                
                # Boxplot
                fig_box = px.box(data, y=var, title=f'Boxplot de {var}')
                st.plotly_chart(fig_box)

                # Resumen de outliers
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total de Datos", len(data))
                with col2:
                    st.metric("Número de Outliers", num_outliers)
                with col3:
                    st.metric("Porcentaje de Outliers", f"{num_outliers/len(data)*100:.2f}%")

                # Mostrar outliers
                if st.checkbox(f"Mostrar outliers de {var}"):
                    st.dataframe(outliers)
    return data
