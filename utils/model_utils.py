# models_utils.py Módulo para utils
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import time
import pickle
import io

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import (
    LinearRegression, LogisticRegression, Lasso, Ridge,
    SGDClassifier, RidgeClassifier, PassiveAggressiveClassifier
)
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingClassifier, AdaBoostClassifier,
    BaggingClassifier, ExtraTreesClassifier, ExtraTreesRegressor
)
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, SVR
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, classification_report, confusion_matrix
)
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
import xgboost as xgb

class ModelTrainer:
    """
    Clase para gestionar el entrenamiento de modelos de machine learning
    """
    @staticmethod
    def get_model_options(problem_type):
        """
        Obtener opciones de modelos según el tipo de problema
        
        Args:
            problem_type (str): Tipo de problema ('classification' o 'regression')
        
        Returns:
            dict: Diccionario de opciones de modelos
        """
        if problem_type == 'regression':
            return ModelTrainer._get_regression_models()
        else:
            return ModelTrainer._get_classification_models()

    @staticmethod
    def _get_regression_models():
        """
        Definir opciones de modelos para regresión
        
        Returns:
            dict: Modelos de regresión con sus parámetros
        """
        return {
            'Regresión Lineal': {
                'model': lambda rs: Pipeline([
                    ('scaler', StandardScaler()),
                    ('regressor', LinearRegression())
                ]),
                'params': {
                    'regressor__fit_intercept': [True, False],
                    'regressor__copy_X': [True],
                    'regressor__positive': [True, False],
                    'scaler__with_mean': [True, False],
                    'scaler__with_std': [True, False]
                }
            },
            'Lasso': {
                'model': lambda rs: Pipeline([
                    ('scaler', StandardScaler()),
                    ('regressor', Lasso(random_state=rs))
                ]),
                'params': {
                    'regressor__alpha': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
                    'regressor__fit_intercept': [True, False],
                    'regressor__max_iter': [1000, 2000, 5000],
                    'regressor__selection': ['cyclic', 'random'],
                    'regressor__tol': [1e-4, 1e-3],
                    'scaler__with_mean': [True, False],
                    'scaler__with_std': [True, False]
                }
            },
            'Ridge': {
                'model': lambda rs: Pipeline([
                    ('scaler', StandardScaler()),
                    ('regressor', Ridge(random_state=rs))
                ]),
                'params': {
                    'regressor__alpha': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
                    'regressor__fit_intercept': [True, False],
                    'regressor__solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
                    'regressor__tol': [1e-4, 1e-3],
                    'scaler__with_mean': [True, False],
                    'scaler__with_std': [True, False]
                }
            },
            'Árbol de Decisión': {
                'model': lambda rs: DecisionTreeRegressor(random_state=rs),
                'params': {
                    'max_depth': [3, 5, 7, 10, 15, None],
                    'min_samples_split': [2, 5, 10, 20],
                    'min_samples_leaf': [1, 2, 4, 8],
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'splitter': ['best', 'random'],
                    'max_features': ['sqrt', 'log2', None]
                }
            },
            'Random Forest': {
                'model': lambda rs: RandomForestRegressor(random_state=rs),
                'params': {
                    'n_estimators': [100, 200, 300, 500],
                    'max_depth': [3, 5, 7, 10, None],
                    'min_samples_split': [2, 5, 10, 20],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', None],
                    'bootstrap': [True, False],
                    'criterion': ['squared_error', 'absolute_error', 'poisson']
                }
            },
            'XGBoost': {
                'model': lambda rs: xgb.XGBRegressor(
                    tree_method='hist',
                    device='cuda',
                    enable_categorical=True,
                    random_state=rs
                ),
                'params': {
                    'n_estimators': [100, 200, 300, 500],
                    'max_depth': [3, 5, 7, 9],
                    'learning_rate': [0.01, 0.05, 0.1, 0.3],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0],
                    'min_child_weight': [1, 3, 5],
                    'gamma': [0, 0.1, 0.2],
                    'reg_alpha': [0, 0.1, 0.5],
                    'reg_lambda': [0.1, 1.0, 5.0]
                }
            }
        }

    @staticmethod
    def _get_classification_models():
        """
        Definir opciones de modelos para clasificación
        
        Returns:
            dict: Modelos de clasificación con sus parámetros
        """
        return {
            'Regresión Logística': {
                'model': lambda rs: LogisticRegression(max_iter=1000, random_state=rs),
                'params': {
                    'C': [0.001, 0.01, 0.1, 1.0, 10.0],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga'],
                    'class_weight': [None, 'balanced'],
                    'warm_start': [True, False],
                    'tol': [1e-4, 1e-3, 1e-2]
                }
            },
            'Random Forest': {
                'model': lambda rs: RandomForestClassifier(random_state=rs),
                'params': {
                    'n_estimators': [100, 200, 300, 500],
                    'max_depth': [3, 5, 7, 10, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'class_weight': [None, 'balanced', 'balanced_subsample'],
                    'criterion': ['gini', 'entropy'],
                    'max_features': ['sqrt', 'log2', None]
                }
            },
            'XGBoost': {
                'model': lambda rs: xgb.XGBClassifier(
                    tree_method='hist',
                    device='cuda',
                    enable_categorical=True,
                    random_state=rs
                ),
                'params': {
                    'n_estimators': [100, 200, 300, 500],
                    'max_depth': [3, 5, 7, 9],
                    'learning_rate': [0.01, 0.05, 0.1, 0.3],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0],
                    'min_child_weight': [1, 3, 5],
                    'gamma': [0, 0.1, 0.2],
                    'reg_alpha': [0, 0.1, 0.5],
                    'reg_lambda': [0.1, 1.0, 5.0],
                    'scale_pos_weight': [1, 2, 3]
                }
            },
            'SVM': {
                'model': lambda rs: SVC(random_state=rs),
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
                    'gamma': ['scale', 'auto', 0.1, 0.01, 0.001],
                    'class_weight': [None, 'balanced'],
                    'probability': [True]
                }
            },
            'Naive Bayes': {
                'model': lambda rs: GaussianNB(),
                'params': {
                    'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
                }
            }
        }

    @staticmethod
    def _determine_problem_type(model):
        """
        Determinar el tipo de problema basado en el modelo
        
        Args:
            model (BaseEstimator): Modelo a evaluar
        
        Returns:
            str: Tipo de problema ('classification', 'regression', 'unknown')
        """
        try:
            # Importaciones dinámicas para evitar dependencias estrictas
            from sklearn.base import ClassifierMixin, RegressorMixin
            
            if hasattr(model, 'predict_proba'):
                return 'classification'
            elif hasattr(model, 'predict'):
                return 'regression'
            else:
                return 'unknown'
        except ImportError:
            # Fallback si las importaciones fallan
            return 'unknown'

    @staticmethod
    def _get_default_scoring(problem_type):
        """
        Obtener la métrica de scoring predeterminada
        
        Args:
            problem_type (str): Tipo de problema
        
        Returns:
            str: Métrica de scoring predeterminada
        """
        scoring_map = {
            'classification': 'accuracy',
            'regression': 'r2',
            'unknown': None
        }
        return scoring_map.get(problem_type, None)

    @staticmethod
    def train_model_pipeline(
        X_train, 
        y_train, 
        model_config, 
        X_test=None, 
        y_test=None, 
        cv=5, 
        scoring=None, 
        random_state=42,  # Añadir explícitamente random_state
        **kwargs
    ):
        """
        Entrenar modelo con validación cruzada y evaluación flexible
        
        Args:
            X_train (array-like): Datos de entrenamiento
            y_train (array-like): Etiquetas de entrenamiento
            model_config (dict): Configuración del modelo
            X_test (array-like, optional): Datos de prueba
            y_test (array-like, optional): Etiquetas de prueba
            cv (int, optional): Número de pliegues para validación cruzada
            scoring (str, optional): Métrica de puntuación
            random_state (int, optional): Semilla aleatoria para reproducibilidad
            **kwargs: Argumentos adicionales
        
        Returns:
            dict: Resultados detallados del entrenamiento
        """
        # Extraer modelo y parámetros
        model_func = model_config.get('model')
        params = model_config.get('params', {})

        # Instanciar el modelo si es una función
        if callable(model_func):
            model = model_func(random_state)
        else:
            model = model_func

        # Verificar que el modelo sea una instancia válida
        if not hasattr(model, 'fit') or not hasattr(model, 'predict'):
            raise ValueError(f"Modelo inválido: {model}. Debe tener métodos 'fit' y 'predict'.")

        # Determinar tipo de problema
        problem_type = ModelTrainer._determine_problem_type(model)
        
        # Configurar scoring
        if scoring is None:
            scoring = ModelTrainer._get_default_scoring(problem_type)

        # Configurar parámetros de GridSearchCV
        grid_search_params = {
            'estimator': model,
            'param_grid': params,
            'cv': cv,
            'scoring': scoring
        }
        
        # Añadir kwargs adicionales
        grid_search_params.update({
            k: v for k, v in kwargs.items() 
            if k in ['n_jobs', 'verbose', 'refit', 'error_score']
        })

        try:
            # Realizar búsqueda de hiperparámetros
            grid_search = GridSearchCV(**grid_search_params)
            grid_search.fit(X_train, y_train)
        except Exception as e:
            return {
                'error': f"Error durante el entrenamiento: {str(e)}",
                'problem_type': problem_type
            }

        # Preparar resultados base
        results = {
            'problem_type': problem_type,
            'best_model': grid_search.best_estimator_,
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }

        # Evaluación en conjunto de prueba
        if X_test is not None and y_test is not None:
            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_test)
            
            # Métricas específicas según el tipo de problema
            if problem_type == 'classification':
                results.update({
                    'test_accuracy': accuracy_score(y_test, y_pred),
                    'classification_report': classification_report(y_test, y_pred, output_dict=True),
                    'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
                    'y_pred': y_pred  # Añadir predicciones para informes posteriores
                })
            elif problem_type == 'regression':
                results.update({
                    'test_mse': mean_squared_error(y_test, y_pred),
                    'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                    'test_mae': mean_absolute_error(y_test, y_pred),
                    'test_r2': r2_score(y_test, y_pred),
                    'y_pred': y_pred  # Añadir predicciones para informes posteriores
                })
            else:
                results['test_predictions'] = y_pred

        return results

    @staticmethod
    def create_class_distribution_plot(y_original):
        """
        Crear un gráfico de distribución de clases
        
        Args:
            y_original (pd.Series): Variable objetivo original
        
        Returns:
            plotly.graph_objs._figure.Figure: Gráfico de distribución de clases
        """
        class_dist = pd.DataFrame({
            'Clase': y_original.value_counts().index,
            'Cantidad': y_original.value_counts().values
        })
        
        fig = px.bar(
            class_dist,
            x='Clase',
            y='Cantidad',
            title='Distribución de clases'
        )
        
        return fig

    @staticmethod
    def process_classification_data(y, random_state):
        """
        Procesar datos de clasificación
        
        Args:
            y (pd.Series): Variable objetivo
            random_state (int): Semilla aleatoria
        
        Returns:
            tuple: Variable objetivo procesada y codificador de etiquetas
        """
        # Codificación de etiquetas
        le = LabelEncoder()
        y_encoded = pd.Series(le.fit_transform(y))
        
        return y_encoded, le

    @staticmethod
    def save_model(model, filename):
        """
        Guardar modelo entrenado en un archivo
        
        Args:
            model: Modelo entrenado
            filename (str): Nombre del archivo
        """
        with open(filename, 'wb') as f:
            pickle.dump(model, f)

    @staticmethod
    def load_model(filename):
        """
        Cargar modelo desde un archivo
        
        Args:
            filename (str): Nombre del archivo
        
        Returns:
            Modelo cargado
        """
        with open(filename, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def get_model_performance_metrics(y_true, y_pred, problem_type):
        """
        Obtener métricas de rendimiento del modelo
        
        Args:
            y_true (pd.Series): Etiquetas verdaderas
            y_pred (pd.Series): Etiquetas predichas
            problem_type (str): Tipo de problema
        
        Returns:
            dict: Métricas de rendimiento
        """
        if problem_type == 'classification':
            return {
                'accuracy': accuracy_score(y_true, y_pred),
                'classification_report': classification_report(y_true, y_pred, output_dict=True)
            }
        else:  # Regresión
            return {
                'mse': mean_squared_error(y_true, y_pred),
                'r2_score': r2_score(y_true, y_pred)
            }

    @staticmethod
    def split_data(X, y, test_size=0.2, random_state=42):
        """
        Dividir datos en conjuntos de entrenamiento y prueba
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Variable objetivo
            test_size (float): Proporción de datos de prueba
            random_state (int): Semilla aleatoria
        
        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    @staticmethod
    def prepare_data_for_ml(df, target_column, problem_type='classification', test_size=0.2, random_state=42):
        """
        Preparar datos para machine learning
        
        Args:
            df (pd.DataFrame): DataFrame de datos
            target_column (str): Columna objetivo
            problem_type (str): Tipo de problema
            test_size (float): Proporción de datos de prueba
            random_state (int): Semilla aleatoria
        
        Returns:
            dict: Diccionario con datos preparados
        """
        # Separar features y target
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Preprocesar datos según el tipo de problema
        if problem_type == 'classification':
            y, label_encoder = ModelTrainer.process_classification_data(y, random_state)
        else:
            label_encoder = None

        # Dividir datos
        X_train, X_test, y_train, y_test = ModelTrainer.split_data(X, y, test_size, random_state)

        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'label_encoder': label_encoder,
            'features': list(X.columns),
            'problem_type': problem_type
        }

    @staticmethod
    def generate_model_comparison_report(trained_models, problem_type):
        """
        Generar informe comparativo de modelos
        
        Args:
            trained_models (dict): Modelos entrenados
            problem_type (str): Tipo de problema
        
        Returns:
            pd.DataFrame: Informe comparativo de modelos
        """
        comparison_data = []

        for model_name, model_info in trained_models.items():
            model_metrics = ModelTrainer.get_model_performance_metrics(
                model_info['y_test'], 
                model_info['y_pred'], 
                problem_type
            )

            model_entry = {
                'Modelo': model_name,
                'Tiempo de Entrenamiento': model_info.get('training_time', 0),
            }

            # Agregar métricas según el tipo de problema
            if problem_type == 'classification':
                model_entry.update({
                    'Precisión': model_metrics['accuracy'],
                    'Precisión (Macro)': model_metrics['classification_report']['macro avg']['precision'],
                    'Recall (Macro)': model_metrics['classification_report']['macro avg']['recall'],
                    'F1-Score (Macro)': model_metrics['classification_report']['macro avg']['f1-score']
                })
            else:
                model_entry.update({
                    'MSE': model_metrics['mse'],
                    'R2 Score': model_metrics['r2_score']
                })

            comparison_data.append(model_entry)

        return pd.DataFrame(comparison_data)

    @staticmethod
    def plot_model_comparison(comparison_df, problem_type):
        """
        Crear gráfico comparativo de modelos
        
        Args:
            comparison_df (pd.DataFrame): DataFrame de comparación de modelos
            problem_type (str): Tipo de problema
        
        Returns:
            plotly.graph_objs._figure.Figure: Gráfico comparativo
        """
        metric_column = 'Precisión' if problem_type == 'classification' else 'R2 Score'
        
        fig = px.bar(
            comparison_df, 
            x='Modelo', 
            y=metric_column,
            title=f'Comparación de Modelos - {metric_column}'
        )
        
        return fig

# Funciones sueltas para importación directa
def get_model_options(problem_type):
    return ModelTrainer.get_model_options(problem_type)

def train_model_pipeline(*args, **kwargs):
    return ModelTrainer.train_model_pipeline(*args, **kwargs)

def process_classification_data(y, random_state=42):
    return ModelTrainer.process_classification_data(y, random_state)

def create_class_distribution_plot(y):
    return ModelTrainer.create_class_distribution_plot(y)