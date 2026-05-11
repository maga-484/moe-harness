"""
selector.py - Módulo de selección de productos/experimentos
Inspirado en el selector de 16.py, pero refactorizado para el Harness.
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from typing import List, Dict, Any, Optional
import os

class ProductSelector:
    """
    Selector de productos/experimentos basado en Random Forest.
    Predice la probabilidad de éxito de un nuevo producto antes de lanzarlo.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Inicializa el selector.
        
        Args:
            model_path: Ruta al modelo entrenado (.pkl). Si no existe, se entrena uno nuevo.
        """
        self.model = None
        self.model_path = model_path
        
        if model_path and os.path.exists(model_path):
            self.model = joblib.load(model_path)
            print(f"✅ Selector cargado desde {model_path}")
    
    def entrenar(self, datos: pd.DataFrame, features: List[str], target: str = 'exito',
                 test_size: float = 0.2, random_state: int = 42) -> Dict[str, float]:
        """
        Entrena el selector con datos históricos.
        
        Args:
            datos: DataFrame con features y target
            features: Lista de columnas a usar como features
            target: Columna objetivo (éxito/fracaso)
            test_size: Proporción para validación
            random_state: Semilla para reproducibilidad
            
        Returns:
            Diccionario con métricas de evaluación
        """
        X = datos[features]
        y = datos[target]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        self.model = RandomForestClassifier(
            n_estimators=50, 
            max_depth=3, 
            random_state=random_state
        )
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        accuracy = (y_pred == y_test).mean()
        
        print(f"📊 Selector entrenado. Precisión: {accuracy*100:.1f}%")
        
        if self.model_path:
            joblib.dump(self.model, self.model_path)
            print(f"💾 Modelo guardado en {self.model_path}")
        
        return {'accuracy': accuracy, 'test_size': len(X_test)}
    
    def predecir(self, features: np.ndarray) -> float:
        """
        Predice probabilidad de éxito para un nuevo producto.
        
        Args:
            features: Array de features [variation, loss_exp, loss_den, ventaja]
            
        Returns:
            Probabilidad de éxito (0.0 a 1.0)
        """
        if self.model is None:
            raise ValueError("Selector no entrenado. Llame a entrenar() primero o cargue un modelo.")
        
        probas = self.model.predict_proba(features)
        
        # Si solo hay una clase
        if probas.shape[1] == 1:
            clases = self.model.classes_
            return 1.0 if clases[0] == 1 else 0.0
        
        return probas[0][1]  # Probabilidad de éxito (clase 1)
    
    def predecir_producto(self, variation: float, loss_exp: float, loss_den: float) -> float:
        """
        Método de alto nivel para predecir un producto.
        
        Args:
            variation: Variación del producto (0-1)
            loss_exp: Pérdida estimada del experto
            loss_den: Pérdida estimada del modelo denso
            
        Returns:
            Probabilidad de éxito
        """
        ventaja = loss_den - loss_exp
        features = np.array([[variation, loss_exp, loss_den, ventaja]])
        return self.predecir(features)
    
    def seleccionar_catalogo(self, catalogo: List[Dict[str, Any]], 
                             umbral: float = 0.6) -> List[Dict[str, Any]]:
        """
        Evalúa un catálogo de productos y selecciona los prometedores.
        
        Args:
            catalogo: Lista de diccionarios con claves 'variation', 'loss_exp', 'loss_den', 'id'
            umbral: Probabilidad mínima para seleccionar (0.6 = 60%)
            
        Returns:
            Lista de productos seleccionados (los que superan el umbral)
        """
        seleccionados = []
        
        for producto in catalogo:
            prob = self.predecir_producto(
                producto['variation'],
                producto['loss_exp'],
                producto['loss_den']
            )
            producto['prob_exito'] = prob
            
            if prob >= umbral:
                seleccionados.append(producto)
                print(f"✅ {producto.get('id', '?')}: SELECCIONADO (prob={prob*100:.1f}%)")
            else:
                print(f"❌ {producto.get('id', '?')}: DESCARTADO (prob={prob*100:.1f}%)")
        
        return seleccionados


# ------------------------------------------------------------
# Ejemplo de uso
# ------------------------------------------------------------
if __name__ == "__main__":
    # Datos históricos de ejemplo (de 13.py)
    datos_historicos = pd.DataFrame({
        'variation': [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
        'loss_exp': [0.0708, 0.0758, 0.0717, 0.0736, 0.0757, 0.0678, 0.0689, 0.0712, 0.0687, 0.0780],
        'loss_den': [0.1023, 0.1029, 0.1021, 0.0995, 0.1052, 0.0984, 0.0998, 0.1015, 0.1009, 0.1050],
    })
    datos_historicos['ventaja'] = datos_historicos['loss_den'] - datos_historicos['loss_exp']
    datos_historicos['exito'] = ((datos_historicos['loss_exp'] < 0.1) & (datos_historicos['ventaja'] > 0)).astype(int)
    
    # Entrenar selector
    selector = ProductSelector(model_path='models/selector_model.pkl')
    selector.entrenar(datos_historicos, features=['variation', 'loss_exp', 'loss_den', 'ventaja'])
    
    # Probar con un catálogo
    catalogo = [
        {'id': 'PROD-001', 'variation': 0.08, 'loss_exp': 0.072, 'loss_den': 0.105},
        {'id': 'PROD-002', 'variation': 0.12, 'loss_exp': 0.075, 'loss_den': 0.103},
        {'id': 'PROD-003', 'variation': 0.18, 'loss_exp': 0.080, 'loss_den': 0.101},
    ]
    
    seleccionados = selector.seleccionar_catalogo(catalogo, umbral=0.6)
    print(f"\n📊 Seleccionados: {len(seleccionados)}/{len(catalogo)}")