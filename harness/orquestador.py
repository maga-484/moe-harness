"""
orquestador.py - Orquestador principal del Harness
Integra selección, lanzamiento y monitoreo de expertos.
"""

import os
import pandas as pd
from typing import List, Dict, Any, Optional
from selector import ProductSelector
from lanzador import SiloLanzador
from monitor import SiloMonitor  # ← Importación necesaria

class MoEHarness:
    """
    Harness para Mixture of Experts.
    Orquesta la selección, lanzamiento y monitoreo de expertos identitarios.
    """
    
    def __init__(self, model_dir: str = "models", expert_dir: str = "expertos",
                 registro_csv: str = "productos_registrados.csv"):
        """
        Args:
            model_dir: Directorio donde está el selector entrenado
            expert_dir: Directorio donde se guardan los expertos
            registro_csv: Archivo CSV con el registro de lanzamientos
        """
        self.selector = ProductSelector(model_path=os.path.join(model_dir, 'selector_model.pkl'))
        self.lanzador = SiloLanzador(directorio_expertos=expert_dir, registro_csv=registro_csv)
        
        # Si el selector no tiene modelo, se puede entrenar con datos históricos
        if self.selector.model is None:
            print("⚠️ Selector no entrenado. Usa entrenar_selector() primero.")
    
    def entrenar_selector(self, datos_historicos: pd.DataFrame,
                          features: List[str] = ['variation', 'loss_exp', 'loss_den', 'ventaja'],
                          target: str = 'exito'):
        """Entrena el selector con datos históricos."""
        self.selector.entrenar(datos_historicos, features=features, target=target)
    
    def pipeline_completo(self, catalogo: List[Dict[str, Any]], umbral: float = 0.6) -> List[Dict]:
        """
        Pipeline completo: selecciona productos, lanza los seleccionados.
        
        Args:
            catalogo: Lista de productos candidatos
            umbral: Probabilidad mínima para seleccionar
            
        Returns:
            Lista de resultados de lanzamiento
        """
        print("\n" + "="*60)
        print("🚀 PIPELINE DEL HARNESS")
        print("="*60)
        
        # 1. Selección
        seleccionados = self.selector.seleccionar_catalogo(catalogo, umbral)
        
        if not seleccionados:
            print("⚠️ No hay productos seleccionados. Pipeline detenido.")
            return []
        
        # 2. Lanzamiento
        resultados = self.lanzador.lanzar_catalogo(seleccionados)
        
        # 3. Resumen
        print("\n" + "="*60)
        print("✅ PIPELINE COMPLETADO")
        print(f"   Productos evaluados: {len(catalogo)}")
        print(f"   Seleccionados: {len(seleccionados)}")
        print(f"   Silos creados: {len(resultados)}")
        print("="*60)
        
        return resultados
    
    def monitorear(self, ciclo_unico: bool = True, verbose: bool = True):
        """
        Ejecuta el monitoreo de silos activos usando SiloMonitor.
        
        Args:
            ciclo_unico: Si True, ejecuta un solo ciclo. Si False, bucle infinito.
            verbose: Si True, imprime resultados en consola.
        """
        monitor = SiloMonitor(registro_csv=self.lanzador.registro_csv,
                              directorio_expertos=self.lanzador.directorio_expertos)
        if ciclo_unico:
            return monitor.monitorear_todos(verbose=verbose)
        else:
            monitor.ejecutar_ciclo(intervalo_segundos=60, ciclos=None)
    
    def monitorear_silos(self, silos_ids: Optional[List[str]] = None, verbose: bool = True):
        """
        Monitorea el rendimiento de los silos activos (versión simplificada).
        (Mantiene compatibilidad con código anterior que usaba este nombre)
        """
        return self.monitorear(ciclo_unico=True, verbose=verbose)


# ------------------------------------------------------------
# Ejemplo de uso completo
# ------------------------------------------------------------
if __name__ == "__main__":
    # 1. Datos históricos (de 13.py)
    datos_historicos = pd.DataFrame({
        'variation': [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
        'loss_exp': [0.0708, 0.0758, 0.0717, 0.0736, 0.0757, 0.0678, 0.0689, 0.0712, 0.0687, 0.0780],
        'loss_den': [0.1023, 0.1029, 0.1021, 0.0995, 0.1052, 0.0984, 0.0998, 0.1015, 0.1009, 0.1050],
    })
    datos_historicos['ventaja'] = datos_historicos['loss_den'] - datos_historicos['loss_exp']
    datos_historicos['exito'] = ((datos_historicos['loss_exp'] < 0.1) & (datos_historicos['ventaja'] > 0)).astype(int)
    
    # 2. Crear harness
    harness = MoEHarness()
    
    # 3. Entrenar selector (si no existe modelo)
    if harness.selector.model is None:
        harness.entrenar_selector(datos_historicos)
    
    # 4. Catálogo de prueba
    catalogo = [
        {'id': 'TEST-001', 'variation': 0.08, 'loss_exp': 0.072, 'loss_den': 0.105},
        {'id': 'TEST-002', 'variation': 0.12, 'loss_exp': 0.075, 'loss_den': 0.103},
        {'id': 'TEST-003', 'variation': 0.18, 'loss_exp': 0.080, 'loss_den': 0.101},
        {'id': 'TEST-004', 'variation': 0.35, 'loss_exp': 0.120, 'loss_den': 0.098},
    ]
    
    # 5. Ejecutar pipeline
    resultados = harness.pipeline_completo(catalogo, umbral=0.6)
    
    # 6. Monitorear silos creados (usando el monitor con simulación)
    if resultados:
        print("\n" + "="*60)
        print("📡 INICIANDO MONITOREO DE SILOS")
        print("="*60)
        # Esto llamará a la simulación de métricas y tomará decisiones
        monitor_resultados = harness.monitorear(ciclo_unico=True, verbose=True)