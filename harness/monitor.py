"""
monitor.py - Módulo de monitoreo de silos
Simula la recolección de señales y toma decisiones automáticas.
"""

import pandas as pd
import numpy as np
import time
import random
import os
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

class SiloMonitor:
    """
    Monitor de silos: recolecta señales, evalúa rendimiento y toma decisiones.
    """
    
    # Umbrales para decisiones (se pueden ajustar)
    UMBRAL_ESCALAR = 0.8      # Probabilidad > 80% → escalar
    UMBRAL_MONITOREO = 0.5     # Probabilidad > 50% → monitorear
    UMBRAL_DESCARTAR = 0.3     # Probabilidad < 30% → descontinuar
    
    def __init__(self, registro_csv: str = "productos_registrados.csv",
                 directorio_expertos: str = "expertos",
                 historial_csv: str = "historial_monitoreo.csv"):
        self.registro_csv = registro_csv
        self.directorio_expertos = directorio_expertos
        self.historial_csv = historial_csv
        self._inicializar_historial()
    
    def _inicializar_historial(self):
        if not os.path.exists(self.historial_csv):
            pd.DataFrame(columns=[
                'timestamp', 'producto_id', 'id_hierarquico', 'hash',
                'visitas', 'guardados', 'preguntas', 'conversion_rate',
                'prob_exito', 'decision', 'accion_tomada'
            ]).to_csv(self.historial_csv, index=False)
    
    def _simular_metricas(self, silo_id: str, random_seed: Optional[int] = None) -> Dict[str, float]:
        if random_seed:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        visitas = max(0, np.random.poisson(lam=50) + np.random.randint(-20, 50))
        guardados = max(0, int(visitas * np.random.uniform(0.01, 0.15)))
        preguntas = max(0, int(visitas * np.random.uniform(0.005, 0.05)))
        conversiones = max(0, int(visitas * np.random.uniform(0.005, 0.10)))
        conversion_rate = conversiones / max(visitas, 1)
        
        prob_exito = min(0.99, 
                         0.2 + 
                         0.3 * (guardados / max(visitas, 1)) + 
                         0.2 * (preguntas / max(visitas, 1)) + 
                         0.3 * conversion_rate)
        
        return {
            'visitas': visitas,
            'guardados': guardados,
            'preguntas': preguntas,
            'conversiones': conversiones,
            'conversion_rate': conversion_rate,
            'prob_exito': prob_exito
        }
    
    def _decidir_accion(self, prob_exito: float) -> Tuple[str, str]:
        if prob_exito >= self.UMBRAL_ESCALAR:
            decision = "ESCALAR"
            accion = "Aumentar publicidad, comprar más stock, escalar producción"
        elif prob_exito >= self.UMBRAL_MONITOREO:
            decision = "MONITOREAR"
            accion = "Seguimiento intensivo, mantener inversión actual"
        elif prob_exito >= self.UMBRAL_DESCARTAR:
            decision = "OBSERVAR"
            accion = "Reducir inversión, preparar plan de contingencia"
        else:
            decision = "DESCONTINUAR"
            accion = "Pausar publicidad, liquidar stock, archivar silo"
        return decision, accion
    
    def _registrar_decision(self, producto_id: str, id_hierarquico: str, hash_val: int,
                            metricas: Dict, decision: str, accion: str):
        nuevo_registro = pd.DataFrame([{
            'timestamp': time.time(),
            'producto_id': producto_id,
            'id_hierarquico': id_hierarquico,
            'hash': hash_val,
            'visitas': metricas['visitas'],
            'guardados': metricas['guardados'],
            'preguntas': metricas['preguntas'],
            'conversion_rate': metricas['conversion_rate'],
            'prob_exito': metricas['prob_exito'],
            'decision': decision,
            'accion_tomada': accion
        }])
        
        if os.path.exists(self.historial_csv):
            existing = pd.read_csv(self.historial_csv)
            updated = pd.concat([existing, nuevo_registro], ignore_index=True)
            updated.to_csv(self.historial_csv, index=False)
        else:
            nuevo_registro.to_csv(self.historial_csv, index=False)
    
    def monitorear_silo(self, producto_id: str, id_hierarquico: str, hash_val: int,
                        verbose: bool = True) -> Dict[str, Any]:
        metricas = self._simular_metricas(producto_id)
        decision, accion = self._decidir_accion(metricas['prob_exito'])
        self._registrar_decision(producto_id, id_hierarquico, hash_val,
                                 metricas, decision, accion)
        if verbose:
            print(f"\n🔍 Monitoreando {producto_id} ({id_hierarquico})")
            print(f"   Visitas: {metricas['visitas']} | Guardados: {metricas['guardados']} | Preguntas: {metricas['preguntas']}")
            print(f"   Probabilidad de éxito: {metricas['prob_exito']*100:.1f}%")
            print(f"   Decisión: {decision} → {accion}")
        return {
            'producto_id': producto_id,
            'id_hierarquico': id_hierarquico,
            'hash': hash_val,
            'metricas': metricas,
            'decision': decision,
            'accion': accion
        }
    
    def monitorear_todos(self, verbose: bool = True) -> List[Dict]:
        if not os.path.exists(self.registro_csv):
            print("⚠️ No hay productos registrados. Ejecuta el lanzador primero.")
            return []
        
        df = pd.read_csv(self.registro_csv)
        resultados = []
        print("\n" + "="*60)
        print("📊 CICLO DE MONITOREO DE SILOS")
        print("="*60)
        
        for _, row in df.iterrows():
            res = self.monitorear_silo(
                producto_id=row['producto_id'],
                id_hierarquico=row['id_hierarquico'],
                hash_val=int(row['hash']),
                verbose=verbose
            )
            resultados.append(res)
        return resultados
    
    def ejecutar_ciclo(self, intervalo_segundos: int = 60, ciclos: int = 1):
        contador = 0
        while ciclos is None or contador < ciclos:
            print(f"\n🕒 Ciclo de monitoreo {contador+1} - {datetime.now().isoformat()}")
            self.monitorear_todos(verbose=True)
            contador += 1
            if ciclos is None or contador < ciclos:
                print(f"\n⏳ Esperando {intervalo_segundos} segundos para el próximo ciclo...")
                time.sleep(intervalo_segundos)


if __name__ == "__main__":
    monitor = SiloMonitor()
    if os.path.exists(monitor.registro_csv):
        resultados = monitor.monitorear_todos()
        print("\n📊 Resumen de decisiones:")
        for r in resultados:
            print(f"   {r['producto_id']}: {r['decision']} (prob={r['metricas']['prob_exito']*100:.1f}%)")
    else:
        print("⚠️ No hay productos registrados. Ejecuta primero el pipeline de lanzamiento.")