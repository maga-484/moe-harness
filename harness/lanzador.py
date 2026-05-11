"""
lanzador.py - Módulo de lanzamiento de productos/experimentos
Crea silos y entrena expertos identitarios para productos seleccionados.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
import csv
import hashlib
import time
from typing import Dict, Any, Optional, Tuple

# ------------------------------------------------------------
# Funciones de ID jerárquico (igual que en identity-experts-moe)
# ------------------------------------------------------------
def hash_jerarquico(a, b, c, d, e, f, g, h):
    return (a * (100**7) + b * (100**6) + c * (100**5) + d * (100**4) + 
            e * (100**3) + f * (100**2) + g * 100 + h)

def id_a_componentes(id_str):
    partes = id_str.split('.')
    return (int(partes[0]), int(partes[1]), int(partes[2]), int(partes[3]),
            int(partes[4]), int(partes[5]), int(partes[6]), int(partes[7]))

def id_a_hash(id_str):
    a,b,c,d,e,f,g,h = id_a_componentes(id_str)
    return hash_jerarquico(a,b,c,d,e,f,g,h)

def generar_id_aleatorio():
    a = random.randint(0, 999)
    b = random.randint(0, 99)
    c = random.randint(0, 99)
    d = random.randint(0, 99)
    e = random.randint(0, 99)
    f = random.randint(0, 99)
    g = random.randint(0, 99)
    h = random.randint(0, 99)
    return f"{a:03d}.{b:02d}.{c:02d}.{d:02d}.{e:02d}.{f:02d}.{g:02d}.{h:02d}"

# ------------------------------------------------------------
# Experto Identitario (tiny MLP)
# ------------------------------------------------------------
class IdentityExpert(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=128, output_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)

# ------------------------------------------------------------
# Generador de datos sintéticos (placeholder para datos reales)
# ------------------------------------------------------------
def generate_silo_data(seed, n_samples=500, variation=0.1):
    """Genera datos sintéticos para entrenar el experto."""
    torch.manual_seed(seed)
    input_dim, output_dim = 32, 32
    X = torch.randn(n_samples, input_dim)
    torch.manual_seed(42)
    W_base = torch.randn(input_dim, output_dim) * 0.8
    torch.manual_seed(seed)
    W_variation = torch.randn(input_dim, output_dim) * variation
    W = W_base + W_variation
    y = torch.mm(X, W) + torch.randn(n_samples, output_dim) * 0.05
    # Normalización
    y_mean = y.mean(dim=0, keepdim=True)
    y_std = y.std(dim=0, keepdim=True) + 1e-8
    y = (y - y_mean) / y_std
    X_mean = X.mean(dim=0, keepdim=True)
    X_std = X.std(dim=0, keepdim=True) + 1e-8
    X = (X - X_mean) / X_std
    return X, y

# ------------------------------------------------------------
# Lanzador (crea silos y entrena expertos)
# ------------------------------------------------------------
class SiloLanzador:
    """
    Lanzador de silos: entrena y guarda un nuevo experto para un producto.
    """
    
    def __init__(self, directorio_expertos: str = "expertos",
                 registro_csv: str = "productos_registrados.csv"):
        """
        Args:
            directorio_expertos: Carpeta donde se guardan los pesos de los expertos
            registro_csv: Archivo CSV con el registro de productos lanzados
        """
        self.directorio_expertos = directorio_expertos
        self.registro_csv = registro_csv
        os.makedirs(directorio_expertos, exist_ok=True)
    
    def crear_silo(self, producto: Dict[str, Any]) -> Tuple[int, str]:
        """
        Crea un silo para un producto seleccionado.
        
        Args:
            producto: Diccionario con claves 'id', 'variation', 'loss_exp', 'loss_den'
                      Opcional: 'id_hierarquico' (si no, se genera uno)
            
        Returns:
            Tuple (hash_value, id_hierarquico)
        """
        # 1. Generar ID jerárquico único
        if 'id_hierarquico' in producto:
            id_hierarquico = producto['id_hierarquico']
        else:
            id_hierarquico = generar_id_aleatorio()
        
        # 2. Calcular hash
        hash_val = id_a_hash(id_hierarquico)
        
        # 3. Crear experto
        experto = IdentityExpert()
        
        # 4. Generar datos sintéticos (en producción, serían datos reales)
        variation = producto.get('variation', 0.1)
        X, y = generate_silo_data(seed=hash_val % 10000, variation=variation)
        
        # 5. Entrenamiento rápido
        optimizer = optim.Adam(experto.parameters(), lr=0.001)
        loss_fn = nn.MSELoss()
        experto.train()
        for epoch in range(30):
            optimizer.zero_grad()
            pred = experto(X)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
        
        # 6. Guardar pesos
        ruta_pesos = os.path.join(self.directorio_expertos, f"{hash_val}.pth")
        torch.save(experto.state_dict(), ruta_pesos)
        
        # 7. Registrar en CSV
        self._registrar(producto, id_hierarquico, hash_val, ruta_pesos)
        
        print(f"   ✅ Silo creado: {id_hierarquico} (hash={hash_val})")
        return hash_val, id_hierarquico
    
    def _registrar(self, producto: Dict, id_hierarquico: str, hash_val: int, ruta_pesos: str):
        """Registra el lanzamiento en un archivo CSV."""
        registro = {
            'producto_id': producto['id'],
            'id_hierarquico': id_hierarquico,
            'hash': hash_val,
            'variation': producto.get('variation', 0.0),
            'loss_exp_estimado': producto.get('loss_exp', 0.0),
            'loss_den_estimado': producto.get('loss_den', 0.0),
            'ruta_pesos': ruta_pesos,
            'created_at': time.time()
        }
        
        file_exists = os.path.isfile(self.registro_csv)
        with open(self.registro_csv, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=registro.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(registro)
    
    def lanzar_catalogo(self, catalogo_seleccionados: list) -> list:
        """
        Lanza un catálogo de productos seleccionados (crea silos para todos).
        """
        resultados = []
        for producto in catalogo_seleccionados:
            print(f"\n--- Lanzando {producto['id']} ---")
            hash_val, id_hier = self.crear_silo(producto)
            resultados.append({'producto': producto, 'hash': hash_val, 'id_hierarquico': id_hier})
        return resultados


# ------------------------------------------------------------
# Ejemplo de uso (integrado con selector)
# ------------------------------------------------------------
if __name__ == "__main__":
    from selector import ProductSelector
    import pandas as pd
    
    # 1. Datos históricos
    datos_historicos = pd.DataFrame({
        'variation': [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
        'loss_exp': [0.0708, 0.0758, 0.0717, 0.0736, 0.0757, 0.0678, 0.0689, 0.0712, 0.0687, 0.0780],
        'loss_den': [0.1023, 0.1029, 0.1021, 0.0995, 0.1052, 0.0984, 0.0998, 0.1015, 0.1009, 0.1050],
    })
    datos_historicos['ventaja'] = datos_historicos['loss_den'] - datos_historicos['loss_exp']
    datos_historicos['exito'] = ((datos_historicos['loss_exp'] < 0.1) & (datos_historicos['ventaja'] > 0)).astype(int)
    
    # 2. Seleccionar productos
    selector = ProductSelector(model_path='models/selector_model.pkl')
    if selector.model is None:
        selector.entrenar(datos_historicos, features=['variation', 'loss_exp', 'loss_den', 'ventaja'])
    
    catalogo = [
        {'id': 'DEMO-001', 'variation': 0.08, 'loss_exp': 0.072, 'loss_den': 0.105},
        {'id': 'DEMO-002', 'variation': 0.12, 'loss_exp': 0.075, 'loss_den': 0.103},
        {'id': 'DEMO-003', 'variation': 0.18, 'loss_exp': 0.080, 'loss_den': 0.101},
    ]
    seleccionados = selector.seleccionar_catalogo(catalogo, umbral=0.6)
    
    # 3. Lanzar los seleccionados
    lanzador = SiloLanzador()
    resultados = lanzador.lanzar_catalogo(seleccionados)
    
    print("\n" + "="*60)
    print("RESUMEN DE LANZAMIENTO")
    print("="*60)
    for r in resultados:
        print(f"  {r['producto']['id']} -> {r['id_hierarquico']} (hash={r['hash']})")
    print("="*60)