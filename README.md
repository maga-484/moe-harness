markdown

# moe-harness

**Orquestador de Mixture of Experts para selección, lanzamiento y monitoreo de expertos identitarios.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

---

## 🚀 ¿Qué es esto?

`moe-harness` es un conjunto de herramientas para **orquestar el ciclo de vida de expertos identitarios** (basados en el repositorio `identity-experts-moe`). Permite:

- **Seleccionar** productos/experimentos prometedores usando un selector entrenado (Random Forest).
- **Lanzar** los seleccionados: crear silos, entrenar expertos tiny y guardarlos con un ID jerárquico `a.b.c.d.e.f.g.h`.
- **Monitorear** los silos en tiempo real (simulación de métricas: visitas, guardados, preguntas) y tomar decisiones automáticas (escalar, monitorear, descontinuar).

Este harness es ideal para automatizar la gestión de miles de expertos especializados en e‑commerce, logística, I+D o cualquier dominio con entidades identificables.

---

## 📦 Estructura del repositorio

moe-harness/
├── README.md
├── requirements.txt
├── LICENSE
├── harness/
│ ├── init.py
│ ├── selector.py # ProductSelector (Random Forest)
│ ├── lanzador.py # SiloLanzador (crea silos y expertos)
│ ├── monitor.py # SiloMonitor (simula señales y decisiones)
│ └── orquestador.py # MoEHarness (integra todo)
├── models/ # Modelos preentrenados (selector y expertos)
├── expertos/ # Pesos de los expertos (.pth)
├── data/ # Archivos CSV con registros
└── tests/ # Pruebas unitarias

text

---

## 🛠️ Instalación

```bash
git clone https://github.com/maga-484/moe-harness.git
cd moe-harness
pip install -r requirements.txt
🧪 Uso rápido
python
from harness.orquestador import MoEHarness
import pandas as pd

# 1. Datos históricos (para entrenar el selector)
datos = pd.DataFrame({
    'variation': [0.05, 0.10, ...],
    'loss_exp': [0.0708, 0.0758, ...],
    'loss_den': [0.1023, 0.1029, ...],
    'exito': [1, 1, ...]
})

# 2. Crear harness
harness = MoEHarness()

# 3. Entrenar selector (opcional, si no existe modelo)
harness.entrenar_selector(datos)

# 4. Catálogo de candidatos
catalogo = [
    {'id': 'PROD-001', 'variation': 0.08, 'loss_exp': 0.072, 'loss_den': 0.105},
    {'id': 'PROD-002', 'variation': 0.12, 'loss_exp': 0.075, 'loss_den': 0.103},
]

# 5. Pipeline completo: selecciona, lanza y monitorea
resultados = harness.pipeline_completo(catalogo, umbral=0.6)
harness.monitorear()   # simula métricas y toma decisiones
🧠 Componentes principales
ProductSelector (selector.py)
Entrena un Random Forest para predecir éxito de un producto a partir de variation, loss_exp, loss_den.

Guarda el modelo en models/selector_model.pkl.

SiloLanzador (lanzador.py)
Genera ID jerárquico único (8 niveles) y su hash O(1).

Crea un IdentityExpert (MLP tiny), lo entrena con datos sintéticos (o reales) y guarda los pesos en expertos/.

Registra el lanzamiento en productos_registrados.csv.

SiloMonitor (monitor.py)
Simula la recolección de métricas (visitas, guardados, preguntas) para cada silo.

Calcula una probabilidad de éxito y decide: ESCALAR, MONITOREAR, OBSERVAR o DESCONTINUAR.

Guarda el histórico de decisiones en historial_monitoreo.csv.

MoEHarness (orquestador.py)
Integra selector, lanzador y monitor.

Ofrece los métodos pipeline_completo() y monitorear().

📊 Ejemplo de salida del monitor
text
🔍 Monitoreando PROD-001 (784.02.86.02.11.24.28.25)
   Visitas: 52 | Guardados: 6 | Preguntas: 2
   Probabilidad de éxito: 24.8%
   Decisión: DESCONTINUAR → Pausar publicidad, liquidar stock, archivar silo
🔗 Dependencias
torch>=2.0.0

numpy>=1.24.0

pandas>=2.0.0

scikit-learn>=1.3.0

joblib>=1.3.0

📄 Licencia
MIT

🙋‍♀️ Autora
Magali Ofelia Gafe – https://github.com/maga-484

Proyecto inspirado en la organización y en arquitecturas de Mixture of Experts eficientes.
```
