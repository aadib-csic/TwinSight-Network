# 🫁 TwinSight Network

**Gemelo Digital Hepático con IA para Planificación Quirúrgica**

> Proyecto de investigación — prueba de concepto desarrollada en el marco del proyecto **Momentum / SGAI-CSIC**
> Entrenamiento en el clúster HPC **[Drago (CSIC)](https://aic.csic.es/servicios/computacion/hpc-drago/)** · NVIDIA H200

---

## ¿Qué es TwinSight Network?

TwinSight Network es un pipeline end-to-end que transforma un CT hepático en un **gemelo digital interactivo** con asistente de IA. Partiendo de imágenes médicas en formato NIfTI, el sistema:

1. **Segmenta automáticamente** vasos hepáticos y tumores mediante nnUNet v2 (entrenado en HPC con GPU H200)
2. **Calcula métricas clínicas** reales: carga tumoral, distancia a vasos, localización anatómica (segmentos de Couinaud)
3. **Genera un gemelo digital** — archivo HTML autocontenido con visualización 3D WebGL interactiva
4. **Integra un asistente IA** (LLM local vía Ollama) para consultas clínicas en lenguaje natural

El resultado final es un único archivo `.html` por paciente, sin servidor, sin base de datos, listo para compartir.

---

## 🏗️ Arquitectura del Pipeline

```
CT Hepático (.nii.gz)
        │
        ├─► TotalSegmentator ──────────────────► Máscara de hígado (.nii.gz)
        │   (zero-shot, sin reentrenamiento)       └─► Volumen hepático real (cc)
        │
        └─► nnUNet v2 (5-fold ensemble) ──────► Segmentación (Clase 1 = vasos,
            Entrenado en Task 08 MSD              Clase 2 = tumor)
            303 CTs · Dice ~0.70
            HPC Drago / NVIDIA H200
                   │
                   ▼
         mesh_utils_tumorsight.py
                   │
         ┌─────────┴──────────┐
         │  Métricas clínicas  │  Renderizado 3D
         │  ─────────────────  │  ────────────────
         │  Carga tumoral (%)  │  Marching Cubes
         │  Dist. a vasos (mm) │  Isotropic resample
         │  Segmentos Couinaud │  Plotly WebGL
         │  Riesgo quirúrgico  │  Silueta hígado
         └─────────┬──────────┘
                   │
                   ▼
         Gemelo Digital (.html)
         ┌──────────────────────────────┐
         │  Visualización 3D interactiva│
         │  Panel de métricas clínicas  │
         │  Chat IA (Qwen2.5:7b/Ollama) │
         └──────────────────────────────┘
```

---

## 📁 Estructura del Repositorio

```
TwinSight-Network/
│
├── README.md                        # Este archivo
├── requirements.txt                 # Dependencias Python
├── .gitignore                       # Excluye datos, checkpoints, HTMLs
│
├── main_tumorsight.py               # 🚀 Punto de entrada principal
│
├── src/
│   ├── mesh_utils_tumorsight.py     # Motor de visualización + generador HTML
│   └── data_utils.py                # Utilidades de datos y preprocesado
│
├── training/
│   └── train_v2.py                  # Scripts de entrenamiento (SwinUNETR/nnUNet)
│
├── agents/
│   ├── twin_agent_v2.py             # Orquestador multiagente
│   ├── react.py                     # Agente ReAct (análisis + mejoras)
│   ├── executor.py                  # Agente ejecutor del pipeline
│   ├── base.py                      # Configuración de modelos
│   └── __init__.py
│
└── docs/
    └── pipeline_diagram.png         # Diagrama del pipeline
```

---

## 🧠 Modelo de Segmentación — nnUNet v2

| Parámetro         | Valor                                      |
|-------------------|--------------------------------------------|
| Framework         | nnUNet v2                                  |
| Dataset           | Task 08 MSD (Medical Segmentation Decathlon) |
| Volumen training  | 303 CTs hepáticos                          |
| Clases            | 1 = vasos hepáticos · 2 = tumor            |
| Configuración     | 3D fullres · 5-fold cross-validation       |
| Dice score        | ~0.70 (ensemble de 5 folds)                |
| Infraestructura   | HPC Drago (CSIC) · NVIDIA H200 × 8        |
| Tiempo training   | ~10 h/fold × 5 folds                       |

> **Nota sobre eficiencia computacional:** El entrenamiento completo (~50 h en una GPU convencional) se realizó en paralelo en el clúster Drago, reduciendo el tiempo de forma significativa. La inferencia en un nuevo paciente tarda segundos en CPU estándar.

### Segmentación del hígado (TotalSegmentator)

La máscara hepática se obtiene mediante **TotalSegmentator** (zero-shot, sin reentrenamiento), lo que permite calcular la carga tumoral real respecto al volumen total del hígado. Esta separación modular (nnUNet para patología + TotalSegmentator para anatomía) es una decisión de diseño intencionada.

---

## 🚀 Cómo Ejecutar el Pipeline

### Prerrequisitos

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Instalar nnUNet v2
pip install nnunetv2

# 3. Instalar TotalSegmentator
pip install TotalSegmentator

# 4. Instalar Ollama y el modelo LLM (para el asistente IA)
# → https://ollama.ai
ollama pull qwen2.5:7b
```

### Variables de Entorno (nnUNet)

```bash
export nnUNet_raw="/ruta/a/nnUNet_raw"
export nnUNet_preprocessed="/ruta/a/nnUNet_preprocessed"
export nnUNet_results="/ruta/a/nnUNet_results"
```

### Ejecutar un Paciente

```bash
# Procesar un paciente específico
python -u main_tumorsight.py --patient hepaticvessel_009

# Procesar una muestra aleatoria de N pacientes
python -u main_tumorsight.py --sample 10
```

### Salida Esperada

```
[TotalSegmentator] hepaticvessel_009 → liver_segmentations/hepaticvessel_009_liver.nii.gz
[nnUNet] Running inference...
[Marching Cubes] Resampling to 1.5mm isotropic...
[HTML] Generating digital twin...

📋 Summary:
  • Liver:        1380.1 cc
  • Tumor load:   14.0 cc
  • Burden:       1.0%
  • Focal lesions: 2
  • Vessel risk:  MODERATE — 7.8mm (5–10mm range)

✅ Output: TwinSight_HTML_Final/hepaticvessel_009_twin.html
```

---

## 💻 Visualizar el Gemelo Digital Localmente

El gemelo digital es un **archivo HTML autocontenido**. Para visualizarlo:

1. **Descargar** el archivo `.html` del clúster/servidor al ordenador local:
   ```bash
   scp usuario@drago:/lustre/.../TwinSight_HTML_Final/hepaticvessel_009_twin.html ./
   ```

2. **Abrir** directamente en el navegador (Chrome recomendado para WebGL):
   ```bash
   # Linux / macOS
   xdg-open hepaticvessel_009_twin.html
   open hepaticvessel_009_twin.html

   # O simplemente arrastrar el archivo al navegador
   ```

3. **Activar el asistente IA** (opcional):
   - Requiere Ollama corriendo en la máquina local: `ollama serve`
   - El modelo `qwen2.5:7b` debe estar descargado
   - El chat se conecta automáticamente a `localhost:11434`

> **Sin servidor, sin internet, sin instalación adicional.** La visualización 3D funciona offline con cualquier navegador moderno.

---

## 🤖 Asistente IA Integrado

Cada gemelo digital incluye un asistente clínico basado en LLM local:

- **Modelo**: Qwen2.5:7b (vía Ollama)
- **Contexto embebido**: datos del paciente (volúmenes, riesgo vascular, coordenadas de lesiones) en JSON dentro del HTML
- **Capacidades**:
  - Resumir el caso clínico
  - Responder preguntas sobre localización anatómica
  - Explicar el riesgo de resección
  - Comparar con umbrales clínicos estándar

**Riesgo vascular** (pre-calculado en Python, no por el LLM):

| Distancia a vaso | Clasificación |
|-----------------|---------------|
| < 5 mm          | 🔴 CRÍTICO    |
| 5 – 10 mm       | 🟡 MODERADO   |
| > 10 mm         | 🟢 BAJO       |

---

## 📊 Métricas Clínicas Calculadas

| Métrica                  | Descripción                                                        |
|--------------------------|--------------------------------------------------------------------|
| **Volumen hepático**     | Volumen total del hígado (TotalSegmentator), en cc                 |
| **Volumen tumoral**      | Suma de lesiones focales segmentadas (nnUNet Clase 2), en cc       |
| **Carga tumoral (%)**    | Tumor / Hígado × 100 — indica extensión de la enfermedad           |
| **Lesiones focales**     | Número de tumores individuales (componentes conectados)            |
| **Distancia a vasos**    | Distancia mínima tumor→vaso en mm (mapa de distancias 3D)          |
| **Riesgo quirúrgico**    | CRÍTICO / MODERADO / BAJO según umbral clínico                     |
| **Segmento Couinaud**    | Localización anatómica aproximada (coordenadas RAS)                |

---

## 🏋️ Entrenamiento del Modelo (Referencia)

> El entrenamiento fue realizado en el **clúster HPC Drago del CSIC** con 8× NVIDIA H200.

```bash
# Preparar dataset Task 08 MSD
nnUNetv2_plan_and_preprocess -d 008 --verify_dataset_integrity

# Entrenar los 5 folds (ejecutar en paralelo en HPC)
for FOLD in 0 1 2 3 4; do
    nnUNetv2_train 008 3d_fullres $FOLD --npz
done

# Encontrar la mejor configuración
nnUNetv2_find_best_configuration 008 -c 3d_fullres
```

Los checkpoints resultantes se colocan en `$nnUNet_results/Dataset008_HepaticVessel/`.

---

## 🔧 Sistema de Agentes (Experimental)

El directorio `agents/` contiene un sistema multiagente para análisis y mejora continua:

```bash
# Modo interactivo
python twin_agent_v2.py

# Comandos disponibles:
#   scan      — Escanear HTMLs generados y extraer métricas
#   analyze   — Analizar resultados del paciente
#   improve   — Sugerir mejoras de parámetros
#   generate  — Generar nuevo gemelo digital
#   knowledge — Mostrar base de conocimiento acumulada
```

---

## 📦 Dependencias Principales

```
nnunetv2
TotalSegmentator
nibabel
numpy
scipy
scikit-image
plotly
torch
monai
```

Ver `requirements.txt` para versiones exactas.

---

## 📄 Referencia y Créditos

Este trabajo se enmarca en el **Proyecto Momentum — SGAI-CSIC** y ha sido desarrollado como prueba de concepto para la aplicación de gemelos digitales en planificación quirúrgica hepática.

**Infraestructura computacional:** Clúster HPC Drago (CSIC) — NVIDIA H200 × 8  
→ https://aic.csic.es/servicios/computacion/hpc-drago/

**Modelos de terceros:**
- [nnUNet v2](https://github.com/MIC-DKFZ/nnUNet) — Isensee et al., Nature Methods 2021
- [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) — Wasserthal et al., Radiology AI 2023
- [Qwen2.5](https://huggingface.co/Qwen/Qwen2.5-7B) — Alibaba DAMO Academy

**Autor:** Ali Adib, PhD — Investigador Postdoctoral, SGAI-CSIC / Proyecto Momentum

---

## ⚠️ Aviso

Este proyecto es una **prueba de concepto de investigación**. No está validado clínicamente ni aprobado para uso diagnóstico o terapéutico. Los resultados no deben utilizarse como base para decisiones médicas reales sin validación adicional.

---

*TwinSight Network · SGAI-CSIC · 2026*
