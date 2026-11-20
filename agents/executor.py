"""
agents/executor.py - Agente EJECUTOR con soporte multi-modelo
Compatible con Python 3.6 (sin capture_output ni text=)
"""

import os
import subprocess
import json
from typing import Dict, Any, Optional, List
from agents.base import get_patient_metrics, list_available_patients, MODELS, get_model_info, list_models

# Ruta del Python del entorno twin_agent
PYTHON_PATH = "/lustre/home/admins/aadib/.conda/envs/twin_agent/bin/python"

# =====================================================================
# CONFIGURACIÓN
# =====================================================================

TEST_IMAGES_DIR = "/lustre/home/admins/aadib/3D_Medical_image/nnUNet_raw/Dataset008_HepaticVessel/imagesTs"
SWIN_TASK03_IMAGES_DIR = "/lustre/home/admins/aadib/3D_Medical_image/data/Task03_Liver/imagesTr"

# =====================================================================
# HERRAMIENTAS DEL AGENTE EJECUTOR
# =====================================================================

def generate_twin(patient_id: str, model: str = "nnunet") -> Dict[str, Any]:
    """
    Genera el gemelo digital de un paciente.
    """
    print(f"🔮 [EJECUTOR] Generando gemelo para {patient_id} con {model}...")

    model_info = get_model_info(model)
    if not model_info:
        return {"status": "failed", "patient": patient_id, "error": f"Modelo {model} no encontrado"}

    script = model_info["script"]

    # Construir comando según modelo
    if model in ["nnunet", "swin_task08"]:
        input_file = f"{TEST_IMAGES_DIR}/{patient_id}_0000.nii.gz"
        if not os.path.exists(input_file):
            return {"status": "failed", "patient": patient_id, "error": f"Archivo no encontrado: {input_file}"}
        cmd = f"{PYTHON_PATH} {script} --file {input_file}"

    elif model == "swin_task03":
        input_file = f"{SWIN_TASK03_IMAGES_DIR}/{patient_id}.nii.gz"
        if os.path.exists(input_file):
            cmd = f"{PYTHON_PATH} {script} --file {input_file}"
        else:
            cmd = f"{PYTHON_PATH} {script} --patient {patient_id}"

    else:
        return {"status": "failed", "patient": patient_id, "error": f"Modelo {model} no soportado"}

    # Variables de entorno
    env = os.environ.copy()
    env['nnUNet_results'] = '/lustre/home/admins/aadib/3D_Medical_image/nnUNet_results'
    env['nnUNet_raw'] = '/lustre/home/admins/aadib/3D_Medical_image/nnUNet_raw'
    env['nnUNet_preprocessed'] = '/lustre/home/admins/aadib/3D_Medical_image/nnUNet_preprocessed'

    print(f"   Ejecutando: {cmd}")

    try:
        # Python 3.6: usar universal_newlines en lugar de text
        result = subprocess.run(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            env=env,
            timeout=1800
        )

        if result.returncode == 0:
            return {"status": "success", "patient": patient_id, "model": model}
        else:
            return {"status": "failed", "patient": patient_id, "model": model, "error": result.stderr}

    except subprocess.TimeoutExpired:
        return {"status": "failed", "patient": patient_id, "model": model, "error": "Timeout (30 minutos)"}
    except Exception as e:
        return {"status": "failed", "patient": patient_id, "model": model, "error": str(e)}


def get_metrics(patient_id: str, model: str = "nnunet") -> Dict[str, Any]:
    """Obtiene las métricas de un paciente."""
    print(f"📊 [EJECUTOR] Obteniendo métricas para {patient_id} ({model})...")
    metrics = get_patient_metrics(patient_id, model)
    if metrics:
        return {"patient_id": patient_id, "model": model, "metrics": metrics}
    else:
        return {"patient_id": patient_id, "model": model, "error": "No se encontraron métricas"}


def compare_patients(patient_id1: str, patient_id2: str, model: str = "nnunet") -> Dict[str, Any]:
    """Compara dos pacientes."""
    print(f"🔍 [EJECUTOR] Comparando {patient_id1} vs {patient_id2} ({model})...")

    m1 = get_patient_metrics(patient_id1, model)
    m2 = get_patient_metrics(patient_id2, model)

    if m1 is None or m2 is None:
        return {"error": "Uno o ambos pacientes no tienen métricas"}

    return {
        "model": model,
        "patient1": {"id": patient_id1, "metrics": m1},
        "patient2": {"id": patient_id2, "metrics": m2},
        "diff": {
            "organ": m1.get("organ_cc", 0) - m2.get("organ_cc", 0),
            "tumor": m1.get("tumor_cc", 0) - m2.get("tumor_cc", 0),
            "burden": m1.get("burden", 0) - m2.get("burden", 0)
        }
    }


def compare_models(patient_id: str) -> Dict[str, Any]:
    """Compara todos los modelos para un mismo paciente."""
    print(f"🔬 [EJECUTOR] Comparando modelos para {patient_id}...")

    results = {}
    for model in ["nnunet", "swin_task08", "swin_task03"]:
        metrics = get_patient_metrics(patient_id, model)
        if metrics:
            results[model] = {
                "organ_cc": metrics.get("organ_cc", 0),
                "tumor_cc": metrics.get("tumor_cc", 0),
                "burden": metrics.get("burden", 0),
                "tumor_count": metrics.get("tumor_count", 0),
                "organ_type": metrics.get("organ_type", "unknown")
            }

    return {"patient_id": patient_id, "models": results}


def list_patients(model: str = "nnunet", limit: int = 30) -> List[str]:
    """Lista los pacientes disponibles para un modelo."""
    patients = list_available_patients(model)
    return patients[:limit]


def batch_process(model: str = "nnunet", limit: int = 10) -> Dict[str, Any]:
    """Procesa múltiples pacientes en lote."""
    print(f"📦 [EJECUTOR] Procesando lote de {limit} pacientes con {model}...")

    patients = list_available_patients(model)[:limit]
    results = []
    success = 0

    for pid in patients:
        result = generate_twin(pid, model)
        results.append(result)
        if result.get("status") == "success":
            success += 1

    return {
        "model": model,
        "total": len(patients),
        "success": success,
        "failed": len(patients) - success,
        "results": results
    }


def list_models() -> List[str]:
    """Lista los modelos disponibles."""
    from agents.base import list_models as _list_models
    return _list_models()
