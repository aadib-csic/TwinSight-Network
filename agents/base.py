"""
agents/base.py - Base para los agentes con soporte multi-modelo
Rutas actualizadas para nnUNet, Swin Task08 y Swin Task03
"""

from typing import Dict, Any, Optional, List
import os
import glob
import json
import re

# =====================================================================
# CONFIGURACIÓN DE MODELOS
# =====================================================================

MODELS = {
    "nnunet": {
        "name": "nnUNet (Task 008)",
        "script": "main_liver_styled.py",
        "mesh": "mesh_utils_V2.py",
        "dataset": "Task 008: Hepatic Vessel",
        "classes": "0=fondo, 1=vasos, 2=tumor",
        "organ_type": "vessels",
        "model_path": "nnUNet_results/Dataset008_HepaticVessel/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth",
        "results_dir": "/lustre/home/admins/aadib/3D_Medical_image/TwinSight_HTML_nnUNet",
        "test_images_dir": "/lustre/home/admins/aadib/3D_Medical_image/nnUNet_raw/Dataset008_HepaticVessel/imagesTs",
        "html_pattern": "{}_twin.html",
        "report_pattern": "report.txt",
        "nifti_pattern": "{}.nii.gz",
        "has_nifti": True,
        "class_mapping": {1: "vessels", 2: "tumor"}
    },
    "swin_task08": {
        "name": "SwinUNETR (Task 08)",
        "script": "main_v2.py",
        "mesh": "mesh_utils_V2.py",
        "dataset": "Task 08: Hepatic Vessel",
        "classes": "0=fondo, 1=vasos, 2=tumor",
        "organ_type": "vessels",
        "model_path": "src/models/swin_unetr_vessels_epoch_150.pt",
        "results_dir": "/lustre/home/admins/aadib/3D_Medical_image/final_3d_twins_vessels",
        "test_images_dir": "/lustre/home/admins/aadib/3D_Medical_image/nnUNet_raw/Dataset008_HepaticVessel/imagesTs",
        "html_pattern": "{}_3d_view.html",
        "report_pattern": "surgical_report.txt",
        "nifti_pattern": "{}_segmentation.nii.gz",
        "has_nifti": True,
        "class_mapping": {1: "vessels", 2: "tumor"}
    },
    "swin_task03": {
        "name": "SwinUNETR (Task 03)",
        "script": "main.py",
        "mesh": "mesh_utils.py",
        "dataset": "Task 03: Liver",
        "classes": "0=fondo, 1=hígado, 2=tumor",
        "organ_type": "liver",
        "model_path": "models/models/swin_unetr_epoch_100.pt",
        "results_dir": "/lustre/home/admins/aadib/3D_Medical_image/final_3d_twins",
        "test_images_dir": "/lustre/home/admins/aadib/3D_Medical_image/data/Task03_Liver/imagesTr",
        "html_pattern": "{}_interactive_twin.html",
        "report_pattern": "surgical_report.txt",
        "nifti_pattern": "{}.nii.gz",
        "has_nifti": False,
        "class_mapping": {1: "liver", 2: "tumor"}
    }
}

# Alias para simplificar
MODELS["swin"] = MODELS["swin_task08"]

# =====================================================================
# FUNCIONES DE RUTAS
# =====================================================================

def get_model_info(model_name: str) -> Optional[Dict]:
    """Obtiene información de un modelo."""
    return MODELS.get(model_name)

def list_models() -> List[str]:
    """Lista los modelos disponibles."""
    return list(MODELS.keys())

def get_results_dir(model: str = "nnunet") -> str:
    """Devuelve el directorio de resultados para un modelo."""
    info = get_model_info(model)
    return info.get("results_dir", "") if info else ""

def get_test_images_dir(model: str = "nnunet") -> str:
    """Devuelve el directorio de imágenes de test para un modelo."""
    info = get_model_info(model)
    return info.get("test_images_dir", "") if info else ""

def get_organ_type(model: str = "nnunet") -> str:
    """Devuelve el tipo de órgano (vessels/liver)."""
    info = get_model_info(model)
    return info.get("organ_type", "organ") if info else "organ"

def has_nifti(model: str = "nnunet") -> bool:
    """Indica si el modelo tiene NIfTI disponible."""
    info = get_model_info(model)
    return info.get("has_nifti", False) if info else False

# =====================================================================
# RUTAS DE ARCHIVOS POR PACIENTE
# =====================================================================

def get_patient_html_path(patient_id: str, model: str = "nnunet") -> str:
    """Devuelve la ruta del HTML de un paciente."""
    results_dir = get_results_dir(model)
    info = get_model_info(model)
    if not info or not results_dir:
        return ""

    html_pattern = info.get("html_pattern", "{}_twin.html")

    possible_paths = [
        os.path.join(results_dir, patient_id, html_pattern.format(patient_id)),
        os.path.join(results_dir, patient_id, "index.html"),
        os.path.join(results_dir, f"{patient_id}_twin.html"),
        os.path.join(results_dir, f"{patient_id}.html"),
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path
    return possible_paths[0]

def get_patient_report_path(patient_id: str, model: str = "nnunet") -> str:
    """Devuelve la ruta del reporte de un paciente."""
    results_dir = get_results_dir(model)
    info = get_model_info(model)
    if not info or not results_dir:
        return ""

    report_pattern = info.get("report_pattern", "report.txt")

    possible_paths = [
        os.path.join(results_dir, patient_id, report_pattern),
        os.path.join(results_dir, patient_id, "report.txt"),
        os.path.join(results_dir, patient_id, "surgical_report.txt"),
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path
    return possible_paths[0]

def get_patient_nifti_path(patient_id: str, model: str = "nnunet") -> str:
    """Devuelve la ruta del NIfTI de un paciente."""
    results_dir = get_results_dir(model)
    info = get_model_info(model)
    if not info or not results_dir:
        return ""

    nifti_pattern = info.get("nifti_pattern", "{}.nii.gz")

    possible_paths = [
        os.path.join(results_dir, patient_id, nifti_pattern.format(patient_id)),
        os.path.join(results_dir, patient_id, f"{patient_id}.nii.gz"),
        os.path.join(results_dir, patient_id, "prediction.nii.gz"),
        os.path.join(results_dir, f"{patient_id}.nii.gz"),
        f"test_predictions/{patient_id}.nii.gz",
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path
    return possible_paths[0]

# =====================================================================
# EXTRACCIÓN DE MÉTRICAS
# =====================================================================

def extract_metrics_from_report(content: str) -> Dict[str, Any]:
    """Extrae métricas de un reporte de texto."""
    metrics = {}

    patterns = [
        (r"(?:Liver|Vascular Tree|Órgano)\s*:\s*([\d.]+)\s*cc", "organ_cc"),
        (r"(?:Tumor|Lesion)\s*(?:Volume|Load)\s*:\s*([\d.]+)\s*cc", "tumor_cc"),
        (r"(?:Tumor Burden|Burden)\s*:\s*([\d.]+)\s*%", "burden"),
        (r"(?:Tumors?|Lesions?)\s*(?:detected|identified|count)?\s*:?\s*(\d+)", "tumor_count"),
    ]

    for pattern, key in patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            try:
                metrics[key] = float(match.group(1)) if "." in match.group(1) else int(match.group(1))
            except ValueError:
                pass

    if "organ_cc" in metrics and "tumor_cc" in metrics and "burden" not in metrics:
        if metrics["organ_cc"] > 0:
            metrics["burden"] = (metrics["tumor_cc"] / metrics["organ_cc"]) * 100
        else:
            metrics["burden"] = 0

    return metrics

def get_patient_metrics(patient_id: str, model: str = "nnunet") -> Optional[Dict]:
    """
    Obtiene métricas de un paciente para un modelo específico.
    Prioriza: NIfTI > Reporte > HTML
    """
    info = get_model_info(model)
    if not info:
        return None

    organ_type = info.get("organ_type", "organ")
    metrics = {
        "organ_type": organ_type,
        "model": model,
        "patient_id": patient_id
    }

    # 1. Intentar desde NIfTI (si existe)
    if has_nifti(model):
        nifti_path = get_patient_nifti_path(patient_id, model)
        if os.path.exists(nifti_path):
            try:
                import nibabel as nib
                import numpy as np

                data = nib.load(nifti_path).get_fdata()
                spacing = (1.5, 1.5, 1.5)
                vox_cc = np.prod(spacing) / 1000.0

                organ_cc = float(np.sum(data == 1) * vox_cc)
                tumor_cc = float(np.sum(data == 2) * vox_cc)
                tumor_count = int(len(np.unique(data)) - 1)

                metrics["organ_cc"] = organ_cc
                metrics["tumor_cc"] = tumor_cc
                metrics["tumor_count"] = tumor_count
                metrics["burden"] = (tumor_cc / organ_cc * 100) if organ_cc > 0 else 0
                metrics["nifti_path"] = nifti_path
                metrics["source"] = "nifti"

                return metrics

            except Exception as e:
                print(f"   ⚠️ Error leyendo NIfTI: {e}")

    # 2. Intentar desde Reporte
    report_path = get_patient_report_path(patient_id, model)
    if os.path.exists(report_path):
        try:
            with open(report_path, "r") as f:
                content = f.read()

            report_metrics = extract_metrics_from_report(content)

            if "organ_cc" in report_metrics and "tumor_cc" in report_metrics:
                metrics.update(report_metrics)
                metrics["report_path"] = report_path
                metrics["source"] = "report"
                return metrics

        except Exception as e:
            print(f"   ⚠️ Error leyendo reporte: {e}")

    # 3. Intentar desde HTML
    html_path = get_patient_html_path(patient_id, model)
    if os.path.exists(html_path):
        metrics["html_path"] = html_path
        metrics["source"] = "html_only"

        try:
            with open(html_path, "r") as f:
                content = f.read()
            tumor_markers = re.findall(r'T\d+', content)
            if tumor_markers:
                metrics["tumor_count"] = len(set(tumor_markers))
            return metrics
        except:
            pass

        return metrics

    return None

# =====================================================================
# LISTAR PACIENTES
# =====================================================================

def list_available_patients(model: str = "nnunet") -> List[str]:
    """Lista pacientes disponibles para un modelo."""
    images_dir = get_test_images_dir(model)
    if not images_dir or not os.path.exists(images_dir):
        return []

    files = glob.glob(os.path.join(images_dir, "*.nii.gz"))
    patients = []
    for f in files:
        basename = os.path.basename(f)
        name = basename.replace("_0000.nii.gz", "").replace(".nii.gz", "")
        if name and not name.startswith("._"):
            patients.append(name)
    return sorted(patients)

def list_patients_with_results(model: str = "nnunet") -> List[str]:
    """Lista pacientes que tienen resultados generados."""
    results_dir = get_results_dir(model)
    if not results_dir or not os.path.exists(results_dir):
        return []

    patient_dirs = glob.glob(os.path.join(results_dir, "*"))
    patients = []
    for d in patient_dirs:
        if os.path.isdir(d):
            name = os.path.basename(d)
            if glob.glob(os.path.join(d, "*.html")) or glob.glob(os.path.join(d, "*.txt")):
                patients.append(name)
    return sorted(patients)

def list_all_patients_with_any_results() -> Dict[str, List[str]]:
    """Lista pacientes con resultados para cada modelo."""
    results = {}
    for model in list_models():
        results[model] = list_patients_with_results(model)
    return results
