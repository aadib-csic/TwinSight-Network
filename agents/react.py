"""
agents/react.py - Agente REACT con análisis de HTML y aprendizaje inverso
"""

import json
import os
import re
import glob
from typing import Dict, Any, List, Optional
from datetime import datetime
from collections import defaultdict

from agents.base import (
    get_patient_metrics,
    list_available_patients,
    list_patients_with_results,
    get_results_dir,
    get_patient_html_path,
    get_patient_report_path,
    get_patient_nifti_path,
    MODELS,
    extract_metrics_from_report
)

# =====================================================================
# CONFIGURACIÓN DE APRENDIZAJE
# =====================================================================

KNOWLEDGE_FILE = "agent_knowledge.json"
LEARNING_HISTORY_FILE = "learning_history.json"

# =====================================================================
# ANÁLISIS DE HTML
# =====================================================================

def analyze_html_file(html_path: str) -> Dict[str, Any]:
    """
    Analiza un archivo HTML y extrae información.
    """
    if not os.path.exists(html_path):
        return {"error": "HTML no encontrado"}

    with open(html_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Extraer marcadores (T1, T2, T3, ...)
    tumor_markers = re.findall(r'T\d+', content)

    # Extraer colores
    colors = re.findall(r'#[\da-fA-F]{6}', content)

    # Detectar estructura 3D
    has_3d = "scene" in content
    has_legend = "legend" in content
    has_markers = "marker" in content or "Scatter3d" in content

    # Contar etiquetas de marcadores únicas
    unique_markers = len(set(tumor_markers))

    return {
        "html_path": html_path,
        "tumor_count": unique_markers,
        "raw_markers_found": len(tumor_markers),
        "colors_used": list(set(colors))[:10],
        "has_3d_view": has_3d,
        "has_legend": has_legend,
        "has_markers": has_markers,
        "html_size": len(content)
    }


def analyze_report_file(report_path: str) -> Dict[str, Any]:
    """
    Analiza un archivo de reporte y extrae métricas.
    """
    if not os.path.exists(report_path):
        return {"error": "Reporte no encontrado", "has_metrics": False}

    with open(report_path, "r", encoding="utf-8") as f:
        content = f.read()

    metrics = {}

    # Patrones para volumen del órgano
    organ_patterns = [
        r"(?:Liver|Hígado|Vascular Tree)\s*[Vv]olume\s*:\s*([\d.]+)\s*cc",
        r"Liver\s+[Vv]olume\s*:\s*([\d.]+)\s*cc",
        r"Hígado\s*:\s*([\d.]+)\s*cc",
        r"Vascular Tree\s*:\s*([\d.]+)\s*cc",
    ]

    for pattern in organ_patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            metrics["organ_cc"] = float(match.group(1))
            break

    # Patrones para tumor
    tumor_patterns = [
        r"(?:Tumor|Lesion)\s*[Vv]olume\s*:\s*([\d.]+)\s*cc",
        r"Tumor\s+[Ll]oad\s*:\s*([\d.]+)\s*cc",
        r"Tumor\s*:\s*([\d.]+)\s*cc",
        r"Tumor\s+[Vv]olume\s+\(cc\)\s*:\s*([\d.]+)",
    ]

    for pattern in tumor_patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            metrics["tumor_cc"] = float(match.group(1))
            break

    # Patrones para burden
    burden_patterns = [
        r"(?:Tumor Burden|Burden)\s*:\s*([\d.]+)\s*%",
        r"Burden\s+is\s+([\d.]+)\s*%",
        r"Tumor\s+[Bb]urden\s*:\s*([\d.]+)\s*%",
    ]

    for pattern in burden_patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            metrics["burden"] = float(match.group(1))
            break

    # Patrones para número de tumores
    count_patterns = [
        r"(?:Tumors?|Lesions?)\s*(?:detected|identified|count)?\s*:?\s*(\d+)",
        r"Number\s+of\s+(?:tumors|lesions)\s*:\s*(\d+)",
        r"(\d+)\s+(?:tumors|lesions)\s+(?:detected|identified)",
    ]

    for pattern in count_patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            metrics["tumor_count"] = int(match.group(1))
            break

    # Si hay organ_cc y tumor_cc pero no burden, calcularlo
    if "organ_cc" in metrics and "tumor_cc" in metrics and "burden" not in metrics:
        if metrics["organ_cc"] > 0:
            metrics["burden"] = (metrics["tumor_cc"] / metrics["organ_cc"]) * 100
        else:
            metrics["burden"] = 0

    return {
        "report_path": report_path,
        "metrics": metrics,
        "has_metrics": bool(metrics)
    }


def detect_discrepancies(html_info: Dict, report_info: Dict) -> List[Dict]:
    """
    Detecta discrepancias entre el HTML y el reporte.
    """
    discrepancies = []

    html_tumors = html_info.get("tumor_count", 0)
    report_metrics = report_info.get("metrics", {})
    report_tumors = report_metrics.get("tumor_count", 0)
    report_organ = report_metrics.get("organ_cc", 0)
    report_tumor_cc = report_metrics.get("tumor_cc", 0)

    # Si no hay reporte, no se puede comparar
    if not report_info.get("has_metrics", False):
        discrepancies.append({
            "type": "no_report",
            "severity": "medium",
            "description": "No se encontró reporte para comparar",
            "suggestion": "Verificar que el reporte se generó correctamente"
        })
        return discrepancies

    # Discrepancia 1: Demasiados marcadores en HTML
    if html_tumors > 0 and report_tumors > 0:
        ratio = html_tumors / report_tumors
        if ratio > 10:
            discrepancies.append({
                "type": "too_many_markers",
                "severity": "critical",
                "description": f"HTML muestra {html_tumors} marcadores pero el reporte indica {report_tumors} tumores (ratio {ratio:.1f}x)",
                "suggestion": "Aumentar MIN_VOXELS_TUMOR para reducir marcadores falsos"
            })

    # Discrepancia 2: Marcadores pero sin tumor en reporte
    if html_tumors > 0 and report_tumors == 0:
        discrepancies.append({
            "type": "markers_without_tumor",
            "severity": "critical",
            "description": f"HTML muestra {html_tumors} marcadores pero el reporte no indica tumores",
            "suggestion": "Aumentar MIN_VOXELS_TUMOR o verificar filtro anatómico"
        })

    # Discrepancia 3: HTML sin marcadores pero con tumor
    if html_tumors == 0 and report_tumors > 0:
        discrepancies.append({
            "type": "missing_markers",
            "severity": "high",
            "description": f"HTML no muestra marcadores pero el reporte indica {report_tumors} tumores",
            "suggestion": "Verificar que los marcadores se generan correctamente en mesh_utils"
        })

    # Discrepancia 4: Tumores en HTML pero sin volumen
    if html_tumors > 0 and report_tumor_cc == 0:
        discrepancies.append({
            "type": "zero_volume_markers",
            "severity": "high",
            "description": f"HTML muestra {html_tumors} marcadores pero el volumen tumoral es 0",
            "suggestion": "Aumentar MIN_VOXELS_TUMOR para eliminar fragmentos sin volumen"
        })

    # Discrepancia 5: Órgano muy bajo
    if report_organ > 0 and report_organ < 50:
        discrepancies.append({
            "type": "organ_too_low",
            "severity": "high",
            "description": f"Órgano: {report_organ:.2f} cc (muy bajo)",
            "suggestion": "Reducir MIN_VOXELS_ORGAN para capturar más órgano"
        })

    return discrepancies


def generate_html_report(html_data: List[Dict]) -> Dict[str, Any]:
    """
    Genera un resumen del análisis de HTML.
    """
    total_patients = len(html_data)
    patients_with_issues = [d for d in html_data if d.get("discrepancies")]
    critical_issues = [d for d in patients_with_issues if d.get("severity") == "critical"]

    # Agrupar por tipo de discrepancia
    issue_types = {}
    for data in html_data:
        for discrepancy in data.get("discrepancies", []):
            issue_type = discrepancy["type"]
            if issue_type not in issue_types:
                issue_types[issue_type] = []
            issue_types[issue_type].append(data["patient"])

    return {
        "total_patients": total_patients,
        "patients_with_issues": len(patients_with_issues),
        "critical_issues": len(critical_issues),
        "issue_types": issue_types,
        "summary": f"Analizados {total_patients} pacientes. {len(patients_with_issues)} con problemas, {len(critical_issues)} críticos."
    }


# =====================================================================
# ESCANEO DE HTML
# =====================================================================

def scan_model_htmls(model: str) -> List[Dict]:
    """
    Escanea los HTML de un modelo específico.
    """
    results_dir = get_results_dir(model)
    if not results_dir or not os.path.exists(results_dir):
        return []

    # Buscar todos los HTML en el directorio
    html_files = glob.glob(os.path.join(results_dir, "*", "*.html"))

    html_data = []
    for html_path in html_files:
        patient = os.path.basename(os.path.dirname(html_path))

        # Analizar HTML
        html_info = analyze_html_file(html_path)

        # Analizar reporte
        report_path = get_patient_report_path(patient, model)
        report_info = analyze_report_file(report_path)

        # Detectar discrepancias
        discrepancies = detect_discrepancies(html_info, report_info)

        html_data.append({
            "patient": patient,
            "model": model,
            "html_path": html_path,
            "html_info": html_info,
            "report_info": report_info,
            "discrepancies": discrepancies,
            "severity": max([d["severity"] for d in discrepancies]) if discrepancies else "none"
        })

    return html_data


def scan_all_htmls() -> Dict[str, Any]:
    """
    Escanea todos los HTML disponibles de todos los modelos.
    """
    results = {}

    models = ["nnunet", "swin_task08", "swin_task03"]
    for model in models:
        print(f"📂 Escaneando {model}...")
        results[model] = scan_model_htmls(model)

    return results


# =====================================================================
# ANÁLISIS DE RESULTADOS DE UN PACIENTE (combinado con HTML)
# =====================================================================

def analyze_results(patient_id: str, model: str = "nnunet") -> Dict[str, Any]:
    """
    Analiza los resultados de un paciente usando HTML + Reporte + NIfTI.
    """
    print(f"🧠 [REACT] Analizando {patient_id} con {model}...")

    # 1. Obtener rutas
    html_path = get_patient_html_path(patient_id, model)
    report_path = get_patient_report_path(patient_id, model)
    nifti_path = get_patient_nifti_path(patient_id, model)

    # 2. Analizar HTML
    html_info = analyze_html_file(html_path) if os.path.exists(html_path) else {"error": "HTML no encontrado"}

    # 3. Analizar Reporte
    report_info = analyze_report_file(report_path) if os.path.exists(report_path) else {"error": "Reporte no encontrado"}

    # 4. Obtener métricas (prioriza NIfTI, fallback reporte)
    metrics = get_patient_metrics(patient_id, model)
    if not metrics and report_info.get("has_metrics"):
        metrics = report_info["metrics"]
        metrics["organ_type"] = "vessels" if model in ["nnunet", "swin_task08"] else "liver"
        metrics["source"] = "report"

    # 5. Detectar discrepancias entre HTML y reporte
    discrepancies = detect_discrepancies(html_info, report_info)

    issues = []

    # 6. Detectar problemas en métricas
    if metrics:
        organ_cc = metrics.get("organ_cc", 0)
        tumor_cc = metrics.get("tumor_cc", 0)
        burden = metrics.get("burden", 0)
        tumor_count = metrics.get("tumor_count", 0)
        organ_type = metrics.get("organ_type", "organ")

        if burden > 100:
            issues.append({
                "type": "burden_impossible",
                "severity": "critical",
                "description": f"Burden {burden:.1f}% > 100% (imposible)",
                "suggestion": f"MIN_VOXELS_{organ_type.upper()} demasiado alto o MIN_VOXELS_TUMOR demasiado bajo",
                "source": "metrics"
            })

        if tumor_count > 100:
            issues.append({
                "type": "false_positives_massive",
                "severity": "critical",
                "description": f"{tumor_count} tumores (ruido)",
                "suggestion": "Aumentar MIN_VOXELS_TUMOR a 100+",
                "source": "metrics"
            })

        if organ_cc > 0 and organ_cc < 50:
            issues.append({
                "type": f"{organ_type}_too_low",
                "severity": "high",
                "description": f"{organ_type.capitalize()}: {organ_cc:.2f} cc (muy bajo)",
                "suggestion": f"Reducir MIN_VOXELS_{organ_type.upper()} a 100-200",
                "source": "metrics"
            })

        if tumor_cc > 200:
            issues.append({
                "type": "tumor_too_high",
                "severity": "medium",
                "description": f"Tumor: {tumor_cc:.2f} cc (muy alto)",
                "suggestion": "Verificar si es real o falso positivo",
                "source": "metrics"
            })

        if tumor_count > 50 and tumor_cc < 50:
            issues.append({
                "type": "small_fragments",
                "severity": "high",
                "description": f"{tumor_count} tumores pero solo {tumor_cc:.2f} cc total",
                "suggestion": "Aumentar MIN_VOXELS_TUMOR a 50+",
                "source": "metrics"
            })

    # 7. Añadir discrepancias del HTML a los issues
    issues.extend(discrepancies)

    # 8. Determinar severidad
    severity = "none"
    if issues:
        severities = [i.get("severity", "medium") for i in issues]
        if "critical" in severities:
            severity = "critical"
        elif "high" in severities:
            severity = "high"
        else:
            severity = "medium"

    return {
        "patient_id": patient_id,
        "model": model,
        "metrics": metrics or {},
        "html_info": html_info,
        "report_info": report_info,
        "issues": issues,
        "severity": severity,
        "nifti_path": nifti_path,
        "html_path": html_path,
        "report_path": report_path
    }


# =====================================================================
# DETECCIÓN DE PATRONES
# =====================================================================

def detect_patterns(patient_data: List[Dict]) -> List[Dict]:
    """
    Detecta patrones en los resultados de múltiples pacientes.
    """
    patterns = []

    # Agrupar por tipo de problema
    issues_by_type = defaultdict(list)
    for data in patient_data:
        for issue in data.get("issues", []):
            issue_type = issue["type"]
            issues_by_type[issue_type].append({
                "patient_id": data["patient_id"],
                "metrics": data["metrics"],
                "model": data.get("model", "nnunet"),
                "issue": issue
            })

    # Analizar cada tipo de problema
    for issue_type, instances in issues_by_type.items():
        if len(instances) >= 2:  # Aparece en al menos 2 pacientes
            # Buscar patrones comunes
            common_metrics = defaultdict(list)
            for inst in instances:
                metrics = inst["metrics"]
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        common_metrics[key].append(value)

            # Calcular promedios
            avg_metrics = {}
            for key, values in common_metrics.items():
                if values:
                    avg_metrics[key] = sum(values) / len(values)

            # Determinar severidad
            severity_counts = defaultdict(int)
            for inst in instances:
                severity_counts[inst["issue"].get("severity", "medium")] += 1

            # Generar recomendación específica
            suggestion = instances[0]["issue"].get("suggestion", "")

            patterns.append({
                "type": issue_type,
                "frequency": len(instances),
                "affected_patients": [i["patient_id"] for i in instances],
                "models": list(set([i["model"] for i in instances])),
                "average_metrics": avg_metrics,
                "severity": max(severity_counts, key=severity_counts.get),
                "common_suggestion": suggestion,
                "confidence": min(1.0, len(instances) / 5)
            })

    return patterns


# =====================================================================
# BASE DE CONOCIMIENTO
# =====================================================================

def load_knowledge() -> Dict:
    """Carga la base de conocimiento del agente."""
    if os.path.exists(KNOWLEDGE_FILE):
        with open(KNOWLEDGE_FILE, "r") as f:
            return json.load(f)
    return {
        "learned_patterns": [],
        "parameter_recommendations": {},
        "model_performance": {},
        "iteration_history": [],
        "html_analysis": {},
        "created_at": datetime.now().isoformat(),
        "last_update": datetime.now().isoformat()
    }


def save_knowledge(knowledge: Dict):
    """Guarda la base de conocimiento."""
    knowledge["last_update"] = datetime.now().isoformat()
    with open(KNOWLEDGE_FILE, "w") as f:
        json.dump(knowledge, f, indent=2, default=str)


def load_learning_history() -> List[Dict]:
    """Carga el historial de aprendizaje."""
    if os.path.exists(LEARNING_HISTORY_FILE):
        with open(LEARNING_HISTORY_FILE, "r") as f:
            return json.load(f)
    return []


def save_learning_history(entry: Dict):
    """Guarda una entrada en el historial de aprendizaje."""
    history = load_learning_history()
    history.append(entry)
    with open(LEARNING_HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2, default=str)


# =====================================================================
# APRENDIZAJE INVERSO DESDE HTML
# =====================================================================

def learn_from_test_results(model: str = None, limit: int = 20) -> Dict[str, Any]:
    """
    APRENDIZAJE INVERSO: Analiza HTMLs, detecta patrones y aprende.
    """
    print("🧠 [REACT] Iniciando aprendizaje inverso...")

    # 1. Escanear HTMLs
    if model:
        print(f"   📂 Escaneando {model}...")
        html_data = scan_model_htmls(model)
    else:
        print("   📂 Escaneando todos los modelos...")
        all_data = scan_all_htmls()
        html_data = []
        for m, data in all_data.items():
            html_data.extend(data)

    if not html_data:
        return {"error": "No se encontraron HTMLs para analizar"}

    print(f"   📊 Analizando {len(html_data)} pacientes...")

    # 2. Agrupar por tipo de discrepancia
    issues_by_type = defaultdict(list)
    for data in html_data:
        for discrepancy in data.get("discrepancies", []):
            issue_type = discrepancy["type"]
            issues_by_type[issue_type].append({
                "patient": data["patient"],
                "model": data["model"],
                "html_info": data["html_info"],
                "report_info": data["report_info"],
                "discrepancy": discrepancy
            })

    # 3. Generar recomendaciones
    print("   💡 Generando recomendaciones...")
    recommendations = {}
    changes_made = False

    # Mapeo de tipos de discrepancia a ajustes de parámetros
    param_mapping = {
        "too_many_markers": {
            "param": "MIN_VOXELS_TUMOR",
            "current": 30,
            "suggested": 100,
            "reason": "Demasiados marcadores en HTML"
        },
        "markers_without_tumor": {
            "param": "MIN_VOXELS_TUMOR",
            "current": 30,
            "suggested": 80,
            "reason": "Marcadores sin tumor en reporte"
        },
        "zero_volume_markers": {
            "param": "MIN_VOXELS_TUMOR",
            "current": 30,
            "suggested": 60,
            "reason": "Marcadores sin volumen"
        },
        "organ_too_low": {
            "param": "MIN_VOXELS_ORGAN",
            "current": 500,
            "suggested": 200,
            "reason": "Órgano muy bajo"
        },
        "too_many_markers_html": {
            "param": "MIN_VOXELS_TUMOR",
            "current": 30,
            "suggested": 100,
            "reason": "Demasiados marcadores en HTML"
        }
    }

    for issue_type, instances in issues_by_type.items():
        if len(instances) >= 2:  # Al menos 2 pacientes con el mismo problema
            if issue_type in param_mapping:
                mapping = param_mapping[issue_type]
                param_name = mapping["param"]

                # Si ya hay una recomendación para este parámetro, usar la más estricta
                if param_name in recommendations:
                    if mapping["suggested"] > recommendations[param_name]["suggested"]:
                        recommendations[param_name] = {
                            "current": mapping["current"],
                            "suggested": mapping["suggested"],
                            "confidence": min(1.0, len(instances) / 5),
                            "reason": f"{mapping['reason']} en {len(instances)} pacientes"
                        }
                else:
                    recommendations[param_name] = {
                        "current": mapping["current"],
                        "suggested": mapping["suggested"],
                        "confidence": min(1.0, len(instances) / 5),
                        "reason": f"{mapping['reason']} en {len(instances)} pacientes"
                    }
                changes_made = True

    # 4. Si no hay cambios, usar LLM para análisis adicional
    if not changes_made:
        llm = get_llm()
        if llm:
            print("   🤖 Usando LLM para análisis adicional...")
            prompt = f"""
            Análisis de HTML de {len(html_data)} pacientes:
            {json.dumps([{
                "patient": d["patient"],
                "html_tumors": d["html_info"].get("tumor_count", 0),
                "discrepancies": [i["type"] for i in d.get("discrepancies", [])]
            } for d in html_data[:10]], indent=2)}

            ¿Qué patrones detectas? ¿Qué parámetros recomiendas ajustar?
            """
            recommendations["llm_analysis"] = llm.invoke(prompt)

    # 5. Guardar conocimiento
    knowledge = load_knowledge()
    knowledge["html_analysis"] = {
        "total_patients": len(html_data),
        "issues_by_type": {k: len(v) for k, v in issues_by_type.items()},
        "recommendations": recommendations,
        "last_analysis": datetime.now().isoformat()
    }
    knowledge["parameter_recommendations"] = recommendations
    save_knowledge(knowledge)

    print(f"   💾 Conocimiento guardado")

    return {
        "status": "success",
        "patients_analyzed": len(html_data),
        "issue_types_found": len(issues_by_type),
        "recommendations": recommendations,
        "knowledge_file": KNOWLEDGE_FILE,
        "changes_made": changes_made
    }


# =====================================================================
# FUNCIONES PARA LLM
# =====================================================================

def get_llm():
    """Obtiene el LLM para explicaciones."""
    try:
        from langchain_ollama import OllamaLLM
        return OllamaLLM(model="qwen2.5:32b-instruct-q8_0")
    except ImportError:
        return None


# =====================================================================
# OTRAS FUNCIONES
# =====================================================================

def summarize_twin(patient_id: str, model: str = "nnunet") -> str:
    """Genera un resumen clínico del gemelo digital."""
    print(f"📋 [REACT] Generando resumen para {patient_id} ({model})...")

    metrics = get_patient_metrics(patient_id, model)
    if not metrics:
        return f"No hay métricas disponibles para {patient_id} con {model}"

    llm = get_llm()
    if not llm:
        organ_type = metrics.get("organ_type", "Órgano")
        return f"Paciente {patient_id} ({model}): {organ_type} {metrics.get('organ_cc', 0):.2f} cc, Tumor {metrics.get('tumor_cc', 0):.2f} cc, Burden {metrics.get('burden', 0):.2f}%"

    prompt = f"""
    Eres un radiólogo experto en oncología.
    Paciente: {patient_id}
    Modelo: {model}
    {metrics.get('organ_type', 'Órgano').capitalize()}: {metrics.get('organ_cc', 0):.2f} cc
    Tumor: {metrics.get('tumor_cc', 0):.2f} cc
    Burden: {metrics.get('burden', 0):.2f}%
    Tumores: {metrics.get('tumor_count', 0)}

    Genera un resumen clínico breve (2-3 frases) que incluya:
    1. Estado del órgano
    2. Evaluación de la carga tumoral
    3. Recomendación quirúrgica si aplica
    """

    return llm.invoke(prompt)


def suggest_improvements(patient_id: str, model: str = "nnunet") -> Dict[str, Any]:
    """Sugiere mejoras para un modelo basado en un paciente."""
    print(f"💡 [REACT] Sugiriendo mejoras para {patient_id} ({model})...")

    analysis = analyze_results(patient_id, model)
    if "error" in analysis:
        return {"patient_id": patient_id, "model": model, "error": analysis["error"]}

    changes = {}
    organ_type = analysis.get("metrics", {}).get("organ_type", "organ")

    for issue in analysis.get("issues", []):
        suggestion = issue.get("suggestion", "")
        if "MIN_VOXELS_TUMOR" in suggestion or "MIN_VOXELS_LIVER" in suggestion or "MIN_VOXELS_VESSELS" in suggestion:
            param = "MIN_VOXELS_TUMOR"
            if "VESSELS" in suggestion:
                param = "MIN_VOXELS_VESSELS"
            elif "LIVER" in suggestion:
                param = "MIN_VOXELS_LIVER"
            elif "ORGAN" in suggestion:
                param = f"MIN_VOXELS_{organ_type.upper()}"
            changes[param] = {"current": 500 if "VESSELS" in param or "ORGAN" in param else 30, "suggested": 200 if "VESSELS" in param or "ORGAN" in param else 100}

        if "DILATION_ITERATIONS" in suggestion:
            changes["DILATION_ITERATIONS"] = {"current": 3, "suggested": 5}

    return {
        "patient_id": patient_id,
        "model": model,
        "analysis": analysis,
        "suggested_changes": changes
    }


def compare_models_for_patient(patient_id: str) -> Dict[str, Any]:
    """Compara todos los modelos para un mismo paciente."""
    print(f"🔬 [REACT] Comparando modelos para {patient_id}...")

    results = {}
    for model in ["nnunet", "swin_task08", "swin_task03"]:
        analysis = analyze_results(patient_id, model)
        if "error" not in analysis:
            metrics = analysis.get("metrics", {})
            results[model] = {
                "organ_cc": metrics.get("organ_cc", 0),
                "tumor_cc": metrics.get("tumor_cc", 0),
                "burden": metrics.get("burden", 0),
                "tumor_count": metrics.get("tumor_count", 0),
                "issues": len(analysis.get("issues", []))
            }

    return {"patient_id": patient_id, "models": results}


def get_learning_status() -> Dict[str, Any]:
    """Devuelve el estado actual del aprendizaje del agente."""
    knowledge = load_knowledge()
    history = load_learning_history()

    return {
        "knowledge_file": KNOWLEDGE_FILE,
        "patterns_learned": len(knowledge.get("learned_patterns", [])),
        "patients_analyzed": knowledge.get("patients_analyzed", 0),
        "iterations": len(history),
        "last_update": knowledge.get("last_update", "Nunca"),
        "current_recommendations": knowledge.get("parameter_recommendations", {}),
        "html_analysis": knowledge.get("html_analysis", {})
    }


def scan_htmls(model: str = None) -> Dict[str, Any]:
    """
    Escanea y analiza todos los HTML disponibles.
    Si se especifica un modelo, solo escanea ese modelo.
    """
    print("🔍 [REACT] Escaneando HTMLs...")

    if model:
        results = {model: scan_model_htmls(model)}
    else:
        results = scan_all_htmls()

    # Generar reporte por modelo
    reports = {}
    for model_name, data in results.items():
        reports[model_name] = generate_html_report(data)

    return {
        "results": results,
        "reports": reports,
        "total_patients": sum(len(data) for data in results.values()),
        "total_issues": sum(r["patients_with_issues"] for r in reports.values())
    }


def suggest_parameters_for_patient(patient_id: str, model: str = "nnunet") -> Dict[str, Any]:
    """
    Sugiere parámetros personalizados basados en el conocimiento acumulado.
    """
    print(f"🎯 [REACT] Generando parámetros personalizados para {patient_id}...")

    analysis = analyze_results(patient_id, model)
    if "error" in analysis:
        return {"error": analysis["error"]}

    knowledge = load_knowledge()
    recommendations = knowledge.get("parameter_recommendations", {})

    suggested_params = {
        "MIN_VOXELS_TUMOR": 30,
        "MIN_VOXELS_VESSELS": 500,
        "DILATION_ITERATIONS": 3
    }

    # Aplicar recomendaciones del conocimiento
    for param, rec in recommendations.items():
        if param in suggested_params:
            suggested_params[param] = rec.get("suggested", suggested_params[param])

    metrics = analysis.get("metrics", {})
    organ_cc = metrics.get("organ_cc", 0)
    tumor_cc = metrics.get("tumor_cc", 0)
    burden = metrics.get("burden", 0)
    tumor_count = metrics.get("tumor_count", 0)
    organ_type = metrics.get("organ_type", "organ")

    # Ajustes específicos del paciente
    if burden > 100:
        suggested_params[f"MIN_VOXELS_{organ_type.upper()}"] = min(
            suggested_params.get(f"MIN_VOXELS_{organ_type.upper()}", 500), 200
        )

    if tumor_count > 50:
        suggested_params["MIN_VOXELS_TUMOR"] = max(suggested_params["MIN_VOXELS_TUMOR"], 100)

    if organ_cc > 0 and organ_cc < 50:
        suggested_params[f"MIN_VOXELS_{organ_type.upper()}"] = min(
            suggested_params.get(f"MIN_VOXELS_{organ_type.upper()}", 500), 150
        )

    if tumor_count > 50 and tumor_cc < 50:
        suggested_params["MIN_VOXELS_TUMOR"] = max(suggested_params["MIN_VOXELS_TUMOR"], 60)

    return {
        "patient_id": patient_id,
        "model": model,
        "analysis": analysis,
        "suggested_parameters": suggested_params,
        "based_on_knowledge": bool(knowledge.get("learned_patterns"))
    }
