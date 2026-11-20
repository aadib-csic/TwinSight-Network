#!/usr/bin/env python3
"""
twin_agent_v2.py - TwinSight Agent V2 (Multi-agente)
Arquitectura: ORQUESTADOR (LLM) + AGENTE EJECUTOR + AGENTE REACT
Soporte: nnUNet, SwinUNETR (Task08), SwinUNETR (Task03)
Análisis de HTML + Aprendizaje inverso
"""

import os
import sys
import json
import argparse
from typing import Dict, Any, List, Optional

# =====================================================================
# IMPORTAR AGENTES
# =====================================================================

# Agente EJECUTOR
from agents.executor import (
    generate_twin,
    get_metrics,
    compare_patients,
    compare_models,
    list_patients,
    batch_process,
    list_models
)

# Agente REACT (con análisis de HTML)
from agents.react import (
    analyze_results,
    suggest_improvements,
    learn_from_test_results,
    compare_models_for_patient,
    summarize_twin,
    suggest_parameters_for_patient,
    get_learning_status,
    scan_htmls,
    scan_model_htmls,
    generate_html_report
)

# Base
from agents.base import MODELS, get_model_info

# =====================================================================
# VERIFICAR LANGGRAPH (opcional)
# =====================================================================

try:
    from langgraph.graph import StateGraph, END
    from langchain_ollama import OllamaLLM
    HAS_LANGGRAPH = True
except ImportError:
    HAS_LANGGRAPH = False
    print("⚠️ LangGraph no instalado. Usando modo CLI simple.")
    print("   Instala: pip install langgraph langchain-ollama")

# =====================================================================
# ORQUESTADOR CON LLM (si LangGraph está disponible)
# =====================================================================

def get_orchestrator_llm():
    """Inicializa el LLM orquestador."""
    try:
        from langchain_ollama import OllamaLLM
        return OllamaLLM(model="qwen2.5:32b-instruct-q8_0")
    except ImportError:
        return None


def orchestrator_decision(query: str, patient_id: str = None) -> Dict[str, Any]:
    """
    El LLM decide qué agente/herramienta ejecutar.
    """
    llm = get_orchestrator_llm()

    if not llm:
        return {"agent": "executor", "tool": "", "error": "LLM no disponible"}

    # Lista de herramientas disponibles
    executor_tools = ["generate_twin", "get_metrics", "compare_patients", "compare_models", "list_patients", "batch_process"]
    react_tools = ["analyze_results", "suggest_improvements", "learn_from_test_results", "compare_models_for_patient", "summarize_twin", "scan_htmls"]

    prompt = f"""
    Eres un orquestador de agentes para gemelos digitales hepáticos.

    AGENTE EJECUTOR (acciones): {', '.join(executor_tools)}
    AGENTE REACT (razonamiento): {', '.join(react_tools)}

    Modelos disponibles: nnunet, swin_task08, swin_task03

    Consulta del usuario: {query}
    Paciente: {patient_id or 'No especificado'}

    Responde SOLO con el formato: "agent:tool:model"
    Ejemplos:
    - "executor:generate_twin:nnunet"
    - "react:analyze_results:swin_task08"
    - "executor:compare_models:nnunet"
    - "react:scan_htmls:nnunet"
    """

    response = llm.invoke(prompt).strip().lower()
    parts = response.split(":")

    agent = parts[0] if len(parts) > 0 else "executor"
    tool = parts[1] if len(parts) > 1 else ""
    model = parts[2] if len(parts) > 2 else "nnunet"

    return {"agent": agent, "tool": tool, "model": model}


def execute_action(decision: Dict[str, Any], patient_id: str = None, **kwargs) -> Dict[str, Any]:
    """
    Ejecuta la acción decidida por el orquestador.
    """
    agent = decision.get("agent", "executor")
    tool = decision.get("tool", "")
    model = decision.get("model", "nnunet")

    if not tool:
        tool = "generate_twin" if patient_id else "list_patients"

    try:
        if agent == "executor":
            if tool == "generate_twin":
                return generate_twin(patient_id, model)
            elif tool == "get_metrics":
                return get_metrics(patient_id, model)
            elif tool == "compare_patients" and "id2" in kwargs:
                return compare_patients(patient_id, kwargs["id2"], model)
            elif tool == "compare_models":
                return compare_models(patient_id)
            elif tool == "list_patients":
                return {"patients": list_patients(model)}
            elif tool == "batch_process":
                return batch_process(model, kwargs.get("limit", 10))
            else:
                return {"error": f"Herramienta {tool} no reconocida en AGENTE EJECUTOR"}

        elif agent == "react":
            if tool == "analyze_results":
                return analyze_results(patient_id, model)
            elif tool == "suggest_improvements":
                return suggest_improvements(patient_id, model)
            elif tool == "learn_from_test_results":
                return learn_from_test_results(model, kwargs.get("limit", 20))
            elif tool == "compare_models_for_patient":
                return compare_models_for_patient(patient_id)
            elif tool == "summarize_twin":
                return {"summary": summarize_twin(patient_id, model)}
            elif tool == "scan_htmls":
                if patient_id:
                    # Si se especifica paciente, analizar ese HTML específico
                    return analyze_results(patient_id, model)
                else:
                    # Escanear todos los HTML del modelo
                    return scan_htmls(model)
            elif tool == "suggest_parameters":
                return suggest_parameters_for_patient(patient_id, model)
            else:
                return {"error": f"Herramienta {tool} no reconocida en AGENTE REACT"}

        else:
            return {"error": f"Agente {agent} no reconocido"}

    except Exception as e:
        return {"error": str(e)}


# =====================================================================
# MODO INTERACTIVO (CLI)
# =====================================================================

def print_help():
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║              TWINSIGHT AGENT V2 - COMANDOS                                   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  📊 ANÁLISIS DE RESULTADOS:                                                  ║
║    analyze <id> [model]       - Analiza resultados de un paciente            ║
║    scan [model]               - Escanea y analiza todos los HTMLs            ║
║    learn [model]              - Aprendizaje inverso desde HTMLs              ║
║    knowledge                  - Muestra el conocimiento del agente           ║
║                                                                              ║
║  🔧 GENERACIÓN DE GEMELOS:                                                   ║
║    generate <id> [model]      - Genera gemelo digital                        ║
║    batch [N] [model]          - Procesa N pacientes en lote                  ║
║    list [model]               - Lista pacientes disponibles                  ║
║                                                                              ║
║  📈 ANÁLISIS Y MÉTRICAS:                                                     ║
║    metrics <id> [model]       - Muestra métricas de un paciente              ║
║    compare <id1> <id2> [model] - Compara dos pacientes                       ║
║    compare_models <id>        - Compara todos los modelos para un paciente   ║
║    summarize <id> [model]     - Resumen clínico                              ║
║                                                                              ║
║  🎯 MEJORAS Y SUGERENCIAS:                                                   ║
║    suggest <id> [model]       - Sugiere mejoras para un paciente             ║
║    suggest_params <id> [model] - Sugiere parámetros personalizados           ║
║                                                                              ║
║  🧠 MODELOS:                                                                 ║
║    models                     - Lista modelos disponibles                    ║
║                                                                              ║
║  ❓ AYUDA:                                                                   ║
║    help                       - Muestra esta ayuda                          ║
║    exit                       - Salir                                       ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  MODELOS DISPONIBLES:                                                        ║
║    nnunet      - nnUNet (Task 008: Hepatic Vessel)                          ║
║    swin_task08 - SwinUNETR (Task 08: Hepatic Vessel)                        ║
║    swin_task03 - SwinUNETR (Task 03: Liver)                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)


def interactive_mode():
    """Modo interactivo con el agente."""
    print("\n" + "=" * 70)
    print("🧠 TWINSIGHT AGENT V2 - Multi-Agente con Análisis de HTML")
    print("=" * 70)
    print(f"Modelos disponibles: {', '.join(list_models())}")
    print("Escribe 'help' para ver comandos.")
    print("=" * 70)

    while True:
        try:
            user_input = input("\n> ").strip()
            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit"]:
                print("👋 Saliendo...")
                break

            if user_input.lower() == "help":
                print_help()
                continue

            parts = user_input.split()
            command = parts[0].lower()

            # =====================================================================
            # COMANDOS DE ANÁLISIS DE HTML
            # =====================================================================

            if command == "scan":
                model = parts[1] if len(parts) > 1 else None
                print(f"🔍 [AGENTE] Escaneando HTMLs de {model or 'todos los modelos'}...")

                if model:
                    results = scan_model_htmls(model)
                    report = generate_html_report(results)
                    print(f"\n📊 RESUMEN DEL ESCANEO ({model}):")
                    print(f"   Pacientes: {report['total_patients']}")
                    print(f"   Con problemas: {report['patients_with_issues']}")
                    print(f"   Críticos: {report['critical_issues']}")
                    if report.get('issue_types'):
                        print("\n   Tipos de problemas:")
                        for issue_type, patients in report['issue_types'].items():
                            print(f"     - {issue_type}: {len(patients)} pacientes")
                else:
                    results = scan_htmls()
                    print(f"\n📊 RESUMEN DEL ESCANEO (todos los modelos):")
                    print(f"   Total pacientes: {results['total_patients']}")
                    print(f"   Total problemas: {results['total_issues']}")
                    for model_name, report in results['reports'].items():
                        if report['total_patients'] > 0:
                            print(f"\n   {model_name}:")
                            print(f"     - Pacientes: {report['total_patients']}")
                            print(f"     - Con problemas: {report['patients_with_issues']}")
                            print(f"     - Críticos: {report['critical_issues']}")
                continue

            elif command == "learn":
                model = parts[1] if len(parts) > 1 else None
                print(f"🧠 [AGENTE] Iniciando aprendizaje inverso para {model or 'todos los modelos'}...")
                result = learn_from_test_results(model, 20)

                if "error" in result:
                    print(f"❌ Error: {result['error']}")
                else:
                    print(f"\n📊 RESULTADO DEL APRENDIZAJE:")
                    print(f"   Pacientes analizados: {result['patients_analyzed']}")
                    print(f"   Tipos de problemas encontrados: {result['issue_types_found']}")

                    recommendations = result.get('recommendations', {})
                    if recommendations:
                        print("\n📊 RECOMENDACIONES DE PARÁMETROS:")
                        for param, change in recommendations.items():
                            if isinstance(change, dict) and "current" in change and "suggested" in change:
                                print(f"   - {param}: {change['current']} → {change['suggested']}")
                                print(f"     Confianza: {change.get('confidence', 0):.0%}")
                                print(f"     Razón: {change.get('reason', '')}")
                            elif param != "llm_analysis":
                                print(f"   - {param}: {change}")
                    if "llm_analysis" in recommendations:
                        print(f"\n🤖 Análisis del LLM:\n{recommendations['llm_analysis']}")
                continue

            elif command == "knowledge":
                status = get_learning_status()
                print("\n📚 BASE DE CONOCIMIENTO DEL AGENTE:")
                print(f"   Archivo: {status.get('knowledge_file')}")
                print(f"   Patrones aprendidos: {status.get('patterns_learned')}")
                print(f"   Pacientes analizados: {status.get('patients_analyzed')}")
                print(f"   Iteraciones: {status.get('iterations')}")
                print(f"   Última actualización: {status.get('last_update')}")

                html_analysis = status.get('html_analysis', {})
                if html_analysis:
                    print(f"\n📊 ANÁLISIS DE HTML:")
                    print(f"   Total pacientes analizados: {html_analysis.get('total_patients', 0)}")
                    issues = html_analysis.get('issues_by_type', {})
                    if issues:
                        print("   Problemas detectados:")
                        for issue_type, count in issues.items():
                            print(f"     - {issue_type}: {count} pacientes")

                recommendations = status.get('current_recommendations', {})
                if recommendations:
                    print("\n📊 RECOMENDACIONES ACTUALES:")
                    for param, change in recommendations.items():
                        if isinstance(change, dict) and "current" in change and "suggested" in change:
                            print(f"   - {param}: {change['current']} → {change['suggested']}")
                continue

            # =====================================================================
            # COMANDOS DE ANÁLISIS
            # =====================================================================

            elif command == "analyze" and len(parts) >= 2:
                patient_id = parts[1]
                model = parts[2] if len(parts) > 2 else "nnunet"
                result = analyze_results(patient_id, model)
                print(json.dumps(result, indent=2, default=str))
                continue

            elif command == "summarize" and len(parts) >= 2:
                patient_id = parts[1]
                model = parts[2] if len(parts) > 2 else "nnunet"
                result = summarize_twin(patient_id, model)
                print(f"\n{result}")
                continue

            # =====================================================================
            # COMANDOS DE GENERACIÓN
            # =====================================================================

            elif command == "generate" and len(parts) >= 2:
                patient_id = parts[1]
                model = parts[2] if len(parts) > 2 else "nnunet"
                result = generate_twin(patient_id, model)
                print(json.dumps(result, indent=2))
                continue

            elif command == "batch":
                limit = int(parts[1]) if len(parts) > 1 else 10
                model = parts[2] if len(parts) > 2 else "nnunet"
                result = batch_process(model, limit)
                print(json.dumps(result, indent=2))
                continue

            elif command == "list":
                model = parts[1] if len(parts) > 1 else "nnunet"
                patients = list_patients(model, 30)
                print(f"\n📂 Pacientes ({model}): {len(patients)}")
                for p in patients[:20]:
                    print(f"  - {p}")
                if len(patients) > 20:
                    print(f"  ... y {len(patients)-20} más")
                continue

            # =====================================================================
            # COMANDOS DE MÉTRICAS
            # =====================================================================

            elif command == "metrics" and len(parts) >= 2:
                patient_id = parts[1]
                model = parts[2] if len(parts) > 2 else "nnunet"
                result = get_metrics(patient_id, model)
                print(json.dumps(result, indent=2))
                continue

            elif command == "compare" and len(parts) >= 3:
                patient_id1 = parts[1]
                patient_id2 = parts[2]
                model = parts[3] if len(parts) > 3 else "nnunet"
                result = compare_patients(patient_id1, patient_id2, model)
                print(json.dumps(result, indent=2))
                continue

            elif command == "compare_models" and len(parts) >= 2:
                patient_id = parts[1]
                result = compare_models(patient_id)
                print(json.dumps(result, indent=2))
                continue

            # =====================================================================
            # COMANDOS DE SUGERENCIAS
            # =====================================================================

            elif command == "suggest" and len(parts) >= 2:
                patient_id = parts[1]
                model = parts[2] if len(parts) > 2 else "nnunet"
                result = suggest_improvements(patient_id, model)
                print(json.dumps(result, indent=2))
                continue

            elif command == "suggest_params" and len(parts) >= 2:
                patient_id = parts[1]
                model = parts[2] if len(parts) > 2 else "nnunet"
                result = suggest_parameters_for_patient(patient_id, model)
                if "error" in result:
                    print(f"❌ Error: {result['error']}")
                else:
                    print(f"\n🎯 PARÁMETROS PERSONALIZADOS PARA {patient_id}:")
                    for param, value in result.get('suggested_parameters', {}).items():
                        print(f"   - {param}: {value}")
                    print(f"\n📊 Basado en {len(result.get('analysis', {}).get('issues', []))} problemas detectados")
                continue

            # =====================================================================
            # COMANDOS DE MODELOS
            # =====================================================================

            elif command == "models":
                print("\n📂 Modelos disponibles:")
                for name, info in MODELS.items():
                    if name not in ["swin"]:  # Saltar alias
                        print(f"  - {name}: {info['name']} ({info['dataset']})")
                continue

            # =====================================================================
            # COMANDO DESCONOCIDO → USAR ORQUESTADOR
            # =====================================================================

            else:
                if HAS_LANGGRAPH:
                    print(f"🧠 [ORQUESTADOR] Procesando: {user_input}")
                    decision = orchestrator_decision(user_input)
                    print(f"   Decisión: {decision}")

                    # Extraer patient_id del comando
                    patient_id = None
                    for word in user_input.split():
                        if word.startswith("hepaticvessel_") or word.startswith("liver_"):
                            patient_id = word
                            break

                    if patient_id:
                        result = execute_action(decision, patient_id)
                    else:
                        result = execute_action(decision)

                    print("\n📊 Resultado:")
                    print(json.dumps(result, indent=2, default=str))
                else:
                    print(f"❌ Comando desconocido: {command}")
                    print("   Usa 'help' para ver los comandos disponibles.")

        except KeyboardInterrupt:
            print("\n👋 Saliendo...")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()


# =====================================================================
# MODO BATCH (línea de comandos)
# =====================================================================

def batch_mode():
    """Modo batch para uso en scripts."""
    parser = argparse.ArgumentParser(description="TwinSight Agent V2")
    parser.add_argument("--scan", type=str, nargs="?", const="all", help="Escanea HTMLs de un modelo (o todos)")
    parser.add_argument("--learn", type=str, nargs="?", const="all", help="Aprendizaje inverso para un modelo (o todos)")
    parser.add_argument("--analyze", type=str, help="Analiza un paciente: patient_id[,model]")
    parser.add_argument("--generate", type=str, help="Genera gemelo: patient_id[,model]")
    parser.add_argument("--metrics", type=str, help="Muestra métricas: patient_id[,model]")
    parser.add_argument("--compare", nargs=2, help="Compara pacientes: id1 id2 [model]")
    parser.add_argument("--compare-models", type=str, help="Compara modelos para un paciente")
    parser.add_argument("--list", type=str, nargs="?", const="nnunet", help="Lista pacientes de un modelo")
    parser.add_argument("--models", action="store_true", help="Lista modelos disponibles")
    parser.add_argument("--knowledge", action="store_true", help="Muestra conocimiento del agente")

    args = parser.parse_args()

    if args.models:
        print(json.dumps(list_models(), indent=2))
        return

    if args.knowledge:
        status = get_learning_status()
        print(json.dumps(status, indent=2, default=str))
        return

    if args.list:
        model = args.list if args.list != "True" else "nnunet"
        patients = list_patients(model)
        print(json.dumps(patients, indent=2))
        return

    if args.scan:
        model = None if args.scan == "all" else args.scan
        if model:
            results = scan_model_htmls(model)
            report = generate_html_report(results)
            print(json.dumps({"model": model, "report": report}, indent=2, default=str))
        else:
            results = scan_htmls()
            print(json.dumps(results, indent=2, default=str))
        return

    if args.learn:
        model = None if args.learn == "all" else args.learn
        result = learn_from_test_results(model)
        print(json.dumps(result, indent=2, default=str))
        return

    if args.analyze:
        parts = args.analyze.split(",")
        patient_id = parts[0]
        model = parts[1] if len(parts) > 1 else "nnunet"
        result = analyze_results(patient_id, model)
        print(json.dumps(result, indent=2, default=str))
        return

    if args.generate:
        parts = args.generate.split(",")
        patient_id = parts[0]
        model = parts[1] if len(parts) > 1 else "nnunet"
        result = generate_twin(patient_id, model)
        print(json.dumps(result, indent=2))
        return

    if args.metrics:
        parts = args.metrics.split(",")
        patient_id = parts[0]
        model = parts[1] if len(parts) > 1 else "nnunet"
        result = get_metrics(patient_id, model)
        print(json.dumps(result, indent=2))
        return

    if args.compare:
        patient_id1 = args.compare[0]
        patient_id2 = args.compare[1]
        model = args.compare[2] if len(args.compare) > 2 else "nnunet"
        result = compare_patients(patient_id1, patient_id2, model)
        print(json.dumps(result, indent=2))
        return

    if args.compare_models:
        result = compare_models(args.compare_models)
        print(json.dumps(result, indent=2))
        return

    print("Usa --help para ver las opciones disponibles.")


# =====================================================================
# MAIN
# =====================================================================

def main():
    if len(sys.argv) > 1:
        batch_mode()
    else:
        interactive_mode()


if __name__ == "__main__":
    main()
