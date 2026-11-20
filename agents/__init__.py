# agents/__init__.py
from .base import MODELS, get_model_info, get_patient_metrics, list_available_patients
from .executor import (
    generate_twin,
    get_metrics,
    compare_patients,
    compare_models,
    list_patients,
    batch_process,
    list_models
)
from .react import (
    analyze_results,
    suggest_improvements,
    learn_from_test_results,  # <--- Cambiado: antes era learn_from_patients
    compare_models_for_patient,
    summarize_twin,
    suggest_parameters_for_patient,
    get_learning_status
)
