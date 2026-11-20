#!/usr/bin/env python3
"""
main_tumorsight.py — TwinSight Network
Pipeline principal: CT hepático → Gemelo Digital HTML

Uso:
    python -u main_tumorsight.py --patient hepaticvessel_009
    python -u main_tumorsight.py --sample 10

Requisitos:
    - nnUNet v2 instalado y variables de entorno configuradas
    - TotalSegmentator instalado
    - Ollama corriendo con qwen2.5:7b (para el asistente IA en el HTML)

Variables de entorno necesarias:
    nnUNet_raw          → directorio con datos crudos
    nnUNet_preprocessed → directorio de preprocesado
    nnUNet_results      → directorio con checkpoints del modelo

Proyecto Momentum / SGAI-CSIC · 2026
"""

import os
import sys
import glob
import argparse
import random
import subprocess
import numpy as np
import nibabel as nib

from src.mesh_utils_tumorsight import calculate_volumes, get_tumor_gps, generate_plotly_html


# ─────────────────────────────────────────────────────────────────
#  CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────

INPUT_DIR           = os.environ.get("TWINSIGHT_INPUT",
                      "data/Task08_HepaticVessel/imagesTr")
NNUNET_RESULTS_DIR  = os.environ.get("nnUNet_results",
                      "nnUNet_results")
OUTPUT_DIR          = "TwinSight_HTML_Final"
LIVER_CACHE_DIR     = "liver_segmentations"

DATASET_ID          = "008"
NNUNET_CONFIG       = "3d_fullres"
NNUNET_TRAINER      = "nnUNetTrainer"
NNUNET_PLANS        = "nnUNetPlans"


# ─────────────────────────────────────────────────────────────────
#  SEGMENTACIÓN DEL HÍGADO (TotalSegmentator, con caché)
# ─────────────────────────────────────────────────────────────────

def get_liver_mask(img_path: str, patient_id: str) -> np.ndarray | None:
    """
    Obtiene la máscara del hígado usando TotalSegmentator.
    Cachea el resultado en LIVER_CACHE_DIR para no recomputar.

    Returns:
        np.ndarray con la máscara binaria del hígado, o None si falla.
    """
    os.makedirs(LIVER_CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(LIVER_CACHE_DIR, f"{patient_id}_liver.nii.gz")

    if os.path.exists(cache_path):
        print(f"  [TotalSegmentator] Usando caché: {cache_path}")
    else:
        print(f"  [TotalSegmentator] Segmentando hígado de {patient_id}...")
        try:
            subprocess.run(
                ["TotalSegmentator", "-i", img_path,
                 "-o", os.path.join(LIVER_CACHE_DIR, patient_id),
                 "--roi_subset", "liver", "--fast"],
                check=True, capture_output=True
            )
            # TotalSegmentator genera liver.nii.gz dentro del subdirectorio
            ts_output = os.path.join(LIVER_CACHE_DIR, patient_id, "liver.nii.gz")
            if os.path.exists(ts_output):
                os.rename(ts_output, cache_path)
                # Limpiar directorio temporal
                os.rmdir(os.path.join(LIVER_CACHE_DIR, patient_id))
        except Exception as e:
            print(f"  [TotalSegmentator] ERROR: {e}")
            return None

    try:
        liver_nib = nib.load(cache_path)
        liver_data = liver_nib.get_fdata().astype(bool)
        print(f"  [TotalSegmentator] Hígado cargado: {liver_data.sum()} vóxeles")
        return liver_data
    except Exception as e:
        print(f"  [TotalSegmentator] Error al cargar caché: {e}")
        return None


# ─────────────────────────────────────────────────────────────────
#  INFERENCIA nnUNet
# ─────────────────────────────────────────────────────────────────

def run_nnunet_inference(img_path: str, patient_id: str) -> np.ndarray | None:
    """
    Ejecuta inferencia nnUNet v2 (ensemble 5 folds) sobre un CT.

    Returns:
        np.ndarray con la segmentación (0=fondo, 1=vasos, 2=tumor)
        o None si la inferencia falla.
    """
    tmp_input  = f"/tmp/nnunet_input_{patient_id}"
    tmp_output = f"/tmp/nnunet_output_{patient_id}"
    os.makedirs(tmp_input,  exist_ok=True)
    os.makedirs(tmp_output, exist_ok=True)

    # nnUNet espera el archivo con sufijo _0000
    link_path = os.path.join(tmp_input, f"{patient_id}_0000.nii.gz")
    if not os.path.exists(link_path):
        os.symlink(os.path.abspath(img_path), link_path)

    print(f"  [nnUNet] Ejecutando inferencia (ensemble 5 folds)...")
    try:
        subprocess.run(
            ["nnUNetv2_predict",
             "-i", tmp_input,
             "-o", tmp_output,
             "-d", DATASET_ID,
             "-c", NNUNET_CONFIG,
             "-f", "all",           # Ensemble de los 5 folds
             "--save_probabilities"],
            check=True, capture_output=True
        )
    except subprocess.CalledProcessError as e:
        print(f"  [nnUNet] ERROR: {e.stderr.decode()[:500]}")
        return None

    # Cargar resultado
    pred_path = os.path.join(tmp_output, f"{patient_id}.nii.gz")
    if not os.path.exists(pred_path):
        print(f"  [nnUNet] No se encontró predicción en {pred_path}")
        return None

    pred_nib  = nib.load(pred_path)
    voxel_data = pred_nib.get_fdata().astype(np.uint8)
    affine     = pred_nib.affine
    spacing    = tuple(abs(affine[i, i]) for i in range(3))

    print(f"  [nnUNet] Predicción cargada: shape={voxel_data.shape}, spacing={spacing}")
    return voxel_data, affine, spacing


# ─────────────────────────────────────────────────────────────────
#  PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────────

def process_patient(img_path: str) -> bool:
    """Procesa un paciente completo: segmentación → métricas → HTML."""
    patient_id = os.path.basename(img_path).replace(".nii.gz", "")
    print(f"\n{'='*60}")
    print(f"[Processing] {patient_id}")
    print(f"{'='*60}")

    try:
        # 1. Segmentación del hígado (TotalSegmentator)
        liver_mask = get_liver_mask(img_path, patient_id)

        # 2. Inferencia nnUNet
        result = run_nnunet_inference(img_path, patient_id)
        if result is None:
            return False
        voxel_data, affine, spacing = result

        # 3. Métricas clínicas
        l_cc, v_cc, t_cc, burden = calculate_volumes(
            voxel_data,
            spacing=spacing,
            liver_mask=liver_mask
        )
        gps_report = get_tumor_gps(voxel_data, affine)

        print(f"\n📋 Summary:")
        print(f"  • Liver:         {l_cc:.1f} cc")
        print(f"  • Vessels:       {v_cc:.1f} cc")
        print(f"  • Tumor load:    {t_cc:.1f} cc")
        print(f"  • Burden:        {burden:.1f}%")
        print(f"  • Focal lesions: {gps_report.get('n_tumors', '?')}")

        # 4. Generar Gemelo Digital HTML
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        html_path = os.path.join(OUTPUT_DIR, f"{patient_id}_twin.html")

        generate_plotly_html(
            voxel_data=voxel_data,
            affine=affine,
            output_path=html_path,
            patient_id=patient_id,
            liver_mask=liver_mask,
            spacing=spacing,
            clinical_data={
                "liver_cc": l_cc,
                "vessels_cc": v_cc,
                "tumor_cc": t_cc,
                "burden_pct": burden,
            }
        )

        print(f"\n✅ Gemelo digital generado: {html_path}")
        return True

    except Exception as e:
        import traceback
        print(f"\n❌ ERROR procesando {patient_id}: {e}")
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="TwinSight Network — Generador de Gemelos Digitales Hepáticos"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--patient", type=str,
                       help="ID del paciente (ej: hepaticvessel_009)")
    group.add_argument("--sample",  type=int,
                       help="Procesar N pacientes aleatorios del INPUT_DIR")
    group.add_argument("--all",     action="store_true",
                       help="Procesar todos los pacientes del INPUT_DIR")
    parser.add_argument("--input_dir", type=str, default=INPUT_DIR,
                        help=f"Directorio de imágenes (default: {INPUT_DIR})")
    args = parser.parse_args()

    input_dir = args.input_dir
    all_images = sorted(glob.glob(os.path.join(input_dir, "*.nii.gz")))

    if not all_images:
        print(f"❌ No se encontraron imágenes en {input_dir}")
        sys.exit(1)

    # Selección de pacientes
    if args.patient:
        targets = [p for p in all_images if args.patient in p]
        if not targets:
            # Intentar path directo
            direct = os.path.join(input_dir, f"{args.patient}.nii.gz")
            if os.path.exists(direct):
                targets = [direct]
            else:
                print(f"❌ Paciente '{args.patient}' no encontrado en {input_dir}")
                sys.exit(1)
    elif args.sample:
        targets = random.sample(all_images, min(args.sample, len(all_images)))
    else:
        targets = all_images

    print(f"\n🚀 TwinSight Network — Procesando {len(targets)} paciente(s)")
    print(f"   Input:  {input_dir}")
    print(f"   Output: {OUTPUT_DIR}/\n")

    success, failed = 0, 0
    for img_path in targets:
        ok = process_patient(img_path)
        if ok:
            success += 1
        else:
            failed += 1

    print(f"\n{'='*60}")
    print(f"COMPLETE  Success:{success}  Failed:{failed}")
    print(f"Output: {os.path.abspath(OUTPUT_DIR)}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
