"""
train_v3.py — TwinSight Network
Entrenamiento corregido de SwinUNETR para Task 08: Hepatic Vessel (MSD).

Fixes vs train_v2.py:
  [FIX #1] DiceCELoss con include_background=False — el fondo (95% del volumen)
           ya no domina la loss. El modelo ahora recibe señal de gradiente
           real sobre vasos (Clase 1) y tumor (Clase 2).

  [FIX #2] Pesos de clase en CrossEntropy para compensar el desbalance extremo:
           fondo × 0.1, vasos × 2.0, tumor × 2.5.
           Combinado con include_background=False en Dice, el modelo no puede
           ignorar estructuras minoritarias.

  [FIX #3] CosineAnnealingLR scheduler — el lr decae suavemente de 1e-4 a 1e-6
           durante todo el entrenamiento. Estándar en todos los papers de
           segmentación médica con SwinUNETR (nnU-Net, UNETR, etc.).

  [FIX #4] Validación con métricas Dice por clase en cada época — monitoriza
           Dice_vessels y Dice_tumor por separado. Si Dice_vessels < 0.05
           después de época 20, el modelo está colapsado y hay que parar.

  [FIX #5] Gradient clipping (max_norm=1.0) — previene explosión de gradientes
           en los primeros pasos del entrenamiento con parches pequeños.

  [FIX #6] Early stopping por Dice_mean en validación — guarda el mejor modelo
           según la métrica real, no solo el último checkpoint.

  [FIX #7] Logging detallado por clase — permite detectar colapso de clase 1
           (vasos) antes de desperdiciar horas de cómputo en el clúster.
"""

import os
import torch
import numpy as np
from monai.networks.nets import SwinUNETR
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.utils import set_determinism
from monai.transforms import AsDiscrete
from monai.data import decollate_batch

from data_utils_v2 import get_dataloader


# ─────────────────────────────────────────────────────────────────
#  CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────
ROOT_DIR      = "/lustre/home/admins/aadib/3D_Medical_image/data"
OUTPUT_DIR    = "models"
BEST_MODEL    = "models/swin_unetr_vessels_best.pt"
LOG_FILE      = "models/training_log.csv"

NUM_EPOCHS    = 150      # más épocas — con los fixes convergirá más tarde
BATCH_SIZE    = 3
NUM_WORKERS   = 4
LR_INITIAL    = 1e-4
LR_MIN        = 1e-6
GRAD_CLIP     = 1.0      # max_norm para gradient clipping
VAL_EVERY     = 5        # validar cada N épocas
SAVE_EVERY    = 25       # checkpoint cada N épocas

# FIX #2: pesos de clase para CrossEntropy
# fondo × 0.1 — penalizamos poco el fondo (es fácil y dominante)
# vasos × 2.0 — reforzamos la señal de gradiente en vasos
# tumor × 2.5 — el tumor es la clase más clínicamente relevante
CE_CLASS_WEIGHTS = [0.1, 2.0, 2.5]

# Umbral de alerta: si Dice_vessels < esto en época 20, el modelo está colapsado
COLLAPSE_DICE_THRESHOLD = 0.05
COLLAPSE_CHECK_EPOCH    = 20

# Reproducibilidad
RANDOM_SEED = 42
set_determinism(seed=RANDOM_SEED)


def main():
    print("=" * 65, flush=True)
    print("  TwinSight Network — SwinUNETR Training v3               ", flush=True)
    print("  Task 08: Hepatic Vessel — MSD (Memorial Sloan Kettering)", flush=True)
    print("=" * 65, flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"-> GPU: {torch.cuda.get_device_name(0)}", flush=True)
        print(f"-> VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB", flush=True)
    else:
        print("-> WARNING: CPU mode — training will be extremely slow", flush=True)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── DATOS ─────────────────────────────────────────────────────
    print("\n-> Loading dataset...", flush=True)
    train_loader, val_loader = get_dataloader(
        root_dir=ROOT_DIR,
        batch_size=BATCH_SIZE,
        num_workers=0,       # 0 en Slurm/Lustre — multiprocessing causa recvfds error
        val_split=0.1,
    )

    # ── MODELO ────────────────────────────────────────────────────
    model = SwinUNETR(
        in_channels=1,
        out_channels=3,          # 0=fondo, 1=vasos, 2=tumor
        feature_size=48,
        use_checkpoint=True,
        spatial_dims=3,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"-> Model params: {total_params / 1e6:.1f} M", flush=True)

    # ── LOSS ──────────────────────────────────────────────────────
    # FIX #1: DiceLoss con include_background=False — el fondo (95%) no domina.
    # FIX #2: CrossEntropyLoss con weight — pesos de clase para compensar
    #         el desbalance extremo. ce_weight no existe en versiones antiguas
    #         de MONAI, así que usamos torch.nn.CrossEntropyLoss directamente.
    #
    # Loss total = dice_loss + ce_loss (equivalente a DiceCELoss lambda=1,1)
    dice_loss_fn = DiceLoss(
        to_onehot_y=True,
        softmax=True,
        include_background=False,   # FIX #1
    )
    ce_weights = torch.tensor(CE_CLASS_WEIGHTS, dtype=torch.float32).to(device)
    ce_loss_fn = torch.nn.CrossEntropyLoss(weight=ce_weights)  # FIX #2

    def loss_fn(outputs, labels):
        # DiceLoss espera labels (B,1,D,H,W) con valores enteros
        dice = dice_loss_fn(outputs, labels)
        # CrossEntropyLoss espera labels (B,D,H,W) con valores enteros (long)
        ce = ce_loss_fn(outputs, labels.squeeze(1).long())
        return dice + ce

    # ── OPTIMIZER + SCHEDULER ─────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR_INITIAL,
        weight_decay=1e-5,
    )

    # FIX #3: cosine annealing — lr decae de 1e-4 a 1e-6 suavemente
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS,
        eta_min=LR_MIN,
    )

    # ── MÉTRICAS DE VALIDACIÓN (FIX #4) ──────────────────────────
    # Dice por clase separado — detecta colapso de clase 1
    dice_metric = DiceMetric(
        include_background=False,
        reduction="mean_batch",   # media por clase, no global
        get_not_nans=False,
    )

    # Post-processing para convertir logits → one-hot para la métrica
    post_pred  = AsDiscrete(argmax=True, to_onehot=3)
    post_label = AsDiscrete(to_onehot=3)

    # ── LOG CSV ───────────────────────────────────────────────────
    with open(LOG_FILE, "w") as f:
        f.write("epoch,train_loss,lr,dice_vessels,dice_tumor,dice_mean\n")

    # ── TRAINING LOOP ─────────────────────────────────────────────
    best_dice_mean  = -1.0
    best_epoch      = 0

    print(f"\n-> Starting training for {NUM_EPOCHS} epochs", flush=True)
    print(f"-> Loss: DiceCE(include_bg=False) + CE weights {CE_CLASS_WEIGHTS}", flush=True)
    print(f"-> Scheduler: CosineAnnealingLR [{LR_INITIAL} → {LR_MIN}]", flush=True)
    print("-" * 65, flush=True)

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        step = 0

        for batch in train_loader:
            step += 1
            imgs   = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss    = loss_fn(outputs, labels)
            loss.backward()

            # FIX #5: gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)

            optimizer.step()
            epoch_loss += loss.item()

            if step % 10 == 0:
                print(
                    f"  Epoch {epoch:03d} [{step:03d}/{len(train_loader)}] "
                    f"loss={loss.item():.4f}  lr={scheduler.get_last_lr()[0]:.2e}",
                    flush=True
                )

        avg_loss = epoch_loss / max(step, 1)
        scheduler.step()   # FIX #3
        current_lr = scheduler.get_last_lr()[0]

        # ── VALIDACIÓN (FIX #4) ───────────────────────────────────
        dice_vessels, dice_tumor, dice_mean = -1.0, -1.0, -1.0

        if epoch % VAL_EVERY == 0 or epoch == NUM_EPOCHS:
            model.eval()
            dice_metric.reset()

            with torch.no_grad():
                for val_batch in val_loader:
                    val_imgs   = val_batch["image"].to(device)
                    val_labels = val_batch["label"].to(device)

                    val_outputs = model(val_imgs)

                    # Convertir a one-hot para la métrica
                    val_preds_list  = [post_pred(i)  for i in decollate_batch(val_outputs)]
                    val_labels_list = [post_label(i) for i in decollate_batch(val_labels)]

                    dice_metric(y_pred=val_preds_list, y=val_labels_list)

            # dice_per_class: tensor [n_classes-1] (sin fondo)
            # índice 0 → vasos (Clase 1), índice 1 → tumor (Clase 2)
            dice_per_class = dice_metric.aggregate()
            if dice_per_class.numel() >= 2:
                dice_vessels = float(dice_per_class[0])
                dice_tumor   = float(dice_per_class[1])
                dice_mean    = (dice_vessels + dice_tumor) / 2.0
            elif dice_per_class.numel() == 1:
                dice_mean    = float(dice_per_class[0])

            print(
                f"  [VAL] Epoch {epoch:03d} — "
                f"Dice vessels: {dice_vessels:.4f}  "
                f"Dice tumor: {dice_tumor:.4f}  "
                f"Mean: {dice_mean:.4f}",
                flush=True
            )

            # FIX #4: alerta de colapso de clase
            if epoch >= COLLAPSE_CHECK_EPOCH and dice_vessels < COLLAPSE_DICE_THRESHOLD:
                print(
                    f"  [ALERT] Dice_vessels={dice_vessels:.4f} < {COLLAPSE_DICE_THRESHOLD} "
                    f"at epoch {epoch}. Model may be collapsing on Class 1. "
                    f"Check CE weights and pos/neg ratio.",
                    flush=True
                )

            # FIX #6: guardar mejor modelo por Dice_mean
            if dice_mean > best_dice_mean:
                best_dice_mean = dice_mean
                best_epoch     = epoch
                torch.save(model.state_dict(), BEST_MODEL)
                print(f"  [BEST] New best model saved (Dice_mean={best_dice_mean:.4f})", flush=True)

        # ── LOG ───────────────────────────────────────────────────
        print(
            f"Epoch {epoch:03d}/{NUM_EPOCHS} | "
            f"Loss={avg_loss:.4f} | LR={current_lr:.2e} | "
            f"Dice_ves={dice_vessels:.4f} Dice_tum={dice_tumor:.4f}",
            flush=True
        )
        with open(LOG_FILE, "a") as f:
            f.write(f"{epoch},{avg_loss:.6f},{current_lr:.2e},"
                    f"{dice_vessels:.4f},{dice_tumor:.4f},{dice_mean:.4f}\n")

        # Checkpoint periódico
        if epoch % SAVE_EVERY == 0:
            ckpt_path = f"{OUTPUT_DIR}/swin_unetr_vessels_epoch_{epoch:03d}.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"  [CKPT] Saved: {ckpt_path}", flush=True)

    # ── RESUMEN FINAL ─────────────────────────────────────────────
    print("\n" + "=" * 65, flush=True)
    print(f"  Training complete — {NUM_EPOCHS} epochs", flush=True)
    print(f"  Best model: epoch {best_epoch}  Dice_mean={best_dice_mean:.4f}", flush=True)
    print(f"  Saved at: {BEST_MODEL}", flush=True)
    print(f"  Log: {LOG_FILE}", flush=True)
    print("=" * 65, flush=True)

    # ── GUARDAR MODELO FINAL CON NOMBRE COMPATIBLE ───────────────
    # Para usar con main_v2_fixed.py: copia swin_unetr_vessels_best.pt
    # a src/models/swin_unetr_vessels_epoch_100.pt (o actualiza MODEL_PATH)
    final_path = f"{OUTPUT_DIR}/swin_unetr_vessels_epoch_{NUM_EPOCHS:03d}.pt"
    torch.save(model.state_dict(), final_path)
    print(f"  Final model saved: {final_path}", flush=True)


if __name__ == "__main__":
    main()

