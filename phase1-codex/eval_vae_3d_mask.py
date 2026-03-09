from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generative.networks.nets import AutoencoderKL
from vessel_ldm_utils import (
    VesselMaskPtDataset,
    dice_score,
    extract_state_dict,
    infer_case_name,
    iou_score,
    save_generated_overview,
    save_json,
    save_nifti_volume,
    save_recon_comparison,
    set_seed,
    split_pt_files_train_val_test,
    vessel_ratio,
    volume_relative_error,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Evaluate 3D AutoencoderKL for hepatic vessel mask caches")
    parser.add_argument("--cache_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True, help="Path to state_dict or training checkpoint")
    parser.add_argument("--output_dir", type=str, default="./phase1-codex/eval_vae_vessel")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test", "all"])
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max_visualizations", type=int, default=20)
    parser.add_argument("--save_volumes", action="store_true")
    parser.add_argument("--max_saved_volumes", type=int, default=200)
    parser.add_argument("--spatial_size", type=int, nargs=3, default=[128, 128, 128])
    parser.add_argument("--latent_channels", type=int, default=3)
    parser.add_argument("--num_channels", type=int, nargs="+", default=[64, 128, 128, 128])
    parser.add_argument("--num_res_blocks", type=int, default=2)
    parser.add_argument("--norm_num_groups", type=int, default=32)
    parser.add_argument("--norm_eps", type=float, default=1e-6)
    parser.add_argument("--attention_levels", type=int, nargs="+", default=[0, 0, 0, 1])
    return parser.parse_args()


def build_autoencoder(args: argparse.Namespace) -> AutoencoderKL:
    return AutoencoderKL(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        latent_channels=args.latent_channels,
        num_channels=tuple(args.num_channels),
        num_res_blocks=args.num_res_blocks,
        norm_num_groups=args.norm_num_groups,
        norm_eps=args.norm_eps,
        attention_levels=tuple(bool(x) for x in args.attention_levels),
        with_encoder_nonlocal_attn=False,
        with_decoder_nonlocal_attn=False,
    )


def get_eval_files(cache_dir: str, split: str, val_ratio: float, test_ratio: float, seed: int) -> list[Path]:
    train_files, val_files, test_files = split_pt_files_train_val_test(
        cache_dir,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    if split == "train":
        return train_files
    if split == "val":
        return val_files
    if split == "test":
        return test_files
    return train_files + val_files + test_files


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    vis_dir = output_dir / "visualizations"
    raw_dir = output_dir / "generated_views"
    vol_dir = output_dir / "volumes"
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    if args.save_volumes:
        vol_dir.mkdir(parents=True, exist_ok=True)
    save_json(vars(args), output_dir / "eval_config.json")

    eval_files = get_eval_files(args.cache_dir, args.split, args.val_ratio, args.test_ratio, args.seed)
    dataset = VesselMaskPtDataset(eval_files, augment=False, spatial_size=tuple(args.spatial_size))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_autoencoder(args).to(device)
    loaded = torch.load(args.model_path, map_location="cpu")
    model.load_state_dict(extract_state_dict(loaded), strict=True)
    model.eval()

    per_case: list[dict[str, float | str]] = []
    vis_count = 0
    vol_count = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", ncols=120):
            gt = batch["mask"].to(device).float()
            recon_raw, _, _ = model(gt)
            recon = torch.sigmoid(recon_raw).float()
            pred_bin = (recon > args.threshold).float()
            gt_bin = (gt > 0.5).float()

            l1 = F.l1_loss(recon, gt, reduction="none").mean(dim=[1, 2, 3, 4]).cpu().numpy()
            mae = torch.abs(recon - gt).mean(dim=[1, 2, 3, 4]).cpu().numpy()
            dice = dice_score(pred_bin, gt_bin).cpu().numpy()
            iou = iou_score(pred_bin, gt_bin).cpu().numpy()
            vre = volume_relative_error(pred_bin, gt_bin).cpu().numpy()
            vessel_pred = vessel_ratio(pred_bin).cpu().numpy()
            vessel_gt = vessel_ratio(gt_bin).cpu().numpy()
            paths = batch["path"]

            for idx, case_path in enumerate(paths):
                case_name = infer_case_name(case_path)
                per_case.append(
                    {
                        "case": case_name,
                        "path": case_path,
                        "dice": float(dice[idx]),
                        "iou": float(iou[idx]),
                        "l1": float(l1[idx]),
                        "mae": float(mae[idx]),
                        "volume_rel_error": float(vre[idx]),
                        "vessel_ratio_pred": float(vessel_pred[idx]),
                        "vessel_ratio_gt": float(vessel_gt[idx]),
                    }
                )

                if vis_count < args.max_visualizations:
                    save_recon_comparison(
                        gt=gt[idx : idx + 1].cpu(),
                        recon=recon[idx : idx + 1].cpu(),
                        out_path=vis_dir / f"{case_name}.png",
                        threshold=args.threshold,
                    )
                    save_generated_overview(
                        mask=recon[idx : idx + 1].cpu(),
                        out_path=raw_dir / f"{case_name}.png",
                        threshold=args.threshold,
                        title="Recon",
                    )
                    vis_count += 1

                if args.save_volumes and vol_count < args.max_saved_volumes:
                    save_nifti_volume(gt[idx : idx + 1].cpu(), vol_dir / f"{case_name}_gt.nii.gz")
                    save_nifti_volume(recon[idx : idx + 1].cpu(), vol_dir / f"{case_name}_recon_raw.nii.gz")
                    save_nifti_volume(pred_bin[idx : idx + 1].cpu(), vol_dir / f"{case_name}_recon_bin.nii.gz")
                    vol_count += 1

    def mean_of(key: str) -> float:
        return float(np.mean([item[key] for item in per_case])) if per_case else 0.0

    summary = {
        "num_cases": len(per_case),
        "split": args.split,
        "threshold": args.threshold,
        "dice_mean": mean_of("dice"),
        "iou_mean": mean_of("iou"),
        "l1_mean": mean_of("l1"),
        "mae_mean": mean_of("mae"),
        "volume_rel_error_mean": mean_of("volume_rel_error"),
        "vessel_ratio_pred_mean": mean_of("vessel_ratio_pred"),
        "vessel_ratio_gt_mean": mean_of("vessel_ratio_gt"),
        "model_path": args.model_path,
    }
    save_json(summary, output_dir / "metrics_summary.json")
    with (output_dir / "per_case_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case",
                "path",
                "dice",
                "iou",
                "l1",
                "mae",
                "volume_rel_error",
                "vessel_ratio_pred",
                "vessel_ratio_gt",
            ],
        )
        writer.writeheader()
        writer.writerows(per_case)

    print("Evaluation finished.")
    print(json.dumps(summary, indent=2))
    print(f"Saved metrics to {output_dir}")


if __name__ == "__main__":
    main()
