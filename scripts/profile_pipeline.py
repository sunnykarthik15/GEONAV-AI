"""Phase 8 pipeline profiler — per-component latency measurement.

Measures the time consumed by each major processing stage on EuRoC MH_01_easy.
Reports mean, median, P95 latency and FPS for each component.

Usage:
    python scripts/profile_pipeline.py

IMPORTANT: All measurements are from a development-machine software benchmark.
They do NOT represent performance on any embedded or UAV onboard compute platform.
"""

import sys
import time
import tracemalloc
from pathlib import Path
from typing import Dict, List
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from geonav.config.settings import VIOConfig, ExtrinsicsConfig
from geonav.datasets.euroc import EurocDataset
from geonav.synchronization.synchronizer import synchronize_streams
from geonav.vio.pipeline import VIOPipeline
from geonav.vio.visual_frontend import VisualFrontEnd
from geonav.vio.imu_propagator import IMUPropagator
from geonav.ml.features import FeatureExtractor
from geonav.ml.inference import MLVelocityCorrector


def _stats(times_ms: List[float]) -> Dict[str, float]:
    """Compute mean, median, P95, min, max from a list of millisecond durations."""
    a = np.array(times_ms, dtype=np.float64)
    return {
        "count": int(len(a)),
        "mean_ms": float(np.mean(a)),
        "median_ms": float(np.median(a)),
        "p95_ms": float(np.percentile(a, 95)),
        "min_ms": float(np.min(a)),
        "max_ms": float(np.max(a)),
        "total_s": float(np.sum(a) / 1000.0),
    }


def main():
    import json

    repo_root = Path(__file__).resolve().parent.parent
    dataset_path = repo_root / "data" / "raw" / "euroc" / "MH_01_easy"
    artifacts_dir = repo_root / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("GEONAV-AI -- PHASE 8 PIPELINE PROFILER")
    print("(Development-machine software benchmark)")
    print("=" * 70)

    dataset = EurocDataset(dataset_path)
    vio_config = VIOConfig()
    if dataset.camera_extrinsics is not None:
        T_BS = dataset.camera_extrinsics
        R_BS, t_BS = T_BS[:3, :3], T_BS[:3, 3]
        vio_config.extrinsics = ExtrinsicsConfig(
            rotation_matrix=(
                (float(R_BS[0, 0]), float(R_BS[0, 1]), float(R_BS[0, 2])),
                (float(R_BS[1, 0]), float(R_BS[1, 1]), float(R_BS[1, 2])),
                (float(R_BS[2, 0]), float(R_BS[2, 1]), float(R_BS[2, 2])),
            ),
            translation=(float(t_BS[0]), float(t_BS[1]), float(t_BS[2])),
        )

    pipeline = VIOPipeline(config=vio_config)
    corrector = MLVelocityCorrector(
        weights_path=artifacts_dir / "phase7_edge_mlp_weights.json",
        scaler_path=artifacts_dir / "phase7_scaler.json",
        max_correction_mps=3.0,
        min_confidence=0.20,
    )
    extractor = FeatureExtractor()

    synced = list(synchronize_streams(
        (img.to_camera_frame() for img in dataset.images()),
        (s.to_imu_sample() for s in dataset.imu()),
    ))
    print(f"Loaded {len(synced)} synchronized frames.\n")

    # Per-component timing accumulators
    t_preprocess: List[float] = []
    t_detect: List[float] = []
    t_flow: List[float] = []
    t_essential: List[float] = []
    t_imu: List[float] = []
    t_fusion: List[float] = []
    t_ml_feat: List[float] = []
    t_ml_infer: List[float] = []
    t_total: List[float] = []

    # Memory tracking
    tracemalloc.start()
    mem_snapshots: List[float] = []

    prev_hybrid = None
    frame_count = 0

    for m in synced:
        t_frame_start = time.perf_counter()

        # --- VIO initialization (skip timing for first frame)
        if not pipeline._is_initialized:
            pipeline.process_measurement(m)
            continue

        frame_count += 1

        # 1. Preprocessing
        t0 = time.perf_counter()
        curr_gray = pipeline.frontend.preprocess_image(m.camera_frame.frame_data)
        t_preprocess.append((time.perf_counter() - t0) * 1e3)

        # 2. Feature detection (if needed)
        t0 = time.perf_counter()
        if pipeline._prev_pts is None or len(pipeline._prev_pts) < pipeline.frontend.min_features:
            pipeline.frontend.detect_features(pipeline._prev_gray if pipeline._prev_gray is not None else curr_gray)
        t_detect.append((time.perf_counter() - t0) * 1e3)

        # 3. Optical flow tracking
        t0 = time.perf_counter()
        if pipeline._prev_gray is not None and pipeline._prev_pts is not None and len(pipeline._prev_pts) > 0:
            pts_p, pts_c = pipeline.frontend.track_features(pipeline._prev_gray, curr_gray, pipeline._prev_pts)
        else:
            pts_p, pts_c = np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)
        t_flow.append((time.perf_counter() - t0) * 1e3)

        # 4. Essential matrix + pose recovery
        t0 = time.perf_counter()
        if len(pts_p) >= pipeline.frontend.min_features:
            pipeline.frontend.estimate_relative_motion(pts_p, pts_c)
        t_essential.append((time.perf_counter() - t0) * 1e3)

        # 5. IMU propagation
        t0 = time.perf_counter()
        p_prev = np.asarray(pipeline._current_state.position)
        v_prev = np.asarray(pipeline._current_state.velocity)
        q_prev = np.asarray(pipeline._current_state.orientation)
        try:
            pipeline.propagator.propagate_interval(p_prev, v_prev, q_prev, m.imu_samples)
        except Exception:
            pass
        t_imu.append((time.perf_counter() - t0) * 1e3)

        # 6. Full pipeline step (fusion + state update)
        t_prev_ts = pipeline._current_state.timestamp if pipeline._current_state else m.camera_frame.timestamp
        t_curr_ts = m.camera_frame.timestamp
        dt = max(t_curr_ts - t_prev_ts, 1e-4)

        t0 = time.perf_counter()
        raw_state = pipeline.process_measurement(m)
        t_fusion.append((time.perf_counter() - t0) * 1e3)

        # 7. ML feature extraction
        t0 = time.perf_counter()
        feat = extractor.extract(
            visual_result=pipeline._last_visual_result,
            pts_prev=pipeline._prev_pts,
            pts_curr=None,
            imu_samples=m.imu_samples,
            current_state=raw_state,
            dt=dt,
        )
        t_ml_feat.append((time.perf_counter() - t0) * 1e3)

        # 8. ML inference + correction
        t0 = time.perf_counter()
        corrected = corrector.correct_state(prev_hybrid, raw_state, feat, dt)
        t_ml_infer.append((time.perf_counter() - t0) * 1e3)
        prev_hybrid = corrected

        t_total.append((time.perf_counter() - t_frame_start) * 1e3)

        # Sample memory every 100 frames
        if frame_count % 100 == 0:
            current, peak = tracemalloc.get_traced_memory()
            mem_snapshots.append(peak / (1024 * 1024))  # MB

    tracemalloc.stop()

    total_s = sum(t_total) / 1000.0
    fps = frame_count / total_s if total_s > 0 else 0.0

    print(f"Profiled {frame_count} frames in {total_s:.2f}s ({fps:.1f} FPS)\n")
    print(f"{'Component':<30} {'Mean':>8} {'Median':>8} {'P95':>8} {'Total':>10}")
    print(f"{'':30} {'ms':>8} {'ms':>8} {'ms':>8} {'s':>10}")
    print("-" * 68)

    components = [
        ("Preprocessing", t_preprocess),
        ("Feature Detection", t_detect),
        ("Optical Flow", t_flow),
        ("Essential Matrix", t_essential),
        ("IMU Propagation", t_imu),
        ("Full Pipeline Step", t_fusion),
        ("ML Feature Extraction", t_ml_feat),
        ("ML Inference + Correction", t_ml_infer),
        ("Total Frame", t_total),
    ]

    profile_stats = {}
    for name, times in components:
        if not times:
            continue
        s = _stats(times)
        print(f"{name:<30} {s['mean_ms']:>8.3f} {s['median_ms']:>8.3f} {s['p95_ms']:>8.3f} {s['total_s']:>10.2f}")
        profile_stats[name] = s

    peak_mem_mb = max(mem_snapshots) if mem_snapshots else 0.0
    print(f"\nPeak tracked memory: {peak_mem_mb:.1f} MB")
    print(f"Total frames processed: {frame_count}")
    print(f"Overall FPS: {fps:.1f}")

    # Model file size
    weights_file = artifacts_dir / "phase7_edge_mlp_weights.json"
    model_kb = weights_file.stat().st_size / 1024 if weights_file.is_file() else 0.0
    print(f"ML model file size: {model_kb:.1f} KB")

    result = {
        "benchmark_type": "Development-machine software benchmark",
        "platform_note": "Results measured on development laptop. Do NOT extrapolate to embedded/UAV hardware.",
        "dataset": "EuRoC MH_01_easy",
        "frames_profiled": frame_count,
        "total_runtime_s": round(total_s, 2),
        "overall_fps": round(fps, 1),
        "peak_memory_mb": round(peak_mem_mb, 1),
        "model_size_kb": round(model_kb, 1),
        "component_latencies_ms": {
            name: {
                "mean_ms": round(s["mean_ms"], 3),
                "median_ms": round(s["median_ms"], 3),
                "p95_ms": round(s["p95_ms"], 3),
                "total_s": round(s["total_s"], 2),
            }
            for name, s in profile_stats.items()
        },
    }

    out_path = artifacts_dir / "phase8_performance_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        import json
        json.dump(result, f, indent=2)
    print(f"\nExported performance metrics to: {out_path}")
    print("\n[DONE] Phase 8 profiling complete.")


if __name__ == "__main__":
    main()
