import argparse
import csv
import gc
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


IMG_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def calc_stats(values):
    values = np.asarray(values, dtype=np.float64)

    return {
        "mean": float(np.mean(values)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "std": float(np.std(values)),
    }


def print_stats(name, values):
    s = calc_stats(values)

    print(f"\n{name}")
    print("-" * 60)
    print(f"Mean : {s['mean']:.3f} ms")
    print(f"P50  : {s['p50']:.3f} ms")
    print(f"P95  : {s['p95']:.3f} ms")
    print(f"Min  : {s['min']:.3f} ms")
    print(f"Max  : {s['max']:.3f} ms")
    print(f"Std  : {s['std']:.3f} ms")

    return s


def get_images(image_dir, num_images):
    image_dir = Path(image_dir)

    images = sorted(
        [
            p
            for p in image_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMG_SUFFIXES
        ]
    )

    if len(images) == 0:
        raise RuntimeError(f"No images found in: {image_dir}")

    if num_images > 0:
        images = images[:num_images]

    return images


def show_active_provider(model, backend_name):
    if backend_name.lower() != "onnx":
        return

    try:
        predictor = model.predictor
        backend = predictor.model
        session = getattr(backend, "session", None)

        if session is not None:
            print("\nActive ONNX providers:")
            for provider in session.get_providers():
                print(f" - {provider}")
        else:
            print("\n[Warning] Could not obtain ONNX Runtime session.")
    except Exception as e:
        print(f"\n[Warning] Could not inspect ONNX providers: {e}")


def benchmark(
    model_path,
    images,
    backend_name,
    imgsz,
    device,
    warmup,
    conf,
    iou,
    max_det,
    output_dir,
):
    print("\n" + "=" * 72)
    print(f"Benchmark backend : {backend_name}")
    print(f"Model             : {model_path}")
    print(f"Images            : {len(images)}")
    print(f"Image size        : {imgsz}")
    print(f"Batch             : 1")
    print(f"Device            : {device}")
    print("=" * 72)

    model = YOLO(str(model_path))

    # ---------------------------------------------------------
    # Warmup
    # ---------------------------------------------------------
    warm_img = cv2.imread(str(images[0]))

    if warm_img is None:
        raise RuntimeError(f"Cannot read image: {images[0]}")

    print(f"\nWarmup: {warmup} iterations")

    for _ in range(warmup):
        model.predict(
            source=warm_img,
            imgsz=imgsz,
            batch=1,
            device=device,
            conf=conf,
            iou=iou,
            max_det=max_det,
            verbose=False,
        )

    # Print actual ONNX Runtime provider after initialization
    show_active_provider(model, backend_name)

    # ---------------------------------------------------------
    # Benchmark
    # ---------------------------------------------------------
    rows = []

    preprocess_times = []
    inference_times = []
    postprocess_times = []
    e2e_times = []

    print("\nRunning benchmark...")

    for index, image_path in enumerate(images, start=1):

        # Disk I/O deliberately excluded from benchmark
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"[Skip] Cannot read: {image_path}")
            continue

        # Make sure previous CUDA work has finished
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t0 = time.perf_counter()

        results = model.predict(
            source=image,
            imgsz=imgsz,
            batch=1,
            device=device,
            conf=conf,
            iou=iou,
            max_det=max_det,
            verbose=False,
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t1 = time.perf_counter()

        e2e_ms = (t1 - t0) * 1000.0

        speed = results[0].speed

        preprocess_ms = float(speed.get("preprocess", 0.0))
        inference_ms = float(speed.get("inference", 0.0))
        postprocess_ms = float(speed.get("postprocess", 0.0))

        preprocess_times.append(preprocess_ms)
        inference_times.append(inference_ms)
        postprocess_times.append(postprocess_ms)
        e2e_times.append(e2e_ms)

        rows.append(
            {
                "image": image_path.name,
                "preprocess_ms": preprocess_ms,
                "inference_ms": inference_ms,
                "postprocess_ms": postprocess_ms,
                "e2e_ms": e2e_ms,
            }
        )

        if index % 50 == 0 or index == len(images):
            print(f"{index}/{len(images)}")

    # ---------------------------------------------------------
    # Save raw measurements
    # ---------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"{backend_name.lower()}_benchmark.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image",
                "preprocess_ms",
                "inference_ms",
                "postprocess_ms",
                "e2e_ms",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------
    print("\n" + "=" * 72)
    print(f"{backend_name} RESULTS")
    print("=" * 72)

    pre_stats = print_stats("Preprocess latency", preprocess_times)
    inf_stats = print_stats("Inference latency", inference_times)
    post_stats = print_stats("Postprocess latency", postprocess_times)
    e2e_stats = print_stats("End-to-end latency", e2e_times)

    inference_fps = 1000.0 / inf_stats["mean"]
    e2e_fps = 1000.0 / e2e_stats["mean"]

    print("\nFPS")
    print("-" * 60)
    print(f"Inference FPS : {inference_fps:.2f}")
    print(f"E2E FPS       : {e2e_fps:.2f}")

    print(f"\nRaw data saved to: {csv_path}")

    summary = {
        "backend": backend_name,
        "inference_mean_ms": inf_stats["mean"],
        "inference_p50_ms": inf_stats["p50"],
        "inference_p95_ms": inf_stats["p95"],
        "inference_fps": inference_fps,
        "e2e_mean_ms": e2e_stats["mean"],
        "e2e_p50_ms": e2e_stats["p50"],
        "e2e_p95_ms": e2e_stats["p95"],
        "e2e_fps": e2e_fps,
    }

    # Release model before next backend
    del model
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary


def print_comparison(pt_result, onnx_result):
    pt_ms = pt_result["inference_mean_ms"]
    onnx_ms = onnx_result["inference_mean_ms"]

    speedup = pt_ms / onnx_ms
    latency_reduction = (pt_ms - onnx_ms) / pt_ms * 100.0

    print("\n")
    print("=" * 72)
    print("FINAL COMPARISON")
    print("=" * 72)

    print(
        f"{'Backend':<15}"
        f"{'Mean(ms)':>12}"
        f"{'P50(ms)':>12}"
        f"{'P95(ms)':>12}"
        f"{'FPS':>12}"
        f"{'E2E FPS':>12}"
    )

    print("-" * 75)

    for r in [pt_result, onnx_result]:
        print(
            f"{r['backend']:<15}"
            f"{r['inference_mean_ms']:>12.3f}"
            f"{r['inference_p50_ms']:>12.3f}"
            f"{r['inference_p95_ms']:>12.3f}"
            f"{r['inference_fps']:>12.2f}"
            f"{r['e2e_fps']:>12.2f}"
        )

    print("\nAcceleration")
    print("-" * 60)
    print(f"ONNX speedup           : {speedup:.3f}x")
    print(f"Latency reduction      : {latency_reduction:.2f}%")

    print("=" * 72)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--pt", required=True)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--images", required=True)

    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--num-images", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=30)

    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=1000)

    parser.add_argument(
        "--output",
        default="outputs/benchmark_runtime",
    )

    args = parser.parse_args()

    images = get_images(args.images, args.num_images)

    print(f"Found {len(images)} test images.")

    output_dir = Path(args.output)

    pt_result = benchmark(
        model_path=args.pt,
        images=images,
        backend_name="PyTorch",
        imgsz=args.imgsz,
        device=args.device,
        warmup=args.warmup,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        output_dir=output_dir,
    )

    onnx_result = benchmark(
        model_path=args.onnx,
        images=images,
        backend_name="ONNX",
        imgsz=args.imgsz,
        device=args.device,
        warmup=args.warmup,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        output_dir=output_dir,
    )

    engine_result = benchmark(
        model_path=args.engine,
        images=images,
        backend_name="TensorRT",
        imgsz=args.imgsz,
        device=args.device,
        warmup=args.warmup,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        output_dir=output_dir,
    )

    # print_comparison(pt_result, onnx_result)
    results = [pt_result, onnx_result, engine_result]

    print("\n")
    print("=" * 82)
    print("FINAL COMPARISON")
    print("=" * 82)

    print(
        f"{'Backend':<15}"
        f"{'Mean(ms)':>12}"
        f"{'P50(ms)':>12}"
        f"{'P95(ms)':>12}"
        f"{'FPS':>12}"
        f"{'E2E FPS':>12}"
    )

    print("-" * 82)

    for r in results:
        print(
            f"{r['backend']:<15}"
            f"{r['inference_mean_ms']:>12.3f}"
            f"{r['inference_p50_ms']:>12.3f}"
            f"{r['inference_p95_ms']:>12.3f}"
            f"{r['inference_fps']:>12.2f}"
            f"{r['e2e_fps']:>12.2f}"
        )

    pt_ms = pt_result["inference_mean_ms"]

    print("\nAcceleration vs PyTorch")
    print("-" * 60)

    for r in [onnx_result, engine_result]:
        speedup = pt_ms / r["inference_mean_ms"]

        reduction = (
                (pt_ms - r["inference_mean_ms"])
                / pt_ms
                * 100.0
        )

        print(
            f"{r['backend']:<12}: "
            f"{speedup:.3f}x speedup, "
            f"{reduction:.2f}% latency reduction"
        )

    print("=" * 82)


if __name__ == "__main__":
    main()