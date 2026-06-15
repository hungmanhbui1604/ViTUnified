import argparse
import os
import time
import numpy as np

from trt_runner import TensorRTRunner


def get_engine_size_mb(engine_path):
    return os.path.getsize(engine_path) / (1024 * 1024)


def benchmark(engine_path, batch_size=1, warmup=50, runs=200):
    disk_mb = get_engine_size_mb(engine_path)

    engine = TensorRTRunner(engine_path)

    x = np.random.randn(batch_size, 3, 224, 224).astype(np.float32)

    # Warmup (also triggers allocation of TensorRT's working buffers)
    for _ in range(warmup):
        engine.infer(x)

    times = []
    for _ in range(runs):
        start = time.perf_counter()
        engine.infer(x)
        end = time.perf_counter()
        times.append((end - start) * 1000.0)

    times = np.array(times)

    latency_per_batch = times.mean()
    latency_per_image = latency_per_batch / batch_size
    throughput = batch_size * 1000.0 / latency_per_batch

    results = {
        "engine_path": engine_path,
        "batch_size": batch_size,
        "latency_per_batch_ms": latency_per_batch,
        "latency_per_image_ms": latency_per_image,
        "throughput_img_s": throughput,
        "min_latency_ms": times.min(),
        "max_latency_ms": times.max(),
        "std_latency_ms": times.std(),
        "disk_size_mb": disk_mb,
    }

    print_results(results)
    return results


def print_results(r):
    print("=" * 50)
    print(f"Engine              : {r['engine_path']}")
    print(f"Batch size          : {r['batch_size']}")
    print(f"Latency / batch     : {r['latency_per_batch_ms']:.3f} ms")
    print(f"Latency / image     : {r['latency_per_image_ms']:.3f} ms")
    print(f"Throughput          : {r['throughput_img_s']:.2f} img/s")
    print(f"Min latency / batch : {r['min_latency_ms']:.3f} ms")
    print(f"Max latency / batch : {r['max_latency_ms']:.3f} ms")
    print(f"Std latency         : {r['std_latency_ms']:.3f} ms")
    print(f"Engine size on disk : {r['disk_size_mb']:.2f} MB")
    print("=" * 50)


def print_comparison(results_list):
    print("\n" + "=" * 78)
    print("COMPARISON")
    print("=" * 78)

    header = f"{'Metric':<22}"
    for r in results_list:
        header += f"{os.path.basename(r['engine_path']):>20}"
    print(header)
    print("-" * 78)

    rows = [
        ("Batch size", "batch_size", "d"),
        ("Latency/batch (ms)", "latency_per_batch_ms", ".3f"),
        ("Latency/image (ms)", "latency_per_image_ms", ".3f"),
        ("Throughput (img/s)", "throughput_img_s", ".2f"),
        ("Disk size (MB)", "disk_size_mb", ".2f"),
    ]

    for label, key, fmt in rows:
        line = f"{label:<22}"
        for r in results_list:
            val = r.get(key)
            if val is None:
                line += f"{'N/A':>20}"
            else:
                line += f"{val:>20{fmt}}"
        print(line)
    print("=" * 78)


def parse_engine_arg(arg):
    """Parse 'path:batch_size' or just 'path' (defaults batch_size=1)."""
    if ":" in arg:
        path, bs = arg.rsplit(":", 1)
        return path, int(bs)
    return arg, 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark TensorRT engines: latency, throughput, and disk size."
    )
    parser.add_argument(
        "--engines",
        nargs="+",
        default=["vu_fp16.engine:1"],
        help="One or more engines as 'path' or 'path:batch_size'. "
             "Example: --engines vu_fp16_b1.engine:1 vu_fp16_b16.engine:16",
    )
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--runs", type=int, default=200)
    args = parser.parse_args()

    all_results = []
    for engine_arg in args.engines:
        path, bs = parse_engine_arg(engine_arg)
        result = benchmark(
            engine_path=path,
            batch_size=bs,
            warmup=args.warmup,
            runs=args.runs,
        )
        all_results.append(result)

    if len(all_results) > 1:
        print_comparison(all_results)
