
import os
import time
import platform
import contextlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np

STUDENT_ID = 230103216
N = 30_000_000
MAX_CORES = os.cpu_count() or 1
CORE_COUNTS = sorted(set([1, 2, MAX_CORES]))

A = None
B = None


def copy_chunk(start, end):
    B[start:end] = A[start:end]


def compute_chunk(start, end):
    x = A[start:end]
    B[start:end] = np.sin(x) * np.cos(x) + np.tan(x)


def run_kernel(kernel, workers):
    chunk_size = (N + workers - 1) // workers
    chunks = [
        (start, min(start + chunk_size, N))
        for start in range(0, N, chunk_size)
    ]

    function = copy_chunk if kernel == "Memory-Bound Copy" else compute_chunk

    start_time = time.perf_counter()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(function, start, end)
            for start, end in chunks
        ]
        for future in futures:
            future.result()

    return time.perf_counter() - start_time


def benchmark_kernel(kernel):
    times = {}

    if kernel == "Memory-Bound Copy":
        B[:1000] = A[:1000]
    else:
        x = A[:1000]
        B[:1000] = np.sin(x) * np.cos(x) + np.tan(x)

    for workers in CORE_COUNTS:
        elapsed = run_kernel(kernel, workers)
        times[workers] = elapsed

    baseline = times[1]
    results = []

    for workers in CORE_COUNTS:
        elapsed = times[workers]
        speedup = baseline / elapsed
        efficiency = speedup / workers

        results.append({
            "workers": workers,
            "time": elapsed,
            "speedup": speedup,
            "efficiency": efficiency,
        })

    return results


def print_results(name, results):
    print(f"\n=== {name} ===")
    print(
        f"{'Cores':>8} | {'Wall Time (s)':>14} | "
        f"{'Speedup':>10} | {'Efficiency':>12}"
    )
    print("-" * 55)

    for row in results:
        print(
            f"{row['workers']:8d} | "
            f"{row['time']:14.4f} | "
            f"{row['speedup']:9.3f}x | "
            f"{row['efficiency'] * 100:11.2f}%"
        )


def classify_memory_scaling(results):
    max_result = max(results, key=lambda r: r["workers"])
    max_workers = max_result["workers"]

    if max_workers > 1 and max_result["efficiency"] < 0.50:
        return (
            "Memory-bound copy shows poor scaling at the maximum worker "
            "count, consistent with possible memory-bandwidth saturation."
        )

    return (
        "The copy kernel did not show strong evidence of memory-channel "
        "saturation under the chosen threshold. Inspect measured results."
    )


def main():
    global A, B

    print("Student ID:", STUDENT_ID)
    print("Operating System:", platform.platform())
    print("CPU Model:", platform.processor() or "Not available")
    print("Logical CPU cores:", MAX_CORES)
    print("Array elements:", f"{N:,}")
    print("Data type: float64")
    print("Approximate size per array: 240 MB")
    print("Worker counts:", CORE_COUNTS)

    rng = np.random.default_rng(STUDENT_ID)
    A = rng.uniform(0.1, 1.0, size=N).astype(np.float64)
    B = np.empty_like(A)

    copy_results = benchmark_kernel("Memory-Bound Copy")
    compute_results = benchmark_kernel("Compute-Bound Trigonometry")

    print_results("Kernel A: B[i] = A[i]", copy_results)
    print_results(
        "Kernel B: sin(A[i]) * cos(A[i]) + tan(A[i])",
        compute_results
    )

    print("\n=== SCALING ANALYSIS ===")
    print(classify_memory_scaling(copy_results))

    copy_max = max(copy_results, key=lambda r: r["workers"])
    compute_max = max(compute_results, key=lambda r: r["workers"])

    print(
        f"Copy kernel efficiency at {copy_max['workers']} workers: "
        f"{copy_max['efficiency'] * 100:.2f}%"
    )
    print(
        f"Compute kernel efficiency at {compute_max['workers']} workers: "
        f"{compute_max['efficiency'] * 100:.2f}%"
    )
    print(
        "Note: low scaling efficiency alone does not prove memory-channel "
        "saturation; it can also reflect thread overhead, CPU limits, or "
        "memory-controller behavior."
    )


if __name__ == "__main__":
    with open("task7_output.txt", "w", encoding="utf-8") as output:
        with contextlib.redirect_stdout(output):
            main()

    with open("task7_output.txt", "r", encoding="utf-8") as output:
        print(output.read(), end="")
