import os
import time
import platform
import contextlib
import io
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext

import numpy as np

try:
    from threadpoolctl import threadpool_limits
except ImportError:
    threadpool_limits = None


STUDENT_ID = 230103216
N = 1024
MAX_WORKERS = os.cpu_count() or 1
WORKERS = min(MAX_WORKERS, 8)
BLOCK_SIZES = [16, 64]
OUTPUT_FILE = "task8_output.txt"


def multiply_non_tiled(a, b, workers):
    result = np.zeros((N, N), dtype=np.float64)
    rows_per_task = (N + workers - 1) // workers

    def calculate_rows(start):
        end = min(start + rows_per_task, N)
        result[start:end, :] = a[start:end, :] @ b

    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(calculate_rows, range(0, N, rows_per_task)))

    return result


def multiply_tiled(a, b, block_size, workers):
    result = np.zeros((N, N), dtype=np.float64)

    tiles = [
        (i, j)
        for i in range(0, N, block_size)
        for j in range(0, N, block_size)
    ]

    def calculate_tile(tile):
        i, j = tile
        i_end = min(i + block_size, N)
        j_end = min(j + block_size, N)

        tile_result = np.zeros(
            (i_end - i, j_end - j), dtype=np.float64
        )

        for k in range(0, N, block_size):
            k_end = min(k + block_size, N)
            tile_result += (
                a[i:i_end, k:k_end] @ b[k:k_end, j:j_end]
            )

        result[i:i_end, j:j_end] = tile_result

    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(calculate_tile, tiles))

    return result


def benchmark(label, function, reference):
    start = time.perf_counter()
    result = function()
    elapsed = time.perf_counter() - start

    correct = np.allclose(result, reference, rtol=1e-9, atol=1e-9)
    print(
        f"{label:<24} | {elapsed:>10.4f} s | "
        f"{'PASS' if correct else 'FAIL'}"
    )

    if not correct:
        raise RuntimeError(f"Incorrect matrix result for: {label}")

    return elapsed


def main():
    rng = np.random.default_rng(STUDENT_ID)
    a = rng.random((N, N), dtype=np.float64)
    b = rng.random((N, N), dtype=np.float64)

    print(f"Student ID: {STUDENT_ID}")
    print(f"Operating System: {platform.platform()}")
    print(f"CPU: {platform.processor() or platform.machine()}")
    print(f"Logical CPU cores: {MAX_WORKERS}")
    print(f"Matrix size: {N} x {N}")
    print("Data type: float64")
    print(f"CPU worker threads: {WORKERS}")
    print("Each matrix occupies approximately 8 MiB.")
    print()

    blas_context = (
        threadpool_limits(limits=1)
        if threadpool_limits is not None
        else nullcontext()
    )

    with blas_context:
        print("Calculating reference result...")
        reference = a @ b

        print("\n=== Matrix Multiplication Benchmark ===")
        print(f"{'Method':<24} | {'Runtime':>12} | Correct")
        print("-" * 54)

        results = {}

        results["Non-tiled"] = benchmark(
            "Non-tiled",
            lambda: multiply_non_tiled(a, b, WORKERS),
            reference,
        )

        for block_size in BLOCK_SIZES:
            label = f"Tiled {block_size}x{block_size}"
            results[label] = benchmark(
                label,
                lambda bs=block_size: multiply_tiled(
                    a, b, bs, WORKERS
                ),
                reference,
            )

    print("\n=== Runtime Comparison ===")
    baseline = results["Non-tiled"]

    for label, elapsed in results.items():
        speedup = baseline / elapsed if elapsed > 0 else float("inf")
        print(
            f"{label:<24} | {elapsed:.4f} s | "
            f"relative to non-tiled: {speedup:.3f}x"
        )

    best = min(results, key=results.get)
    print(f"\nFastest measured method: {best}")
    print(
        "Note: runtime is measured directly. Cache misses are not "
        "counted because hardware performance counters are not used."
    )
    print(
        "Results depend on CPU, NumPy/BLAS implementation, thread "
        "scheduling, and system load."
    )


if __name__ == "__main__":
    output = io.StringIO()

    with contextlib.redirect_stdout(output):
        main()

    report = output.getvalue()

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        file.write(report)

    print(report, end="")
    print(f"\nOutput saved to {OUTPUT_FILE}")
