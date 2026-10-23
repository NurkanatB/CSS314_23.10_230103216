import os
import time
import platform
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import psutil
from numba import njit

STUDENT_ID = 230103216
ITERATIONS = 50_000_000
CACHE_LINE_BYTES = 64
INT64_BYTES = 8
PADDING = CACHE_LINE_BYTES // INT64_BYTES


@njit(nogil=True)
def increment_shared(counts, index, iterations):
    for _ in range(iterations):
        counts[index] += 1


@njit(nogil=True)
def increment_local(counts, index, iterations):
    local_count = 0

    for _ in range(iterations):
        local_count += 1

    counts[index] = local_count


def run_benchmark(worker_count, padded):
    if padded:
        counts = np.zeros(
            worker_count * PADDING,
            dtype=np.int64
        )
        indices = [i * PADDING for i in range(worker_count)]
        worker_function = increment_shared
        label = "Padded Shared"
    else:
        counts = np.zeros(worker_count, dtype=np.int64)
        indices = list(range(worker_count))
        worker_function = increment_shared
        label = "Unpadded Shared"

    start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(worker_function, counts, index, ITERATIONS)
            for index in indices
        ]

        for future in futures:
            future.result()

    elapsed = time.perf_counter() - start

    expected = ITERATIONS
    valid = all(counts[index] == expected for index in indices)

    return elapsed, valid, label


def run_local_benchmark(worker_count):
    counts = np.zeros(worker_count, dtype=np.int64)

    start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(increment_local, counts, i, ITERATIONS)
            for i in range(worker_count)
        ]

        for future in futures:
            future.result()

    elapsed = time.perf_counter() - start
    valid = bool(np.all(counts == ITERATIONS))

    return elapsed, valid


def main():
    physical_cores = psutil.cpu_count(logical=False) or 1
    logical_cores = os.cpu_count() or 1

    print("Student ID:", STUDENT_ID)
    print("Operating System:", platform.platform())
    print("CPU Model:", platform.processor() or "Not available")
    print("Physical CPU Cores:", physical_cores)
    print("Logical CPU Cores:", logical_cores)
    print("Threads:", physical_cores)
    print("Increments per thread:", f"{ITERATIONS:,}")
    print()

    warmup = np.zeros(8, dtype=np.int64)
    increment_shared(warmup, 0, 1)
    increment_local(warmup, 1, 1)

    unpadded_time, unpadded_valid, _ = run_benchmark(
        physical_cores, padded=False
    )

    padded_time, padded_valid, _ = run_benchmark(
        physical_cores, padded=True
    )

    local_time, local_valid = run_local_benchmark(physical_cores)

    print("Results")
    print("-" * 76)
    print(f"{'Method':<25} {'Wall Time (s)':>15} {'Correct':>12}")
    print("-" * 76)
    print(
        f"{'Unpadded Shared':<25} "
        f"{unpadded_time:>15.4f} "
        f"{str(unpadded_valid):>12}"
    )
    print(
        f"{'Padded Shared':<25} "
        f"{padded_time:>15.4f} "
        f"{str(padded_valid):>12}"
    )
    print(
        f"{'Local Counter':<25} "
        f"{local_time:>15.4f} "
        f"{str(local_valid):>12}"
    )
    print("-" * 76)

    ratio = unpadded_time / local_time if local_time > 0 else float("inf")
    print(f"Execution ratio (Unpadded / Local): {ratio:.2f}x")
    print(
        "All results correct:",
        unpadded_valid and padded_valid and local_valid
    )


if __name__ == "__main__":
    main()
