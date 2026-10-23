import os
import platform
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import shared_memory

import numpy as np

STUDENT_ID = 230103216
N = 20_000_000
P = 0.95

_worker_array = None
_worker_shm = None


def init_worker(shm_name, size):
    global _worker_array, _worker_shm

    _worker_shm = shared_memory.SharedMemory(name=shm_name)
    _worker_array = np.ndarray(
        (size,),
        dtype=np.float64,
        buffer=_worker_shm.buf
    )


def sum_chunk(bounds):
    start, end = bounds
    return float(np.sum(_worker_array[start:end]))


def benchmark(worker_count, shm_name):
    chunk_count = worker_count * 4
    chunk_size = (N + chunk_count - 1) // chunk_count

    chunks = [
        (start, min(start + chunk_size, N))
        for start in range(0, N, chunk_size)
    ]

    start_time = time.perf_counter()

    with ProcessPoolExecutor(
        max_workers=worker_count,
        initializer=init_worker,
        initargs=(shm_name, N)
    ) as executor:
        partial_sums = list(executor.map(sum_chunk, chunks))

    wall_time = time.perf_counter() - start_time
    total_sum = sum(partial_sums)

    return wall_time, total_sum


def main():
    max_cores = os.cpu_count() or 1

    print("Student ID:", STUDENT_ID)
    print("Operating System:", platform.platform())
    print("CPU Model:", platform.processor() or "Not available")
    print("Logical CPU Cores:", max_cores)
    print("Array Size:", f"{N:,}")
    print("Data Type: float64")
    print("Execution: Multiprocessing")
    print()

    rng = np.random.default_rng(STUDENT_ID)
    data = rng.random(N, dtype=np.float64)

    shm = shared_memory.SharedMemory(create=True, size=data.nbytes)

    try:
        shared_array = np.ndarray(
            data.shape,
            dtype=data.dtype,
            buffer=shm.buf
        )
        shared_array[:] = data
        del shared_array
        del data

        worker_counts = sorted(set([
            1, 2, 4, max_cores, max_cores * 2
        ]))

        results = []
        baseline_time = None
        reference_sum = None

        print("Running benchmarks...")
        print()

        for workers in worker_counts:
            wall_time, total_sum = benchmark(workers, shm.name)

            if baseline_time is None:
                baseline_time = wall_time
                reference_sum = total_sum

            speedup = baseline_time / wall_time

            amdahl_bound = 1 / ((1 - P) + P / workers)

            results.append((
                workers,
                wall_time,
                speedup,
                amdahl_bound,
                total_sum
            ))

        print("Cores | Wall Time (s) | Speedup | Amdahl Upper Bound")
        print("-" * 62)

        for workers, wall_time, speedup, bound, total_sum in results:
            print(
                f"{workers:5d} | "
                f"{wall_time:13.4f} | "
                f"{speedup:7.2f}x | "
                f"{bound:18.2f}x"
            )

        print()
        print("P =", P)
        print("Baseline sum:", reference_sum)
        print("All sums match baseline:", all(
            np.isclose(total_sum, reference_sum, rtol=1e-12, atol=1e-8)
            for _, _, _, _, total_sum in results
        ))

    finally:
        shm.close()
        shm.unlink()


if __name__ == "__main__":
    mp.freeze_support()
    main()
