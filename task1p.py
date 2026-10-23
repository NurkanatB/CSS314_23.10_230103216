
import os
import time
import multiprocessing as mp
from multiprocessing import shared_memory

import numpy as np

STUDENT_ID = 230103216
MATRIX_SIZE = 8000


def get_cpu_counts():
    logical = os.cpu_count() or 1

    try:
        import psutil
        physical = psutil.cpu_count(logical=False) or max(1, logical // 2)
    except ImportError:
        physical = max(1, logical // 2)

    return logical, physical


def parallel_sum(args):
    shm_name, start, end, mode = args

    shm = shared_memory.SharedMemory(name=shm_name)
    try:
        matrix = np.ndarray(
            (MATRIX_SIZE, MATRIX_SIZE),
            dtype=np.float64,
            buffer=shm.buf
        )

        if mode == "row":
            result = matrix[start:end, :].sum(dtype=np.float64)
        else:
            result = matrix[:, start:end].sum(dtype=np.float64)

        return float(result)
    finally:
        shm.close()


def run_benchmark(shm_name, workers, mode):
    ranges = []

    for i in range(workers):
        start = i * MATRIX_SIZE // workers
        end = (i + 1) * MATRIX_SIZE // workers
        ranges.append((shm_name, start, end, mode))

    start_time = time.perf_counter()

    with mp.Pool(processes=workers) as pool:
        partial_sums = pool.map(parallel_sum, ranges)

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    return sum(partial_sums), elapsed_ms


def main():
    logical, physical = get_cpu_counts()
    workers = min(physical, MATRIX_SIZE)

    print(f"Student ID: {STUDENT_ID}")
    print(f"Logical cores: {logical}")
    print(f"Physical cores: {physical}")
    print(f"Parallel workers: {workers}")
    print(f"Matrix size: {MATRIX_SIZE} x {MATRIX_SIZE}")

    rng = np.random.default_rng(STUDENT_ID)
    matrix = rng.random(
        (MATRIX_SIZE, MATRIX_SIZE),
        dtype=np.float64
    )

    shm = shared_memory.SharedMemory(
        create=True,
        size=matrix.nbytes
    )

    shared_matrix = np.ndarray(
        matrix.shape,
        dtype=np.float64,
        buffer=shm.buf
    )
    shared_matrix[:] = matrix
    del shared_matrix
    del matrix

    try:
        row_sum, row_ms = run_benchmark(shm.name, workers, "row")
        col_sum, col_ms = run_benchmark(shm.name, workers, "column")

        penalty = col_ms / row_ms if row_ms > 0 else float("inf")

        output = [
            f"Student ID: {STUDENT_ID}",
            f"Logical cores: {logical}",
            f"Physical cores: {physical}",
            f"Parallel workers: {workers}",
            f"Matrix size: {MATRIX_SIZE} x {MATRIX_SIZE}",
            f"Mode A (Row-Major) runtime: {row_ms:.2f} ms",
            f"Mode B (Column-Major / Strided) runtime: {col_ms:.2f} ms",
            f"Cache penalty factor: {penalty:.4f}",
            f"Mode A sum: {row_sum:.10f}",
            f"Mode B sum: {col_sum:.10f}",
            f"Sum difference: {abs(row_sum - col_sum):.10f}",
        ]

        print("\n" + "\n".join(output))

        with open("task1_output.txt", "w", encoding="utf-8") as file:
            file.write("\n".join(output) + "\n")

        print("\nResults saved to task1_output.txt")

    finally:
        shm.close()
        shm.unlink()


if __name__ == "__main__":
    mp.freeze_support()
    main()
