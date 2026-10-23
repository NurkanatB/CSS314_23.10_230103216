import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

STUDENT_ID = 230103216
ARRAY_SIZE = 2_000_000
THREAD_COUNT = 4


def serial_prefix_sum(data):
    output = np.empty(len(data), dtype=np.int64)
    running_sum = 0

    for i in range(len(data)):
        running_sum += int(data[i])
        output[i] = running_sum

    return output


def compute_local_scan(chunk):
    local_output = np.cumsum(chunk, dtype=np.int64)
    local_sum = int(local_output[-1])
    return local_output, local_sum


def apply_offset(local_output, offset):
    if offset:
        local_output += offset
    return local_output


def parallel_prefix_sum(data):
    chunks = np.array_split(data, THREAD_COUNT)

    with ThreadPoolExecutor(max_workers=THREAD_COUNT) as executor:
        local_results = list(executor.map(compute_local_scan, chunks))

    local_scans = [result[0] for result in local_results]
    chunk_sums = [result[1] for result in local_results]

    offsets = []
    running_total = 0

    for chunk_sum in chunk_sums:
        offsets.append(running_total)
        running_total += chunk_sum

    with ThreadPoolExecutor(max_workers=THREAD_COUNT) as executor:
        adjusted = list(
            executor.map(apply_offset, local_scans, offsets)
        )

    return np.concatenate(adjusted)


def main():
    rng = np.random.default_rng(STUDENT_ID)
    data = rng.integers(1, 11, size=ARRAY_SIZE, dtype=np.int64)

    print(f"Student ID: {STUDENT_ID}")
    print(f"Array size: {ARRAY_SIZE:,}")
    print(f"Threads: {THREAD_COUNT}")
    print("Running serial prefix scan...")

    start = time.perf_counter()
    serial_output = serial_prefix_sum(data)
    serial_ms = (time.perf_counter() - start) * 1000

    print("Running parallel chunked prefix scan...")

    start = time.perf_counter()
    parallel_output = parallel_prefix_sum(data)
    parallel_ms = (time.perf_counter() - start) * 1000

    speedup = serial_ms / parallel_ms if parallel_ms > 0 else float("inf")

    try:
        assert parallel_output[-1] == serial_output[-1], (
            f"Final sum mismatch: parallel={parallel_output[-1]}, "
            f"serial={serial_output[-1]}"
        )
    except AssertionError as error:
        status = "FAIL"
        output = [
            f"Student ID: {STUDENT_ID}",
            f"Array size: {ARRAY_SIZE}",
            f"Serial Time (ms): {serial_ms:.2f}",
            f"Parallel Time (ms): {parallel_ms:.2f}",
            f"Speedup: {speedup:.4f}x",
            f"Verification Status: {status}",
            str(error),
        ]

        print("\n" + "\n".join(output))

        with open("task3p_output.txt", "w", encoding="utf-8") as file:
            file.write("\n".join(output) + "\n")

        sys.exit(1)

    output = [
        f"Student ID: {STUDENT_ID}",
        f"Array size: {ARRAY_SIZE}",
        f"Threads: {THREAD_COUNT}",
        f"Serial Time (ms): {serial_ms:.2f}",
        f"Parallel Time (ms): {parallel_ms:.2f}",
        f"Speedup: {speedup:.4f}x",
        "Verification Status: PASS",
        f"Serial final prefix sum: {serial_output[-1]}",
        f"Parallel final prefix sum: {parallel_output[-1]}",
    ]

    print("\n" + "\n".join(output))

    with open("task3p_output.txt", "w", encoding="utf-8") as file:
        file.write("\n".join(output) + "\n")

    print("\nResults saved to task3p_output.txt")


if __name__ == "__main__":
    main()
