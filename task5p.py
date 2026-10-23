
import hashlib
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

STUDENT_ID = 230103216
TASK_COUNT = 10_000
HASHES_PER_TASK = 500
PAYLOAD = b"A" * 1024


def get_core_count():
    try:
        import psutil
        physical = psutil.cpu_count(logical=False)
        if physical:
            return physical
    except ImportError:
        pass

    return os.cpu_count() or 1


def cpu_task(task_id):
    digest = PAYLOAD

    for _ in range(HASHES_PER_TASK):
        digest = hashlib.sha256(PAYLOAD + digest[:0]).digest()

    return digest[0] + task_id % 1


def run_benchmark(pool_size):
    start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=pool_size) as executor:
        checksum = sum(executor.map(cpu_task, range(TASK_COUNT)))

    elapsed_ms = (time.perf_counter() - start) * 1000
    return elapsed_ms, checksum


def main():
    cores = get_core_count()

    requested_sizes = [
        1,
        max(1, cores // 2),
        cores,
        cores * 2,
        cores * 4,
        cores * 16,
        256,
    ]

    pool_sizes = list(dict.fromkeys(requested_sizes))

    print(f"Student ID: {STUDENT_ID}")
    print(f"Physical cores used for sizing: {cores}")
    print(f"CPU tasks: {TASK_COUNT:,}")
    print(f"SHA-256 operations per task: {HASHES_PER_TASK}")
    print(f"Pool sizes: {pool_sizes}")
    print("\nRunning benchmarks...")

    results = []

    for pool_size in pool_sizes:
        times = []
        checksums = []

        for _ in range(3):
            elapsed_ms, checksum = run_benchmark(pool_size)
            times.append(elapsed_ms)
            checksums.append(checksum)

        median_ms = statistics.median(times)
        results.append({
            "pool": pool_size,
            "time": median_ms,
            "checksum": checksums[0],
        })

        print(f"Completed pool size {pool_size}")

    baseline = next(r["time"] for r in results if r["pool"] == 1)

    best_smaller_time = float("inf")
    first_worsening_pool = None

    for result in results:
        if result["time"] > best_smaller_time:
            first_worsening_pool = result["pool"]
            break
        best_smaller_time = min(best_smaller_time, result["time"])

    lines = [
        f"Student ID: {STUDENT_ID}",
        f"Physical cores used for sizing: {cores}",
        f"CPU tasks: {TASK_COUNT}",
        f"Hashes per task: {HASHES_PER_TASK}",
        "Payload size: 1024 bytes",
        "Timing statistic: median of 3 runs",
        "",
        f"{'Pool Size':>10} | {'Execution Time (ms)':>20} | "
        f"{'Speedup':>10} | Status",
        "-" * 75,
    ]

    for result in results:
        speedup = baseline / result["time"] if result["time"] > 0 else 0

        if result["pool"] == 1:
            status = "BASELINE"
        elif result["pool"] == first_worsening_pool:
            status = "FIRST WORSENING"
        elif result["time"] < baseline:
            status = "FASTER THAN BASELINE"
        else:
            status = "SLOWER THAN BASELINE"

        lines.append(
            f"{result['pool']:>10} | "
            f"{result['time']:>20.2f} | "
            f"{speedup:>9.3f}x | {status}"
        )

    lines.append("")
    if first_worsening_pool is not None:
        lines.append(
            "First observed worsening pool size: "
            f"{first_worsening_pool}"
        )
        lines.append(
            "Definition: first tested pool size slower than the "
            "best result among smaller tested pool sizes."
        )
    else:
        lines.append(
            "First observed worsening pool size: not detected "
            "in the tested pool sizes."
        )

    lines.append(
        "Note: timing alone cannot prove OS context switching "
        "is the sole cause of a slowdown."
    )

    report = "\n".join(lines)
    print("\n" + report)

    with open("task5p_output.txt", "w", encoding="utf-8") as file:
        file.write(report + "\n")

    print("\nResults saved to task5p_output.txt")


if __name__ == "__main__":
    main()
