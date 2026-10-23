
import math
import random
import threading
import time

STUDENT_ID = 230103216
TOTAL_POINTS = 40_000_000
THREAD_COUNT = 4


def monte_carlo_worker(count, rng, lock=None):
    inside = 0

    for _ in range(count):
        if lock is None:
            x = rng.random()
            y = rng.random()
        else:
            with lock:
                x = rng.random()
                y = rng.random()

        if x * x + y * y <= 1.0:
            inside += 1

    return inside


def run_shared_prng():
    rng = random.Random(STUDENT_ID)
    lock = threading.Lock()

    base, remainder = divmod(TOTAL_POINTS, THREAD_COUNT)
    counts = [base + (1 if i < remainder else 0)
              for i in range(THREAD_COUNT)]
    results = [0] * THREAD_COUNT

    def worker(index):
        results[index] = monte_carlo_worker(
            counts[index], rng, lock
        )

    threads = [
        threading.Thread(target=worker, args=(i,))
        for i in range(THREAD_COUNT)
    ]

    start = time.perf_counter()

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    elapsed_ms = (time.perf_counter() - start) * 1000
    pi_estimate = 4.0 * sum(results) / TOTAL_POINTS

    return pi_estimate, abs(pi_estimate - math.pi), elapsed_ms


def run_independent_prng():
    base, remainder = divmod(TOTAL_POINTS, THREAD_COUNT)
    counts = [base + (1 if i < remainder else 0)
              for i in range(THREAD_COUNT)]
    results = [0] * THREAD_COUNT

    def worker(index):
        seed = STUDENT_ID * 10 + index
        rng = random.Random(seed)
        results[index] = monte_carlo_worker(counts[index], rng)

    threads = [
        threading.Thread(target=worker, args=(i,))
        for i in range(THREAD_COUNT)
    ]

    start = time.perf_counter()

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    elapsed_ms = (time.perf_counter() - start) * 1000
    pi_estimate = 4.0 * sum(results) / TOTAL_POINTS

    return pi_estimate, abs(pi_estimate - math.pi), elapsed_ms


def main():
    print(f"Student ID: {STUDENT_ID}")
    print(f"Total sample points: {TOTAL_POINTS:,}")
    print(f"Threads: {THREAD_COUNT}")
    print("Starting benchmark...")

    shared_pi, shared_error, shared_ms = run_shared_prng()
    independent_pi, independent_error, independent_ms = (
        run_independent_prng()
    )

    speedup = (
        shared_ms / independent_ms
        if independent_ms > 0
        else float("inf")
    )

    output = [
        f"Student ID: {STUDENT_ID}",
        f"Total sample points: {TOTAL_POINTS}",
        f"Threads: {THREAD_COUNT}",
        "",
        "Shared PRNG (mutex-protected):",
        f"Computed pi: {shared_pi:.12f}",
        f"Absolute error: {shared_error:.12f}",
        f"Wall time: {shared_ms:.2f} ms",
        "",
        "Independent PRNG (one generator per thread):",
        f"Computed pi: {independent_pi:.12f}",
        f"Absolute error: {independent_error:.12f}",
        f"Wall time: {independent_ms:.2f} ms",
        "",
        f"Speedup factor (shared / independent): {speedup:.4f}x",
    ]

    print("\n" + "\n".join(output))

    with open("task2p_output.txt", "w", encoding="utf-8") as file:
        file.write("\n".join(output) + "\n")

    print("\nResults saved to task2p_output.txt")


if __name__ == "__main__":
    main()
