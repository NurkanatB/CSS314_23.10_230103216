import os
import time
import platform
from concurrent.futures import ThreadPoolExecutor

import numpy as np

STUDENT_ID = 230103216
SIZES = [100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000]
REPEATS = 5
WORKERS = os.cpu_count() or 1
A = 2.5
B = 1.5


def serial_operation(x):
    return A * x + B


def parallel_operation(x, pool):
    chunks = [chunk for chunk in np.array_split(x, WORKERS) if len(chunk)]

    futures = [
        pool.submit(serial_operation, chunk)
        for chunk in chunks
    ]
    return [future.result() for future in futures]


def measure_serial(x):
    start = time.perf_counter()
    y = serial_operation(x)
    elapsed = time.perf_counter() - start
    return elapsed, y


def measure_parallel(x, pool):
    start = time.perf_counter()
    parts = parallel_operation(x, pool)
    elapsed = time.perf_counter() - start
    return elapsed, parts


def main():
    print("Student ID:", STUDENT_ID)
    print("Operating System:", platform.platform())
    print("CPU Model:", platform.processor() or "Not available")
    print("Logical CPU cores:", WORKERS)
    print("Parallel workers:", WORKERS)
    print("Operation: y[i] = a * x[i] + b")
    print(f"a = {A}, b = {B}")
    print(f"Repeats per configuration: {REPEATS}")
    print()

    results = []

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for n in SIZES:
            rng = np.random.default_rng(STUDENT_ID + n)
            x = rng.random(n, dtype=np.float64)

            serial_times = []
            parallel_times = []

            serial_operation(x[:min(n, 100)])

            for _ in range(REPEATS):
                elapsed, y_serial = measure_serial(x)
                serial_times.append(elapsed)

                elapsed, parallel_parts = measure_parallel(x, pool)
                parallel_times.append(elapsed)

            serial_time = float(np.median(serial_times))
            parallel_time = float(np.median(parallel_times))
            speedup = serial_time / parallel_time
            faster = parallel_time < serial_time

            y_parallel = np.concatenate(parallel_parts)
            correct = np.allclose(y_serial, y_parallel)

            results.append({
                "n": n,
                "serial": serial_time,
                "parallel": parallel_time,
                "speedup": speedup,
                "faster": faster,
                "correct": correct,
            })

    print(
        f"{'N':>12} | {'Serial (ms)':>12} | "
        f"{'Parallel (ms)':>14} | {'Speedup':>9} | "
        f"{'Parallel faster?':>16} | Correct"
    )
    print("-" * 92)

    for r in results:
        print(
            f"{r['n']:12,} | "
            f"{r['serial'] * 1000:12.4f} | "
            f"{r['parallel'] * 1000:14.4f} | "
            f"{r['speedup']:8.3f}x | "
            f"{str(r['faster']):>16} | "
            f"{r['correct']}"
        )

    print("\nCROSSOVER ANALYSIS")
    states = [r["faster"] for r in results]
    transitions = []

    for i in range(1, len(results)):
        if states[i] != states[i - 1]:
            lower_n = results[i - 1]["n"]
            upper_n = results[i]["n"]

            if states[i]:
                print(
                    f"Parallel changes from SLOWER to FASTER "
                    f"between N={lower_n:,} and N={upper_n:,}."
                )
            else:
                print(
                    f"Parallel changes from FASTER to SLOWER "
                    f"between N={lower_n:,} and N={upper_n:,}."
                )
            transitions.append((lower_n, upper_n))

    if not transitions:
        if all(states):
            print("Parallel was faster at every tested size.")
        elif not any(states):
            print("Parallel was slower or equal at every tested size.")
        else:
            print(
                "No single slower-to-faster transition was found; "
                "results may fluctuate across sizes."
            )

    print(
        "\nNote: the crossover is reported as a range between "
        "tested sizes, not an exact untested N."
    )


if __name__ == "__main__":
    import contextlib

    with open("task5_output.txt", "w", encoding="utf-8") as output:
        with contextlib.redirect_stdout(output):
            main()

    with open("task5_output.txt", "r", encoding="utf-8") as output:
        print(output.read(), end="")
