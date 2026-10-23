import os
import time
import threading
import platform
from concurrent.futures import ThreadPoolExecutor
import contextlib
import sys

STUDENT_ID = 230103216
THREADS = 4
INCREMENTS_PER_THREAD = 2_000_000
EXPECTED = THREADS * INCREMENTS_PER_THREAD

unsafe_counter = 0

safe_counter = 0
counter_lock = threading.Lock()


def unsafe_worker():
    global unsafe_counter

    for _ in range(INCREMENTS_PER_THREAD):
        current = unsafe_counter
        unsafe_counter = current + 1


def safe_worker():
    global safe_counter

    for _ in range(INCREMENTS_PER_THREAD):
        with counter_lock:
            safe_counter += 1


def main():
    global unsafe_counter, safe_counter

    print("Student ID:", STUDENT_ID)
    print("Operating System:", platform.platform())
    print("Logical CPU cores:", os.cpu_count())
    print("Threads:", THREADS)
    print("Increments per thread:", f"{INCREMENTS_PER_THREAD:,}")
    print()

    unsafe_counter = 0
    start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=THREADS) as executor:
        futures = [
            executor.submit(unsafe_worker)
            for _ in range(THREADS)
        ]
        for future in futures:
            future.result()

    unsafe_time = time.perf_counter() - start
    actual_unsafe = unsafe_counter

    lost_updates = EXPECTED - actual_unsafe
    lost_percentage = (lost_updates / EXPECTED) * 100

    print("=== UNSYNCHRONIZED COUNTER ===")
    print(f"Expected final value: {EXPECTED:,}")
    print(f"Actual final value:   {actual_unsafe:,}")
    print(f"Lost updates:         {lost_updates:,}")
    print(f"Updates lost (%):     {lost_percentage:.4f}%")
    print(f"Wall-clock time:      {unsafe_time:.4f} seconds")
    print()

    safe_counter = 0
    start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=THREADS) as executor:
        futures = [
            executor.submit(safe_worker)
            for _ in range(THREADS)
        ]
        for future in futures:
            future.result()

    safe_time = time.perf_counter() - start
    actual_safe = safe_counter

    print("=== SYNCHRONIZED COUNTER (LOCK) ===")
    print(f"Expected final value: {EXPECTED:,}")
    print(f"Actual final value:   {actual_safe:,}")
    print(f"All updates preserved: {actual_safe == EXPECTED}")
    print(f"Wall-clock time:       {safe_time:.4f} seconds")

    penalty = safe_time - unsafe_time
    if unsafe_time > 0:
        penalty_percent = (penalty / unsafe_time) * 100
    else:
        penalty_percent = 0.0

    print()
    print("=== SYNCHRONIZATION COST ===")
    print(f"Time penalty: {penalty:.4f} seconds")
    print(f"Time penalty (%): {penalty_percent:.2f}%")
    print()
    print(
        "Note: CPython's GIL may prevent this test from showing "
        "lost updates consistently. Results depend on Python version "
        "and runtime scheduling."
    )


if __name__ == "__main__":
    with open("task6_output.txt", "w", encoding="utf-8") as output:
        with contextlib.redirect_stdout(output):
            main()

    with open("task6_output.txt", "r", encoding="utf-8") as output:
        print(output.read(), end="")
