
import multiprocessing as mp
from multiprocessing import shared_memory
import statistics
import time

import numpy as np

STUDENT_ID = 230103216
ARRAY_SIZES = [100_000, 1_000_000, 10_000_000]
TRIALS = 3


def queue_receiver(queue, connection):
    connection.send("READY")
    array = queue.get()
    connection.send(("RECEIVED", array.size))
    connection.close()


def shared_memory_receiver(connection):
    connection.send("READY")
    name, size = connection.recv()

    shm = shared_memory.SharedMemory(name=name)
    try:
        array = np.ndarray((size,), dtype=np.int64, buffer=shm.buf)
        first_value = int(array[0]) if size else 0
        connection.send(("RECEIVED", size, first_value))
    finally:
        shm.close()
        connection.close()


def benchmark_queue(array):
    context = mp.get_context("spawn")
    queue = context.Queue()
    parent_conn, child_conn = context.Pipe()

    process = context.Process(
        target=queue_receiver,
        args=(queue, child_conn)
    )
    process.start()
    child_conn.close()

    try:
        if parent_conn.recv() != "READY":
            raise RuntimeError("Queue receiver failed to become ready")

        start = time.perf_counter()
        queue.put(array)
        response = parent_conn.recv()
        elapsed_ms = (time.perf_counter() - start) * 1000

        if response != ("RECEIVED", array.size):
            raise RuntimeError("Queue transfer verification failed")

        process.join()
        if process.exitcode != 0:
            raise RuntimeError("Queue receiver exited with an error")

        return elapsed_ms
    finally:
        if process.is_alive():
            process.terminate()
            process.join()
        queue.close()
        queue.join_thread()
        parent_conn.close()


def benchmark_shared_memory(array):
    context = mp.get_context("spawn")
    shm = shared_memory.SharedMemory(create=True, size=array.nbytes)
    parent_conn, child_conn = context.Pipe()

    try:
        shared_array = np.ndarray(
            array.shape, dtype=np.int64, buffer=shm.buf
        )
        shared_array[:] = array

        process = context.Process(
            target=shared_memory_receiver,
            args=(child_conn,)
        )
        process.start()
        child_conn.close()

        if parent_conn.recv() != "READY":
            raise RuntimeError("Shared-memory receiver failed to become ready")

        start = time.perf_counter()
        parent_conn.send((shm.name, array.size))
        response = parent_conn.recv()
        elapsed_ms = (time.perf_counter() - start) * 1000

        expected = ("RECEIVED", array.size, int(array[0]))
        if response != expected:
            raise RuntimeError("Shared-memory verification failed")

        process.join()
        if process.exitcode != 0:
            raise RuntimeError("Shared-memory receiver exited with an error")

        return elapsed_ms
    finally:
        if 'process' in locals() and process.is_alive():
            process.terminate()
            process.join()
        parent_conn.close()
        shm.close()
        shm.unlink()


def main():
    results = [
        "CPU IPC / Shared Memory Transfer Benchmark",
        f"Student ID: {STUDENT_ID}",
        f"Trials per path: {TRIALS}",
        "Timing excludes receiver startup and initial shared-memory allocation.",
        "Path A includes queue serialization and data transfer.",
        "Path B measures descriptor IPC; array data is shared without copying.",
        "",
    ]

    print("\n".join(results))

    for size in ARRAY_SIZES:
        print(f"Benchmarking N={size:,} ...")
        rng = np.random.default_rng(STUDENT_ID + size)
        array = rng.integers(
            1, 101, size=size, dtype=np.int64
        )

        queue_times = []
        shared_times = []

        for _ in range(TRIALS):
            queue_times.append(benchmark_queue(array))
            shared_times.append(benchmark_shared_memory(array))

        queue_ms = statistics.median(queue_times)
        shared_ms = statistics.median(shared_times)
        ratio = queue_ms / shared_ms if shared_ms > 0 else float("inf")

        block = [
            f"N = {size:,}",
            f"Path A (Queue / Pickle): {queue_ms:.3f} ms",
            f"Path B (Shared Memory): {shared_ms:.3f} ms",
            f"Path A / Path B ratio: {ratio:.2f}x",
            "",
        ]
        results.extend(block)
        print("\n".join(block))

    with open("task4p_output.txt", "w", encoding="utf-8") as file:
        file.write("\n".join(results) + "\n")

    print("Results saved to task4p_output.txt")


if __name__ == "__main__":
    mp.freeze_support()
    main()
