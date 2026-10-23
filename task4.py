
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

STUDENT_ID = 230103216
TOTAL_ITEMS = 1000
WORKERS = 4


def do_work(i):
    time.sleep(0.0001 * (i % 20))


def run_static():
    results = []
    lock = threading.Lock()
    start_time = time.perf_counter()

    def worker(worker_id, items):
        worker_start = time.perf_counter()

        for i in items:
            do_work(i)

        finish = time.perf_counter() - start_time

        with lock:
            results.append((worker_id, finish))

    block_size = TOTAL_ITEMS // WORKERS

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = []

        for worker_id in range(WORKERS):
            begin = worker_id * block_size
            end = (worker_id + 1) * block_size
            futures.append(
                executor.submit(worker, worker_id, range(begin, end))
            )

        for future in futures:
            future.result()

    total_time = time.perf_counter() - start_time
    finish_times = [finish for _, finish in results]
    completion_delta = max(finish_times) - min(finish_times)

    return total_time, completion_delta


def run_dynamic(chunk_size):
    task_queue = Queue()
    results = []
    lock = threading.Lock()
    start_time = time.perf_counter()

    for begin in range(0, TOTAL_ITEMS, chunk_size):
        task_queue.put(
            range(begin, min(begin + chunk_size, TOTAL_ITEMS))
        )

    def worker(worker_id):
        while True:
            try:
                chunk = task_queue.get_nowait()
            except Exception:
                break

            for i in chunk:
                do_work(i)

            task_queue.task_done()

        finish = time.perf_counter() - start_time

        with lock:
            results.append((worker_id, finish))

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = [
            executor.submit(worker, worker_id)
            for worker_id in range(WORKERS)
        ]

        for future in futures:
            future.result()

    total_time = time.perf_counter() - start_time
    finish_times = [finish for _, finish in results]
    completion_delta = max(finish_times) - min(finish_times)

    return total_time, completion_delta


def main():
    print("Student ID:", STUDENT_ID)
    print("Items:", TOTAL_ITEMS)
    print("Workers:", WORKERS)
    print("Workload: sleep(0.0001 * (i % 20))")
    print()

    print(
        f"{'Strategy':<25}"
        f"{'Total Time (s)':>17}"
        f"{'Completion Delta (s)':>24}"
    )
    print("-" * 66)

    static_time, static_delta = run_static()
    print(
        f"{'A: Static':<25}"
        f"{static_time:>17.4f}"
        f"{static_delta:>24.4f}"
    )

    dynamic1_time, dynamic1_delta = run_dynamic(1)
    print(
        f"{'B: Dynamic (chunk=1)':<25}"
        f"{dynamic1_time:>17.4f}"
        f"{dynamic1_delta:>24.4f}"
    )

    dynamic5_time, dynamic5_delta = run_dynamic(5)
    print(
        f"{'B: Dynamic (chunk=5)':<25}"
        f"{dynamic5_time:>17.4f}"
        f"{dynamic5_delta:>24.4f}"
    )

    print()
    print("Completion delta = last worker finish - first worker finish")


if __name__ == "__main__":
    import contextlib

    with open("task4_output.txt", "w", encoding="utf-8") as output:
        with contextlib.redirect_stdout(output):
            main()

    with open("task4_output.txt", "r", encoding="utf-8") as output:
        print(output.read(), end="")
