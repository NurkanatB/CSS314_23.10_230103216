
import os
import platform
import time
import numpy as np

STUDENT_ID = 230103216

print("Student ID:", STUDENT_ID)
print("Operating System:", platform.platform())
print("CPU Model:", platform.processor() or "Not available")
print("Logical CPU Cores:", os.cpu_count())

N = 20_000_000
rng = np.random.default_rng(STUDENT_ID)
data = rng.random(N, dtype=np.float64)

print("Array Size:", len(data))
print("Data Type:", data.dtype)

start_time = time.perf_counter()

total = 0.0
for value in data:
    total += float(value)

mean = total / N
squared_sum = 0.0

for value in data:
    difference = float(value) - mean
    squared_sum += difference * difference

std_dev = (squared_sum / N) ** 0.5

end_time = time.perf_counter()
execution_time_ms = (end_time - start_time) * 1000

print("Sum:", total)
print("Mean:", mean)
print("Population Standard Deviation:", std_dev)
print("Execution Time (ms):", round(execution_time_ms, 2))
