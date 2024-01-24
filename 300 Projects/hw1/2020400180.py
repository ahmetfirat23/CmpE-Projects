import random
import math
import time

def algorithm(arr, n):
    y = 0
    for i in range(0, n):
        if arr[i] == 0:
            for j in range(i, n):
                y += 1
                k = n
                while k > 0:
                    k = k // 3
                    y += 1 # basic operation
        elif arr[i] == 1:
            for m in range(i, n):
                y += 1
                for l in range(m, n):
                    for t in range(n, 0, -1):
                        for z in range(n, 0, -t): #confusing
                            y += 1 # basic operation
        else:
            y += 1
            p = 0
            while p < n:
                for j in range(0, p**2):
                    y += 1 # basic operation
                p += 1

# Measure the runtime of the algorithm
def measure_time(arr, n):
    start = time.time()
    algorithm(arr, n)
    end = time.time()
    return end - start

# These lambda functions are used to find the switch point in the worst case
fn = lambda n: math.floor(math.log(n) / math.log(3))
find_switch_4 = lambda n: math.floor(n + 1/2 - math.sqrt((2*n**2 - 3*n +1) / (3*math.log(n)) + 1/4))

# Best case
for n, i in zip([1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150], range(17)):
    arr = [0 for x in range(n)]
    curr_time = measure_time(arr, n)
    print(f"Case: best Size: {n} Elapsed Time (s): {curr_time}")

# Worst case
for n, i in zip([1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150], range(17)):
    border = 2 if n == 1 else find_switch_4(n)
        
    arr = [1 if (x <= border) else 2 for x in range(n)]
    curr_time = measure_time(arr, n)
    print(f"Case: worst Size: {n} Elapsed Time (s): {curr_time}")

# Average case
run_count = 10
for n, i in zip([1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150], range(17)):
    run_time = 0
    for j in range(run_count):
        arr = [random.randint(0, 2) for i in range(n)]
        curr_time = measure_time(arr, n)
        print(f"Case: average_{j} Size: {n} Elapsed Time (s): {curr_time}")
        run_time += curr_time
    print(f"Case: average Size: {n} Elapsed Time (s): {run_time / run_count}")
