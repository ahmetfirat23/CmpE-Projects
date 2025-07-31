from homework1 import collect
from multiprocessing import Process


if __name__ == "__main__":
    processes = []
    for i in range(10):
        p = Process(target=collect, args=(i, 100))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()