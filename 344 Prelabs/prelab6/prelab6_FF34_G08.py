import numpy as np

from cachesim import CacheSimulator, Cache, MainMemory
from argparse import ArgumentParser


def make_cache() -> CacheSimulator:
    mem = MainMemory()
    l3 = Cache("L3", 20480, 16, 64, "LRU")                           
    mem.load_to(l3)
    mem.store_from(l3)
    l2 = Cache("L2", 512, 8, 64, "LRU", store_to=l3, load_from=l3)  
    l1 = Cache("L1", 64, 8, 64, "LRU", store_to=l2, load_from=l2) 
    cs = CacheSimulator(l1, mem)
    return cs


parser = ArgumentParser()
parser.add_argument('-a', '--algorithm', type=str, choices=['simple', 'recursive'])
parser.add_argument('-N', '--N', type=int)
parser.add_argument('-K', '--K', type=int)
args = parser.parse_args()

algorithm, N, K = args.algorithm, args.N, args.K

cs1 = make_cache()
cs2 = make_cache()

rnd_vals1 = np.random.rand(N, N)
rnd_vals2 = np.random.rand(N, N)

# WRITE YOUR CODE BELOW #

# block index to row-major index
def normalize_index(i, j, N, K) -> tuple:
    block_row = i // K
    inside_block_row = i % K

    block_column = j // K
    inside_block_column = j % K

    row_block_size = N * K
    column_block_size = K * K

    idx = block_row * row_block_size + block_column * column_block_size + inside_block_row * K + inside_block_column

    normalized_i = idx // N
    normalized_j = idx % N

    return (normalized_i, normalized_j)

def fill_row_major_arrays(A, B, N, cs):
  for i in range(N):
    for j in range(N):
        idx = i * N + j

        A[i][j] = rnd_vals1[i][j]
        cs.store(idx) #store operation at position i * N + j in row major to memory(A1)

        B[i][j] = rnd_vals2[i][j]
        cs.store(N * N + idx) #store operation at position N * N + i * N + j in row major to memory(B1)


def fill_block_arrays(A, B, N, cs):
  for i in range(N):
    for j in range(N):
        (normalized_i, normalized_j) = normalize_index(i, j, N, K)

        A[normalized_i][normalized_j] = rnd_vals1[i][j]
        
        cs.store(normalized_i * N + normalized_j) #store operation at position i * N + j in row major to memory(A1)

        B[normalized_i][normalized_j] = rnd_vals2[i][j]
        cs.store(N * N + normalized_i * N + normalized_j) #store operation at position N * N + i * N + j in row major to memory(B1)


def matrix_multiply_row_major(A, B, C, N, cs):
  for i in range(N):
    for j in range(N):
        for k in range(N):
            mul1 = A[i, k]
            cs.load(N * i + k) #load operation at position N * i + k to row major cache

            mul2 = B[k, j]
            cs.load(N * N + N * k + j) #load operation at position N * k + j to row major cache

            C[i, j] += mul1 * mul2
            cs.store(2 * N * N + N * i + j) #store operation at position 2 * N * N + N * i + j in row major to memory(C1)


def matrix_multiply_blocks(A, B, C, N, cs):
  for i in range(N):
    for j in range(N):
        for k in range(N):
            (n_i, n_k) = normalize_index(i, k, N, K)
            mul1 = A[n_i, n_k]
            cs.load(N * n_i + n_k) #load operation at position N * i + k to row major cache

            (n_k, n_j) = normalize_index(k, j, N, K)
            mul2 = B[n_k, n_j]
            cs.load(N * N + N * n_k + n_j) #load operation at position N * k + j to row major cache

            C[i, j] += mul1 * mul2
            cs.store(2 * N * N + N * n_i + n_j) #store operation at position 2 * N * N + N * i + j in row major to memory(C2)


def recursive_matrix_multiply_row_major(A, B, result, i, j, k, n, cs):
    if n == 1:
        mul1 = A[i, k]
        cs.load(n * i + k) #load operation at position N * i + k to row major cache

        mul2 = B[k, j]
        cs.load(n * n + n * k + j) #load operation at position N * k + j to row major cache

        result[i, j] += mul1 * mul2
        cs.store(2 * n * n + n * i + j) #store operation at position 2 * N * N + N * i + j in row major to memory(C1)
        return

    mid = n // 2

    # Recursive matrix multiplications
    recursive_matrix_multiply_row_major(A, B, result, i, j, k, mid, cs)
    recursive_matrix_multiply_row_major(A, B, result, i, j, k + mid, mid, cs)

    recursive_matrix_multiply_row_major(A, B, result, i, j + mid, k, mid, cs)
    recursive_matrix_multiply_row_major(A, B, result, i, j + mid, k + mid, mid, cs)

    recursive_matrix_multiply_row_major(A, B, result, i + mid, j, k, mid, cs)
    recursive_matrix_multiply_row_major(A, B, result, i + mid, j, k + mid, mid, cs)

    recursive_matrix_multiply_row_major(A, B, result, i + mid, j + mid, k, mid, cs)
    recursive_matrix_multiply_row_major(A, B, result, i + mid, j + mid, k + mid, mid, cs)


def recursive_matrix_multiply_blocks(A, B, result, i, j, k, n, cs):
    if n == 1:

        (n_i, n_k) = normalize_index(i, k, N, K)
        mul1 = A[n_i, n_k]
        cs.load(n * n_i + n_k) #load operation at position N * i + k to row major cache

        (n_k, n_j) = normalize_index(k, j, N, K)
        mul2 = B[n_k, n_j]
        cs.load(n * n + n * n_k + n_j) #load operation at position N * k + j to row major cache

        result[i, j] += mul1 * mul2
        cs.store(2 * n * n + n * n_i + n_j) #store operation at position 2 * N * N + N * i + j in row major to memory(C1)
        return

    mid = n // 2

    # Recursive matrix multiplications
    recursive_matrix_multiply_blocks(A, B, result, i, j, k, mid, cs)
    recursive_matrix_multiply_blocks(A, B, result, i, j, k + mid, mid, cs)

    recursive_matrix_multiply_blocks(A, B, result, i, j + mid, k, mid, cs)
    recursive_matrix_multiply_blocks(A, B, result, i, j + mid, k + mid, mid, cs)

    recursive_matrix_multiply_blocks(A, B, result, i + mid, j, k, mid, cs)
    recursive_matrix_multiply_blocks(A, B, result, i + mid, j, k + mid, mid, cs)

    recursive_matrix_multiply_blocks(A, B, result, i + mid, j + mid, k, mid, cs)
    recursive_matrix_multiply_blocks(A, B, result, i + mid, j + mid, k + mid, mid, cs)
   

A1 = np.zeros([N, N])
B1 = np.zeros([N, N])
C1 = np.zeros([N, N])

fill_row_major_arrays(A1, B1, N, cs1)

if (algorithm == 'simple'):
  matrix_multiply_row_major(A1, B1, C1, N, cs1)
elif (algorithm == 'recursive'):
  recursive_matrix_multiply_row_major(A1, B1, C1, 0, 0, 0, N, cs1)


A2 = np.zeros([N, N])
B2 = np.zeros([N, N])
C2 = np.zeros([N, N])

fill_block_arrays(A2, B2, N, cs2)

if (algorithm == 'simple'):
  matrix_multiply_blocks(A2, B2, C2, N, cs2)
elif (algorithm == 'recursive'):
  recursive_matrix_multiply_blocks(A2, B2, C2, 0, 0, 0, N, cs2)

# WRITE YOUR CODE ABOVE #

print('Row major array')
cs1.print_stats()


print('Block array')
cs2.print_stats()