import time
import numpy as np
# from tqdm import tqdm
import argparse
# from interruptingcow import timeout

chess_board = []

for i in range(8):
    chess_board.append([-1, -1, -1, -1, -1, -1, -1, -1] )


def random_move(chess_board, row, col, move_count):
    moves = [[row + 2, col + 1], [row + 2, col - 1], [row - 2, col + 1], [row - 2, col - 1], [row + 1, col + 2], [row + 1, col - 2], [row - 1, col + 2], [row - 1, col - 2]]
    np.random.shuffle(moves)
    for move in moves:
        if move[0] >= 0 and move[0] < 8 and move[1] >= 0 and move[1] < 8 and chess_board[move[0]][move[1]] == -1:
            chess_board[move[0]][move[1]] = move_count
            return move
    return []

    
def run_test(chess_board, run, p, file):
    move_count = 0
    row_pos = np.random.randint(0, 8)
    col_pos = np.random.randint(0, 8)
    chess_board[row_pos][col_pos] = move_count
    print(f"Run {run}: starting from ({row_pos}, {col_pos})", file=file)
    move_count += 1
    threshold = np.ceil(64 * p)

    success_count = 0
    while True:
        move = random_move(chess_board, row_pos, col_pos, move_count)
        if (move == []):
            break
        row_pos = move[0]
        col_pos = move[1]
        print(f"Stepping into  ({row_pos}, {col_pos})", file=file)
        move_count += 1
        if move_count >= threshold:
            print(f"Successful - Tour length : {move_count}", file=file)
            success_count = 1
            break

    if (success_count == 0):
        print(f"Unsuccessful - Tour length : {move_count}", file=file)
        
    for i in range(8):
        for j in range(8):
            print(f"{chess_board[i][j]:3}", end=" ", file=file)
        print(file=file)
    return success_count



def run_test_recursive(chess_board, p, k):
    move_count = 0
    row_pos = np.random.randint(0, 8)
    col_pos = np.random.randint(0, 8)
    chess_board[row_pos][col_pos] = 1
    move_count += 1
    threshold = np.ceil(64 * p)
    success_count = 0
    for i in range(k, 0, -1):
        move = random_move(chess_board, row_pos, col_pos, move_count)
        if (move == []):
            return 0
        row_pos = move[0]
        col_pos = move[1]
        move_count += 1
    ret_val = recursive_move(chess_board, row_pos, col_pos, move_count, threshold)
    if ret_val == -1:
        success_count = 0

    else:
        success_count = 1
    return success_count


def recursive_move(chess_board, row, col, move_count, threshold):
    moves = [[row + 2, col + 1], [row + 2, col - 1], [row - 2, col + 1], [row - 2, col - 1], [row + 1, col + 2], [row + 1, col - 2], [row - 1, col + 2], [row - 1, col - 2]]

    for move in moves:
        if move[0] >= 0 and move[0] < 8 and move[1] >= 0 and move[1] < 8 and chess_board[move[0]][move[1]] == -1:
            chess_board[move[0]][move[1]] = 1
            if move_count+1 >= threshold:
                return 1           
            ret_val = recursive_move(chess_board, move[0], move[1], move_count + 1, threshold)

            if ret_val == -1:
                chess_board[move[0]][move[1]] = -1
            else:
                return 1
    return -1


def reset_board(chess_board):
    for i in range(8):
        for j in range(8):
            chess_board[i][j] = -1

def run_tests(chess_board, num_runs, p, file):
    num_of_success = 0
    for i in range(num_runs):
        reset_board(chess_board)
        num_of_success += run_test(chess_board, i + 1, p, file)
        print(file=file)
    reset_board(chess_board)
    return num_of_success

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("part", type=str, default="part1")
    part = argparser.parse_args().part

    if part == "part1":   
        num_runs = 100000
        with open('results_p07.txt', 'w') as file:
            start_time1 = time.time()
            success_count_1 = run_tests(chess_board, num_runs, 0.7, file)
            end_time1 = time.time()
            # print(f"Time taken for p = 0.7 : {end_time1 - start_time1}")
            print("LasVegas Algorithm With p = 0.7")
            print(f"Number of successful tours : {success_count_1}")
            print(f"Number of trials : {num_runs}")
            print(f"Probability of a successful tour : {success_count_1 / num_runs}")
            print()
        with open('results_p08.txt', 'w') as file:
            start_time2 = time.time()
            success_count_2 = run_tests(chess_board, num_runs, 0.8, file)
            end_time2 = time.time()
            # print(f"Time taken for p = 0.8 : {end_time2 - start_time2}")
            print("LasVegas Algorithm With p = 0.8")
            print(f"Number of successful tours : {success_count_2}")
            print(f"Number of trials : {num_runs}")
            print(f"Probability of a successful tour : {success_count_2 / num_runs}")
            print()
        with open('results_p085.txt', 'w') as file:
            start_time3 = time.time()
            success_count_3 = run_tests(chess_board, num_runs, 0.85, file)
            end_time3 = time.time()
            # print(f"Time taken for p = 0.85 : {end_time3 - start_time3}")
            print("LasVegas Algorithm With p = 0.85")
            print(f"Number of successful tours : {success_count_3}")
            print(f"Number of trials : {num_runs}")
            print(f"Probability of a successful tour : {success_count_3 / num_runs}")
    elif part == "part2":
        for i in [0.7, 0.8, 0.85]:
            print(f"--- p = {i} ---")
            for k in [0, 2, 3]:
                success_count = 0
                start_time = time.time()
                for _ in range(100000):
                    success_count += run_test_recursive(chess_board, i, k)
                    reset_board(chess_board)
                end_time = time.time()   
                print(f"LasVegas Algorithm With p = {i}, k = {k}")
                print(f"Number of successful tours : {success_count}")
                print(f"Number of trials : {100000}")
                print(f"Probability of a successful tour : {success_count / 100000}")
                print()
                # print(f"Time taken : {end_time - start_time}")
    # elif part == "part2-3":
    #     for i in tqdm([0.7, 0.8, 0.85]):
    #         for k in [15]:
    #             success_count = 0
    #             start_time = time.time()
    #             for _ in tqdm(range(1000)):
    #                 try:
    #                     with timeout(30, exception=RuntimeError):
    #                         success_count += run_test_recursive(chess_board, i, k)
    #                 except RuntimeError:
    #                     pass
    #                 reset_board(chess_board)
    #             end_time = time.time()
    #             print(f"--- p = {i} ---")
    #             print(f"LasVegas Algorithm With p = {i}, k = {k}")
    #             print(f"Number of successful tours : {success_count}")
    #             print(f"Number of trials : {1000}")
    #             print(f"Probability of a successful tour : {success_count / 10000}")
    #             print(f"Time taken : {end_time - start_time}")
    # elif part == "part3":
    #     with open('part3.txt', 'w') as file:
    #         start_time1 = time.time()
    #         success_count_1 = run_tests(chess_board, 100000, 1, file)
    #         end_time1 = time.time()
    #     print(f"Time taken for p = 1 : {end_time1 - start_time1}")
    #     print("LasVegas Algorithm With p = 1")
    #     print(f"Number of successful tours : {success_count_1}")
    #     print(f"Number of trials : {100000}")
    #     print(f"Probability of a successful tour : {success_count_1 / 100000}")
    #     print()      

    #     i = 1
    #     for k in tqdm([35, 40]):
    #         success_count = 0
    #         start_time = time.time()
    #         for _ in tqdm(range(100000)):
    #             try:
    #                 with timeout(600, exception=RuntimeError):
    #                     success_count += run_test_recursive(chess_board, i, k)
    #                     # if success_count == 1:
    #                     #     break
    #             except RuntimeError:
    #                 pass
    #             reset_board(chess_board)
    #         end_time = time.time()
    #         print(f"--- p = {i} ---")
    #         print(f"LasVegas Algorithm With p = {i}, k = {k}")
    #         print(f"Number of successful tours : {success_count}")
    #         print(f"Number of trials : {100000}")
    #         print(f"Probability of a successful tour : {success_count / 100000}")
    #         print(f"Time taken : {end_time - start_time}")

