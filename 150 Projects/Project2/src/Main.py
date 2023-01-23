glass_size = int(input())
straw_pos = int(input())

# DO_NOT_EDIT_ANYTHING_ABOVE_THIS_LINE

for sip_num in range(glass_size + 1):
    # Check if the line available for sipping
    if straw_pos <= 2 * (glass_size - sip_num + 1):
        # Draw straw alone
        for i in range(straw_pos):
            for _ in range(i):
                print(' ', end='')
            print('o')

        # Draw intersection of straw and glass
        for k in range(sip_num):
            for _ in range(k):
                print(' ', end='')
            print('\\', end='')
            for _ in range(straw_pos - 1):
                print(' ', end='')
            print('o', end='')
            for _ in range(2 * (glass_size - k) - straw_pos):
                print(' ', end='')
            print('/')

        # Draw only glass
        for j in range(glass_size - sip_num):
            j = j + sip_num
            for _ in range(j):
                print(' ', end='')
            print('\\', end='')
            for _ in range(2 * (glass_size - j)):
                print('*', end='')
            print('/')

        # Draw fixed part of the glass
        for _ in range(glass_size):
            print(' ', end='')
        print('--')
        for _ in range(glass_size):
            for _ in range(glass_size):
                print(' ', end='')
            print('||')

# DO_NOT_EDIT_ANYTHING_BELOW_THIS_LINE
