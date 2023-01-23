
glass_size = int(input())
straw_pos = int(input())

# DO_NOT_EDIT_ANYTHING_ABOVE_THIS_LINE


# Print asterisk character n times
def print_asterisk_n_times(n):
    if n == 0:
        return
    print('*', end='')
    print_asterisk_n_times(n-1)


# Print space character n times
def print_space_n_times(n):
    if n == 0:
        return
    print(' ', end='')
    print_space_n_times(n-1)


# Draw straw alone
def draw_straw(_straw_pos):
    if _straw_pos == 0:
        return
    draw_straw(_straw_pos-1)
    print_space_n_times(_straw_pos-1)
    print('o')


# Draw intersection of straw and glass
def draw_intersection(_sip_num, _glass_size, _straw_pos):
    if _sip_num == 0:
        return
    draw_intersection(_sip_num-1, _glass_size, _straw_pos)
    print_space_n_times(_sip_num-1)
    print('\\', end='')
    print_space_n_times(_straw_pos - 1)
    print('o', end='')
    print_space_n_times(2 * (_glass_size - _sip_num + 1) - _straw_pos)
    print('/')


# Draw only glass
def draw_glass(_glass_size, _sip_num):
    if _sip_num == _glass_size:
        return
    n = _sip_num
    print_space_n_times(n)
    print('\\', end='')
    print_asterisk_n_times(2 * (_glass_size - n))
    print('/')
    draw_glass(_glass_size, _sip_num + 1)


# Draw fixed part of the glass
def draw_fixed(_glass_size, m=0):
    if m == _glass_size:
        return
    print_space_n_times(_glass_size)
    print('||')
    draw_fixed(_glass_size, m+1)


# Act as main function
def draw_whole_glass(_glass_size, _straw_pos, sip_num=0):
    if sip_num == _glass_size + 1:
        return
    # Check if the line available for sipping
    if _straw_pos <= 2 * (_glass_size - sip_num + 1):

        draw_straw(_straw_pos)

        draw_intersection(sip_num, _glass_size, _straw_pos)

        draw_glass(_glass_size, sip_num)

        # Complete fixed part of the glass
        print_space_n_times(_glass_size)
        print('--')
        draw_fixed(_glass_size)

    draw_whole_glass(_glass_size, _straw_pos, sip_num+1)


draw_whole_glass(glass_size, straw_pos)

# DO_NOT_EDIT_ANYTHING_BELOW_THIS_LINE

