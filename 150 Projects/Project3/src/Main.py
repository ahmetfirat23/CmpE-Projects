
# Ultimate Battleships

def print_ships_to_be_placed():
    print("Ships to be placed:", end=" ")
    if FILE_OUTPUT_FLAG:
        f.write("Ships to be placed: ")


# elem expected to be a single list element of a primitive type.
def print_single_element(elem):
    print(str(elem), end=" ")
    if FILE_OUTPUT_FLAG:
        f.write(str(elem) + " ")


def print_empty_line():
    print()
    if FILE_OUTPUT_FLAG:
        f.write("\n")


# n expected to be str or int.
def print_player_turn_to_place(n):
    print("It is Player {}'s turn to place their ships.".format(n))
    if FILE_OUTPUT_FLAG:
        f.write("It is Player {}'s turn to place their ships.\n".format(n))


def print_to_place_ships():
    print("Enter a name, coordinates and orientation to place a ship (Example: Carrier 1 5 h) :", end=" ")
    if FILE_OUTPUT_FLAG:
        f.write("Enter a name, coordinates and orientation to place a ship (Example: Carrier 1 5 h) : \n")
        # There is a \n because we want the board to start printing on the next line.


def print_incorrect_input_format():
    print("Input is in incorrect format, please try again.")
    if FILE_OUTPUT_FLAG:
        f.write("Input is in incorrect format, please try again.\n")


def print_incorrect_coordinates():
    print("Incorrect coordinates given, please try again.")
    if FILE_OUTPUT_FLAG:
        f.write("Incorrect coordinates given, please try again.\n")


def print_incorrect_ship_name():
    print("Incorrect ship name given, please try again.")
    if FILE_OUTPUT_FLAG:
        f.write("Incorrect ship name given, please try again.\n")


def print_incorrect_orientation():
    print("Incorrect orientation given, please try again.")
    if FILE_OUTPUT_FLAG:
        f.write("Incorrect orientation given, please try again.\n")


# ship expected to be str.
def print_ship_is_already_placed(ship):
    print(ship, "is already placed, please try again.")
    if FILE_OUTPUT_FLAG:
        f.write(ship + " is already placed, please try again.\n")


# ship expected to be str.
def print_ship_cannot_be_placed_outside(ship):
    print(ship, "cannot be placed outside the board, please try again.")
    if FILE_OUTPUT_FLAG:
        f.write(ship + " cannot be placed outside the board, please try again.\n")


# ship expected to be str.
def print_ship_cannot_be_placed_occupied(ship):
    print(ship, "cannot be placed to an already occupied space, please try again.")
    if FILE_OUTPUT_FLAG:
        f.write(ship + " cannot be placed to an already occupied space, please try again.\n")


def print_confirm_placement():
    print("Confirm placement Y/N :", end=" ")
    if FILE_OUTPUT_FLAG:
        f.write("Confirm placement Y/N : \n")


# n expected to be str or int.
def print_player_turn_to_strike(n):
    print("It is Player {}'s turn to strike.".format(n))
    if FILE_OUTPUT_FLAG:
        f.write("It is Player {}'s turn to strike.\n".format(n))


def print_choose_target_coordinates():
    print("Choose target coordinates :", end=" ")
    if FILE_OUTPUT_FLAG:
        f.write("Choose target coordinates : ")


def print_miss():
    print("Miss.")
    if FILE_OUTPUT_FLAG:
        f.write("Miss.\n")


# n expected to be str or int.
def print_type_done_to_yield(n):
    print("Type done to yield your turn to player {} :".format(n), end=" ")
    if FILE_OUTPUT_FLAG:
        f.write("Type done to yield your turn to player {} : \n".format(n))


def print_tile_already_struck():
    print("That tile has already been struck. Choose another target.")
    if FILE_OUTPUT_FLAG:
        f.write("That tile has already been struck. Choose another target.\n")


def print_hit():
    print("Hit!")
    if FILE_OUTPUT_FLAG:
        f.write("Hit!\n")


# n expected to be str or int.
def print_player_won(n):
    print("Player {} has won!".format(n))
    if FILE_OUTPUT_FLAG:
        f.write("Player {} has won!\n".format(n))


def print_thanks_for_playing():
    print("Thanks for playing.")
    if FILE_OUTPUT_FLAG:
        f.write("Thanks for playing.\n")


# my_list expected to be a 3-dimensional list, formed from two 2-dimensional lists containing the boards of each player.
def print_3d_list(my_list):
    first_d = len(my_list[0])
    for row_ind in range(first_d):
        second_d = len(my_list[0][row_ind])
        print("{:<2}".format(row_ind+1), end=' ')
        for col_ind in range(second_d):
            print(my_list[0][row_ind][col_ind], end=' ')
        print("\t\t\t", end='')
        print("{:<2}".format(row_ind+1), end=' ')
        for col_ind in range(second_d):
            print(my_list[1][row_ind][col_ind], end=' ')
        print()
    print("", end='   ')
    for row_ind in range(first_d):
        print(row_ind + 1, end=' ')
    print("\t\t", end='   ')
    for row_ind in range(first_d):
        print(row_ind + 1, end=' ')
    print("\nPlayer 1\t\t\t\t\t\tPlayer 2")
    print()

    if FILE_OUTPUT_FLAG:
        first_d = len(my_list[0])
        for row_ind in range(first_d):
            second_d = len(my_list[0][row_ind])
            f.write("{:<2} ".format(row_ind + 1))
            for col_ind in range(second_d):
                f.write(my_list[0][row_ind][col_ind] + " ")
            f.write("\t\t\t")
            f.write("{:<2} ".format(row_ind + 1))
            for col_ind in range(second_d):
                f.write(my_list[1][row_ind][col_ind] + " ")
            f.write("\n")
        f.write("   ")
        for row_ind in range(first_d):
            f.write(str(row_ind + 1) + " ")
        f.write("\t\t   ")
        for row_ind in range(first_d):
            f.write(str(row_ind + 1) + " ")
        f.write("\nPlayer 1\t\t\t\t\t\tPlayer 2\n")
        f.write("\n")


def print_rules():
    print("Welcome to Ultimate Battleships")
    print("This is a game for 2 people, to be played on two 10x10 boards.")
    print("There are 5 ships in the game:  Carrier (occupies 5 spaces), Battleship (4), Cruiser (3), Submarine (3), and Destroyer (2).")
    print("First, the ships are placed. Ships can be placed on any unoccupied space on the board. The entire ship must be on board.")
    print("Write the ship's name, followed by an x y coordinate, and the orientation (v for vertical or h for horizontal) to place the ship.")
    print("If a player is placing a ship with horizontal orientation, they need to give the leftmost coordinate.")
    print("If a player is placing a ship with vertical orientation, they need to give the uppermost coordinate.")
    print("Player 1 places first, then Player 2 places. Afterwards, players take turns (starting from Player 1) to strike and sink enemy ships by guessing their location on the board.")
    print("Guesses are again x y coordinates. Do not look at the board when it is the other player's turn.")
    print("The last player to have an unsunk ship wins.")
    print("Have fun!")
    print()

    if FILE_OUTPUT_FLAG:
        f.write("Welcome to Ultimate Battleships\n")
        f.write("This is a game for 2 people, to be played on two 10x10 boards.\n")
        f.write(
            "There are 5 ships in the game:  Carrier (occupies 5 spaces), Battleship (4), Cruiser (3), Submarine (3), and Destroyer (2).\n")
        f.write(
            "First, the ships are placed. Ships can be placed on any unoccupied space on the board. The entire ship must be on board.\n")
        f.write(
            "Write the ship's name, followed by an x y coordinate, and the orientation (v for vertical or h for horizontal) to place the ship.\n")
        f.write("If a player is placing a ship with horizontal orientation, they need to give the leftmost coordinate.\n")
        f.write("If a player is placing a ship with vertical orientation, they need to give the uppermost coordinate.\n")
        f.write(
            "Player 1 places first, then Player 2 places. Afterwards, players take turns (starting from Player 1) to strike and sink enemy ships by guessing their location on the board.\n")
        f.write("Guesses are again x y coordinates. Do not look at the board when it is the other player's turn.\n")
        f.write("The last player to have an unsunk ship wins.\n")
        f.write("Have fun!\n")
        f.write("\n")


# Create the game
board_size = 10
f = open('UltimateBattleships.txt', 'w')
FILE_OUTPUT_FLAG = True  # You can change this to True to also output to a file so that you can check your outputs with diff.

print_rules()

# Remember to use list comprehensions at all possible times.
# If we see you populate a list that could be done with list comprehensions using for loops, append/extend/insert etc. you will lose points.

# Make sure to put comments in your code explaining your approach and the execution.

# We defined all the functions above for your use so that you can focus only on your code and not the formatting.
# You need to call them in your code to use them of course.

# If you have questions related to this homework, direct them to utku.bozdogan@boun.edu.tr please.

# Do not wait until the last day or two to start doing this homework, it requires serious effort.

try:  # The entire code is in this try block, if there ever is an error during execution, we can safely close the file.
    # DO_NOT_EDIT_ANYTHING_ABOVE_THIS_LINE

    ship_len = {'Carrier': 5, 'Battleship': 4, 'Cruiser': 3, 'Submarine': 3, 'Destroyer': 2}
    # Create game table as 4d list including both player's table acc to what they see
    game_table = [[[['-' for column in range(10)] for row in range(10)] for player_sight in range(2)] for player in range(2)]
    scores = [0 for player in range(2)]

    # Loop two times for two player to make placements
    for placer_num in range(2):
        who_is_placing = placer_num + 1  # Player1 or Player2
        player_table = game_table[placer_num][placer_num]  # Reference to the table to be used
        remaining_ships = [ship for ship in ship_len.keys()]
        is_placing = True

        while is_placing:
            print_3d_list(game_table[placer_num])
            print_ships_to_be_placed()
            [print_single_element(ship) for ship in remaining_ships]
            print_empty_line()
            print_player_turn_to_place(who_is_placing)
            print_to_place_ships()

            move = input().strip().split()  # Raw player move

            # Ordinary checks for input validity
            if len(move) < 4 or not move[1].isdecimal() or not move[2].isdecimal():
                print_incorrect_input_format()

            elif not 0 < int(move[1]) <= 10 or not 0 < int(move[2]) <= 10:
                print_incorrect_coordinates()

            elif move[0].lower() not in [ship.lower() for ship in ship_len.keys()]:
                print_incorrect_ship_name()

            elif move[3].lower() not in ('v', 'h'):
                print_incorrect_orientation()

            # Further checks for table's availability
            else:
                # Format the valid input for table
                move[0] = move[0][0].upper() + move[0][1:].lower()
                move[1], move[2] = int(move[1])-1, int(move[2])-1
                move[3] = move[3].lower()

                if move[0].lower() not in [ship.lower() for ship in remaining_ships]:
                    print_ship_is_already_placed(move[0][0].upper()+move[0][1:].lower())
                    continue

                elif (move[3] == 'v' and move[2] + ship_len[move[0]] - 1 > 9) or (move[3] == 'h' and move[1] + ship_len[move[0]] - 1 > 9):
                    print_ship_cannot_be_placed_outside(move[0])
                    continue

                # Comprehension returns empty list (False) if there is no intersection
                elif (move[3] == 'v' and [True for row in player_table[move[2]:move[2] + ship_len[move[0]]] if row[move[1]] == '#']) \
                        or (move[3] == 'h' and [True for column in player_table[move[2]][move[1]:move[1]+ship_len[move[0]]] if column == '#']):
                    print_ship_cannot_be_placed_occupied(move[0])
                    continue

                # Place the ship
                if move[3] == 'v':
                    for row in range(move[2], move[2]+ship_len[move[0]]):
                        player_table[row][move[1]] = '#'

                else:
                    for column in range(move[1], move[1]+ship_len[move[0]]):
                        player_table[move[2]][column] = '#'

                remaining_ships.remove(move[0])

                # End placement for current player
                if len(remaining_ships) == 0:
                    print_3d_list(game_table[placer_num])
                    print_confirm_placement()
                    confirm = input()

                    while True:
                        # Yield to the next player or pass to battle phase
                        if confirm.lower() == 'y':
                            is_placing = False
                            break

                        # Reset the table used and restart the placement for current player
                        elif confirm.lower() == 'n':
                            game_table[placer_num] = [[['-' for column in range(10)] for row in range(10)] for player_sight in range(2)]
                            player_table = game_table[placer_num][placer_num]
                            remaining_ships = [ship for ship in ship_len.keys()]
                            break

                        # Ask for confirmation until valid input entered
                        print_confirm_placement()
                        confirm = input()

    # Initialize variables for battle phase
    who_is_playing = 1  # Player 1
    winning_score = sum(ship_len.values())  # All ship tiles must be hit to get winning score

    # Current player continues to play when gets a score
    # Only check whether current player reached enough score
    while scores[who_is_playing-1] < winning_score:
        # References to the tables to be used
        marking_table = game_table[who_is_playing-1][2-who_is_playing]  # Only for marking
        player_table = game_table[who_is_playing-1]  # Only for printing
        enemy_table = game_table[2-who_is_playing][2-who_is_playing]  # For marking and checking

        print_3d_list(player_table)
        print_player_turn_to_strike(who_is_playing)
        print_choose_target_coordinates()
        strike = input().strip().split()

        # Checks for input validity and table availability
        if len(strike) < 2 or not strike[0].isdecimal() or not strike[1].isdecimal():
            print_incorrect_input_format()

        elif not 0 < int(strike[0]) <= 10 or not 0 < int(strike[1]) <= 10:
            print_incorrect_coordinates()

        elif marking_table[int(strike[1])-1][int(strike[0])-1] != '-':
            print_tile_already_struck()

        # Mark the strike on table
        else:
            strike = [int(num)-1 for num in strike]  # Format strike for marking

            if enemy_table[strike[1]][strike[0]] == '#':
                marking_table[strike[1]][strike[0]] = '!'
                enemy_table[strike[1]][strike[0]] = '!'
                scores[who_is_playing-1] += 1  # Player gains one score per hit
                print_hit()

            else:
                marking_table[strike[1]][strike[0]] = 'O'
                enemy_table[strike[1]][strike[0]] = 'O'
                print_miss()

                # Ask for confirmation until 'done' typed
                while True:
                    print_type_done_to_yield(3-who_is_playing)
                    yld = input().strip()
                    if yld.lower() == 'done':
                        who_is_playing = 3-who_is_playing  # Yield to next player
                        break

    # Game ending prints
    print_3d_list(game_table[who_is_playing-1])
    print_player_won(who_is_playing)
    print_thanks_for_playing()


    # DO_NOT_EDIT_ANYTHING_BELOW_THIS_LINE
except:
    f.close()

