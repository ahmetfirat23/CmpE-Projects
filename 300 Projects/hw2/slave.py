"""
Student Names: Ahmet Fırat Gamsız, Ali Tarık Şahin	
Student Numbers: 
Group No: 40
"""

from mpi4py import MPI
import math

# get parent communicator
comm = MPI.Comm.Get_parent()
# get intracommunicator to send messages to other machines
sender_comm = MPI.COMM_WORLD

# get rank and size
size = comm.Get_size()
rank = comm.Get_rank()

# receive initial data from master
machine_id = comm.recv(source=0, tag=0)
n_cycles = comm.recv(source=0, tag=1)
threshold = comm.recv(source=0, tag=2)
enhance_wear = comm.recv(source=0, tag=3)
reverse_wear = comm.recv(source=0, tag=4)
chop_wear = comm.recv(source=0, tag=5)
trim_wear = comm.recv(source=0, tag=6)
split_wear = comm.recv(source=0, tag=7)   
children_ids = comm.recv(source=0, tag=8)
parent_id = comm.recv(source=0, tag=9)
operation = comm.recv(source=0, tag=10)
machine_input = comm.recv(source=0, tag=11)

total_cycles = n_cycles
accumulated_wear = 0

# basic machine functions
def add_strings(strings):
    result = ""
    for s in strings:
        result += s
    return result

 
def enhance(string):
    global accumulated_wear
    accumulated_wear += enhance_wear
    string = string[0] + string + string[-1]
    return string


def reverse(string):
    global accumulated_wear
    accumulated_wear += reverse_wear
    string = string[::-1]
    return string


def chop(string):
    global accumulated_wear
    accumulated_wear += chop_wear
    if len(string) == 1:
        return string
    string = string[:-1]
    return string


def trim(string):
    global accumulated_wear
    accumulated_wear += trim_wear
    if len(string) <= 2:
        return string
    string = string[1:-1]
    return string


def split(string):
    global accumulated_wear
    accumulated_wear += split_wear
    if len(string) == 1:
        return string
    string = string[:math.ceil(len(string)/2)]
    return string


# check if the machine needs to be repaired
def check_wear(operation, n_cycles):
    global accumulated_wear
    # set the wf according to last operation
    if operation == "enhance":
        wf = enhance_wear
    elif operation == "reverse":
        wf = reverse_wear
    elif operation == "chop":
        wf = chop_wear
    elif operation == "trim":
        wf = trim_wear
    elif operation == "split":
        wf = split_wear
    if accumulated_wear >= threshold:
        # calculate C and send to master non-blocking
        C = (accumulated_wear - threshold + 1) * wf
        comm.send(f"{machine_id}-{C}-{total_cycles - n_cycles + 1}", dest=0, tag=0)
        # reset accumulated wear
        accumulated_wear = 0

# string to function dictionary for machine functions
func_dict = {"enhance": enhance, "reverse": reverse, "chop": chop, "trim": trim, "split": split}


# terminal machine
if machine_id == 1:
    while n_cycles > 0:
        # if there is no children, just add the string
        if len(children_ids) == 0:
            product = add_strings([machine_input])

        else:
            children_products = []
            # this loops children in the order of their ids
            for child_id in children_ids:
                # receive product from children with blocking
                children_products.append(sender_comm.recv(source=child_id, tag=child_id))
            # add the strings of children
            product = add_strings(children_products)
        # send result to control room
        comm.send(product, dest=0, tag=1)
        # this machine never needs to be repaired
        n_cycles -= 1

# machine with even id
elif machine_id % 2 == 0:
    # list of functions for even machines
    func_altering = ['enhance', "split", "chop"]
    # get index of initial operation
    idx = func_altering.index(operation)

    while n_cycles > 0:
        # if there is no children, just add the string
        if len(children_ids) == 0:
            product = add_strings([machine_input])

        else:
            children_products = []
            # this loops children in the order of their ids
            for child_id in children_ids:
                # receive product from children with blocking
                children_products.append(sender_comm.recv(source=child_id, tag=child_id))
            # add the strings of children
            product = add_strings(children_products)
        # get operation from dictionary and apply
        product = func_dict[operation](product)
        # send result to parent with blocking
        sender_comm.send(product, dest=parent_id, tag=machine_id)
        # check if the machine needs to be repaired
        check_wear(operation, n_cycles)
        # decrease number of cycles
        n_cycles -= 1
        # get next operation
        idx = (idx + 1) % 3
        operation = func_altering[idx]

# machine with odd id 
# (this is the same as even machine so no extra comments)    
else:
    func_altering = ['reverse', "trim"]
    idx = func_altering.index(operation)

    while n_cycles > 0:
        if len(children_ids) == 0:
            product = add_strings(machine_input)
        else:
            children_products = []
            for child_id in children_ids:
                children_products.append(sender_comm.recv(source=child_id, tag=child_id))
            product = add_strings(children_products)

        product = func_dict[operation](product)
        sender_comm.send(product, dest=parent_id, tag=machine_id)

        check_wear(operation, n_cycles)

        n_cycles -= 1
        
        idx = (idx + 1) % 2
        operation = func_altering[idx]

# disconnect from parent communicator
comm.Disconnect()
# disconnect from sender communicator
sender_comm.Disconnect()
# finalize MPI
MPI.Finalize()
