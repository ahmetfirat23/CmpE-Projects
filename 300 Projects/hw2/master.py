"""
Student Names: Ahmet Fırat Gamsız, Ali Tarık Şahin	
Student Numbers: 
Group No: 40
"""

from mpi4py import MPI
import sys
import argparse

# standard argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('input')
parser.add_argument('output')
args = parser.parse_args()
input_file = args.input
output_file = args.output

with open(input_file, 'r') as f:
    lines = f.readlines()
# empty output file
with open(output_file, 'w') as f:
    pass

# save variables from input file
n_machines = int(lines[0])
n_cycles = int(lines[1])
enhance_wear, reverse_wear, chop_wear, trim_wear, split_wear = [int(x) for x in lines[2].split()]
threshold = int(lines[3])

# key is the machine id, value consist of a list parent id and children ids.
# first id is the parent and the rest are children
# terminal machine has no children and its parent is the control room (master)
children_dict = {1: [0]}
# initial operation of each machine
# terminal machine has no initial operation
initial_op_dict = {1: "-"}
for i in range(n_machines-1):
    machine_id, parent_id, initial_op = int(lines[4+i].split()[0]), int(lines[4+i].split()[1]), lines[4+i].split()[2]
    # set initial operation of the machine
    initial_op_dict.update({machine_id: initial_op})
    # set parent of the machine
    if machine_id not in children_dict.keys():
        children_dict.update({machine_id: [parent_id]})
    else:
        children_dict[machine_id][0] = parent_id

    # in case if child comes before parent in the input file
    if parent_id not in children_dict.keys():
        # -1 is a placeholder for parent id
        children_dict.update({parent_id: [-1]})
    # if parent already exist, append the new child
    children_dict[parent_id].append(machine_id)

# detect leafs (terminal machines)
leafs = []
for key, value in children_dict.items():
    if len(value) == 1:
        leafs.append(key)
# sort leafs so each leaf gets the correct input (string)
leafs.sort()

# last lines of the input file are the initial strings of the leafs
# those are given in the order of leaf ids
string_dict = {}
for j in range(len(leafs)):
    string = lines[4+n_machines-1+j].strip()
    string_dict.update({leafs[j]: string})

# initialize communicator and spawn processes (machines/slaves)
# n+1 since we have n machines and 1 control room (master) 
comm = MPI.COMM_WORLD.Spawn(sys.executable,
                           args=['slave.py'],
                            maxprocs=n_machines+1)

# get rank and size of the communicator
rank = comm.Get_rank()
size = comm.Get_size()

# send initial data to each machine
for i in range(1, n_machines+1):
    machine_id = i
    comm.send(machine_id, dest=i, tag=0)
    comm.send(n_cycles, dest=i, tag=1)
    comm.send(threshold, dest=i, tag=2)
    comm.send(enhance_wear, dest=i, tag=3)
    comm.send(reverse_wear, dest=i, tag=4)
    comm.send(chop_wear, dest=i, tag=5)
    comm.send(trim_wear, dest=i, tag=6)
    comm.send(split_wear, dest=i, tag=7)
    adjacency = children_dict[machine_id]
    parent_id = adjacency[0]
    children_ids = adjacency[1:]
    children_ids.sort()
    comm.send(children_ids, dest=i, tag=8)
    comm.send(parent_id, dest=i, tag=9)
    comm.send(initial_op_dict[machine_id], dest=i, tag=10)
    if (i in leafs):
        comm.send(string_dict[i], dest=i, tag=11)
    else:
        # send empty string to non-leaf machines
        comm.send("", dest=i, tag=11)

log_list = []
f = open(output_file, 'a')
while True:
    # if n_cycles is zero, then all machines are done
    if n_cycles == 0:
        break
    # if there is a log from any machine, receive it
    # logs are sent with tag 0
    if comm.iprobe(source=MPI.ANY_SOURCE, tag=0):
        # continue receiving until there is no log
        while comm.iprobe(source=MPI.ANY_SOURCE, tag=0):
            # save logs to print at the end
            logs = comm.recv(source=MPI.ANY_SOURCE, tag=0)
            log_list.append(logs)

    # if there is a product from terminal machine, receive it
    # products are sent with tag 1
    if comm.iprobe(source=1, tag=1):
            product = comm.recv(source=1, tag=1)
            # write product to output file
            f.write(product + "\n")
            f.flush()
            n_cycles -= 1 

# print logs
for log in log_list:
    f.write(log + "\n")
f.close()

# finalize MPI
MPI.Finalize()
