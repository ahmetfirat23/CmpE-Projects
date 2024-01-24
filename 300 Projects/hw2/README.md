# How to run the code

1. Install the required libraries.
2. Put the input file at the same directory as the code.
3. Run the code with the following command:
```
mpiexec -n 1 python3 master.py <input_file_name>.txt <output_file_name>.txt
```
Note 1: Depending on your python installation, you may need to use "python" instead of "python3" in the above command.

Note 2: You may need to use --oversubscribe option with mpiexec if you receive "not enough slots available" error.

Note 3: Sometimes program terminates with segmentation fault. We receive this error on macOS with M2 processor. However, this error occurs only after program finishes its execution and it does not affect the results. We didn't see any problem caused by this error so it is safe to ignore it.


# Required libraries
You need to install mpi4py library. You can install it with the following command:
```
pip3 install mpi4py
```
Note 1: Depending on your python installation, you may need to use "pip" instead of "pip3" in the above commands.

Note 2: Please not that to install mpi4py, you need to have MPI installed on your system. Please refer to mpi4py documentation for more details: [mpi4py documentation](https://mpi4py.readthedocs.io/en/stable/install.html).


