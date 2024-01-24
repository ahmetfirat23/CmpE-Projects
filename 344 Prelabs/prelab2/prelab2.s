.globl _start

.equ M, 7

.data
D: .word 5,7,0, 8,24,0, 11,44,0, 11,18,0, 36,2,0, 63,27,0, 19,24,0

.text
gcd:

# Insert gcd function below.
    rem t5, a0, a1              # a0 and a1 are provided arguments e.g. 5, 7
    mv a0, a1                   # This algorithm is called euclidian algorithm
    mv a1, t5                   # At the end when a1 is 0, the gcd of a0 and a1 is stored at a0
    bnez a1, gcd        


# Insert gcd function above.

	ret

coprime:

# Insert coprime function below.
    addi sp, sp, -4             # Allocate space in stack pointer for return address
    sw ra, 4(sp)                # Record ra in stack pointer, which will return to the line below where this function is called in _start

	li t2, 12                   # 3*4 byte for triplets in array

	mul t2, t2, a3	            # Make the jump for 3*4 byte (This will give the starting index for each triplet. e.g. 12*0 = 0, 12*1 = 12) 

	add t2, a2, t2              # Add the index to array address to find out the exact locations of the triplets
	lw a0, (t2)                 # Now a0 points to the first element (e.g. a0-> 5 in 5, 7, 0)
	lw a1, 4(t2)                # Now a1 points to the second element (e.g. a1-> 7 in 5, 7, 0)

	jal gcd                     # Call gcd to find out the gcd of (a0) and (a1)

    lw ra, 4(sp)                # Restore the recorded return address
    addi sp, sp, 4              # Reallocate the space

	li t4, 1                    # Set temporary variable to 1 for the following line
	sgt t5, a0, t4              # This instruction checks if a0, the register holding the result of gcd, is greater than 1, if so, sets t5 to 1, else t5 is 0
	add t5, t5, 1               # If t5 is 1, which means gcd > 1, so we set t5 to 1 + 1 = 2, else 0 + 1 = 1
	sw t5, 8(t2)                # Access the third element of the triplet and put the value
	
    add a3, a3, 1               # Increment the counter for M
	blt a3, a4, coprime         # Call itself recursively until M loops are done.


# Insert coprime function above.

	ret

_start:

# Insert _start function below.
    addi sp, sp, -4             # Allocate space in stack pointer for return address
    sw ra, 0(sp)                # Put current record address (ra) to this allocated space

	la t0, D                    # Load array to t0 
	li t1, 0                    # counter for M (will be incremented until M is reached)
	li t6, M                    # Load M

    mv a2, t0                   # Load function paramaters
    mv a3, t1                   # Load function paramaters
    mv a4, t6                   # Load function paramaters

	jal coprime                 # Call coprime and record (ra)

    lw ra, 0(sp)                # Restore the recorded return address
    addi sp, sp, 4              # Reallocate the space


# Insert _start function above.

	ret

.end