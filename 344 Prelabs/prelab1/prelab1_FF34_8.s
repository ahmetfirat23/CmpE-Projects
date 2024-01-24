.section .data

arr:    .byte 8, 3, 12, 15, 26, 0

.section .text

_start:
    la t0, arr
    li t1, 0 #array size
    add t2, x0, t0
COUNT:
    lb t3, 0(t2)
    beq t3, x0, OUTER_LOOP 
    add t2, t2, 1
    add t1, t1, 1
    j COUNT

OUTER_LOOP:
    beq t1, x0, DONE
    add t2, x0, t0 # i = 0
    li t5, 1
    j INNER_LOOP

INNER_LOOP:

    beq t5, t1, MOVE_LOOP

    lb t3, 0(t2)
    lb t4, 1(t2)

    bgt t3, t4, SWAP
    addi t2, t2, 1
    addi t5, t5, 1
    j INNER_LOOP

SWAP:
    sb t3, 1(t2)
    sb t4, 0(t2)
    addi t2, t2, 1
    addi t5, t5, 1
    j INNER_LOOP

MOVE_LOOP:
    addi t1, t1, -1
    j OUTER_LOOP

DONE:
    lb a0, (t0)
    lb a1, 1(t0)
    lb a2, 2(t0)
    lb a3, 3(t0)
    lb a4, 4(t0)
    lb a5, 5(t0)
    j .