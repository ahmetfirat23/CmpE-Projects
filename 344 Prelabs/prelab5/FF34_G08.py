import cache_module
import numpy

# Prepare an RGB image containing 3 colour channels.
ROW = 1024
COL = 2048
Channel = 3
image = numpy.random.randint(0, 256, size=(ROW, COL, Channel), 
dtype=numpy.int64)

# Prepare a mask for the convolution operation.
mask_size = 3
up = 10
down = -10
mask = numpy.random.randint(down, up + 1, size=(mask_size, mask_size), 
dtype=numpy.int64)

# Prepare an empty result image. You will fill this empty array with your code.
result = numpy.zeros([ROW, COL, Channel], dtype=numpy.int64)

# Configuration for the cache simulator module.
l3 = ["L3", 16384, 16, 64, "LRU"]
l2 = ["L2", 4096, 8, 64, "LRU"]
l1 = ["L1", 1024, 4, 64, "LRU"]
m = 256 * 1024 * 1024
cm = cache_module.cache_module(l1, l2, l3, m)

###### WRITE YOUR CODE BELOW. ######

# 1. Load the image into the memory
for i in range(Channel):
    for j in range(ROW):
        for k in range(COL):
            val = image[j, k, i]
            # j + 2 -> padding of first column
            # COL -> padding of first row
            # k -> iterates in a row
            # j * COL -> gets to the next row
            # i * ((COL+1) * (ROW+1)) -> gets to the next channel
            # 8 -> since rgb values are 8 bits (2^8=256), we can just use 1 byte for each value and leave the remaining 7 bytes empty
            idx = 8*((j+2) + COL + k + j * COL  + i * ((COL+1) * (ROW+1)))
            cm.write(idx, val)

# 2. Traverse the image array and apply the mask. Write the results into the memory through the write function. Do not fill the result array in this step.
iter_var = 0
for i in range(Channel):
    for j in range(ROW):
        for k in range(COL):
            res = 0
            # iterate over the mask
            for x in range(mask_size):
                for y in range(mask_size):
                    idx = 8*((j+2) + COL + (k+y-1) + (j+x-1) * COL  + i * ((COL+1) * (ROW+1)))
                    val = cm.read(idx)
                    res += val * mask[x][y]
            # divide the resulting 64 bit value into 8 bit values and write them into the memory
            bytes_arr = numpy.array([res], dtype=numpy.int64).tobytes()
            uint8_arr = numpy.frombuffer(bytes_arr, dtype=numpy.uint8)
            for m in range(8):
                cm.write(iter_var+m, uint8_arr[m])
            iter_var += 8

# 3. Load the result image from memory through the read function.
iter_var = 0
for i in range(Channel):
    for j in range(ROW):
        for k in range(COL):
            # convert 8 bit values into 64 bit values and write them into the result array        
            uint8_arr = numpy.zeros(8, dtype=numpy.uint8)
            for m in range(8):
                uint8_arr[m] = cm.read(iter_var + m)
            iter_var += 8
            result[j, k, i] = numpy.frombuffer(uint8_arr.tobytes(), dtype=numpy.int64)[0]

###### WRITE YOUR CODE ABOVE. ######

cm.finish()
