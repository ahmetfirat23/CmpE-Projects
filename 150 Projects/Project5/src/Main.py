
# return img, nested list
def read_ppm_file(f):
    fp = open(f)
    fp.readline()  # reads P3 (assume it is P3 file)
    lst = fp.read().split()
    n = 0
    n_cols = int(lst[n])
    n += 1
    n_rows = int(lst[n])
    n += 1
    max_color_value = int(lst[n])
    n += 1
    img = []
    for r in range(n_rows):
        img_row = []
        for c in range(n_cols):
            pixel_col = []
            for i in range(3):
                pixel_col.append(int(lst[n]))
                n += 1
            img_row.append(pixel_col)
        img.append(img_row)
    fp.close()
    return img, max_color_value


# Works
def img_printer(img):
    row = len(img)
    col = len(img[0])
    cha = len(img[0][0])
    for i in range(row):
        for j in range(col):
            for k in range(cha):
                print(img[i][j][k], end=" ")
            print("\t|", end=" ")
        print()


filename = input()
operation = int(input())


# DO_NOT_EDIT_ANYTHING_ABOVE_THIS_LINE

img, max_color_value = read_ppm_file(filename)
max_color_value = int(max_color_value)


def min_max_normalization(img, old_max):
    new_min = int(input())
    new_max = int(input())
    old_min = 0

    # Loop through each pixel's channel and apply the given formula
    for i in range(len(img)):
        for j in range(len(img[i])):
            for k in range(3):
                img[i][j][k] = round((img[i][j][k]-old_min)/(old_max-old_min)*(new_max-new_min)+new_min, 4)
    img_printer(img)


def z_score_normalization(img):
    # Loop for means
    channel_means = [0, 0, 0]
    for channel in range(3):
        total = 0
        n = 0
        for i in range(len(img)):
            for j in range(len(img[i])):
                total += img[i][j][channel]
                n += 1
        channel_means[channel] = total/n

    # Loop for deviations
    channel_deviations = [0, 0, 0]
    for channel in range(3):
        total = 0
        n = 0
        for i in range(len(img)):
            for j in range(len(img[i])):
                total += (img[i][j][channel]-channel_means[channel])**2
                n += 1
        channel_deviations[channel] = (total/n)**(1/2) + 0.000001

    # Loop through each pixel's channel and apply the given formula
    for i in range(len(img)):
        for j in range(len(img[i])):
            for k in range(3):
                img[i][j][k] = round((img[i][j][k]-channel_means[k])/channel_deviations[k],4)
    img_printer(img)


def black_white(img):
    # Loop through each pixel and assign the channels' average
    for i in range(len(img)):
        for j in range(len(img[i])):
            total = 0
            for k in range(3):
                total += img[i][j][k]
            img[i][j] = [total//3]*3
    img_printer(img)


# Utility function that returns clamped value between given values
def clamp(num):
    if num < 0:
        return 0
    elif num > max_color_value:
        return max_color_value
    else:
        return num


# Utility function that converts filter to 2d-list
def read_filter(f):
    filter = []
    for line in f:
        filter += [line.split()]
    return filter, len(filter)


def convolution(img):
    f = input()
    with open(f) as filter:
            filter, n = read_filter(filter)
    stride = int(input())
    padding = n//2
    new_img = []

    # Loop through appropriate pixels, apply the filter and append to new list
    index = 0
    for i in range(padding, len(img)-padding, stride):
        new_img.append([])
        for j in range(padding, len(img[i])-padding, stride):
            rgb = []
            for k in range(3):
                val = 0
                for x in range(-padding, padding+1):
                    for y in range(-padding, padding+1):
                        val += img[i+x][j+y][k]*float(filter[padding+x][padding+y])
                val = clamp(int(val))
                rgb.append(val)
            new_img[index].append(rgb)
        index += 1
    img_printer(new_img)


def padded_convolution(img):
    f = input()
    with open(f) as filter:
            filter, n = read_filter(filter)
    stride = int(input())
    padding = n//2
    new_img = []

    # Add vertical padding
    for i in range(len(img)):
        for count in range(padding):
            img[i].append([0, 0, 0])
            img[i].insert(0, [0, 0, 0])

    # Add horizontal padding
    for i in range(padding):
        img.append([[0, 0, 0]]*len(img[0]))
        img.insert(0, [[0, 0, 0]] * len(img[0]))

    # Apply convolution
    index = 0
    for i in range(padding, len(img)-padding, stride):
        new_img.append([])
        for j in range(padding, len(img[i])-padding, stride):
            rgb = []
            for k in range(3):
                val = 0
                for x in range(-padding, padding+1):
                    for y in range(-padding, padding+1):
                        val += img[i+x][j+y][k]*float(filter[padding+x][padding+y])
                val = clamp(int(val))
                rgb.append(val)
            new_img[index].append(rgb)
        index += 1

    img_printer(new_img)


# Scan the image vertically in given direction
# Set next pixel equal to current pixel if pixels differ by less than the range
def vertical_iter(img, rng, x=0, y=0, v=1):
    if (v == 1 and x+1 < len(img)) or (v == -1 and x >= 1):
        change = True
        for i in range(3):
            if abs(img[x][y][i]-img[x+v][y][i]) >= rng:
                change = False
        if change:
            img[x+v][y] = img[x][y]
        return vertical_iter(img, rng, x+v, y, v)
    else:
        return x, y, v*-1


# Compare right pixel
def horizontal_iter(img, rng, x=0, y=0):
    change = True
    for i in range(3):
        if abs(img[x][y][i]-img[x][y+1][i]) >= rng:
            change = False
    if change:
        img[x][y+1] = img[x][y]
    return x, y+1


def quantization(img, rng):
    direction = 1
    x = 0
    y = 0
    # Follow the call order through all pixels
    while y < len(img):
        # Go up or down
        x, y, direction = vertical_iter(img, rng, x, y, direction)
        if y + 1 < len(img):
            # Go right pixel
            x, y = horizontal_iter(img, rng, x, y)
        else:
            # Break the loop
            y += 1
    img_printer(img)


# Scan the image vertically in given direction
# Set next pixel's channel equal to current one if they differ by less than the range
def d3_vertical_iter(img, rng, z, x=0, y=0, v=1):
    if (v == 1 and x+1 < len(img)) or (v == -1 and x >= 1):
        if abs(img[x][y][z]-img[x+v][y][z]) < rng:
            img[x+v][y][z] = img[x][y][z]
        return d3_vertical_iter(img, rng, z, x+v, y, v)
    else:
        return x, y, z, v*-1


# Compare adjacent pixel's channel in given direction
def d3_horizontal_iter(img, rng, z, x=0, y=0, h=1):
    if abs(img[x][y][z]-img[x][y+h][z]) < rng:
        img[x][y+h][z] = img[x][y][z]
    return x, y+h, z


# Compare next channel of same pixel
def d3_diagonal_iter(img, rng, z, x, y):
    if abs(img[x][y][z]-img[x][y][z+1]) < rng:
        img[x][y][z+1] = img[x][y][z]
    return x, y, z+1


def d3_quantization(img, rng):
    direction = 1
    h_direction = 1
    x = 0
    y = 0
    z = 0
    # Follow the call order through all channels
    while y < len(img) and z < 3:
        # Go up or down
        x, y, z, direction = d3_vertical_iter(img, rng, z, x, y, direction)
        # Go right or left
        x, y, z = d3_horizontal_iter(img, rng, z, x, y, h_direction)

        if(y + 1 == len(img) or y == 0) and z < 2:
            # At the right and left end points
            # Scan horizontally and switch to next channel
            # Reverse the horizontal direction
            x, y, z, direction = d3_vertical_iter(img, rng, z, x, y, direction)
            h_direction *= -1
            x, y, z = d3_diagonal_iter(img, rng, z, x, y)

        elif y + 1 == len(img) and z == 2:
            # At the right end with last channel
            # Scan horizontally and break the loop
            x, y, z, direction = d3_vertical_iter(img, rng, z, x, y, direction)
            z += 1
            y += 1
    img_printer(img)


# Run the given operation
if operation == 1:
    min_max_normalization(img, max_color_value)
elif operation == 2:
    z_score_normalization(img)
elif operation == 3:
    black_white(img)
elif operation == 4:
    convolution(img)
elif operation == 5:
    padded_convolution(img)
elif operation == 6:
    rng = int(input())
    quantization(img, rng)
elif operation == 7:
    rng = int(input())
    d3_quantization(img, rng)


# DO_NOT_EDIT_ANYTHING_BELOW_THIS_LINE

