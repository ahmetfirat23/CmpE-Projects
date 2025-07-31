from IPython.display import display

class Inverter:
    def get_rank(self):
        res = 0
        zero_indices = []
        for i in range(len(self.matrix)):
            if self.matrix[i][i] == 0:
                zero_indices.append(i)
                continue
            res += 1
        return res, zero_indices

    def gaussian_elimination(self):
        n = len(self.matrix)
        for i in range(n):
            flag = True
            if self.matrix[i][i] == 0:
                for j in range(i, n):
                    if self.matrix[j][i] != 0:
                        tmp = self.matrix[j]
                        self.matrix[j] = self.matrix[i]
                        self.matrix[i] = tmp
                        flag = False

                        tmp = self.identity_matrix[i]
                        self.identity_matrix[i] = self.identity_matrix[j]
                        self.identity_matrix[j] = tmp

                        break
                if flag:  # if all the columns are 0 continue with the other column
                    continue

            factor = self.matrix[i][i]  # we made sure that factor is not zero
            for j in range(n):  # devide the row with factor so that pivot will be 1
                self.matrix[i][j] /= factor
                if j != n:
                    self.identity_matrix[i][j] /= factor

            for j in range(n):
                if i == j:
                    continue
                factor = self.matrix[j][i]
                for k in range(n):
                    self.matrix[j][k] -= factor * self.matrix[i][k]  # subtract rows
                    if -1e-10 <self.matrix[j][k] < 1e-10:
                        self.matrix[j][k] = 0
                    if k != n:
                        self.identity_matrix[j][k] -= factor * self.identity_matrix[i][k]

    # This function creates identity matrix in order to be able to get inverse of matrix A after applying
    # the same row operations with the matrix A
    def create_identity(self):
        identity = []
        for i in range(self.n):
            tmp = []
            for j in range(self.n):
                if i == j:
                    tmp.append(1)
                else:
                    tmp.append(0)
            identity.append(tmp)
        return identity

    def invert(self, matrix):
        self.matrix = matrix
        self.n = len(self.matrix)
        self.identity_matrix = self.create_identity()

        self.gaussian_elimination()

        rank, zero_indices = self.get_rank()
        if rank != self.n:
            for i in zero_indices:
                if self.matrix[i][self.n] != 0:
                    print("Inconsistent problem")
                    return
 
        return self.identity_matrix


inverter = Inverter()

def create_identity(n):
    identity = []
    for i in range(n):
        tmp = []
        for j in range(n):
            if i == j:
                tmp.append(1.0)
            else:
                tmp.append(0.0)
        identity.append(tmp)
    return identity

def get_input(file_path):
    with open(file_path, "r") as f:
        m, n = f.readline().split()
        c = [float(i) for i in f.readline().strip().split()]
        A,b = [],[]
        for row in f.readlines():
            line = [float(i) for i in row.strip().split()]
            A.append(line[:-1])
            b.append(line[-1])
                
        return int(m), int(n), c, A,b
    
def arg_min(arr):
    min_index = 0
    for i in range(len(arr)):
        if arr[i] < arr[min_index]:
            min_index = i
    return min_index

def min_ratio(ratios):
    min_index = 0
    x_count = 0
    for i in range(len(ratios)):
        if type(ratios[i]) == str:
            x_count += 1
            if min_index == i:
                min_index += 1
            
        elif ratios[i] < ratios[min_index]:
            min_index = i
    if x_count == len(ratios):
        raise Exception("Unbounded problem")
    return min_index


if __name__ == "__main__":
    m,n,c,A,b =get_input("test.txt")

    c = [i * -1 for i in c]
    c.extend([0.0 for i in range(m)])

    B = create_identity(m)
    B_inverse = inverter.invert(B)

    for i in range(m):
        A[i].extend(B_inverse[i])

    nbv = {i:i for i in range(n)}
    bv = {i:n+i for i in range(m)}

    cB = [0 for i in range(m)]

    while True:

        price_out_factor = [0 for i in range(m)]
        for i in range(m):
            for j in range(m):
                price_out_factor[i] += cB[j] * B_inverse[i][j]
        display(price_out_factor)
        
        c_star = [0 for i in range(n)]
        for index, nonbasic in nbv.items():
            c_star[index] = c[nonbasic]
            for j in range(m):
                c_star[index] -= price_out_factor[j] * A[j][nonbasic]
        index = arg_min(c_star)
        x_i = nbv[index]    
        display(c_star)

        A_star = [0 for i in range(m)]
        for i in range(m):
            for j in range(m):
                A_star[i] += B_inverse[j][i] * A[j][x_i]    
        display(A_star)

        b_star = [0 for i in range(m)]
        for i in range(m):
            for j in range(m):
                b_star[i] += B_inverse[j][i] * b[j]    
        display(b_star)

        row_ratios = {i:"x" for i in range(m)}
        for i in range(m):
            if A_star[i] > 0:
                row_ratios[i] = b_star[i] / A_star[i]

        row = min_ratio(row_ratios)    
        display(row_ratios)
        display(row)

        all_nonnegative = True
        for coeff in c_star:
            if coeff < 0:
                all_nonnegative = False
                break
        if (all_nonnegative):
            print("Solution found")
            print("Optimal solution:")
            z = 0
            for basic in bv:
                print("x", bv[basic]+1, end=" ")
                print("= ", b_star[basic], end="\n")
            for i in range(m):
                z += price_out_factor[i] * b[i]
            
            print("z = ",z)
            break

        tmp = bv[row]
        bv[row] = x_i
        nbv[index] = tmp   

        B[row] = [A[j][x_i] for j in range(m)]
        display(B)
        B_copy = [[float(x) for x in row] for row in B]
        B_inverse = inverter.invert(B_copy)
        display(B)
        display(B_copy)
        display(B_inverse)
        cB[row] = c[x_i]
        display(cB)