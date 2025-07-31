class MinimumEditDistance():
    def __init__(self, insert=None, delete=None, substitute=None):
        if insert != None:
            self.insert = insert
        if delete != None:
            self.delete = delete
        if substitute != None:
            self.substitute = substitute

    def insert(self, source, target):
        return 1
    
    def delete(self, source, target):
        return 1
    
    def subsitute(self, source, target):
        if source[-1] == target[-1]:
            return 0
        else:
            return 1
        
    def recursive_distance(self, source, target):
        if len(source) == 0:
            return len(target)
        elif len(target) == 0:
            return len(source)
        else:
            return min(self.distance(source[:-1], target) + self.insert(source, target),
                       self.distance(source, target[:-1]) + self.delete(source, target),
                       self.distance(source[:-1], target[:-1]) + self.subsitute(source, target))
    
    def dynamic_distance(self, source, target):
        n = len(source) + 1
        m = len(target) + 1
        distance_matrix = [[0 for i in range(m)] for j in range(n)]
        for i in range(1, n):
            distance_matrix[i][0] = distance_matrix[i-1][0] + self.delete(source[i-1], target[0])
        for j in range(1, m):
            distance_matrix[0][j] = distance_matrix[0][j-1] + self.insert(source[0], target[j-1])
        for i in range(1, n):
            for j in range(1, m):
                distance_matrix[i][j] = min(
                    distance_matrix[i-1][j] + self.delete(source[:i], target[:j]),distance_matrix[i][j-1] + self.insert(source[:i], target[:j]),
                    distance_matrix[i-1][j-1] + self.subsitute(source[:i], target[:j]))
        return distance_matrix[n-1][m-1]


        
from datetime import datetime
if __name__ == "__main__":
    med = MinimumEditDistance()
    print(datetime.now())
    print(med.recursive_distance("extermination", "intention"))
    print(datetime.now())
    
    print(datetime.now())
    print(med.dynamic_distance("extermination", "intention"))
    print(datetime.now())

# Output:
# 2023-10-01 15:54:08.214751
# 6
# 2023-10-01 15:54:28.877673
# 2023-10-01 15:54:28.877673
# 6
# 2023-10-01 15:54:28.878682