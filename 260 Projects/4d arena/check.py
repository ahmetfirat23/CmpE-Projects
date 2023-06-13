import os
import sys


def read_dic(path):
    results = os.listdir(path)
    dic = {}
    for test in results:
        f = open(os.path.join(path, test), "r")
        lines = f.readlines()
        f.close()
        dic[test] = lines
    return dic


def check(truth, query, points):
    result = {}
    query_keys = query.keys()
    for key in truth:
        a = truth[key]
        if key in query_keys:
            b = query[key]
            if type(a) == list:
                result[key] = (a == b)
            elif type(a) == float:
                result[key] = (abs(a-b) < 1e-8)
        else:
            result[key] = False
    grade = 0
    for key in sorted(truth):
        if key in result:
            if result[key]:
                grade += points[key]
            print(f"{key}: {result[key]}", file=open(os.path.join(sys.argv[2], "result.txt"), "a"))
    return grade


if __name__ == "__main__":
    points = {
        "test-1-1.txt": 5,
        "test-1-2.txt": 5,
        "test-2-1.txt": 5,
        "test-2-2.txt": 5,
        "test-3-1.txt": 3,
        "test-3-2.txt": 3,
        "test-3-3.txt": 4,
        "test-4-1.txt": 3,
        "test-4-2.txt": 3,
        "test-4-3.txt": 4,
        "test-5-1.txt": 5,
        "test-5-2.txt": 5,
        "test-6-1.txt": 2.5,
        "test-6-2.txt": 2.5,
        "test-6-3.txt": 2.5,
        "test-6-4.txt": 2.5,
        "test-7-1.txt": 5,
        "test-7-2.txt": 5,
        "test-7-3.txt": 5,
        "test-7-4.txt": 5,
        "test-8-1.txt": 4,
        "test-8-2.txt": 4,
        "test-8-3.txt": 4,
        "test-8-4.txt": 4,
        "test-8-5.txt": 4,
    }
    result_file = os.path.join(sys.argv[2], "result.txt")
    if os.path.exists(result_file):
        os.remove(result_file)
    truth_dict = read_dic(sys.argv[1])
    query_dict = read_dic(sys.argv[2])
    point = check(truth_dict, query_dict, points)
    print(sys.argv[2], point)
