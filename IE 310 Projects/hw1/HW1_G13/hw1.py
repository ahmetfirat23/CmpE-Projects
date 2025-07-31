from sys import maxsize

minimum = maxsize
for x in range(31):
    for y in range(31):
        for z in range(31):
            for a in range(31):
                curr = 690*x + 545*y + 1020*z + 785*a

                if ((20*x + 30*y + 10*z + 25*a) <= 1000) \
                and ((160*x + 100*y + 200*z + 75*a) <= 8000) \
                and ((30*x + 35*y + 60*z + 80*a) <= 4000) \
                and ((35*x + 45*y + 70*z + 0*a) >= 2100) \
                and ((55*x + 42*y + 0*z + 90*a) >= 1800) \
                and curr < minimum:
                    minimum = curr
                    solution = [x,y,z,a]


minimum_with_ex = maxsize
for x in range(31):
    for y in range(31):
        for z in range(31):
            for a in range(31):
                for b in range(201):
                    curr = 690*x + 545*y + 1020*z + 785*a + 30*b

                    if ((20*x + 30*y + 10*z + 25*a) <= 1000 + b) \
                    and ((160*x + 100*y + 200*z + 75*a) <= 8000) \
                    and ((30*x + 35*y + 60*z + 80*a) <= 4000) \
                    and ((35*x + 45*y + 70*z + 0*a) >= 2100) \
                    and ((55*x + 42*y + 0*z + 90*a) >= 1800) \
                    and curr < minimum_with_ex:
                        minimum_with_ex = curr
                        solution_with_ex = [x,y,z,a,b]

print("Minimum cost without extra hour: " + str(minimum))
print("Solution: " + str(solution) + " (for the number of hr/week of processors 1,2,3 and 4 respectively)")

print("Minimum cost with extra hour: " + str(minimum_with_ex))
print("Solution: " + str(solution_with_ex) + " (for the number of hr/week of processors 1,2,3 and 4 and number of hours of overtime respectively)")