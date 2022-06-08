# #import numpy
# import numpy
# n = input()
# n = [float(i) for i in n.split(' ')]
# print(n)
# print(numpy.polyval(n,0))

n = int(input())
s = set(map(int, input().split()))
nc = int(input())
p = "pop"
r = "remove"
d = "discard"
for i in range(nc):
    c = input().split()
    if len(c)>1:
        if c[0] == r:
            s.remove(c[1])
        elif c[0] == d:
            s.discard(c[1])
    else:
        if c[0] == p:
            s.pop()
    print(s)