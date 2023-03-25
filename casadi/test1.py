from casadi import *

x = SX.sym('x', 3, 3)
y = SX.sym('y', 3)
rhs = x@y+DM([1, 1, 1])
print(y)
print(rhs)
f = Function('f', [x, y], [rhs], ['x1', 'y1'], ['rhs'])
res = f(x1=DM_eye(3), y1=5*DM.ones(3))
print(res)

print(DM_eye(3)+DM_eye(3))
