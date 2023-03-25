import numpy as np
from scipy.optimize import linprog
from scipy import linalg as lin
from math import *
from matplotlib import pyplot as plt
import casadi as ca


def lqr_p_k_cal(a: np.matrix, b: np.matrix, q: np.matrix, r: np.matrix):
    p = np.mat(lin.solve_discrete_are(a, b, q, r))  # 求Riccati方程
    k = lin.inv(r + b.T * p * b) * b.T * p * a

    return p, k
# 通过离散lqr计算出终端惩罚矩阵P和用来镇定实际状态和名义系统状态之差的矩阵K


def arg_cal(a: np.matrix):
    a_list = a.tolist()[0]
    cos_arg = a_list[0] / sqrt(a_list[0] ** 2 + a_list[1] ** 2)

    if float(a_list[1]) > 0:
        arg = acos(cos_arg)
    else:
        arg = -acos(cos_arg)

    return arg
# 计算向量的辐角，用于给多边形的边排序


def edges_sort(a: np.matrix, b: np.matrix):
    new_a, new_b = a.copy(), b.copy()
    if a.shape[0] < 2:
        return new_a
    else:
        last_exchange_index = 0
        sort_border = new_a.shape[0] - 1

        for i in range(0, new_a.shape[0] - 1):
            is_sorted = True
            for j in range(0, sort_border):
                if arg_cal(new_a[j, :]) > arg_cal(new_a[j + 1, :]):
                    is_sorted = False
                    new_a[[j, j + 1], :] = new_a[[j + 1, j], :]
                    new_b[:, [j, j + 1]] = new_b[:, [j + 1, j]]
                    last_exchange_index = j  # 记住每一遍最后一个产生换序动作的序号，说明这之后的项无须排序，下一轮排序可以忽略
            sort_border = last_exchange_index
            if is_sorted:  # 如果走完一遍发现没有产生换序动作，那么说明已经排序完成，跳出循环
                break

    return new_a, new_b
# 冒泡排序算法，用于给多边形的边排序


def vertex_cal(a: np.matrix, b: np.matrix):
    a_sort, b_sort = edges_sort(a, b)
    vertex = []

    for i in range(0, a_sort.shape[0] - 1):  # 按顺序两两求交点
        a_sol = a_sort[[i, i + 1], :].getA()
        b_sol = b_sort[:, [i, i + 1]].getA()[0]
        aug_mat = np.hstack((a_sort[[i, i + 1], :], b_sort[:, [i, i + 1]].T))
        if np.linalg.matrix_rank(a_sort[[i, i + 1], :]) == np.linalg.matrix_rank(aug_mat) and \
                np.linalg.matrix_rank(aug_mat) == b_sort[:, [i, i + 1]].shape[1]:
            vertex.append(lin.solve(a_sol, b_sol))

    a_sol = a_sort[[0, -1], :].getA()  # 计算第一条边和最后一条边的交点
    b_sol = b_sort[:, [0, -1]].getA()[0]
    aug_mat = np.hstack((a_sort[[0, -1], :], b_sort[:, [0, -1]].T))
    if np.linalg.matrix_rank(a_sort[[0, -1], :]) == np.linalg.matrix_rank(aug_mat) and \
            np.linalg.matrix_rank(aug_mat) == b_sort[:, [0, -1]].shape[1]:
        vertex.append(lin.solve(a_sol, b_sol))

    return np.mat(vertex)
# 根据已经排好的边的顺序，相邻边求交点，得出顶点，这种方法只适合凸多边形


def poly_plot(a: np.matrix, b: np.matrix, color):
    vertex = vertex_cal(a, b)

    vertex_list = vertex.T.tolist()

    plt.plot(vertex_list[0], vertex_list[1], color)
    plt.plot([vertex_list[0][0], vertex_list[0][-1]],
             [vertex_list[1][0], vertex_list[1][-1]], color)
# 通过顶点画多边形


def all_zero(a: np.matrix, b: np.matrix):
    c = np.hstack((a, b.T))
    delete_line = []

    for i in range(0, c.shape[0]):
        if (c[i, :-1] == 0.).all():
            delete_line.append(i)

    c = np.delete(c, delete_line, 0)

    new_a = np.mat(np.delete(c, -1, 1))
    new_b = np.mat(c[:, -1].T)

    return new_a, new_b
# 去除系数全为零的行


def collinear(a: np.matrix, b: np.matrix):
    c = np.hstack((a, b.T))
    delete_line = []

    for i in range(0, c.shape[0]):
        for j in range(i + 1, c.shape[0]):
            test_mat = np.vstack((c[i, :], c[j, :]))
            if np.linalg.matrix_rank(test_mat) < 2:
                delete_line.append(j)

    c = np.delete(c, delete_line, 0)

    new_a = np.mat(np.delete(c, -1, 1))
    new_b = np.mat(c[:, -1].T)

    return new_a, new_b
# 去除一个线性不等式矩阵(A,b)中共线的边


def redundant_term(a: np.matrix, b: np.matrix):
    delete_line = []
    for i in range(0, a.shape[0]):
        c = a[i, :]
        a_bar = np.mat(np.delete(a, i, 0))
        b_bar = np.mat(np.delete(b, i, 1))
        bounds = [(None, None)] * c.shape[1]
        res = linprog(-c, A_ub=a, b_ub=b, bounds=bounds,
                      method='revised simplex')
        res_bar = linprog(-c, A_ub=a_bar, b_ub=b_bar,
                          bounds=bounds, method='revised simplex')

        if 0.0001 > res.fun - res_bar.fun > -0.0001:
            delete_line.append(i)
        # 有它没它都一样的话，说明该项冗余

    new_a = np.mat(np.delete(a, delete_line, 0))
    new_b = np.mat(np.delete(b, delete_line, 1))

    return new_a, new_b


def xf_cal(d_a: np.matrix, d_b: np.matrix, a_k: np.matrix):
    t = 0
    a_cons, b_cons = d_a, d_b  # 初始状态

    while True:
        max_res = []

        for i in range(0, d_a.shape[0]):
            c = d_a[i, :] * (a_k ** (t + 1))
            bounds = [(None, None)] * d_a.shape[1]
            res = linprog(-c, A_ub=a_cons, b_ub=b_cons,
                          bounds=bounds, method='revised simplex')
            max_res.append(-res.fun)
        # 检验Ot+1是否与Ot相等

        if ((np.mat(max_res) - d_b) <= 0).all():
            break  # 若相等则跳出循环

        t = t + 1

        a_cons = np.mat(np.vstack((a_cons, d_a * a_k ** t)))
        b_cons = np.mat(np.hstack((b_cons, d_b)))
        # 若不是则令Ot = Ot+1继续循环

    a_cons, b_cons = collinear(a_cons, b_cons)
    a_cons, b_cons = redundant_term(a_cons, b_cons)  # 得到结果后还需除去冗余项
    # 计算方法是增加t，直到Ot == Ot+1，于是有O∞ = Ot
    return a_cons, b_cons
# 计算终端约束区域Xf


def mpc_m_build(a: np.matrix, n: int) -> np.matrix:

    m = np.mat([[1, 0], [0, 1]])
    a_i = np.mat([[1, 0], [0, 1]])

    for i in range(0, n):
        a_i = a_i * a
        m = np.vstack((m, a_i))

    return m
# 求解Xk = M*xk + C*Uk中的M矩阵，便于之后求解可行域


def mpc_c_build(a: np.matrix, b: np.matrix, n: int) -> np.matrix or int:

    if n == 0:
        c = 0
        return c

    else:
        row_delta = b
        row_delta_d = b
        c = np.vstack((np.mat(np.zeros((b.shape[0], b.shape[1]))), b))

        for i in range(0, n - 1):
            row_delta_d = a * row_delta_d
            row_delta = np.hstack((row_delta_d, row_delta))
            c = np.hstack((c, np.mat(np.zeros((c.shape[0], b.shape[1])))))
            c = np.vstack((c, row_delta))

        return c
# 求解Xk = M*xk + C*Uk中的C矩阵，便于之后求解可行域


def mpc_g_h_build(x_a: np.matrix, x_b: np.matrix, xf_a: np.matrix, xf_b: np.matrix, n: int):

    g = xf_a
    h = xf_b
    row_delta = x_a

    for i in range(0, n):
        row_delta = np.hstack(
            (row_delta, np.mat(np.zeros((row_delta.shape[0], xf_a.shape[1])))))
        g = np.hstack((np.mat(np.zeros((g.shape[0], x_a.shape[1]))), g))
        g = np.vstack((row_delta, g))
        h = np.hstack((x_b, h))

    return g, h
# 产生关于Xk的约束不等式组


def uk_a_b_build(u_inf, u_sup, n: int):
    uk_a = np.mat(np.zeros((4 * n, 2 * n)))
    for i in range(0, n):
        uk_a[4 * i:4 * (i + 1), 2 * i:2 * (i + 1)
             ] = np.mat([[1, 0], [-1, 0], [0, 1], [0, -1]])

    uk_b = np.mat([[u_sup, -u_inf] * 2 * n])

    return uk_a, uk_b
# 产生关于Uk的约束不等式组


def feasible_set(a: np.matrix,
                 b: np.matrix,
                 x_a: np.matrix,
                 x_b: np.matrix,
                 u_inf,
                 u_sup,
                 xf_a: np.matrix,
                 xf_b: np.matrix,
                 n: int
                 ):
    m = mpc_m_build(a, n)
    c = mpc_c_build(a, b, n)
    g, h = mpc_g_h_build(x_a, x_b, xf_a, xf_b, n)
    uk_a, uk_b = uk_a_b_build(u_inf, u_sup, n)
    a_bar = np.vstack((np.hstack((g * m, g * c)),
                       np.hstack((np.mat(np.zeros((uk_a.shape[0], m.shape[1]))), uk_a))))
    b_bar = np.hstack((h, uk_b))
    # 这里先求出了M，C矩阵，于是知道了Xk = M*x + C*Uk，之后将约束条件转化为G*Xk <= h，再包含A_Uk*Uk <= b_Uk
    # 于是有
    # [G*M G*C ][x ]      [h   ]
    # [        ][  ]  <=  [    ]
    # [ 0  A_Uk][Uk]      [b_Uk]
    # a_bar, b_bar分别代表上面两个矩阵

    for k in range(0, b.shape[1] * n):  # 这里控制输入u有几维就是几乘n
        pos_a = []
        pos_b = []
        neg_a = []
        neg_b = []
        zero_a = []
        zero_b = []

        for i in range(0, a_bar.shape[0]):
            if a_bar[i, -1] > 0:
                pos_a.append((a_bar[i, :-1] / a_bar[i, -1]).tolist()[0])
                pos_b.append(b_bar[0, i] / a_bar[i, -1])
            elif a_bar[i, -1] < 0:
                neg_a.append((a_bar[i, :-1] / (-a_bar[i, -1])).tolist()[0])
                neg_b.append(b_bar[0, i] / (-a_bar[i, -1]))
            else:
                zero_a.append(a_bar[i, :-1].tolist()[0])
                zero_b.append(b_bar[0, i])

        pos_a = np.mat(pos_a)
        pos_b = np.mat([pos_b])
        neg_a = np.mat(neg_a)
        neg_b = np.mat([neg_b])
        zero_a = np.mat(zero_a)
        zero_b = np.mat([zero_b])

        new_a = []
        new_b = []

        for i in range(0, pos_a.shape[0]):
            for j in range(0, neg_a.shape[0]):
                new_a.append((pos_a[i, :] + neg_a[j, :]).tolist()[0])
                new_b.append(pos_b[0, i] + neg_b[0, j])

        for i in range(0, zero_a.shape[0]):
            new_a.append(zero_a[i, :].tolist()[0])
            new_b.append(zero_b[0, i])

        new_a = np.mat(new_a)
        new_b = np.mat([new_b])

        a_bar, b_bar = all_zero(new_a, new_b)
        a_bar, b_bar = collinear(a_bar, b_bar)
        a_bar, b_bar = redundant_term(a_bar, b_bar)

    # 以上是傅里叶-莫茨金消元法

    return a_bar, b_bar

# 求解可行域，主要思路是构造满足条件的关于x，Uk的不等式组，通过Xk = M*x + C*Uk建立关系，之后利用傅里叶-莫茨金消元法
# 将不等式组投影到x的平面上，便知道了使优化问题有解的x的取值范围


def dm2arr(dm):
    return np.array(dm.full())
# CasADi求解器类型转换


def mpc_optimization(sys_para, X_bou, U_bou, x_ini, xf):
    xf_num = xf['A'].shape[0]

    x_1 = ca.SX.sym('x_1')
    x_2 = ca.SX.sym('x_2')
    states = ca.vertcat(x_1, x_2)  # 控制器中的状态
    n_states = states.numel()  # 行数

    u_1 = ca.SX.sym('u_1')
    u_2 = ca.SX.sym('u_2')
    controls = ca.vertcat(u_1, u_2)  # 控制器中的输入
    n_controls = controls.numel()  # 行数

    X = ca.SX.sym('X', n_states, sys_para['n'] + 1)  # 控制器中的状态变量汇总起来
    U = ca.SX.sym('U', n_controls, sys_para['n'])  # 控制器中的输入变量汇总起来
    ini = ca.SX.sym('ini', n_states)  # 储存每次求优化时的初状态

    A = ca.SX(sys_para['A'])
    B = ca.SX(sys_para['B'])  # 状态矩阵

    Q = ca.SX(sys_para['Q'])
    R = ca.SX(sys_para['R'])
    P = ca.SX(sys_para['P'])  # 三个权重矩阵

    Xf_A = ca.SX(xf['A'])  # 终端约束

    st_fun = A @ states + B @ controls

    f = ca.Function('f', [states, controls], [st_fun])  # 对应状态方程中的f()

    cost_fn = 0

    g = X[:, 0] - ini  # 初值约束

    for k in range(0, sys_para['n']):
        st = X[:, k]
        con = U[:, k]
        cost_fn = cost_fn + st.T @ Q @ st + con.T @ R @ con  # 构造代价函数

        st_next = X[:, k + 1]
        st_next_val = f(st, con)
        g = ca.vertcat(g, st_next - st_next_val)  # 构造约束

    cost_fn = cost_fn + X[:, sys_para['n']
                          ].T @ P @ X[:, sys_para['n']]  # 还应加上终端代价函数

    g = ca.vertcat(g, Xf_A @ X[:, sys_para['n']])  # 还有终端区域约束，输入约束加在bounds里

    opt_variables = ca.vertcat(
        X.reshape((-1, 1)), U.reshape((-1, 1)))  # 将x和u均视为优化问题的变量

    nlp_prob = {
        'f': cost_fn,
        'x': opt_variables,
        'g': g,
        'p': ini
    }

    opts = {
        'ipopt': {
            'max_iter': 2000,
            'print_level': 0,
            'acceptable_tol': 1e-8,
            'acceptable_obj_change_tol': 1e-6
        },
        'print_time': 0
    }  # 求解优化时的设置

    solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

    lbx = ca.DM.zeros(
        (n_states * (sys_para['n'] + 1) + n_controls * sys_para['n'], 1))
    ubx = ca.DM.zeros(
        (n_states * (sys_para['n'] + 1) + n_controls * sys_para['n'], 1))

    lbx[0: n_states * (sys_para['n'] + 1)        : n_states] = X_bou['x_1']['inf']  # x_1 的下界
    lbx[1: n_states * (sys_para['n'] + 1)        : n_states] = X_bou['x_2']['inf']  # x_2 的下界

    ubx[0: n_states * (sys_para['n'] + 1)        : n_states] = X_bou['x_1']['sup']  # x_1 的上界
    ubx[1: n_states * (sys_para['n'] + 1)        : n_states] = X_bou['x_2']['sup']  # x_2 的上界

    lbx[n_states * (sys_para['n'] + 1):] = U_bou['inf']  # u 的下界
    ubx[n_states * (sys_para['n'] + 1):] = U_bou['sup']  # u 的上界

    lbg = ca.DM.zeros((n_states * (sys_para['n'] + 1) + xf_num, 1))
    ubg = ca.DM.zeros((n_states * (sys_para['n'] + 1) + xf_num, 1))

    for i in range(0, xf_num):
        lbg[n_states * (sys_para['n'] + 1) + i] = -ca.inf
        ubg[n_states * (sys_para['n'] + 1) +
            i] = float(xf['b'][0, i])  # x(N)在终端区域内

    args = {'lbg': lbg,
            'ubg': ubg,
            'lbx': lbx,
            'ubx': ubx
            }  # 将约束的值赋进去

    state_ini = ca.DM([x_ini[0], x_ini[1]])  # 系统初状态，要与解优化问题时的初值分清楚

    u0 = ca.DM.zeros((n_controls, sys_para['n']))
    X0 = ca.repmat(state_ini, 1, sys_para['n'] + 1)  # 求解优化过程中的初值

    cat_states = dm2arr(state_ini)  # 实际状态
    cat_controls = dm2arr(u0[:, 0])  # 实际控制量

    for k in range(0, int(sys_para['T'] / sys_para['d_t'])):

        args['p'] = ca.vertcat(state_ini)  # 每一时刻都把实际状态作为优化问题的参数传回去
        args['x0'] = ca.vertcat(ca.reshape(X0, n_states * (sys_para['n'] + 1), 1),
                                ca.reshape(u0, n_controls * sys_para['n'], 1))
        # 用上一时刻的解的变换作为下一次求优化的初值，减少求优化的计算量，防止优化问题从可行域外开始解

        sol = solver(
            x0=args['x0'],
            lbx=args['lbx'],
            ubx=args['ubx'],
            lbg=args['lbg'],
            ubg=args['ubg'],
            p=args['p']
        )

        u = ca.reshape(sol['x'][n_states * (sys_para['n'] + 1):],
                       n_controls, sys_para['n'])  # 控制器内预测的控制输入
        X0 = ca.reshape(sol['x'][: n_states * (sys_para['n'] + 1)],
                        n_states, sys_para['n'] + 1)  # 控制器内预测的状态

        cat_controls = np.vstack((cat_controls, dm2arr(u[:, 0])))  # 记录下实际控制输入

        state_ini = f(state_ini, u[:, 0])  # 更新状态
        cat_states = np.dstack((cat_states, state_ini))  # 记录下实际状态

        u0 = ca.horzcat(
            u[:, 1:],
            ca.reshape(u[:, -1], -1, 1)
        )

        # 这两步是利用到上一步预测的一些结果到下一时刻求优化的初值中去
        X0 = ca.horzcat(X0[:, 1:], ca.reshape(X0[:, -1], -1, 1))

    res = {'x_1': cat_states[0, :][0],
           'x_2': cat_states[1, :][0],
           'u': cat_controls.T[0]
           }
    return res
# tube mpc在线优化


if __name__ == "__main__":
    system_parameter = {'A': np.mat([[2, 1], [0, 2]]),
                        'B': np.mat([[1, 0], [0, 1]]),
                        'Q': np.mat([[1e-5, 0], [0, 1e-5]]),
                        'R': np.mat([[1, 0], [0, 1]]),
                        'n': 3,  # 模型预测控制步数
                        'T': 5,  # 系统仿真总时间
                        'd_t': 0.1}  # 系统采样时间

    X_bounds = {'x_1': {'inf': -5, 'sup': 5},
                'x_2': {'inf': -ca.inf, 'sup': ca.inf}}

    U_bounds = {'inf': -1, 'sup': 1}

    x_initial = [0, -0.75]

    system_parameter['P'], K = lqr_p_k_cal(system_parameter['A'],
                                           system_parameter['B'],
                                           system_parameter['Q'],
                                           system_parameter['R'])

    A_K = system_parameter['A'] - system_parameter['B'] * K

    A_X = np.mat([[1, 0],
                  [-1, 0]])
    b_X = np.mat([[5, 5]])

    A_Kx = np.vstack((K, -K))
    b_Kx = np.mat([[1, 1, 1, 1]])

    A_D = np.mat(np.vstack((A_X, A_Kx)))
    b_D = np.mat(np.hstack((b_X, b_Kx)))

    Xf = {'A': (xf_cal(A_D, b_D, A_K))[0], 'b': (xf_cal(A_D, b_D, A_K))[1]}

    sim_result = mpc_optimization(
        system_parameter, X_bounds, U_bounds, x_initial, Xf)

    A_fea_set, b_fea_set = feasible_set(system_parameter['A'],
                                        system_parameter['B'],
                                        A_X,
                                        b_X,
                                        U_bounds['inf'],
                                        U_bounds['sup'],
                                        Xf['A'],
                                        Xf['b'],
                                        system_parameter['n']
                                        )

    poly_plot(Xf['A'], Xf['b'], 'r')
    plt.plot(sim_result['x_1'], sim_result['x_2'], 'b')
    poly_plot(A_fea_set, b_fea_set, 'g')
    plt.grid(True)
    plt.show()
