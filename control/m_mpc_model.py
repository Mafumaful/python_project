import numpy as np
import scipy.signal
from scipy.optimize import minimize
import matplotlib.pyplot as plt


class system:
    def __init__(self):
        # define the time settings
        self.DeltaT = 0.1  # continuous system rate
        self.dt = 1.0  # sample rate
        self.durations = 100  # durations euals 40s
        self.steps = int(self.durations/self.DeltaT)

        # transfer funciton of the system
        # define the dynamic of the model

        self.model = 1

        # model 1
        if self.model == 1:
            self.G = scipy.signal.lti([1], [0.5, 1, 0])
            self.x0 = np.mat([[-20.0], [-5.0]])
            self.Q = np.diag([0.0, 40.0])
            self.R = np.diag([0.001])

        # model 2
        if self.model == 2:
            A = np.mat([[0, 1, 0], [0, 0, 1], [0, 0, -1/self.DeltaT]])
            B = np.mat([[0], [0], [1/self.DeltaT]])
            C = np.mat([[0, 1, 0]])
            D = np.mat([0])
            self.G = scipy.signal.lti(A, B, C, D)
            self.x0 = np.mat([[20.0], [5.0], [10.0]])
            self.Q = np.diag([20.0, 1.0, 0])
            self.R = np.diag([0.05])

        # define transfer state
        ss = self.G.to_ss()
        self.A = ss.A
        self.B = ss.B
        self.C = ss.C
        self.D = ss.D

        print(">>>>>>>>>the param of the system>>>>>>>>>>>>>")
        print("A:\r\n", self.A, "\r\n")
        print("B:\r\n", self.B, "\r\n")
        print("C:\r\n", self.C, "\r\n")
        print("D:\r\n", self.D, "\r\n")
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

        # MPC control param
        self.M = int(10)  # Control horizon
        self.P = int(20)  # Predict horizon

        # define the xs and us
        self.xs = []
        self.us = []

        # define the continuous time
        self.tcontinuous = np.linspace(0, self.durations, self.steps)

    def update_state(self, x, u, dt):
        # update the next step according to u,x,dt
        increment = dt*self.A@x+dt*self.B@u
        return x+increment

    def state_2_pos(self, x, u=0):
        return self.C@x+self.D@u

    def simulate(self):
        u = None
        x = self.x0
        u_seq = np.ones(self.P)
        count = 0

        for i in range(self.steps):
            if u is not None:
                u_seq = u_opt

            def objective(u_seq):
                cost = 0.0
                xk = x
                uk = np.mat([0])

                for k in range(self.M):
                    uk = np.mat(u_seq[k])
                    cost = cost + 0.5 * \
                        (2*xk.T@self.Q*xk+uk.T@self.R@uk)
                    xk = self.update_state(xk, uk, self.dt)

                for k in range(self.P-self.M):
                    cost = cost + 0.5 * \
                        (2*xk.T@self.Q*xk+uk.T@self.R@uk)
                    xk = self.update_state(xk, uk, self.dt)
                return cost[0, 0]

            # use first control input, dt is the sample rate, DeltaT is the continuous system rate
            if int(i) % int(self.dt/self.DeltaT) == 0:
                # get the desired value u
                res = minimize(objective, u_seq,
                               options={"maxfun": 3}, method="L-BFGS-B")
                # res = minimize(objective, u_seq)
                # print(res)
                u_opt = res.x
                u = np.mat(u_opt[0])

            # update the state
            y = self.state_2_pos(x, u)
            x = self.update_state(x, u, self.DeltaT)
            # print(x)

            # record the value
            self.us.append(u[0, 0])
            self.xs.append(y[0, 0])


if __name__ == '__main__':
    car_ss = system()
    car_ss.simulate()

    # plot the figure
    fig, axes = plt.subplots(nrows=2)
    axes[0].plot(car_ss.tcontinuous, car_ss.xs)
    axes[0].set_title('state')
    axes[1].plot(car_ss.tcontinuous, car_ss.us)
    axes[1].set_title('input')
    plt.show()
