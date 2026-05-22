import numpy as np
import matplotlib.pyplot as plt


def gaussian_pulse(x, mean, sigma):
    return np.exp(-(x - mean) ** 2 / (2 * sigma ** 2))


class FDTD1D:
    def __init__(self, N, time_steps, dx, dt, eps, mu,
                 src_pos, probe_left, probe_right):

        self.N = N
        self.T = time_steps

        self.dx = dx
        self.dt = dt

        self.E = np.zeros(N)
        self.H = np.zeros(N - 1)

        self.eps = eps
        self.mu = mu

        self.src = src_pos
        self.pl = probe_left
        self.pr = probe_right

        self.reflected = np.zeros(time_steps)
        self.transmitted = np.zeros(time_steps)

    def step(self, t):

        self.H += (self.dt / (self.mu[:-1] * self.dx)) * (
            self.E[1:] - self.E[:-1]
        )

        self.E[1:-1] += (self.dt / (self.eps[1:-1] * self.dx)) * (
            self.H[1:] - self.H[:-1]
        )

        self.E[0] = self.E[1]
        self.E[-1] = self.E[-2]

        self.E[self.src] = gaussian_pulse(t, 30, 8)

        self.reflected[t] = self.E[self.pl]
        self.transmitted[t] = self.E[self.pr]

    def run(self):
        for t in range(self.T):
            self.step(t)

N = 300
T = 500

dx = 1.0
c = 1.0
dt = 0.99 * dx / c

eps = np.ones(N)
mu = np.ones(N)

i1, i2 = 120, 180
eps[i1:i2] = 4.0
mu[i1:i2] = 2.0

src_pos = 60
probe_left = 100
probe_right = 220

sim = FDTD1D(
    N=N,
    time_steps=T,
    dx=dx,
    dt=dt,
    eps=eps,
    mu=mu,
    src_pos=60,
    probe_left=100,
    probe_right=220
)

sim.run()

t = np.arange(T)

plt.figure()
plt.plot(t, sim.reflected)
plt.plot(t, sim.transmitted)
plt.xlabel("Time step")
plt.ylabel("Amplitude")
plt.legend(["Reflected", "Transmitted"])
plt.grid()
plt.show()

A_ref = np.max(np.abs(sim.reflected))
A_tr = np.max(np.abs(sim.transmitted))

print("Max reflected amplitude:", A_ref)
print("Max transmitted amplitude:", A_tr)
print("R/T ratio:", A_ref / A_tr)
