import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def gaussian_pulse(x, mean, sigma):
    return np.exp(-(x - mean) ** 2 / (2 * sigma ** 2))


class FDTD1D_Maxwell:
    """
    1D FDTD (Yee):
        dE/dt = (1/eps) dH/dx
        dH/dt = (1/mu)  dE/dx
    """

    def __init__(self, x_dim, time_tot, c, dx, S,
                 obs_probe, record_stop_time,
                 eps=1.0, mu=1.0):

        self.x_dim = x_dim
        self.time_tot = time_tot
        self.c = c
        self.dx = dx
        self.S = S
        self.dt = S * dx / c

        self.eps = eps
        self.mu = mu

        # Поля (стагерена сітка Yee)
        self.E = np.zeros(x_dim)
        self.H = np.zeros(x_dim - 1)

        self.obs_probe = obs_probe
        self.record_stop_time = record_stop_time
        self.signal_obs = np.zeros(time_tot)

        self.t_step = 0

    def step(self):
        # --- Оновлення H (півкроку в просторі) ---
        self.H[:] = self.H[:] + (self.dt / (self.mu * self.dx)) * \
                    (self.E[1:] - self.E[:-1])

        # --- Оновлення E ---
        self.E[1:-1] = self.E[1:-1] + (self.dt / (self.eps * self.dx)) * \
                       (self.H[1:] - self.H[:-1])

        # Граничні умови (ідеальні провідники)
        self.E[0] = 0.0
        self.E[-1] = 0.0

        if self.t_step < self.record_stop_time:
            self.signal_obs[self.t_step] = self.E[self.obs_probe]

        self.t_step += 1

    def run(self):
        for _ in range(self.time_tot):
            self.step()


# --- Параметри моделювання ---
x_dim = 200
time_tot = 400
c = 1.0
dx = 1.0
S = 0.99

source_pos = x_dim // 2
obs_probe = int(0.75 * x_dim)
record_stop_time = 100

x = np.arange(0, x_dim, dx)

# --- Початкове E поле (імпульс), H = 0 ---
sigma_0 = 5
E0 = gaussian_pulse(x, source_pos, sigma_0)

# --- Ініціалізація ---
sim = FDTD1D_Maxwell(x_dim, time_tot, c, dx, S,
                     obs_probe, record_stop_time)

sim.E[:] = E0


# ============================================================
#                      АНІМАЦІЯ
# ============================================================

fig, ax = plt.subplots()
ax.set_xlim(0, x_dim)
ax.set_ylim(-1.2, 1.2)
ax.set_title("1D FDTD (Maxwell): E та H")
ax.set_xlabel("x")
ax.set_ylabel("Поле")

lineE, = ax.plot([], [], lw=2, label="E")
lineH, = ax.plot([], [], lw=2, linestyle="--", label="H")
ax.legend()


def update(frame):
    sim.step()
    lineE.set_data(np.arange(sim.x_dim), sim.E)
    lineH.set_data(np.arange(sim.x_dim - 1) + 0.5, sim.H)
    return lineE, lineH


ani = animation.FuncAnimation(fig, update,
                              frames=time_tot,
                              interval=20,
                              blit=True,
                              repeat=False)

plt.show()
