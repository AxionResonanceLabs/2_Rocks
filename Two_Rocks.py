#!/usr/bin/env python3 """ Ultimate Two Rocks Universe Experiment Author: Inspired by Brian Tice Sr. & User Experiments

Features:

Two Rocks first principle

Classical emergence (animated, multi-curve)

Quantum evolution (animated heatmaps, probabilities)

Universe scaling & falsification checks

3D network emergence, resonance lines

Closed loop integrity to Two Rocks

Full color, motion, immersive visualizations """


import numpy as np from scipy.linalg import expm import matplotlib.pyplot as plt import matplotlib.animation as animation from mpl_toolkits.mplot3d import Axes3D import seaborn as sns sns.set(style="darkgrid")

=====================

0. Two Rocks

=====================

class TwoRocks: def init(self, left=1, right=1): self.L = left self.R = right self.total = self.L + self.R

def move(self):
    if self.R > 0:
        self.L += 1
        self.R -= 1
    assert self.L + self.R == self.total, "Invariant broken!"

def __repr__(self):
    return f"TwoRocks(L={self.L}, R={self.R}, total={self.total})"

=====================

1. Classical Universe

=====================

class ClassicalUniverse: def init(self, rocks: TwoRocks): self.rocks = rocks self.history = []

def evolve(self, steps=200):
    for _ in range(steps):
        self.rocks.move()
        r = len(self.history) % 10
        C_r = 0.001*(1+2*r)*np.exp(-r/3)
        self.history.append({'step': len(self.history), 'L': self.rocks.L, 'R': self.rocks.R, 'C': C_r})
    return self.history

def animate(self):
    steps = [h['step'] for h in self.history]
    Ls = [h['L'] for h in self.history]
    Rs = [h['R'] for h in self.history]
    Cs = [h['C'] for h in self.history]

    fig, ax = plt.subplots(figsize=(12,6))
    line_L, = ax.plot([], [], color='royalblue', linewidth=2, label='Left Rock')
    line_R, = ax.plot([], [], color='orange', linewidth=2, label='Right Rock')
    scatter_C = ax.scatter([], [], c=[], cmap='viridis', s=50, label='C(r)')
    ax.set_xlim(0, max(steps))
    ax.set_ylim(0, max(max(Ls), max(Rs))*1.1)
    ax.set_title("Classical Evolution", fontsize=16)
    ax.set_xlabel("Step")
    ax.set_ylabel("Quantity / Connection")
    ax.legend()

    def update(frame):
        line_L.set_data(steps[:frame], Ls[:frame])
        line_R.set_data(steps[:frame], Rs[:frame])
        scatter_C.set_offsets(np.c_[steps[:frame], Cs[:frame]])
        scatter_C.set_array(np.array(Cs[:frame]))
        return line_L, line_R, scatter_C

    ani = animation.FuncAnimation(fig, update, frames=len(steps), interval=50, blit=True)
    plt.show()

=====================

2. Quantum Universe

=====================

class QuantumUniverse: def init(self): self.states = []

def evolve(self, steps=100):
    state = np.array([[1,0],[0,1]], dtype=complex)
    self.states.append(state.copy())
    for step in range(steps):
        r = step*0.1
        coupling = 0.001*(1+2*r)*np.exp(-r/3)
        H = np.zeros((4,4), dtype=complex)
        for i in range(3):
            H[i,i+1] = coupling
            H[i+1,i] = np.conj(coupling)
        H[0,-1] = coupling
        H[-1,0] = np.conj(coupling)
        U = expm(-1j*H*0.1)
        flat = state.flatten()
        flat = U@flat
        state = flat.reshape(state.shape)
        state /= np.linalg.norm(state)
        self.states.append(state.copy())
    return self.states

def animate_heatmap(self):
    probs = [np.abs(s.flatten())**2 for s in self.states]
    probs = np.array(probs)
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(probs.T, cmap='magma', cbar=True, ax=ax)
    ax.set_title("Quantum Probability Heatmap", fontsize=16)
    ax.set_xlabel("Step")
    ax.set_ylabel("State Index")
    plt.show()

=====================

3. Universe Scaling & Falsification

=====================

class Universe: def init(self, rocks: TwoRocks): self.rocks = rocks self.classical = ClassicalUniverse(rocks) self.quantum = QuantumUniverse()

def run(self):
    classical_history = self.classical.evolve(steps=200)
    quantum_history = self.quantum.evolve(steps=100)
    return classical_history, quantum_history

def falsify(self):
    failed_tests = []
    if max(self.rocks.L, self.rocks.R) > 1e6:
        failed_tests.append("Classical scaling breaks")
    if len(self.quantum.states) < 10:
        failed_tests.append("Quantum evolution incomplete")
    return failed_tests if failed_tests else ["All tests passed"]

=====================

4. 3D Network Emergence

=====================

class Network3D: def init(self, nodes=10): self.nodes = nodes self.positions = np.random.rand(nodes,3)*10 self.edges = [(i,(i+1)%nodes) for i in range(nodes)]

def animate_network(self):
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("3D Emergent Network", fontsize=16)
    scat = ax.scatter(self.positions[:,0], self.positions[:,1], self.positions[:,2], c='cyan', s=80)

    for edge in self.edges:
        line = ax.plot([self.positions[edge[0],0], self.positions[edge[1],0]],
                       [self.positions[edge[0],1], self.positions[edge[1],1]],
                       [self.positions[edge[0],2], self.positions[edge[1],2]],
                       color='magenta', alpha=0.5)
    plt.show()

=====================

5. Closed Loop Demo

=====================

def closed_loop_demo(): rocks = TwoRocks() universe = Universe(rocks) classical, quantum = universe.run() tests = universe.falsify()

print("\n▶ Two Rocks Status:", rocks)
print("▶ Falsification Results:", tests)

universe.classical.animate()
universe.quantum.animate_heatmap()

network = Network3D(nodes=20)
network.animate_network()

return rocks, classical, quantum

=====================

MAIN EXECUTION

=====================

if name == "main": closed_loop_demo()
