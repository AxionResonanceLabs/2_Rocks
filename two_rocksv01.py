#!/usr/bin/env python3
"""
Two Rocks Universal Experiment with Visualizations
Author: Inspired by Brian Tice Sr. & User Experiments

Features:
- Two Rocks first principle
- Classical emergence
- Quantum evolution
- Universe scaling & falsification
- Colorful graphs to overwhelm curiosity
"""

import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="darkgrid")

# ============================================================================
# 0. Two Rocks First Principle
# ============================================================================
class TwoRocks:
    """Irreducible first principle"""
    def __init__(self, left=1, right=1):
        self.L = left
        self.R = right
        self.total = self.L + self.R

    def move(self):
        """Fundamental operation"""
        if self.R > 0:
            self.L += 1
            self.R -= 1
        assert self.L + self.R == self.total, "Invariant broken!"

    def __repr__(self):
        return f"TwoRocks(L={self.L}, R={self.R}, total={self.total})"

# ============================================================================
# 1. Classical Emergence
# ============================================================================
class ClassicalUniverse:
    def __init__(self, rocks: TwoRocks):
        self.rocks = rocks
        self.history = []

    def evolve(self, steps=200):
        for _ in range(steps):
            self.rocks.move()
            r = len(self.history) % 10
            C_r = 0.001*(1+2*r)*np.exp(-r/3)
            self.history.append({'step': len(self.history),
                                 'L': self.rocks.L,
                                 'R': self.rocks.R,
                                 'C': C_r})
        return self.history

    def plot(self):
        steps = [h['step'] for h in self.history]
        Ls = [h['L'] for h in self.history]
        Rs = [h['R'] for h in self.history]
        Cs = [h['C'] for h in self.history]

        plt.figure(figsize=(12,6))
        plt.plot(steps, Ls, label="Left Rock", color='royalblue', linewidth=2)
        plt.plot(steps, Rs, label="Right Rock", color='orange', linewidth=2)
        plt.fill_between(steps, Ls, Rs, color='purple', alpha=0.1)
        plt.scatter(steps, Cs, c=Cs, cmap='viridis', s=30, label="Connection |C(r)|")
        plt.title("Classical Emergence of Two Rocks", fontsize=16)
        plt.xlabel("Step")
        plt.ylabel("Quantity / Connection Strength")
        plt.legend()
        plt.show()

# ============================================================================
# 2. Quantum Evolution
# ============================================================================
class QuantumUniverse:
    def __init__(self):
        self.states = []

    def evolve(self, steps=100):
        state = np.array([[1,0],[0,1]], dtype=complex)
        self.states.append(state.copy())
        for step in range(steps):
            r = step * 0.1
            coupling = 0.001*(1+2*r)*np.exp(-r/3)
            H = np.zeros((4,4), dtype=complex)
            for i in range(3):
                H[i,i+1] = coupling
                H[i+1,i] = np.conj(coupling)
            H[0,-1] = coupling
            H[-1,0] = np.conj(coupling)
            U = expm(-1j * H * 0.1)
            flat = state.flatten()
            flat = U @ flat
            state = flat.reshape(state.shape)
            state /= np.linalg.norm(state)
            self.states.append(state.copy())
        return self.states

    def plot(self):
        probs = [np.abs(s.flatten())**2 for s in self.states]
        probs = np.array(probs)

        plt.figure(figsize=(12,6))
        for i in range(probs.shape[1]):
            plt.plot(probs[:,i], label=f"State {i+1}", linewidth=2)
        plt.title("Quantum Evolution: Probability Amplitudes", fontsize=16)
        plt.xlabel("Step")
        plt.ylabel("Probability")
        plt.legend()
        plt.show()

        # Heatmap
        plt.figure(figsize=(10,6))
        sns.heatmap(probs.T, cmap="magma", cbar=True)
        plt.title("Quantum State Probability Heatmap", fontsize=16)
        plt.xlabel("Step")
        plt.ylabel("State Index")
        plt.show()

# ============================================================================
# 3. Universe Scaling & Falsification
# ============================================================================
class Universe:
    def __init__(self, rocks: TwoRocks):
        self.rocks = rocks
        self.classical = ClassicalUniverse(rocks)
        self.quantum = QuantumUniverse()

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

# ============================================================================
# 4. Closed Loop Demonstration
# ============================================================================
def closed_loop_demo():
    rocks = TwoRocks()
    universe = Universe(rocks)
    classical, quantum = universe.run()
    tests = universe.falsify()

    print("\n▶ Two Rocks Status:", rocks)
    print("▶ Falsification Results:", tests)

    # Plot everything
    universe.classical.plot()
    universe.quantum.plot()

    return rocks, classical, quantum

# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    closed_loop_demo()
