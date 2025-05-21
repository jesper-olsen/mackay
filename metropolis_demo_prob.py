import numpy as np
import matplotlib.pyplot as plt
import itertools

n_states = 21
initial_positions = [10, 17]

# Create the transition matrix
P = np.zeros((n_states, n_states))
for i in range(n_states):
    if i > 0:
        P[i, i - 1] += 0.5
    if i < n_states - 1:
        P[i, i + 1] += 0.5
P[0, 0] = 0.5
P[n_states - 1, n_states - 1] = 0.5

# Initialize distributions for each initial position
distributions = []
for pos in initial_positions:
    dist = np.zeros(n_states)
    dist[pos] = 1.0
    distributions.append(dist)

positions = np.arange(n_states)

# Manual step-by-step visualization
for t in itertools.count():
    plt.clf()
    for i, dist in enumerate(distributions):
        plt.plot(positions, dist, label=f"Start at {initial_positions[i]}")
    plt.title(f"Time step t = {t}")
    plt.xlabel("Position")
    plt.ylabel("Probability")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.ylim(bottom=0, top=max(np.max(dist) for dist in distributions) * 1.1)  # dynamic scaling
    plt.pause(0.01)  # needed for plot to update
    input("Press Enter for next step...")

    # Update distributions
    for i in range(len(distributions)):
        distributions[i] = distributions[i] @ P
