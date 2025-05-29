# Metropolis simulation of a system with 21 states.
# Target distribution is uniform.
# Proposal density is 0.5/0.5 over left/right states.
# Rejection at either end.
#
# T time steps, step size eps = 1
# => Variance <delta x^2> = sum_t = T
# Expected time to hit a wall: <x^2> = L^2 => T = (L/eps)^2
#
# Example from Video 13:
# https://videolectures.net/videos/mackay_course_13
# https://www.inference.org.uk/mackay/itprnn/code/metrop/
#
import random
import sys
import time
import matplotlib.pyplot as plt
import itertools

l, r = 0, 20
period = 20

x = (r - l) // 2
counters = [0] * (r - l + 1)

print(f"Pausing every {period} iterations - press enter to continue")
for i in itertools.count(1):
    for _ in range(period):
        x += 1 if random.random() > 0.5 else -1
        x = max(l, min(r, x))
        counters[x - l] += 1

        s1 = " " * (x - l)
        s2 = " " * (r - x)
        s3 = " Bonk!" if x == l or x == r else ""
        print(f"|{s1}*{s2}|{s3}")

    plt.clf()
    plt.bar(range(l, r + 1), counters, label="")
    plt.title(f"1D Random Walk Histogram at t={i*period}")
    plt.xlabel("Position")
    plt.ylabel("Visits")
    plt.pause(0.1)  # seconds

    s = f" at t={i * period} ".center(23, "-")
    input(s)
