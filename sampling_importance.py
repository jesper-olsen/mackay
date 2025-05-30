############################################################################################
#  Markov Chain Monte Carlo - Importance Sampling
#
#  References
#
#   * Chapter 29 in "Information Theory, Inference, and Learning Algorithms", David MacKay
#     https://www.inference.org.uk/itprnn/book.html
#   * Lecture Video 12 (start)
#     https://www.inference.org.uk/itprnn_lectures/09_mackay.mp4
#   * Old octave demo:
#     https://www.inference.org.uk/mackay/itprnn/code/mcmc/
#
############################################################################################

import numpy as np
import matplotlib.pyplot as plt
import argparse
from collections import namedtuple

# Constants for plotting and sampling
GRIDX_MIN, GRIDX_MAX = -5, 5
GRIDX_RANGE = GRIDX_MAX - GRIDX_MIN
PHI_SCALE = 0.2
UNIFORM_HIGH = 1.0

SampleData = namedtuple("SampleData", [
    "step", "x_scaled", "w", "wphi", "estimate", "sumwphi", "sumw"
])

def phi(x):
    """Feature function phi(x) = 0.2 * (x + 5)."""
    return PHI_SCALE * (x + 5.0)

def Q_uniform(x, high=UNIFORM_HIGH):
    return high * np.ones_like(x)

def Q_normal(x, mu, sigma, scale=1.0):
    return scale * np.exp(-((x - mu) ** 2) / (2 * sigma**2))

def pstar1(x):
    return np.exp(0.4 * ((x - 0.4) ** 2) - 0.08 * (x**4))

def pstar2(x):
    return np.exp(1.15 * np.sin(6.0 * x) - 0.3 * ((x - 0.65) ** 2) - 0.01 * (x**4))

def setup_reference_distributions(pstar_fn, spacing=0.01):
    x = np.arange(GRIDX_MIN, GRIDX_MAX + spacing, spacing)
    y = np.column_stack((x, pstar_fn(x)))
    xphi = np.column_stack((x, phi(x)))
    return x, y, xphi

class ImportancePlotter:
    def __init__(self):
        self.fig, self.axs = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [1, 2]})
        plt.ion()

    def update(self, sampling_method, y, xphi, xQ, xQs, ws, R):
        self.axs[0].clear()
        self.axs[1].clear()

        self.axs[0].set_title(f"{int(ws[-1].step)} samples")
        self.axs[0].plot([w.step for w in ws], [w.estimate for w in ws], color="steelblue", label="Estimate of ⟨φ(x)⟩")
        self.axs[0].set_ylabel("⟨φ(x)⟩")
        self.axs[0].set_xlim([0, R])
        self.axs[0].legend()
        self.axs[0].grid(True)

        self.axs[1].plot(y[:, 0], y[:, 1], "b-", label="Target P*(x)")
        self.axs[1].plot(xphi[:, 0], xphi[:, 1], "m-", label="φ(x)")
        self.axs[1].plot(np.array(xQs)[:, 0], np.array(xQs)[:, 1], "g.", alpha=0.5, label="Samples")
        if sampling_method == 2 and xQ is not None:
            self.axs[1].plot(xQ[:, 0], xQ[:, 1], color="purple", label="Proposal Q*(x)")
        self.axs[1].legend()
        self.axs[1].set_xlabel("x")
        self.axs[1].set_ylabel("Value")
        self.axs[1].grid(True)

        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)
        input("Press Return to continue...")

    def finalize(self):
        plt.ioff()
        plt.show()

def importance_sampling(pause, sampling_method, xmin, xmax, mu, sigma, R, pstar_fn):
    spacing = 0.01
    x, y, xphi = setup_reference_distributions(pstar_fn, spacing)

    q_funcs = {
        1: lambda x: Q_uniform(x),
        2: lambda x: Q_normal(x, mu, sigma, 2.0),
    }
    sample_funcs = {
        1: lambda: np.random.uniform(xmin, xmax),
        2: lambda: np.random.normal(mu, sigma),
    }

    q_fn = q_funcs[sampling_method]
    sample_fn = sample_funcs[sampling_method]
    xQ = np.column_stack((x, q_fn(x))) if sampling_method == 2 else None

    plotter = ImportancePlotter() if pause else None
    xQs = []
    ws = []
    sumwphi = sumw = 0

    for r in range(R):
        x_sample = sample_fn()
        if sampling_method == 1 and not (xmin <= x_sample <= xmax):
            continue

        q = q_fn(np.array([x_sample])).item() # python scalar
        w = pstar_fn(x_sample) / q
        phi_val = phi(x_sample)
        wphi = w * phi_val
        sumwphi += wphi
        sumw += w

        xQs.append([x_sample, phi_val])
        estimate = sumwphi / sumw
        x_scaled = ((r / R) * GRIDX_RANGE) + GRIDX_MIN
        ws.append(SampleData(r, x_scaled, w, wphi, estimate, sumwphi, sumw))

        if pause and (r < 5 or (r < 60 and r % 10 == 0) or r % 20 == 0):
            plotter.update(sampling_method, y, xphi, xQ, xQs, ws, R)

    if not pause:
        plt.figure(figsize=(10, 4))
        plt.plot([w.step for w in ws], [w.estimate for w in ws], label="Estimate of ⟨φ(x)⟩")
        plt.title(f"Importance Sampling Estimate ({R} steps)")
        plt.xlabel("Sample")
        plt.ylabel("⟨φ(x)⟩")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        plotter.update(sampling_method, y, xphi, xQ, xQs, ws, R)
        print("Done")
        plotter.finalize()

    return sumwphi / sumw

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MacKay-style Importance Sampling Demo")
    parser.add_argument("--steps", type=int, default=50, help="Number of sampling steps")
    parser.add_argument("--sigma", type=float, default=1.0, help="Gaussian proposal stddev")
    parser.add_argument("--method", type=int, default=1, help="Sampling method: 1=Uniform, 2=Gaussian")
    parser.add_argument("--fast", action="store_true", help="Run without pause (faster)")
    parser.add_argument("--pstar2", action="store_true", help="Use second target distribution")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    pstar_fn = pstar2 if args.pstar2 else pstar1

    result = importance_sampling(
        pause=not args.fast,
        sampling_method=args.method,
        xmin=GRIDX_MIN,
        xmax=GRIDX_MAX,
        mu=0,
        sigma=args.sigma,
        R=args.steps,
        pstar_fn=pstar_fn,
    )

    print(f"\nEstimated ⟨φ(x)⟩: {result:.6f}")
