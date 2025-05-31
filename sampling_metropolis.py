############################################################################################
#  Markov Chain Monte Carlo - Metropolis Method
#
#  References
#
#   * Chapter 29 in "Information Theory, Inference, and Learning Algorithms", David MacKay
#     https://www.inference.org.uk/itprnn/book.html
#   * Lecture Video 12 @ 53:50 (Metropolis)
#     https://www.inference.org.uk/itprnn_lectures/09_mackay.mp4
#   * Old octave demo:
#     https://www.inference.org.uk/mackay/itprnn/code/mcmc/
#
############################################################################################

import numpy as np
import matplotlib.pyplot as plt
import argparse

def pstar1(x):
    return np.exp(0.4 * ((x - 0.4) ** 2) - 0.08 * (x**4))

def pstar2(x):
    return np.exp(1.15 * np.sin(6.0 * x) - 0.3 * ((x - 0.65) ** 2) - 0.01 * (x**4))


def phi(x):
    # phi is a function whose mean we want
    return 0.2 * (x + 5.0)
    # return np.exp(-x**2 / 2) / np.sqrt(2 * np.pi) # Standard normal density


def density(pstar):
    # Generate data
    spacing = 0.1
    x = np.arange(-5, 5 + spacing, spacing)
    y = np.column_stack((x, pstar(x)))
    xphi = np.column_stack((x, phi(x)))

    # Figure 1: P*(x) and φ(x)
    plt.figure()
    plt.plot(y[:, 0], y[:, 1], "b-", label="P*(x)")
    plt.plot(xphi[:, 0], xphi[:, 1], "r-", label="φ(x)")
    plt.title("P*(x) and φ(x)")
    plt.xlabel("x")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.show()


def distribution(pstar):
    x = np.arange(-5, 5.2, 0.2)  # Include endpoint as in Octave
    y = pstar(x)
    plt.figure()
    plt.plot(x, y, "o-", label="P*(x)")  # 'o-' for point + line (mimics impulses)
    plt.xlabel("x")
    plt.ylabel("P*(x)")
    plt.title("Distribution P*(x)")
    plt.xlim([-5, 5])
    plt.ylim([0, 3.1])
    plt.legend()
    plt.grid(True)
    plt.show()


def Qnorm(x, mu, sigma, h):
    """
    Qnorm: a normal (Gaussian) density function with peak height h

    Parameters:
        x (array-like): input values
        mu (float): mean of the distribution
        sigma (float): standard deviation
        h (float): peak height

    Returns:
        ndarray: density values at x
    """
    x = np.asarray(x)
    return h * np.exp(-((x - mu) ** 2) / (2 * sigma**2))


def metplota(r, xt, pt, xprop, pprop, xQ, y, xQs, pausing):
    plt.ion()
    plt.figure(1)
    plt.clf()
    plt.ylim([-0.4, 3.1])
    plt.xlim([-5, 5])
    plt.title(f"{r - 1} samples" if r >= 1 else "")

    plt.plot(y[:, 0], y[:, 1], "b-", label="P*(x)")
    plt.plot(xQ[:, 0], xQ[:, 1], "g-", label="Q*(x'; x_t)")
    plt.axvline(xt, color="k", linestyle="--", label="xt")
    plt.plot(xt, pt, "ro", label="xt, pt")
    plt.axvline(xprop, color="m", linestyle="--", label="xprop")
    plt.plot([xt, xprop], [pt, pprop], "co-", label="Current to Proposal")

    if r >= 2:
        x_data = np.array(xQs)
        plt.plot(x_data[:, 0], x_data[:, 4], "mo", label="Accepted Samples")

    plt.xlabel("x")
    plt.ylabel("Probability")
    plt.title("Metropolis Sampling Step")
    plt.legend()
    plt.grid(True)
    plt.draw()
    plt.pause(0.001)

    if pausing:
        input(f"Step {r} - Press Enter to continue...")


def metropolis(pausing, xmin, xmax, mu, sigma, R, pstar):
    defheight = -0.25
    inc = 0.05
    high = 2.0
    topline = 3
    botline = 0.1

    gxmax = 5
    gxmin = -5
    grange = gxmax - gxmin

    x_range = np.linspace(-5, 5, 200)
    y_vals = pstar(x_range)
    phi_vals = phi(x_range)

    y = np.column_stack((x_range, y_vals))
    xphi = np.column_stack((x_range, phi_vals))

    xQs = []
    rej = 0
    a = 0  # accept count
    sumphi = 0.0
    xt = mu
    pt = pstar(xt)
    height = defheight

    for r in range(1, R + 1):
        xq_range = np.linspace(xmin, xmax, 200)
        xQ_vals = Qnorm(xq_range, xt, sigma, high)
        xQ = np.column_stack((xq_range, xQ_vals))

        # Propose new sample
        xprop = np.random.randn() * sigma + xt
        pprop = pstar(xprop)

        if pausing and (r < 50 or (r < 60 and r % 10 == 0) or r % 20 == 0):
            metplota(r, xt, pt, xprop, pprop, xQ, y, xQs, pausing)

        # Metropolis acceptance
        if pprop > pt or np.random.rand() < (pprop / pt):
            a += 1
            xt = xprop
            pt = pprop
            height = defheight
        else:
            rej += 1
            height += inc

        ph = phi(xt)
        sumphi += ph
        xQs.append([xt, ph, r, a, height, sumphi / r])

        if pausing and r < 50:
            metplota(r, xt, pt, xprop, pprop, xQ, y, xQs, pausing)

    xQs = np.array(xQs)

    if not pausing:
        metplota(R, xt, pt, xprop, pprop, xQ, y, xQs, False)

    print("Done")

    acceptance_rate = a / R
    average_phi = sumphi / R
    print(f"Acceptance rate: {acceptance_rate:.3f}")
    print(f"Estimated <φ>: {average_phi:.6f}")
    plt.show()

    return xQs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Metropolis MCMC demo")
    parser.add_argument(
        "--steps", type=int, default=100, help="Number of sampling steps"
    )
    parser.add_argument(
        "--sigma", type=float, default=1.5, help="Proposal distribution stddev"
    )
    parser.add_argument("--fast", action="store_true", help="No pause between steps")
    parser.add_argument("--pstar2", action="store_true", help="density pstar2")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    pstar = pstar2 if args.pstar2 else pstar1
    # density(pstar)
    # distribution(pstar)

    metropolis(
        pausing=not args.fast,
        xmin=-15,
        xmax=15,
        mu=0.0,
        sigma=args.sigma,
        R=args.steps,
        pstar=pstar,
    )
