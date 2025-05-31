############################################################################################
#  Monte Carlo Methods - Rejection Sampling
#
#  References
#
#   * Chapter 29 in "Information Theory, Inference, and Learning Algorithms", David MacKay
#     https://www.inference.org.uk/itprnn/book.html
#   * Lecture Video 12 
#     https://www.inference.org.uk/itprnn_lectures/09_mackay.mp4
#   * Old octave demo:
#     https://www.inference.org.uk/mackay/itprnn/code/mcmc/
#
############################################################################################

import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.stats import norm


def pstar1(x):
    return np.exp(0.4 * (x - 0.4) ** 2 - 0.08 * x ** 4)


def pstar2(x):
    return np.exp(1.15 * np.sin(6.0 * x) - 0.3 * (x - 0.65) ** 2 - 0.01 * x ** 4)


def phi(x):
    return 0.2 * (x + 5.0)


def standard(pstar, spacing=0.05):
    x = np.arange(-5, 5 + spacing, spacing)
    y = np.column_stack((x, pstar(x)))
    return x, y


def plot_initial(y, xQ):
    plt.figure()
    plt.ylim([-0.2, 4.1])
    plt.xlim([-5, 5])
    plt.plot(y[:, 0], y[:, 1], 'b-', label='P*(x)', linewidth=1.5)
    plt.plot(xQ[:, 0], xQ[:, 1], 'r-', label='cQ*(x)', linewidth=1.5)
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('density')
    plt.title('Rejection Sampling Comparison')
    plt.grid(True)
    plt.show()


def pause(msg, enabled=True):
    if enabled:
        input(msg)


def plot_step(y, xQ, accepted, rejected, r, a, pause_enabled):
    plt.clf()
    plt.ylim([-0.2, 4.1])
    plt.xlim([-5, 5])
    plt.title(f"{r} samples, {a} accepted")

    plt.plot(y[:, 0], y[:, 1], 'b-', label='P*(x)')
    plt.plot(xQ[:, 0], xQ[:, 1], 'r-', label='cQ*(x)')

    if accepted.size > 0:
        plt.plot(accepted[:, 0], accepted[:, 1], 'g.', label='accepted raw')
        plt.plot(accepted[:, 0], accepted[:, 6], 'g*', label='accepted φ')

    if rejected.size > 0:
        plt.plot(rejected[:, 0], rejected[:, 1], 'mo', label='rejected')

    plt.legend()
    plt.xlabel('x')
    plt.ylabel('density')
    plt.grid(True)
    plt.draw()
    plt.pause(0.01)
    pause("Press Enter to continue...", pause_enabled)


def rejection_sampling(xmin, xmax, mu, sigma, R, plotting, pstar):
    high = 4.0
    height = -0.1

    x_vals = np.linspace(xmin, xmax, 300)
    q_pdf = lambda x: high * norm.pdf(x, mu, sigma)
    xQ = np.column_stack((x_vals, q_pdf(x_vals)))

    _, y = standard(pstar)
    accepted = []
    rejected = []
    a = 0
    rej = 0
    sum_phi = 0.0

    if plotting >= 1:
        plt.ion()
        plot_initial(y, xQ)
        pause("Press Enter to start sampling...", plotting > 1)

    for r in range(1, R + 1):
        # Sample from proposal until x in domain
        while True:
            x = np.random.randn() * sigma + mu
            if xmin <= x <= xmax:
                break

        q = q_pdf(x)
        u = np.random.rand() * q
        w = pstar(x)

        if u > w:
            rej += 1
            rejected.append([x, phi(x), u, rej, w])
        else:
            ph = phi(x)
            a += 1
            sum_phi += ph
            estimate = sum_phi / a
            accepted.append([x, ph, u, a, w, estimate, height, r])

        if plotting > 1 and (r < 10 or (r < 60 and r % 10 == 0) or r % 20 == 0):
            plot_step(y, xQ, np.array(accepted), np.array(rejected), r, a, True)

    if plotting >= 1:
        plot_step(y, xQ, np.array(accepted), np.array(rejected), R, a, False)
        plt.ioff()

    if a > 0 and plotting >= 1:
        pause("See estimate plot...", plotting > 1)
        accepted_arr = np.array(accepted)
        plt.figure()
        plt.plot(accepted_arr[:, 3], accepted_arr[:, 5], label='Estimate of <φ>', linewidth=1.5)
        plt.xlabel('Sample #')
        plt.ylabel('Estimate of <φ>')
        plt.title('Running Estimate from Accepted Samples')
        plt.grid(True)
        plt.legend()
        plt.show()

    return np.array(accepted) if a > 0 else np.array([]), a, rej


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MacKay-style Rejection Sampling Demo")
    parser.add_argument("--steps", type=int, default=50, help="Number of sampling steps")
    parser.add_argument("--sigma", type=float, default=3.0, help="Proposal stddev")
    parser.add_argument("--pstar2", action="store_true", help="Use pstar2 instead of pstar1")
    parser.add_argument("--plot", type=int, default=2, choices=[0, 1, 2], help="Plotting level: 0=none, 1=final only, 2=interactive")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    np.random.seed(args.seed)
    pstar = pstar2 if args.pstar2 else pstar1

    accepted, a, rej = rejection_sampling(
        xmin=-15,
        xmax=15,
        mu=0,
        sigma=args.sigma,
        R=args.steps,
        plotting=args.plot,
        pstar=pstar
    )

    if accepted.shape[0] > 0:
        final_estimate = accepted[-1, 6]
        print(f"\nFinal estimate of <φ>: {final_estimate:.4f}")
        print(f"Accepted: {a} / {args.steps} ({a / args.steps:.2%})")
    else:
        print("No samples accepted.")
