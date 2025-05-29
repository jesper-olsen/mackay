############################################################################################
#  Markov Chain Monte Carlo - Rejection Sampling
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

# Global configuration
spacing = 0.05


def pstar1(x):
    return np.exp(0.4 * (x - 0.4) ** 2 - 0.08 * x ** 4)

def pstar2(x):
    return np.exp(1.15 * np.sin(6.0 * x) - 0.3 * (x - 0.65) ** 2 - 0.01 * x ** 4)


def phi(x):
    return 0.2 * (x + 5.0)


def Qnorm(x, mu, sigma, h):
    return h * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))


def standard():
    global spacing
    x = np.arange(-5, 5 + spacing, spacing)
    y = np.column_stack((x, pstar(x)))
    xphi = np.column_stack((x, phi(x)))
    return x, y, xphi


def rejplot0(y, xQ):
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

def rejplot(y, xQ, xQa, xQr, a, rej, r, pstar):
    plt.clf()
    plt.ylim([-0.2, 4.1])
    plt.xlim([-5, 5])

    if r >= 1:
        plt.title(f"{r} samples, {a} accepted")

    if rej >= 1 and a >= 1:
        plt.plot(y[:, 0], y[:, 1], 'b-', label='P*(x)')
        plt.plot(xQ[:, 0], xQ[:, 1], 'r-', label='cQ*(x)')
        if xQa.ndim == 2 and xQa.shape[0] > 0:
            plt.plot(xQa[:, 0], xQa[:, 1], 'g.', label='')
            plt.plot(xQa[:, 0], xQa[:, 6], 'g*', label='accepted')
        if xQr.ndim == 2 and xQr.shape[0] > 0:
            plt.plot(xQr[:, 0], xQr[:, 1], 'mo', label='rejected')
    elif rej >= 1:
        plt.plot(y[:, 0], y[:, 1], 'b-', label='P*(x)')
        plt.plot(xQ[:, 0], xQ[:, 1], 'r-', label='cQ*(x)')
        if xQr.ndim == 2 and xQr.shape[0] > 0:
            plt.plot(xQr[:, 0], xQr[:, 1], 'mo', label='rejected')
    elif a >= 1:
        plt.plot(y[:, 0], y[:, 1], 'b-', label='P*(x)')
        plt.plot(xQ[:, 0], xQ[:, 1], 'r-', label='cQ*(x)')
        if xQa.ndim == 2 and xQa.shape[0] > 0:
            plt.plot(xQa[:, 0], xQa[:, 1], 'g.', label='')
            plt.plot(xQa[:, 0], xQa[:, 6], 'g*', label='accepted')
    else:
        print("starting rejplot...")

    plt.legend()
    plt.xlabel('x')
    plt.ylabel('density')
    plt.grid(True)
    plt.draw()
    plt.pause(0.01)
    input("Press Enter to continue...")


def rejection(style, xmin, xmax, mu, sigma, R, plotting, pstar):
    height = -0.1
    high = 4.0

    _, y, _ = standard()
    x_vals = np.linspace(xmin, xmax, 300)
    xQ = np.column_stack((x_vals, Qnorm(x_vals, mu, sigma, high)))

    xQa = []  # accepted
    xQr = []  # rejected
    a = 0
    rej = 0

    if plotting >= 1:
        plt.ion() # interactive mode
        rejplot0(y, xQ)
        if plotting >= 2:
            input('press return to continue...')
        else:
            print(f"generating {R} now")

    sumphi = 0.0

    for r in range(1, R + 1):
        x = xmax + 1.0
        while x > xmax or x < xmin:
            x = np.random.randn() * sigma + mu
            q = Qnorm(x, mu, sigma, high)
            u = np.random.rand() * q

        w = pstar(x)
        if u > w:
            rej += 1
            xQr.append([x, phi(x), u, rej, w])
        else:
            ph = phi(x)
            a += 1
            sumphi += ph
            estimate = sumphi / a
            xQa.append([x, ph, u, a, w, estimate, height, r])

        if plotting > 1 and (r < 10 or (r < 60 and r % 10 == 0) or r % 20 == 0):
            rejplot(y, xQ, np.array(xQa), np.array(xQr), r, a, rej, pstar)

    if plotting == 1:
        rejplot(y, xQ, np.array(xQa), np.array(xQr), R, a, rej, pstar)
        plt.ioff() # interactive mode off

    if a > 0:
        if plotting >= 1:
            input("see estimate (press return)")
            xQa_arr = np.array(xQa)
            plt.plot(xQa_arr[:, 3], xQa_arr[:, 5], label='estimate of <phi>', linewidth=1.5)
            plt.legend()
            plt.xlabel('Sample #')
            plt.ylabel('Estimate of <φ>')
            plt.title('Running Estimate from Accepted Samples')
            plt.grid(True)
            plt.show()

        if plotting == 2:
            input("see x,y again (press return)")
            rejplot(y, xQ, np.array(xQa), np.array(xQr), R, a, rej, pstar)

    return np.array(xQa) if a > 0 else np.array([])

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Rejection sampling MCMC demo")
    parser.add_argument(
        "--steps", type=int, default=50, help="Number of sampling steps"
    )
    parser.add_argument(
        "--sigma", type=float, default=3.0, help="Proposal distribution stddev"
    )
    parser.add_argument("--fast", action="store_true", help="No pause between steps")
    parser.add_argument("--pstar2", action="store_true", help="density pstar2")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
    pstar = pstar2 if args.pstar2 else pstar1


    xQa = rejection(
        style=None,     # Not used, but required for compatibility
        xmin=-15,
        xmax=15,
        mu=0,
        sigma=3.0,
        R=args.steps,          # Number of samples
        plotting=2,      # 0=no plot, 1=final plot, 2=interactive plotting
        pstar=pstar
    )

    if xQa.shape[0] > 0:
        final_estimate = xQa[-1, 6]
        print(f"Final estimate of <phi>: {final_estimate:.4f}")

