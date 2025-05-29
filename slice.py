############################################################################################
#  Markov Chain Monte Carlo - Slice Sampling
#
#  References
#
#   * Chapter 29 in "Information Theory, Inference, and Learning Algorithms", David MacKay
#     https://www.inference.org.uk/itprnn/book.html
#   * Lecture Video 13 
#     https://www.inference.org.uk/itprnn_lectures/13_mackay.mp4
#   * Old octave demo:
#     https://www.inference.org.uk/mackay/itprnn/code/mcmc/
#
############################################################################################

import numpy as np
import matplotlib.pyplot as plt
import argparse

import matplotlib.pyplot as plt
import numpy as np


def slplota(xQs, y, xt, pt, u, r, pausing):
    plt.figure("slplota")
    plt.clf()

    plt.plot(y[:, 0], y[:, 1], "b-", label="P*(x)")

    if r >= 2:
        plt.plot(xQs[:, 0], xQs[:, 4], "ro", markersize=4, label="Previous Samples")

    plt.plot([xt, xt], [0, pt], "k-", linewidth=1.5, label="x(t)")
    plt.axhline(y=u, color="m", linestyle="--", label="Slice level")

    plt.xlim(np.min(y[:, 0]), np.max(y[:, 0]))
    plt.ylim(0, np.max(y[:, 1]) * 1.1)
    plt.title(f"{r} samples")
    plt.legend()
    plt.grid(True)

    plt.show(block=False)
    plt.pause(0.001)

    if pausing:
        input("Sample uniformly... (press Enter)")


def slplotb(xQs, r, y, tele1, tele2, tele3):
    plt.figure("slplotb")
    plt.clf()

    # Plot the target density
    plt.plot(y[:, 0], y[:, 1], "b-", label="P*(x)")

    # Plot previous samples
    if r >= 2:
        plt.plot(xQs[:, 0], xQs[:, 4], "mo", label="samples")

    # Vertical line at current sample (tele1)
    plt.plot(tele1[:, 0], tele1[:, 1], "k-", linewidth=1.5, label="x(t)")

    # Horizontal slice level (tele2)
    plt.plot(tele2[:, 0], tele2[:, 1], "g--", linewidth=1.5, label="u level")

    # Slice interval extensions (tele3)
    plt.plot(tele3[:, 0], tele3[:, 1], "r-.", linewidth=1.5, label="interval")

    plt.title(f"{r} samples")
    plt.xlabel("x")
    plt.ylabel("P*(x)")
    plt.legend()
    plt.grid(True)

    plt.show(block=False)
    plt.pause(0.001)


def slplotc(
    xQs, y, xt, pt, u, xnew, xleft, xright, pprop, r, pausing, specialprinting, logscale
):
    plt.figure("slplotc")
    plt.clf()

    # Axes limits depending on flags
    if logscale:
        plt.yscale("log")
        plt.ylim(1e-4, 3.1)
    else:
        plt.yscale("linear")
        plt.ylim(-0.2, 3.1)

    plt.xlim(-5, 5)
    if specialprinting:
        plt.xlim(-4, 4)
        plt.ylim(-0.2, 1.75)

    # Plot P*(x)
    plt.plot(y[:, 0], y[:, 1], "b-", label="P*(x)")

    # Previous samples
    if r >= 2:
        plt.scatter(xQs[:, 0], xQs[:, 4], s=20, c="black", label="Previous Samples")

    # Vertical line at xt
    plt.plot([xt, xt], [0, pt], "k-", label="x(t)")

    # Slice level (tele2) and interval (tele3)
    plt.plot([xleft, xright], [u, u], "--", color="gray", label="slice level")
    plt.plot([xleft, xright], [u, u], "g-", label="interval")

    # Proposed sample (tele4)
    plt.scatter([xnew], [u], c="red", s=50, label="x prop")

    plt.xlabel("x")
    plt.ylabel("P*(x)")
    plt.title(f"Slice sampling iteration {r}")
    plt.legend()
    plt.grid(True)

    plt.show(block=False)
    plt.pause(0.001)

    if pausing:
        input("Press Enter to continue...")

def phi(x):
    return 0.2 * (x + 5.0)


def iranges(logscale=False, specialprinting=False):
    plt.clf()
    plt.grid(True)
    plt.xlim([-5, 5])
    if logscale:
        plt.ylim([1e-4, 3.1])
        plt.yscale("log")
    else:
        plt.ylim([-0.2, 3.1])

    if specialprinting:
        plt.xlim([-4, 4])
        plt.ylim([-0.2, 1.75])


def slplotd(xQs):
    plt.clf()
    iranges(logscale=False, specialprinting=False)

    xplot = np.linspace(-5, 5, 200)
    yplot = pstar(xplot)
    plt.plot(xplot, yplot, label="P*(x)", linewidth=1.5)

    if len(xQs) > 0:
        plt.scatter(xQs[:, 0], xQs[:, 6], s=20, label="Samples")
        plt.title(f"{len(xQs)} samples")
    else:
        plt.title("No samples")

    plt.legend()
    plt.show()


def slice(pausing, xmin, xmax, mu, sigma, R, plotting, logscale, pstar):
    defheight = -0.1
    xQs = []
    sumphi = 0.0

    xt = mu
    pt = pstar(xt)

    xplot = np.linspace(xmin, xmax, 200)
    yplot = pstar(xplot)
    y = np.column_stack((xplot, yplot))

    for r in range(1, R + 1):
        u = np.random.rand() * pt
        xprop = (np.random.rand() - 0.5) * sigma + xt
        xleft = xprop - 0.5 * sigma
        xright = xprop + 0.5 * sigma
        pleft = pstar(xleft)
        pright = pstar(xright)

        # Expand interval until ends are below slice height
        while (pleft > u) or (pright > u):
            if pleft > u:
                xleft -= sigma
                pleft = pstar(xleft)
            if pright > u:
                xright += sigma
                pright = pstar(xright)

        # Sample within the slice
        pprop = 0.0
        while pprop < u:
            xnew = xleft + np.random.rand() * (xright - xleft)
            pprop = pstar(xnew)
            if pprop < u:
                if xnew < xt:
                    xleft = xnew
                else:
                    xright = xnew

        xt = xnew
        pt = pprop

        ph = phi(xt)
        sumphi += ph
        xQs.append([xt, ph, r, 0, defheight, sumphi / r, u])

        if plotting == 1:
            slplota(np.array(xQs), y, xt, pt, u, r, pausing)
        elif plotting == 2:
            # Prepare tele1, tele2, tele3 for slplotb
            tele1 = np.array([[xt, 0], [xt, pt]])
            tele2 = np.array([[np.min(y[:, 0]), u], [np.max(y[:, 0]), u]])
            tele3 = np.array([[xleft, u], [xright, u]])
            slplotb(np.array(xQs), r, y, tele1, tele2, tele3)
        elif plotting == 3:
            slplotc(
                np.array(xQs),
                y,
                xt,
                pt,
                u,
                xnew,
                xleft,
                xright,
                pprop,
                r,
                pausing,
                False,
                logscale,
            )

        if pausing:
            input("Press Enter to continue...")

    xQs = np.array(xQs)

    # if plotting:
    input("summary...")
    slplotd(xQs)
    plt.ioff()  # Turn off interactive mode
    plt.show(block=True)  # Show and wait until the window is closed
    return xQs


if __name__ == "__main__":

    def pstar1(x):
        return np.exp(0.4 * ((x - 0.4) ** 2) - 0.08 * (x**4))

    def pstar2(x):
        return np.exp(1.15 * np.sin(6.0 * x) - 0.3 * ((x - 0.65) ** 2) - 0.01 * (x**4))

    plt.ion()  # Enable interactive mode globally
    parser = argparse.ArgumentParser(description="Metropolis MCMC demo")
    parser.add_argument("--steps", type=int, default=5, help="Number of sampling steps")
    parser.add_argument(
        "--sigma", type=float, default=1.5, help="Proposal distribution stddev"
    )
    parser.add_argument("--fast", action="store_true", help="No pause between steps")
    parser.add_argument("--logscale", action="store_true", help="Use logscale")
    parser.add_argument("--pstar2", action="store_true", help="density pstar2")
    args = parser.parse_args()
    pstar = pstar2 if args.pstar2 else pstar1
    logscale = False
    print(f"Slice sampling ({args.steps})")
    slice(
        not args.fast,
        xmin=-15,
        xmax=15,
        mu=-0.9,
        sigma=args.sigma,
        R=args.steps,
        plotting=3,
        logscale=args.logscale,
        pstar=pstar,
    )
