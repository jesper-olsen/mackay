###########################################################################################
#  Markov Chain Monte Carlo - Gibbs Sampling
#
#  References
#
#   * Chapter 29 in "Information Theory, Inference, and Learning Algorithms", David MacKay
#     https://www.inference.org.uk/itprnn/book.html
#   * Lecture Video 12 @ 1:08:20
#     https://www.inference.org.uk/itprnn_lectures/12_mackay.mp4
#   * Old octave demo (DEMOskye):
#     https://www.inference.org.uk/mackay/itprnn/code/gibbsdemo/
#
###########################################################################################


import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.stats import gamma


def inverse_gamma_sample(N, a):
    beta = gamma_sample_scaled(0.5 * (N - 3.0), a)
    return np.sqrt(1.0 / beta)


def gamma_sample_scaled(N, a):
    return gamma_sample(N) / a


def gamma_sample(a):
    return gamma.rvs(a + 1, scale=1)


def compute_posterior(mu, sigma, xbar, v, N, exponent):
    D = N * ((mu - xbar) ** 2 + v) / (2 * sigma**2)
    S = sigma**exponent
    return np.exp(-D) / S


def plot_contour(wlt, wl, zz):
    plt.clf()
    if zz is not None and len(zz) >= 3:
        x, y, z = zz
        plt.contour(x, y, z, 20)
    if wlt is not None:
        plt.plot(wlt[:, 0], wlt[:, 1], "b.-", label="Trajectory")
    if wl is not None:
        plt.plot(wl[:, 0], wl[:, 1], "ro", label="Logged Weights")
    plt.xlabel("mu")
    plt.ylabel("sigma")
    plt.legend()
    plt.show(block=False)


def step_plot(state, mu, sigma):
    if state["doplot"]:
        plot_contour(state["wlt"], state["wl"], state["zz"])
        ans = input(f"{state['T']}: ready? (0 to skip plotting): ")
        if ans == "0":
            state["doplot"] = False


def gibbs_sampler(state):
    sN = np.sqrt(state["N"])
    mu, sigma = state["mu"], state["sigma"]

    for _ in range(state["L"]):
        step_plot(state, mu, sigma)

        mu = state["xbar"] + np.random.randn() * sigma / sN
        if state["wltlog"]:
            state["wlt"][state["ltt"], :] = [mu, sigma, -0.5, state["ltt"]]
            state["ltt"] += 1

        step_plot(state, mu, sigma)

        sigma = inverse_gamma_sample(
            state["N"] + 1, 0.5 * state["N"] * ((mu - state["xbar"]) ** 2 + state["v"])
        )

        if state["wltlog"]:
            state["wlt"][state["ltt"], :] = [mu, sigma, -0.5, state["ltt"]]
            state["ltt"] += 1

        state["T"] += 1
        state["wl"][state["logtime"], :] = [mu, sigma, -0.5, state["T"]]
        state["logtime"] += 1

        if state["verbose"]:
            print(f"[mu, sigma] = [{mu}, {sigma}]")

    return mu, sigma


def demo_skye():
    parser = argparse.ArgumentParser(description="MacKay-style Gibbs Sampling Demo")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)


    # Configuration and logging
    state = {
        "mu": 0.1,
        "sigma": 1.0,
        "xbar": 1.0,
        "v": 0.2,
        "N": 5,
        "L": 30,
        "T": 0,
        "logtime": 0,
        "ltt": 0,
        "wl": np.full((70, 4), np.nan),  # weight vector log
        "wlt": np.full((100, 4), np.nan),  # trajectory log
        "wltlog": True,
        "verbose": False,
        "doplot": True,
        "zz": None,
    }

    # Posterior surface setup
    xmin, xmax, dx = 0, 2.0, 0.05
    smin, smax, dls = 0.18, 1.8, 0.1
    x = np.arange(xmin, xmax + dx, dx)
    ls = np.arange(np.log(smin), np.log(smax) + dls, dls)
    s = np.exp(ls)
    X, S = np.meshgrid(x, s)
    D = np.outer(
        np.ones(len(s)), state["N"] * ((x - state["xbar"]) ** 2 + state["v"])
    ) / (2 * np.outer(s**2, np.ones(len(x))))
    exponent = state["N"]
    Z = np.exp(-D) / np.outer(s**exponent, np.ones(len(x)))

    fig = plt.figure(1)
    plt.clf()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, S, Z, cmap="viridis", edgecolor="none", shade=True)
    ax.set_xlabel(r"$\mu$")
    ax.set_ylabel(r"$\sigma$")
    ax.set_zlabel(r"$P(\mu, \sigma)$")
    ax.set_title("Joint Posterior Density")
    ax.view_init(30, 60)
    plt.grid(True)
    plt.show(block=False)

    input("Ready for contour plot? ")

    plt.figure(2)
    plt.clf()
    plt.contour(X, S, Z, 10)
    plt.xlabel(r"$\mu$")
    plt.ylabel(r"$\sigma$")
    plt.title("Posterior Contours")
    plt.tight_layout()
    plt.show(block=False)

    state["zz"] = (X, S, Z)

    # Gibbs sampling
    mu, sigma = gibbs_sampler(state)

    # Final plot
    fig = plt.figure(3)
    plt.clf()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, S, Z, cmap="viridis", edgecolor="none", alpha=0.6)
    ax.set_xlabel(r"$\mu$")
    ax.set_ylabel(r"$\sigma$")
    ax.set_zlabel(r"$P(\mu, \sigma)$")
    ax.set_title("Gibbs Samples on Posterior")
    ax.view_init(30, 60)

    wl = state["wl"][~np.isnan(state["wl"][:, 0])]
    wlt = state["wlt"][~np.isnan(state["wlt"][:, 0])]
    exponent = state["N"]

    z_samples = compute_posterior(
        wl[:, 0], wl[:, 1], state["xbar"], state["v"], state["N"], exponent
    )
    ax.plot3D(wl[:, 0], wl[:, 1], z_samples, "gx", label="Samples")

    if wlt.shape[0] > 0:
        ax.plot3D(
            wlt[: state["ltt"], 0],
            wlt[: state["ltt"], 1],
            wlt[: state["ltt"], 2],
            "ro",
            label="Trajectory",
        )

    ax.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    demo_skye()
