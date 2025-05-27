import numpy as np
import matplotlib.pyplot as plt
########################################################
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
########################################################

spacing = 0.1

def phi(x):
    # phi is a function whose mean we want
    return 0.2 * (x+5.0)
    #return np.exp(-x**2 / 2) / np.sqrt(2 * np.pi) # Standard normal density

def density(pstar):
    # Generate data
    x = np.arange(-5, 5 + spacing, spacing)
    y = np.column_stack((x, pstar(x)))
    xphi = np.column_stack((x, phi(x)))


    # Figure 1: P*(x) and φ(x)
    plt.figure()
    plt.plot(y[:, 0], y[:, 1], 'b-', label='P*(x)')
    plt.plot(xphi[:, 0], xphi[:, 1], 'r-', label='φ(x)')
    plt.title('P*(x) and φ(x)')
    plt.xlabel('x')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()

def distribution(pstar):
    x = np.arange(-5, 5.2, 0.2)  # Include endpoint as in Octave
    y = pstar(x)
    plt.figure()
    plt.plot(x, y, 'o-', label='P*(x)')  # 'o-' for point + line (mimics impulses)
    plt.xlabel('x')
    plt.ylabel('P*(x)')
    plt.title('Distribution P*(x)')
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
    return h * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

def metplota(r, xt, pt, xprop, pprop, xQ, y, xQs, pausing):
    plt.ion()
    plt.figure(1)
    plt.clf()
    plt.ylim([-0.4, 3.1])
    plt.xlim([-5, 5])
    plt.title(f"{r - 1} samples" if r >= 1 else "")

    plt.plot(y[:, 0], y[:, 1], "b-", label="P*(x)")
    plt.plot(xQ[:, 0], xQ[:, 1], "g-", label="Q*(x'; x_t)")
    plt.axvline(xt, color='k', linestyle='--', label="xt")
    plt.plot(xt, pt, "ro", label="xt, pt")
    plt.axvline(xprop, color='m', linestyle='--', label="xprop")
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
        input("Press Enter to continue...")


# def metplot(xQs):
#     xs = xQs[:, 0]
#     ys = xQs[:, 1]
#     plt.figure()
#     plt.plot(xs, ys, 'o-', label='Sampled φ(x)')
#     plt.xlabel('x')
#     plt.ylabel('φ(x)')
#     plt.title('Metropolis Samples')
#     plt.grid(True)
#     plt.legend()
#     plt.show()

def metropolis(pausing, xmin, xmax, mu, sigma, R, plotting, pstar):
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

        if plotting > 1 and (r < 50 or (r < 60 and r % 10 == 0) or r % 20 == 0):
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


        if plotting > 1 and r < 50:
            metplota(r, xt, pt, xprop, pprop, xQ, y, xQs, pausing)

    xQs = np.array(xQs)

    if plotting == 1:
        metplota(R, xt, pt, xprop, pprop, xQ, y, xQs, False)

    print("Done")
    plt.pause(-1)
    return xQs


if __name__=="__main__":
    def pstar(x):
        return np.exp(0.4 * ((x - 0.4)**2) - 0.08 * (x**4))
        #return np.exp(1.15 * np.sin(6.0 * x) - 0.3 * ((x - 0.65)**2) - 0.01 * (x**4))

    #density(pstar)
    #distribution(pstar)

    pausing = False
    pausing = True
    xmin,xmax = -15,15
    mu = 0
    sigma = 1.5
    R=100        # number of samples
    plotting = 1 # everything at once
    plotting = 2 # incremental updates
    metropolis(pausing, xmin, xmax, mu, sigma, R, plotting, pstar)



