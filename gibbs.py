import numpy as np
from scipy.stats import gamma
import matplotlib.pyplot as plt

def inversegamma(N, a):
    """
    inversegamma(N, a)
    creates a sample from 1/s^N * exp( - a/(s^2) )
    """
    beta = dgamma(0.5 * (N-3.0), a)
    sigma = np.sqrt(1.0 / beta)
    return sigma

def dgamma(N, a):
    """
    dgamma(N, a)
    creates a sample from b^N exp( - a b )
    """
    beta = sgamma(N)
    b = beta / a
    return b

def sgamma(a):
    """
    sgamma(a) ~ sample from b^a * exp(-b), i.e., Gamma(a+1, 1)
    """
    return gamma.rvs(a + 1, scale=1)

def gibbs(mu, sigma, L, xbar, v, N, 
          wl, wlt, wltlog, ltt, verbose, T, logtime, zz, doplot):
    sN = np.sqrt(N)

    wl[logtime, :] = [mu, sigma, -0.5, T]
    logtime += 1

    if wltlog:
        wlt[ltt, :] = [mu, sigma, -0.5, ltt]
        ltt += 1

    for l in range(L):
        if doplot:
            import matplotlib.pyplot as plt
            plt.clf()
            if zz is not None and len(zz) >= 3:
                x, y, z = zz[0], zz[1], zz[2]
                plt.contour(x, y, z, 20)

            if wlt is not None:
                plt.plot(wlt[:, 0], wlt[:, 1], 'b.-', label='Trajectory')
            if wl is not None:
                plt.plot(wl[:, 0], wl[:, 1], 'ro', label='Logged Weights')

            plt.xlabel('mu')
            plt.ylabel('sigma')
            plt.legend()
            plt.show(block=False)
            ans = input("ready? (0 to skip plotting): ")
            if ans == '0':
                doplot = False

        # Step 1: sample mu from Gaussian
        mu = xbar + np.random.randn() * sigma / sN

        if wltlog:
            wlt[ltt, :] = [mu, sigma, -0.5, ltt]
            ltt += 1

        if doplot:
            plt.clf()
            if zz is not None and len(zz) >= 3:
                x, y, z = zz[0], zz[1], zz[2]
                plt.contour(x, y, z, 20)
            if wlt is not None:
                plt.plot(wlt[:, 0], wlt[:, 1], 'b.-', label='Trajectory')
            if wl is not None:
                plt.plot(wl[:, 0], wl[:, 1], 'ro', label='Logged Weights')
            plt.xlabel('mu')
            plt.ylabel('sigma')
            plt.legend()
            plt.show(block=False)
            ans = input("ready? (0 to skip plotting): ")
            if ans == '0':
                doplot = False

        # Step 2: sample sigma from inverse gamma
        sigma = inversegamma(N + 1, 0.5 * N * ((mu - xbar)**2 + v))

        if wltlog:
            wlt[ltt, :] = [mu, sigma, -0.5, ltt]
            ltt += 1

        # Log weight
        T += 1
        wl[logtime, :] = [mu, sigma, -0.5, T]
        logtime += 1

        if verbose:
            print(f"[mu, sigma] = [{mu}, {sigma}]")

    return mu, sigma, logtime, ltt

# --- "Global" loggers and settings ---
dT = 1
T = 0
dT0 = 1
T0 = 0
logtime = 0   # Python, zero-based
wl = []   # weight vector log
wlt = []  # trajectory log
wltlog = True  # enable trajectory logging
ltt = 0

autos = False
verbose = False
logsy = True   # log y-axis for sigma
arrows = False

doplot = True

# --- Initial conditions and data ---
mu = 0.1
sigma = 1.0
xbar = 1.0
v = 0.2
N = 5
L = 30

# --- Define plotting range ---
xmin, xmax, dx = 0, 2.0, 0.05
smin, smax, dls = 0.18, 1.8, 0.1

x = np.arange(xmin, xmax + dx, dx)
ls = np.arange(np.log(smin), np.log(smax) + dls, dls)
s = np.exp(ls)

X, S = np.meshgrid(x, s)

D = np.outer(np.ones(len(s)), N * ((x - xbar) ** 2 + v)) / (2 * np.outer(s ** 2, np.ones(len(x))))

if logsy:
    exponent = N
else:
    exponent = N + 1

SO = np.outer(s ** exponent, np.ones(len(x)))
Z = np.exp(-D) / SO

# --- Plotting surface ---
fig = plt.figure(1)
plt.clf()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, S, Z, cmap='viridis', edgecolor='none', shade=True)
ax.set_xlabel(r'$\mu$')
ax.set_ylabel(r'$\sigma$')
ax.set_zlabel(r'$P(\mu, \sigma)$')
ax.set_title('Joint Posterior Density')
ax.view_init(30, 60)
plt.grid(True)
plt.show(block=False)

# Save meshgrid for plotting trajectories later
zz = (X, S, Z)

input("Ready for contour plot? ")

# --- Plotting contour ---
plt.figure(2)
plt.clf()
plt.contour(X, S, Z, 10)
plt.xlabel(r'$\mu$')
plt.ylabel(r'$\sigma$')
plt.title('Posterior Contours')
plt.tight_layout()
plt.show(block=False)

# --- Set random seed ---
see = 0.1123456
np.random.seed(int(see * 10**7))  # best effort, as Octave's "rand('seed', x)" isn't directly compatible

# --- Optional: arrows to indicate sigma_N and sigma_{N-1} ---
if arrows:
    h1 = np.sqrt(v)
    h2 = np.sqrt(v * N / (N - 1))
    plt.plot([1, 2], [h1, h1], 'r--', label=r'$\sigma_N$')
    plt.plot([1, 2], [h2, h2], 'g--', label=r'$\sigma_{N-1}$')
    plt.legend()
    plt.show(block=False)

# --- Initialize sample logging arrays ---
# wl = np.zeros((L * 2 + 10, 4))
# wlt = np.zeros((L * 3 + 10, 4))
wl = np.full((L * 2 + 10, 4), np.nan)
wlt = np.full((L * 3 + 10, 4), np.nan)


# --- Call Gibbs sampler ---
# The gibbs function should be updated to not return mu,sigma but to update wl, wlt and globals by reference (or could return them)
mu, sigma, logtime, ltt  = gibbs(mu, sigma, L, xbar, v, N, 
                  wl, wlt, wltlog, ltt, verbose, T, logtime, zz, doplot)

# --- Final plot of trajectory and samples ---
fig = plt.figure(3)
plt.clf()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, S, Z, cmap='viridis', edgecolor='none', alpha=0.6)
ax.set_xlabel(r'$\mu$')
ax.set_ylabel(r'$\sigma$')
ax.set_zlabel(r'$P(\mu, \sigma)$')
ax.set_title('Gibbs Samples on Posterior')
ax.view_init(30, 60)

wl = wl[~np.isnan(wl[:, 0])]
wlt = wlt[~np.isnan(wlt[:, 0])]

print("wlt shape", wlt.shape)
print("wl shape", wl.shape)
print("wl",wl)
if wlt.shape[0] > 0 and np.any(wlt):
    ax.plot3D(wlt[:ltt, 0], wlt[:ltt, 1], wlt[:ltt, 2], 'ro', label="Trajectory")

# if wl.shape[0] > 0 and np.any(wl):
#     ax.plot3D(wl[:logtime, 0], wl[:logtime, 1], wl[:logtime, 2], 'gx', label="Samples")
# Project wl samples onto Z surface
mu_samples = wl[:logtime, 0]
sigma_samples = wl[:logtime, 1]

print("mu_samples:", mu_samples)
print("sigma_samples:", sigma_samples)

# Evaluate posterior (Z) at the sample points
def posterior(mu, sigma):
    D = N * ((mu - xbar)**2 + v) / (2 * sigma**2)
    S = sigma ** exponent
    return np.exp(-D) / S

z_samples = posterior(mu_samples, sigma_samples)
print("z_samples", z_samples[:10])


# Plot the samples at their actual posterior value
ax.plot3D(mu_samples, sigma_samples, z_samples, 'gx', label="Samples")  # accepted samples
#ax.plot3D(mu_samples, sigma_samples, z_samples,  'gx', label="Samples", markersize=6, zorder=10)



ax.legend()
plt.grid(True)
plt.show()


