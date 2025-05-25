import numpy as np
import matplotlib.pyplot as plt

########################################################
#
#  Inferring a Gaussian
#  Video 9 18:50
#                            See also itprnn/code/gaussian/index.html
########################################################


POINTS = [0.63, 1.0, 0.4, 1.37, 1.6]

# Define  P(x|mu,sigma)   a.k.a.   P(x|m,s)
def P(x, m, s):
    return np.exp(-(x - m)**2 / (2 * s**2)) / (s * np.sqrt(2 * np.pi))

def plot_gaussians_varying_mu(x, s, mus):
    plt.figure()
    for m in mus:
        y = P(x, m, s)
        plt.plot(x, y, label=f'μ={m}, σ²={s**2:.2f}')
    plt.xlabel('x')
    plt.ylabel('P(x | μ, σ)')
    plt.title('Gaussian PDFs with varying μ')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_gaussians_varying_sigma(x, m, sigmas):
    plt.figure()
    for s in sigmas:
        y = P(x, m, s)
        plt.plot(x, y, label=f'μ={m}, σ²={s**2:.2f}')
    plt.xlabel('x')
    plt.ylabel('P(x | μ, σ)')
    plt.title('Gaussian PDFs with varying σ')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_data_and_gaussians(x, models, data_y, dy, xlim):
    plt.figure()
    for m, s in models:
        y = P(x, m, s)
        plt.plot(x, y, label=f'μ={m}, σ²={s**2:.2f}')
    x_vals = np.array(POINTS)
    plt.plot(x_vals, [data_y] * len(POINTS), 'o', color='purple')
    plt.hlines([data_y - dy, data_y + dy], xmin=xlim[0], xmax=xlim[1], colors='gray', linestyles='dashed')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Data Points and Gaussian Curves')
    plt.grid(True)
    plt.legend()
    plt.show()


def main():
    x = np.linspace(-3 - 4, 3 + 4, 500)  # fixed s = 1

    plot_gaussians_varying_mu(x, s=1, mus=[-3, -2, -1, 0, 1, 2, 1])
    plot_gaussians_varying_sigma(x, m=-2, sigmas=[2, 1.7, 1.4, 1.1, 0.8, 0.67, 0.5])
    plot_data_and_gaussians(
        x,
        models=[(3,1.6), (2,0.5), (1,1.2), (0,0.8), (-1,1.5), (-2,0.7), (-3,2)],
        data_y=0.7245,
        dy=0.025,
        xlim=(-6, 6)
    )

if __name__ == "__main__":
    main()



    

