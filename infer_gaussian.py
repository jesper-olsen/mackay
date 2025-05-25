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

if __name__ == "__main__":
    s=1 # standard deviation
    # Create x values over a range (say, m ± 4*s)
    x = np.linspace(-3 - 4*s, 3 + 4*s, 500)

    # Calculate y values (likelihood)

    plt.figure()
    for m in [-3, -2, -1, 0, 1, 2, 1]:
        y = P(x, m, s)
        plt.plot(x, y, label=f'μ={m}, σ²={s**2:.2f}')
    plt.xlabel('x')
    plt.ylabel('P(x | μ, σ)')
    plt.title('Example Gaussian Likelihood Functions')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure()
    m=-2
    for s in [2, 1.7, 1.4, 1.1, 0.8, 0.67, 0.5]:
        y = P(x, m, s)
        plt.plot(x, y, label=f'μ={m}, σ²={s**2:.2f}')
    plt.xlabel('x')
    plt.ylabel('P(x | μ, σ)')
    plt.title('Example Gaussian Likelihood Functions')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure()
    for (m,s) in [(3,1.6), (2,0.5), (1,1.2), (0,0.8), (-1,1.5), (-2,0.7), (-3,2)]:
        y = P(x, m, s)
        plt.plot(x, y, label=f'μ={m}, σ²={s**2:.2f}')
    plt.xlabel('x')
    plt.ylabel('P(x | μ, σ)')
    plt.title('Random Gaussian Likelihood Functions')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot data - two horizontal lines above and below
    plt.figure()
    for (m,s) in [(3,1.6), (2,0.5), (1,1.2), (0,0.8), (-1,1.5), (-2,0.7), (-3,2)]:
        y = P(x, m, s)
        plt.plot(x, y, label=f'μ={m}, σ²={s**2:.2f}')
    xm =-6
    xM =6
    yD =0.7245
    dy =0.025
    dat = np.array([(x,yD) for x in POINTS])
    x_vals = dat[:, 0]  # second column (x)
    y_vals = dat[:, 1]  # first column (yD)
    plt.plot(x_vals, y_vals, linestyle='None', marker='o', color='purple')
    # Plot horizontal lines at yD ± dy
    plt.hlines([yD - dy, yD + dy], xmin=xm, xmax=xM, colors='gray', linestyles='dashed', linewidth=1)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Data Points + potential gaussians')
    plt.grid(True)
    plt.show()
    




    

