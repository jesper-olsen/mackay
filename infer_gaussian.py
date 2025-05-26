import numpy as np
import matplotlib.pyplot as plt

########################################################
#  Inferring a Gaussian
# 
#  References
# 
#   * Chapter 21 in "Information Theory, Inference, and Learning Algorithms", David MacKay
#     https://www.inference.org.uk/itprnn/book.html
#   * Lecture Video 9 @ 18:50
#     https://www.inference.org.uk/itprnn_lectures/09_mackay.mp4
#   * Old gnuplot demo:
#     https://www.inference.org.uk/itprnn/code/gaussian/
# 
########################################################


DATA = [0.63, 1.0, 0.4, 1.37, 1.6]

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
    x_vals = np.array(DATA)
    plt.plot(x_vals, [data_y] * len(DATA), 'o', color='purple')
    plt.hlines([data_y - dy, data_y + dy], xmin=xlim[0], xmax=xlim[1], colors='gray', linestyles='dashed')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Data Points and Gaussian Curves')
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_data_and_gaussians_vlines(x, models, data_y):
    plt.figure()
    for m, s in models:
        y = P(x, m, s)
        plt.plot(x, y, label=f'μ={m}, σ²={s**2:.2f}')
    x_vals = np.array(DATA)
    plt.plot(x_vals, [data_y] * len(DATA), 'o', color='purple')

   # Draw 5 vertical lines 
    plt.vlines(DATA, ymin=0, ymax=0.8, colors='red', linestyles='dashed', linewidth=2)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Data Points and Gaussian Curves')
    plt.grid(True)
    plt.legend()
    plt.show()

#Gaussians + data + likelihood for data
def plot_data_and_gaussians_vlines_like(x, models, data_y):
    plt.figure()
    for m, s in models:
        y = P(x, m, s)
        plt.plot(x, y, label=f'μ={m}, σ²={s**2:.2f}')
    x_vals = np.array(DATA)
    plt.plot(x_vals, [data_y] * len(DATA), 'o', color='purple')

   # Draw 5 vertical lines 
    plt.vlines(DATA, ymin=0, ymax=0.8, colors='red', linestyles='dashed', linewidth=2)

    for m,s in models:
        y_vals = [P(x, m, s) for x in DATA]
        plt.plot(x_vals, y_vals, 'o')

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
    models=[(3,1.6), (2,0.5), (1,1.2), (0,0.8), (-1,1.5), (-2,0.7), (-3,2)]
    plot_data_and_gaussians(
        x,
        models=models,
        data_y=0.7245,
        dy=0.025,
        xlim=(-6, 6)
    )
    plot_data_and_gaussians_vlines(x, models=models, data_y=0.7245)
    plot_data_and_gaussians_vlines_like(x, models=models, data_y=0.7245)

if __name__ == "__main__":
    main()



    

