import numpy as np
import matplotlib.pyplot as plt

# === Global Config ===
func = 0       # set to 1 to use alternate pstar
spacing = 0.01
logscale = 0
showestimate = 1

# === Core Functions ===

def pstar(x):
    global func
    x = np.asarray(x)
    if func == 0:
        return np.exp(0.4 * (x - 0.4) ** 2 - 0.08 * x ** 4)
    else:
        return np.exp(1.15 * np.sin(6.0 * x) - 0.3 * (x - 0.65) ** 2 - 0.01 * x ** 4)

def phi(x):
    x = np.asarray(x)
    return 0.2 * (x + 5.0)

def Quni(x, high):
    return high * np.ones_like(x)

def Qnorm(x, mu, sigma, h):
    x = np.asarray(x)
    return h * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

# === Precomputations ===
def standard():
    global x, y, xphi
    x = np.arange(-5, 5 + spacing, spacing)
    y = np.column_stack((x, pstar(x)))
    xphi = np.column_stack((x, phi(x)))

# === Plotting ===
def impplot0(style, xQ=None, xQs=None, ws=None):
    plt.figure(figsize=(12, 5))
    plt.title("Comparison of Distributions")
    plt.xlabel("x")
    plt.ylabel("Value")
    plt.grid(True)
    
    plt.plot(y[:, 0], y[:, 1], 'b-', label="P*(x)")
    if style == 1:
        plt.plot(xphi[:, 0], xphi[:, 1], 'm-', label="phi(x)")
    else:
        if xQ is not None:
            plt.plot(xQ[:, 0], xQ[:, 1], 'g-', label="Q*(x)")
        plt.plot(xphi[:, 0], xphi[:, 1], 'm-', label="phi(x)")
    
    if xQs is not None:
        plt.plot(xQs[:, 0], xQs[:, 1], 'g.', alpha=0.5)
    
    plt.legend()
    plt.tight_layout()
    plt.show()

# def impplot(style, xQ, xQs, ws, R):
#     fig, axs = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [1, 2]})

#     # Top subplot: estimate of <phi>
#     axs[0].set_title(f"{R} samples")
#     axs[0].plot(ws[:, 0], ws[:, 4], color='steelblue', label="estimate of <phi>")
#     axs[0].set_ylabel("<phi>")
#     axs[0].set_xlim([0, R])
#     axs[0].legend()
#     axs[0].grid(True)

#     # Bottom subplot: distributions
#     axs[1].plot(y[:, 0], y[:, 1], 'b-', label="P*(x)")
#     axs[1].plot(xphi[:, 0], xphi[:, 1], 'm-', label="phi(x)")
#     axs[1].plot(xQs[:, 0], xQs[:, 1], 'g.', alpha=0.5, label="Samples")
#     if style == 2 and xQ is not None:
#         axs[1].plot(xQ[:, 0], xQ[:, 1], color='purple', label="Q*(x)")
#     axs[1].legend()
#     axs[1].set_xlabel("x")
#     axs[1].set_ylabel("Value")
#     axs[1].set_title("Distribution functions")
#     axs[1].grid(True)

#     plt.tight_layout()
#     plt.show()

# Setup persistent figure and axes
fig = None
axs = None

def impplot(style, xQ, xQs, ws, R):
    global fig, axs

    if fig is None:
        plt.ion()  # Turn on interactive mode
        fig, axs = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [1, 2]})
    else:
        for ax in axs:
            ax.clear()

    # Top subplot: estimate of <phi>
    axs[0].set_title(f"{int(ws[-1, 0])} samples")
    axs[0].plot(ws[:, 0], ws[:, 4], color='steelblue', label="estimate of <phi>")
    axs[0].set_ylabel("<phi>")
    axs[0].set_xlim([0, R])
    axs[0].legend()
    axs[0].grid(True)

    # Bottom subplot: distributions
    axs[1].plot(y[:, 0], y[:, 1], 'b-', label="P*(x)")
    axs[1].plot(xphi[:, 0], xphi[:, 1], 'm-', label="phi(x)")
    axs[1].plot(np.array(xQs)[:, 0], np.array(xQs)[:, 1], 'g.', alpha=0.5, label="Samples")
    if style == 2 and xQ is not None:
        axs[1].plot(xQ[:, 0], xQ[:, 1], color='purple', label="Q*(x)")
    axs[1].legend()
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("Value")
    axs[1].set_title("Distribution functions")
    axs[1].grid(True)

    plt.tight_layout()
    plt.draw()
    plt.pause(0.01)

    input("Press Return to continue...")

# === Importance Sampling Main Function ===
def importance(style, xmin, xmax, mu, sigma, R, plotting):
    global y, xphi

    height = -0.1
    high = 1.0
    range_ = xmax - xmin
    gxmin, gxmax = -5, 5
    grange = gxmax - gxmin

    standard()
    
    if style == 1:
        xQ = np.column_stack((x, Quni(x, high)))
    elif style == 2:
        xQ = np.column_stack((x, Qnorm(x, mu, sigma, 2.0)))
    else:
        raise ValueError("Unsupported style")

    xQs = []
    ws = []

    if plotting >= 1:
        impplot0(style, xQ)

    sumwphi = 0.0
    sumw = 0.0
    r = 0
    while r < R:
        if style == 1:
            x_sample = np.random.rand() * range_ + xmin
            q = high
        elif style == 2:
            x_sample = np.random.randn() * sigma + mu
            q = Qnorm(x_sample, mu, sigma, high)
        else:
            raise ValueError("Unsupported style")

        if x_sample < xmin or x_sample > xmax:
            continue

        w = pstar(x_sample) / q
        phi_val = phi(x_sample)
        wphi = w * phi_val
        sumwphi += wphi
        sumw += w
        r += 1

        xQs.append([x_sample, phi_val, height, r, w, wphi])
        ws.append([r, ((r / R) * grange) + gxmin, w, wphi, sumwphi / sumw, sumwphi, sumw])

        if plotting > 1 and (r < 5 or (r < 60 and r % 10 == 0) or r % 20 == 0):
            impplot(style, xQ, np.array(xQs), np.array(ws), R)

    if plotting == 1:
        impplot(style, xQ, np.array(xQs), np.array(ws), R)

    return np.array(ws)

if __name__ == "__main__":
    R100 = 50  
    R1000 = 2000 
    wsu = importance(1,-5,5,0,1,R100,2)
    # ws1 = importance(2,-5,5,0,1,R1000,1);
    # ws03 = importance(2,-5,5,0,1.0/3.0,R100,2);
    # ws3 = importance(2,-15,15,0,3.0,R100,2);



