import matplotlib.pyplot as plt
import numpy as np

# Set standard LaTeX font settings using built-in mathtext engine
plt.rcParams.update({
    "text.usetex": False,            
    "mathtext.fontset": "cm",        # Force native Computer Modern LaTeX font
    "font.family": "serif",          # Match typical serif document layouts
    "font.size": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})

# Define the Haar wavelet function centred at x=0
def haarWaveletCentred(t):
    return np.piecewise(t, 
                        [t < -1, (t >= -1) & (t < 0), (t >= 0) & (t < 1), t >= 1], 
                        [0, -1/np.sqrt(2), 1/np.sqrt(2), 0])

# Generate high-resolution time points around x=0
t = np.linspace(-1.5, 1.5, 1000)
psi = haarWaveletCentred(t)

# A standard LaTeX two-column width is roughly 3.39 inches
figWidthInches = 3.39
figHeightInches = 1.40

fig, ax = plt.subplots(figsize=(figWidthInches, figHeightInches))
ax.plot(t, psi, color='blue', linewidth=1.5, label=r'$\psi^{(H)}$')

# Keep the horizontal baseline but remove the vertical axis line through the origin
ax.axhline(0, color='black', linewidth=0.8)

# Set clean limits and isolate the specific integer ticks on the x-axis
ax.set_xlim(-1.5, 1.5)
ax.set_xticks([-1, 0, 1])

# Isolate 0 as the only y-tick label
ax.set_yticks([0])

# plot the legend
ax.legend(loc='upper left', fontsize=10, frameon=False)

# Use tight layout with zero padding to prevent any unwanted cropping in LaTeX
plt.tight_layout(pad=0.1)

# Display the plot
plt.show()
