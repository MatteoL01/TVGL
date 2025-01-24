import numpy as np
import matplotlib.pyplot as plt
import tvgl

# Load data
X = np.loadtxt("testdata.txt")

# Set parameters
alpha = 10
beta = 10
penalty_type = "L1"
slice_size = 100

# Run TVGL
model = tvgl.TVGL(alpha, beta, penalty_type, slice_size)
model.fit(X)

# Function to plot matrices
def plot_matrices(matrices, title, cmap="viridis"):
    num_matrices = len(matrices)
    fig, axes = plt.subplots(1, num_matrices, figsize=(4 * num_matrices, 4))
    fig.suptitle(title, fontsize=16)
    for i, (matrix, ax) in enumerate(zip(matrices, axes)):
        im = ax.imshow(matrix, cmap=cmap)
        ax.set_title(f"Slice {i + 1}")
        ax.axis("off")
        fig.colorbar(im, ax=ax)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

# Plot covariance matrices
plot_matrices(model.covariance_set, title="Covariance Matrices", cmap="coolwarm")

# Plot precision matrices
plot_matrices(model.precision_set, title="Precision Matrices", cmap="coolwarm")
