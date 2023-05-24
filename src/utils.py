import numpy as np
from skimage import exposure
import matplotlib.pyplot as plt
from matplotlib import rcParams

def quick_look_gen(images, clr='k', dim=16, savename=None):
    """
    Generate a quick look visualization grid for a set of images.
    Args:
        images (Tensor): Input tensor containing the images to visualize.
        clr (str, optional): Color of the grid lines. Defaults to 'k' (black).
        dim (int, optional): Dimension of the grid (number of rows and columns). Defaults to 16.
        savename (str, optional): Name of the file to save the visualization as a PDF. Defaults to None (no saving).
    Returns:
        None
    """
    rcParams['font.family'] = 'serif'
    rcParams['font.size'] = 17
    fig = plt.figure(figsize=(12, 12))
    gs = fig.add_gridspec(dim, dim, wspace=0, hspace=-0.5)
    for i in range(dim):
        for j in range(dim):
            ind = (i * dim) + j
            ax = fig.add_subplot(gs[i, j])
            im = images[ind].numpy()

            # Enhance image contrast
            im = exposure.equalize_hist(im)
            im = exposure.rescale_intensity(im)
            p2, p98 = np.percentile(im, (2, 98))
            im = exposure.rescale_intensity(im, in_range=(p2, p98))

            # Adjust sharp borders
            mean_value = np.mean(im)
            indices = np.where(im == np.min(im))
            im[indices] = mean_value

            ax.imshow(im, cmap='binary', alpha=0.8)
            plt.xticks([])
            plt.yticks([])
            ax.spines['bottom'].set_color(clr)
            ax.spines['top'].set_color(clr)
            ax.spines['right'].set_color(clr)
            ax.spines['left'].set_color(clr)

    if savename is not None:
        plt.savefig(f'../assets/{savename}.pdf', bbox_inches='tight')

    plt.show()
    plt.close(fig)
    return None