import numpy as np
import matplotlib.pyplot as plt

class CTgraph_plot:

    def __init__(self):
        pass

    def plotImages(imageDataset,stop):
            # define the figure size and grid layout properties
        grid = int(np.minimum(np.ceil(np.sqrt(imageDataset.NR_OF_IMAGES)),9))

    #    fig, axs = plt.subplots(nrows = grid, ncols = grid, figsize = (12.3, 9),
    #                            subplot_kw = {'xticks': [], 'yticks': []})

    #    numbers = np.arange(imageDataset.NR_OF_IMAGES)
#        fig.subplots_adjust(left = 0.03, right = 0.97, hspace = 0.3, wspace = 0.05)

#        for ax, image_nr in zip(axs.flat, numbers):
            # this line shows the standard set
#            ax.imshow(imageDataset.getNoisyImage(image_nr))
            # this line shows the permuted set
#            ax.imshow(imageDataset.getNoisyImage(image_nr, True))
#            ax.set_title(str(image_nr))
#        plt.tight_layout()
#        plt.savefig("./images/imgdatasetPer.pdf", dpi = 150)

        fig2, axs2 = plt.subplots(nrows = grid, ncols = grid, figsize = (12.3, 9),
                                subplot_kw = {'xticks': [], 'yticks': []})

        numbers = np.arange(imageDataset.NR_OF_IMAGES)
        fig2.subplots_adjust(left = 0.03, right = 0.97, hspace = 0.3, wspace = 0.05)

        for ax, image_nr in zip(axs2.flat, numbers):
            ax.imshow(imageDataset.getNoisyImage(image_nr))
            ax.set_title(str(image_nr))

        plt.tight_layout()
        plt.savefig("./images/imgdataset1.pdf", dpi = 150)

        if stop:
            plt.show()
        else:
            plt.show(block = False)
