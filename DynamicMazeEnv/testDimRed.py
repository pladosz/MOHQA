import numpy as np
from fastTSNE import TSNE
from sklearn import datasets
from gym_CTMaze.envs.CTMaze_plot import CTMaze_plot
from gym_CTMaze.envs import CTMaze_env
from gym_CTMaze.envs.CTMaze_conf import CTMaze_conf
from gym_CTMaze.envs.CTMaze_images import CTMaze_images
from learning.createDataset import CustomDatasetFromImages


configuration = CTMaze_conf("maze.json")
conf_data = configuration.getParameters()
imageDataset = CTMaze_images(conf_data)
print(imageDataset.image.shape)
imageDataset.image.shape
#plotting
CTMaze_plot.plotImages(imageDataset, False)

imgDataset = CustomDatasetFromImages(imageDataset)

#vectorisedImages = imageDataset.image.reshape(64,144)
#vectorisedImages.shape
imgDataset.imgdata.shape
vecImg = imgDataset.imgdata.reshape(320,144)
import torch
torch.randperm(144)

tsne = TSNE(n_components = 2, perplexity = 21, learning_rate = 100, early_exaggeration = 12,	n_jobs = 4, initialization = 'random', metric = 'euclidean',	n_iter = 750, early_exaggeration_iter = 250, neighbors = 'exact',	negative_gradient_method = 'bh', min_num_intervals = 10, )


embedding = tsne.fit(vecImg)
embedding.shape
import matplotlib.pyplot as plt
plt.scatter(embedding[:,0],embedding[:,1])
