# load in the test.pt file and display the images and labels

import torch

# Load the test images and labels
test_images = torch.load("data/processed/test_images.pt")
test_labels = torch.load("data/processed/test_labels.pt")
# Display the images and labels

import matplotlib.pyplot as plt
import numpy as np

# display th efirst image 

plt.imshow(test_images[1].permute(1, 2, 0), cmap='gray')
plt.title(test_labels[1])
plt.show()


