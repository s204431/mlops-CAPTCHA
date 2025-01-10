# load in the test.pt file and display the images and labels

import torch

test_data = torch.load("data/processed/class_names.pt")
print(test_data)


# plot one image and label 

#import matplotlib.pyplot as plt

#image, label = test_data[0]
#plt.imshow(image.permute(1, 2, 0))
#plt.title(label)

#print("Label:", label)
#plt.show()


#print("Image size:", image.size())
