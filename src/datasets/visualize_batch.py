import matplotlib.pyplot as plt
from test_dataloader import train_loader

images, labels = next(iter(train_loader))

img = images[0].permute(1, 2, 0).numpy()

plt.imshow(img)
plt.title(labels[0].item())

plt.show()