from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from lib32.NodeGraph import *
import lib32.DataTools as dt

image = Image.open('../data/cat.bmp')
pixel = np.array(image)
gray = pixel.sum(axis=2).astype(np.float32) / (255 * 3)

B1 = np.zeros(8, np.float32)
W1 = np.array(
	[[[[1, 1, 1], [0, 0, 0], [0, 0, 0]],
	  [[0, 0, 0], [1, 1, 1], [0, 0, 0]],
	  [[0, 0, 0], [0, 0, 0], [1, 1, 1]],
	  [[1, 0, 0], [1, 0, 0], [1, 0, 0]],
	  [[0, 1, 0], [0, 1, 0], [0, 1, 0]],
	  [[0, 0, 1], [0, 0, 1], [0, 0, 1]],
	  [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
	  [[0, 0, 1], [0, 1, 0], [1, 0, 0]]]],
	np.float32
)
W1 = np.transpose(W1, (2, 3, 0, 1))

B2 = np.zeros(8, np.float32)
W2 = np.zeros((3, 3, 8, 8))
for i in range(8):
	W2[:, :, i] = W1[:, :, 0]

ng = NodeGraphBatch()
sset = ng.add(StartSet2D((gray.shape[0], gray.shape[1]), 1))

conv1 = ng.add(Conv2D((3, 3), 8), sset)
conv1.initB, conv1.initW = B1, W1
relu1 = ng.add(Relu2D(), conv1)
mp1 = ng.add(MaxPool2D((2, 2)), relu1)

conv2 = ng.add(Conv2D((3, 3), 8), mp1)
conv2.initB, conv2.initW = B2, W2
relu2 = ng.add(Relu2D(), conv2)
mp2 = ng.add(MaxPool2D((2, 2)), relu2)

conv3 = ng.add(Conv2D((3, 3), 8), mp2)
conv3.initB, conv3.initW = B2.copy(), W2.copy()
relu3 = ng.add(Relu2D(), conv3)
mp3 = ng.add(MaxPool2D((2, 2)), relu3)

conv4 = ng.add(Conv2D((3, 3), 8), mp3)
conv4.initB, conv4.initW = B2.copy(), W2.copy()
relu4 = ng.add(Relu2D(), conv4)
mp4 = ng.add(MaxPool2D((2, 2)), relu4)

conv5 = ng.add(Conv2D((3, 3), 8), mp4)
conv5.initB, conv5.initW = B2.copy(), W2.copy()
relu5 = ng.add(Relu2D(), conv5)
mp5 = ng.add(MaxPool2D((2, 2)), relu5)

flat = ng.add(Flat2D(), mp5)
loss = ng.add(Mse1D(), flat)
ng.compile()
x = dt.addChannel(np.array([gray]))
y = dt.addChannel(np.zeros((1, 1), np.float32))
ng.epochMax = 0
ng.fit(x, y)
mp1out = mp1.trY[0, :, :].sum(axis=2)
mp1out /= mp1out.max()
mp2out = mp2.trY[0, :, :].sum(axis=2)
mp2out /= mp2out.max()
mp3out = mp3.trY[0, :, :].sum(axis=2)
mp3out /= mp3out.max()
mp4out = mp4.trY[0, :, :].sum(axis=2)
mp4out /= mp4out.max()
mp5out = mp5.trY[0, :, :].sum(axis=2)
mp5out /= mp5out.max()
print(mp1out.shape)
print(mp2out.shape)
print(mp3out.shape)
print(mp4out.shape)
print(mp5out.shape)

fig, ax = plt.subplots(1, 6)
ax[0].imshow(gray, cmap='gray')
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[1].imshow(mp1out, cmap='gray')
ax[1].set_xticks([])
ax[1].set_yticks([])
ax[2].imshow(mp2out, cmap='gray')
ax[2].set_xticks([])
ax[2].set_yticks([])
ax[3].imshow(mp3out, cmap='gray')
ax[3].set_xticks([])
ax[3].set_yticks([])
ax[4].imshow(mp4out, cmap='gray')
ax[4].set_xticks([])
ax[4].set_yticks([])
ax[5].imshow(mp5out, cmap='gray')
ax[5].set_xticks([])
ax[5].set_yticks([])
plt.show()
