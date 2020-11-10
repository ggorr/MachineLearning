from sklearn import datasets
import matplotlib.pyplot as plt

from lib_ch1st.NodeGraph import *
import lib_ch1st.DataTools as dt

im_in = datasets.load_sample_images().images[0] / 255
trX = dt.channelFirst(np.array([im_in]))
sset = StartSet2D(trX.shape[1:3], trX.shape[0])
conv = Conv2D((3, 3), 3, biased=False)
conv.addPrevSets(sset)
conv.compile()
conv.preFit(regularizer=None, momentum=None)
conv.B = np.zeros((conv.ychs, 1, 1, 1))
conv.W = np.zeros((conv.ychs, conv.xchs, conv.filSiz[0], conv.filSiz[1]))
conv.W[0] = np.array([[[0, 0, 0], [.33, .33, .33], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]], np.float)
conv.W[1] = np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [.33, .33, .33], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]], np.float)
conv.W[2] = np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [.33, .33, .33]]], np.float)
conv.prePropTr(1)
conv.resetForPush()
sset.pushTr(trX)

# print(conv.trY.shape)
im_out = dt.channelLast(conv.trY)[0]
plt.subplot(121)
plt.imshow(im_in)
plt.subplot(122)
plt.imshow(im_out)
plt.tight_layout()
plt.show()

# print(np.max(im_in), np.max(im_out))
