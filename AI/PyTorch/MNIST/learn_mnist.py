import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml

mnist_X, mnist_y = fetch_openml('mnist_784', version=1, data_home=".", return_X_y=True)

X = mnist_X.astype(np.float32) / 255
y = mnist_y.astype(np.int32)

def sample_show(num):
    plt.imshow(X[num].reshape(28, 28), cmap='gray')
    print("The lapel of this figure is {:.0f}.".format(y[num]))
    plt.show()
