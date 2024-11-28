import os
import numpy as np
import matplotlib.pyplot as plt


def f():
    print(f"当前文件夹:{os.getcwd()}")
    path = os.path.join(os.path.dirname(__file__), "datas", "image2.png")
    print (path)
    print(os.path.abspath(path))
    img = plt.imread(path)
    print(img)
    print(type(img))
    print(img.shape)

    img1 = np.reshape(img, -1)
    print(img1)
    print(img1.shape)


if __name__ == '__main__':
    f()
