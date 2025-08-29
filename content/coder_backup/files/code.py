import numpy as np
import matplotlib.pyplot as plt

N = 1000
pts = np.random.gaussian(N, 2)  # 2D standard Gaussian

plt.scatter(pts[:, 0], pts[:, 1], s=8)
plt.title("1000 Gaussian-drawn 2D points")
plt.xlabel("x")
plt.ylabel("y")
plt.savefig("file0.png")