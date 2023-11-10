from medmnist import PneumoniaMNIST
from medmnist import PathMNIST
import numpy as np
import matplotlib.pyplot as plt

# dataset1 = PneumoniaMNIST(split='train',download=True,root="Datasets/")
data = np.load('/Users/anlly/Desktop/ucl/Applied Machine Learning Systems-I/AMLS assignment/AMLS_assignment23_24-SN23043574/Datasets/pathmnist.npz')
print(len(data['val_images']))
print(data['val_images'])
for i in range(9):
    print(np.count_nonzero(data['val_labels'].flatten() == i))
# for i in data['train_images']:
#     plt.plot(i)
#     plt.show()

# python -m medmnist save --flag=pneumoniamnist --postfix=png --folder=Datasets/ --root=Datasets/
# python -m medmnist save --flag=pathmnist --postfix=png --folder=Datasets/ --root=Datasets/
# dataset2 = PathMNIST(split='train',download=True,root="Datasets/")

