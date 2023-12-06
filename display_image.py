from numpy import clip
from matplotlib import pyplot as plt
from torch import Tensor

def imshow(inp: Tensor) -> None:
    #Imshow for Tensor.
    
    inp = inp.cpu().numpy()
    inp = inp.transpose((1, 2, 0))
    mean = array([0.485, 0.456, 0.406])
    std = array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = clip(inp, 0, 1)
    plt.imshow(inp)
    plt.show()