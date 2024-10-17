import numpy as np
from typing import List, Tuple, Callable
import random as r

class NeuralNetwork:
    
    def __init__(
        self, 
        image_size:Tuple[int, int],
        n_hl1:int = 128, 
        n_hl2:int = 64,
        epochs:int=10,
        l_rate:float=0.001
    )->None:

        self.epochs = epochs
        self.l_rate = l_rate 
        
        self.il_size = image_size[0] * image_size[1]
        self.ol_size = 10
        self.hl1_size = n_hl1
        self.hl2_size = n_hl2
        
        self.w1 = np.random.randn(self.hl1_size, self.il_size)  * np.sqrt(1. / self.hl1_size),
        self.w2 = np.random.rand(self.hl2_size, self.hl1_size)
        self.w3 = np.random.rand(self.ol_size, self.hl2_size)
        
        print(self.w1)
        

NeuralNetwork((28, 28))        

    
