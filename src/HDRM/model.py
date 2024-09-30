import numpy as np
from typing import List, Tuple, Callable

class DigitRecognitionModel:
    
    def __init__(
        self, 
        image_size:Tuple[int, int],
        neuron:Callable,
        n_hl1:int = 16, 
        n_hl2:int = 16,
        )->None:
        if not isinstance(image_size, Tuple[int, int]) \
        or not isinstance(n_hl1, int) \
        or not isinstance(n_hl2, int) \
        or not isinstance(neuron, Callable):
            raise ValueError("Wrong argument's type for initialization.")
        
        self.il_size = image_size[0] * image_size[1]
        self.ol_size = 10
        self.hl1_size = n_hl1
        self.hl2_size = n_hl2
        self.neuron = neuron
        
        self.input_l: List[Callable] = [self.neuron for k in range(self.il_size)]
        self.hidden_l1: List[Callable] = [self.neuron for k in range(self.hl1_size)]
        self.hidden_l2: List[Callable] = [self.neuron for k in range(self.hl2_size)]
        self.output_l: List[Callable] = [self.neuron for k in range(self.ol_size)]
        
    def normalize(self):
        pass
    
    def forward(self):
        pass
    
    def train(self):
        pass
    
    def save(self):
        pass
    
    