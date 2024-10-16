import numpy as np
import random as rnd
from typing import List

class NeuronNetwork:
    pass

class Neuron(NeuronNetwork):
    
    def __init__(self, value):
        self.value = value if value >= 0 and value <= 1 else rnd.random()
        self.weight = 0
    
    
    def __call__(self, last_layer:List['Neuron'])->int:
        pass