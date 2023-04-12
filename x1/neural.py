from random import random
from typing import Union, Iterable
from math import exp

Number = Union[int, float]

def sigmoid(x: Number) -> Number:
    return 1 / (1 + exp(-x))

class Neuron:
    def __init__(self) -> None:
        self.bias = random()

        self.inputs = []
        self.output: Number = None

        self.inputAxons = []
        self.outputAxons = []
        
        self.error: Number = 0

    def feedforward(self) -> Number:
        output = sigmoid(sum(self.inputs))
        for axon in self.outputAxons:
            axon.feedforward(output)
        return output

    def backpropagate(self) -> Number:
        for axon in self.inputAxons:
            axon.backpropagate(self.error)
        self.bias /= self.error

class Axon:
    def __init__(self, feedforwardTo: Neuron, backpropagateTo: Neuron) -> None:
        self.feedforward = feedforwardTo
        self.feedforward.inputAxons.append(self)
        self.backpropagate = backpropagateTo
        self.backpropagate.outputAxons.append(self)

        self.weight = random()

    def feedforward(self, value: Number) -> None:
        self.feedforward.inputs.append(value * self.weight)

    def backpropagate(self, error: Number) -> None:
        self.backpropagate.error += error * self.weight