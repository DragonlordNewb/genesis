from typing import Union, Iterable
from math import sqrt, exp, log, cos, pi, ceil
from random import random, randint

Number = Union[int, float]
INPUT = 0
OUTPUT = -1
LEARNING_RATE = 0.01

def generateNormal(mu: Number, sigma: Number):
    u1 = random()
    u2 = random()
    z1 = sqrt(-2 * log(u1)) * cos(2 * pi * u2)
    return mu + sigma * z1  # transform to desired mean and variance

def sigmoid(x: Number) -> Number:
	return 1 / (1 + exp(-1 * x))

# === NEURAL NETWORKING BASE CLASSES === # 

class Neuron:
	def __init__(self):
		self.bias = generateNormal(0, 1)
		self.inputAxons: list[Axon] = []
		self.outputAxons: list[Axon] = []
		self.inputs: list[Number] = []
		self.output: Number = 0
		self.error: Number = 0

	def compute(self) -> None:
		self.output = sigmoid(sum(self.inputs)) + self.bias
	
	def receiveInput(self, value: Number) -> None:
		self.inputs.append(value)

	def receiveError(self, error: Number) -> None:
		self.error += error
	
	def forward(self) -> None:
		self.compute()
		for axon in self.outputAxons:
			axon.forward(self.output)

	def backward(self) -> None:
		for axon in self.inputAxons:
			axon.backward(self.error)

	def reset(self) -> None:
		self.inputs: list[Number] = []
		self.output: Number = 0
		self.error: Number = 0
		
class Axon:
	def __init__(self, inputNeuron: Neuron, outputNeuron: Neuron) -> None:
		self.inputNeuron = inputNeuron
		self.inputNeuron.outputAxons.append(self)
		self.outputNeuron = outputNeuron
		self.outputNeuron.inputAxons.append(self)
		self.weight = abs(generateNormal(0, 1))

	def forward(self, value: Number) -> None:
		self.outputNeuron.receiveInput(value)

	def backward(self, error: Number) -> None:
		self.inputNeuron.receiveError(error)
		self.weight /= error * LEARNING_RATE

# === FEEDFORWARD NEURAL NETWORKING === #

class FeedforwardLayer:
	def __init__(self, size: int=5) -> None:
		self.neurons = []
		for _ in range(size):
			self.neurons.append(Neuron())
			
	def __len__(self) -> int:
		return len(self.neurons)

	def __iter__(self) -> Iterable:
		self.n = -1
		return self

	def __next__(self) -> Neuron:
		self.n += 1
		if self.n >= len(self):
			raise StopIteration
		return self[self.n]

	def __getitem__(self, index: int) -> Neuron:
		return self.neurons[index]

	def compute(self) -> None:
		for neuron in self:
			neuron.compute()
	
	def receiveInputs(self, values: list[Number]) -> None:
		values = list(values)
		if len(self) != len(values):
			raise ValueError("Bad number of values (" + str(len(values)) + ", " + str(values) + ") for layer of size " + str(len(self)) + ".")
		for neuron, value in zip(self, values):
			neuron.receiveInput(value)

	def receiveErrors(self, *errors: list[Number]) -> None:
		errors = list(errors)
		if len(self) != len(errors):
			raise ValueError("Bad number of errors (" + str(len(errors)) + ", " + str(errors) + ") for layer of size " + str(len(self)) + ".")
		for neuron, error in zip(self, errors):
			neuron.receiveError(error)
	
	def forward(self) -> None:
		self.compute()
		for neuron in self:
			neuron.forward()

	def backward(self) -> None:
		for neuron in self:
			neuron.backward()

	def reset(self) -> None:
		for neuron in self:
			neuron.reset()

	def outputs(self) -> list[Number]:
		return [neuron.output for neuron in self]

	def errors(self) -> list[Number]:
		return [neuron.error for neuron in self]

class FeedforwardNeuralNetwork:
	def __init__(self, *sizes: list[int]) -> None:
		self.layers = []
		for size in sizes:
			self.layers.append(FeedforwardLayer(size=size))

		self.axons = self._assembleAxons()

	def __len__(self) -> int:
		return len(self.layers)

	def __iter__(self) -> Iterable:
		self.n = -1
		return self

	def __next__(self) -> Neuron:
		self.n += 1
		if self.n >= len(self):
			raise StopIteration
		return self[self.n]

	def __getitem__(self, index: int) -> Neuron:
		return self.layers[index]

	def __reversed__(self) -> Iterable:
		return [layer for layer in self][::-1]

	def _assembleAxons(self) -> list[Axon]:
		axons = []
		for index, layer in enumerate(self):
			if layer is self[OUTPUT]:
				return axons
			for inputNeuron in layer:
				for outputNeuron in self[index + 1]:
					axons.append(Axon(inputNeuron=inputNeuron, outputNeuron=outputNeuron))

	def reset(self) -> None:
		for layer in self:
			layer.reset()

	def predict(self, values: list[Number]) -> list[Number]:
		self.reset()
		self[INPUT].receiveInputs(values)
		for layer in self:
			layer.forward()
		return self[OUTPUT].outputs()

	def adjust(self, expected: list[Number]) -> list[Number]:
		outputLayer = self[OUTPUT]
		if len(expected) != len(outputLayer):
			raise ValueError("Bad number of values (" + str(len(values)) + ", " + str(values) + ") for layer of size " + str(len(outputLayer)) + ".")

		for index, neuron in enumerate(outputLayer):
			neuron.receiveError(neuron.output - expected[index])
		
		for layer in reversed(self):
			layer.backward()

		return outputLayer.errors()

	def adapt(self, inputs: list[Number], outputs: list[Number]) -> list[Number]:
		self.predict(inputs)
		errors = abs(sum(self.adjust(outputs)))
		return errors

	def train(self, inputSets, outputSets, epochs: int=1000) -> list[Number]:
		for inputSet, outputSet in zip(inputSets, outputSets):
			print(inputSet, " maps to ", outputSet)

		errorsOverTime = []
		for _ in range(epochs):
			for inputs, outputs in zip(inputSets, outputSets):
				err = self.adapt(inputs=inputs, outputs=outputs)
				errorsOverTime.append(err)
		return errorsOverTime