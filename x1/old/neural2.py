from typing import Union, Iterable
from math import sqrt, exp, log, cos, pi, ceil
from random import random, randint

Number = Union[int, float]
LEARNING_RATE: Number = 0.01
INPUT = 0
OUTPUT = -1

def generateNormal(mu: Number, sigma: Number):
    u1 = random()
    u2 = random()
    z1 = sqrt(-2 * log(u1)) * cos(2 * pi * u2)
    return mu + sigma * z1  # transform to desired mean and variance

def sigmoid(x: Number) -> Number:
	return 1 / (1 + exp(-1 * x))

class Axon:
	def __init__(self, front: object, back: object) -> None:
		self.weight = generateNormal(1, 1)
		self.front = front
		self.back = back
		if self not in self.front.inputAxons:
			self.front.inputAxons.append(self)
		if self not in self.back.outputAxons:
			self.back.outputAxons.append(self)

	def __repr__(self) -> str:
		"<Axon connecting " + repr(self.front) + " to " + repr(self.back) + ">"

	def forward(self, value: Number) -> None:
		self.front.receive(value * self.weight)

	def backward(self, error: Number) -> None:
		self.back.accumulate(error * self.weight)
		self.weight /= error * LEARNING_RATE

class Neuron:
	def __init__(self, inputCount: int) -> None:
		self._inputCount = inputCount
		self._layer: int = 0
		self.bias = generateNormal(0, sqrt(1 / inputCount)) # Xavier initialization
		self.inputAxons = []
		self.outputAxons = []
		self.inputs = []
		self.output: Number = 0
		self.error: Number = 0

	def __repr__(self) -> str:
		return "<Neuron bias=" + str(self.bias) + ">"

	def receive(self, value: Number) -> None:
		self.inputs.append(value)

	def accumulate(self, error: Number) -> None:
		self.error += error
		self.bias -= error * LEARNING_RATE

	def compute(self) -> None:
		self.output = sigmoid(sum(self.inputs) + self.bias)

	def forward(self) -> None:
		self.compute()
		for axon in self.outputAxons:
			axon.forward(self.output)

	def backward(self) -> None:
		for axon in self.inputAxons:
			axon.backward(self.error)

	def reset(self) -> None:
		self.inputs = []
		self.output = 0
		self.error = 0

# ======= FEEDFORWARD NEURAL NETWORKING ======= #

class FeedforwardLayer:
	def __init__(self, size: int=5, inputCount: int=1) -> None:
		self.neurons = []
		for _ in range(size):
			self.neurons.append(Neuron(inputCount=inputCount))
		self.axons = []

	def __repr__(self) -> str:
		return "<FeedforwardLayer of size " + str(len(self)) + ">"

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

	def receive(self, values: Union[Number, list[Number]]) -> None:
		if type(values) in [int, float]:
			for neuron in self:
				neuron.receive(value=values)
		elif type(values) == list:
			for neuron, value in zip(self, values):
				neuron.receive(value=value)

	def accumulate(self, errors: Union[Number, list[Number]]) -> None:
		if type(errors) in [int, float]:
			for neuron in self:
				neuron.accumulate(error=errors)
		elif type(errors) == list:
			for neuron, error in zip(self, errors):
				neuron.accumulate(error=error)

	def forward(self) -> None:
		for neuron in self:
			neuron.forward()

	def backward(self) -> None:
		for neuron in self:
			neuron.backward()

	def reset(self) -> None:
		for neuron in self:
			neuron.reset()

	def output(self) -> list[Number]:
		for neuron in self:
			neuron.compute()
		return [neuron.output for neuron in self]

class ScalarTrainingDataset:
	def __init__(self, *items: list[tuple[list[Number], list[Number]]]) -> None:
		self.items = items

	def __len__(self) -> int:
		return len(self.items)

	def __iter__(self) -> Iterable:
		self.n = -1
		return self

	def __next__(self) -> Neuron:
		self.n += 1
		if self.n >= len(self):
			raise StopIteration
		return self.items[self.n] # (input, output) tuple

	def split(self, trainingFraction: float=0.8) -> None:
		limit = ceil(trainingFraction * len(self))
		testing = [item for item in self]
		training = []
		while len(training) < limit:
			training.append(testing.pop(randint(0, len(testing) - 1)))
		return training, testing

class FeedforwardNeuralNetwork:
	def __init__(self, *sizes: list[int]) -> None:
		inputCount = 1
		self.layers = []
		for size in sizes:
			self.layers.append(FeedforwardLayer(size=size, inputCount=inputCount))
			inputCount = size

	def __repr__(self) -> str:
		return "<FeedforwardNeuralNetwork with " + str(len(self)) + " layers>"

	def __len__(self) -> int:
		return len(self.layers)

	def __getitem__(self, index: Union[int, str]) -> FeedforwardLayer:
		return self.layers[index]

	def __iter__(self) -> Iterable:
		self.n = -1
		return self

	def __next__(self) -> Neuron:
		self.n += 1
		if self.n >= len(self):
			raise StopIteration
		return self[self.n]

	def __reversed__(self) -> list[FeedforwardLayer]:
		return [layer for layer in self][::-1]
		
	def receive(self, values: Union[Number, list[Number]]) -> None:
		self[INPUT].receive(values)

	def accumulate(self, errors: Union[Number, list[Number]]) -> None:
		self[OUTPUT].accumulate(errors)

	def reset(self) -> None:
		for layer in self:
			layer.reset()

	def feedforward(self, values: Union[Number, list[Number]]) -> list[Number]:
		self.reset()
		self.receive(values)
		for layer in self:
			layer.forward()
			# if only moving forward were this easy for me
		return self[OUTPUT].output()

	def calculateErrors(self, desiredOutputs: Union[Number, list[Number]]) -> list[Number]:
		if type(desiredOutputs) != list:
			desiredOutputs = [desiredOutputs for _ in range(len(self[OUTPUT]))]

		outputLayer = self[OUTPUT]
		output = []
		for index, desiredOutput in enumerate(desiredOutputs):
			output.append(outputLayer[index].output - desiredOutput)
		return output

	def backpropagate(self, errors: Union[Number, list[Number]]) -> None:
		self.accumulate(errors)
		for layer in reversed(self):
			layer.backward()
			# if only going back were this easy for me

	def adapt(self, inputs: list[Number], outputs: list[Number]) -> list[Number]:
		outputs = self.feedforward(values=inputs)
		errors = self.calculateErrors(desiredOutputs=outputs)
		self.backpropagate(errors=errors)
		return errors

	def train(self, dataset: ScalarTrainingDataset, trainingFraction: float=0.8, iterations: int=100) -> Number:
		training, testing = dataset.split(trainingFraction=trainingFraction)

		initialError = 0
		for inputs, outputs in testing:
			self.feedforward(inputs)
			errors = self.calculateErrors(outputs)
			initialError += sum(errors)

		for _ in range(iterations):
			for inputs, outputs in training:
				self.adapt(inputs=inputs, outputs=outputs)

		finalError = 0
		for inputs, outputs in testing:
			self.feedforward(inputs)
			errors = self.calculateErrors(outputs)
			finalError += sum(errors)

		improvement = finalError - initialError
		print(initialError, "to", finalError)
		return improvement