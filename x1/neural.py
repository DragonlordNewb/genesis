from random import random, choice
from typing import Union, Iterable
from math import exp, ceil

Number = Union[int, float]
INPUT = 0
OUTPUT = -1

def sigmoid(x: Number) -> Number:
	return 1 / (1 + exp(-x))

class TrainingDataset:
	def __init__(self, *pairs: set[tuple[list[Number], list[Number]]]) -> None:
		self.pairs = pairs

	def __len__(self) -> int:
		return len(self.pairs)

	def __iter__(self) -> Iterable:
		return iter(self.pairs)

	def split(self, trainingFraction: float=0.8) -> tuple[list[tuple[list[Number], list[Number]]], list[tuple[list[Number], list[Number]]]]:
		# i apologize in advance for that typing up there
		trainingCount = ceil(len(self) * trainingFraction)
		testing = [pair for pair in self]
		training = []
		while len(training) < trainingCount:
			training.append(
				testing.pop(randint(0, len(testing) - 1)) # random element from testing, removing the element as well
			)

		return training, testing

class NeuralCell:
	def __init__(self) -> None:
		self.inputAxons = []
		self.outputAxons = []
		
		self.inputs = []
		self.output: Number = None

		self.bias = random()

		self.error: Number = 0

	def feedforward(self) -> Number:
		self.output = self.computeOutput()
		for axon in self.outputAxons:
			axon.feedforward(self.output)
		return self.output

	def backpropagate(self) -> Number:
		error = self.computeError()
		for axon in self.inputAxons:
			axon.backpropagate(error)
		self.bias /= error

	def reset(self) -> None:
		self.inputs = []
		self.output = None
		self.error = 0

class Neuron(NeuralCell):
	def computeOutput(self) -> Number:
		return sigmoid(sum(self.inputs) + self.bias)

	def computeError(self) -> Number:
		return self.error

class LSTM(NeuralCell):
	def __init__(self, forgetThreshold: Number=1) -> None:
		self.state = 0
		self.forgetThreshold = forgetThreshold

		NeuralCell.__init__(self)

	def computeOutput(self) -> Number:
		out = sigmoid(sum(self.inputs) + self.bias) + self.state
		self.state = out
		return out

	def computeError(self) -> Number:
		if self.error >= self.forgetThreshold:
			self.state = 0

		return self.error

class Axon:
	def __init__(self, feedforwardTo: NeuralCell, backpropagateTo: NeuralCell) -> None:
		self.feedforward = feedforwardTo
		self.feedforward.inputAxons.append(self)
		self.backpropagate = backpropagateTo
		self.backpropagate.outputAxons.append(self)

		self.weight = random()

	def feedforward(self, value: Number) -> None:
		self.feedforward.inputs.append(value * self.weight)

	def backpropagate(self, error: Number) -> None:
		self.backpropagate.error += error * self.weight

# === FEEDFORWARD === #

class FeedforwardHiddenLayer:
	def __init__(self, *neurons: set[NeuralCell]) -> None:
		self.neurons = list(neurons)
		self.axons = []

	def __len__(self) -> int:
		return len(self.neurons)

	def __iter__(self) -> Iterable:
		return iter(self.neurons)

	def feedforward(self) -> None:
		for neuron in self:
			neuron.feedforward()

	def backpropagate(self) -> None:
		for neuron in self:
			neuron.backpropagate()

	def connect(self, nextLayer: object) -> None:
		for neuron in self:
			for output in nextLayer:
				self.axons.append(Axon(feedforwardTo=output, backpropagateTo=neuron))

	def reset(self) -> None:
		for neuron in self:
			neuron.reset()

class FeedforwardInputLayer(FeedforwardHiddenLayer):
	def receiveInputs(self, *inputs: set[Number]) -> None:
		inputs = list(inputs)
		if len(inputs) != len(self):
			raise IndexError("Must provide exactly 1 error for each neuron in output layer.")
		for neuron, value in zip(self, inputs):
			neuron.inputs.append(value)

class FeedforwardOutputLayer(FeedforwardHiddenLayer):
	def receiveErrors(self, *errors: set[Number]) -> None:
		errors = list(errors)
		if len(errors) != len(self):
			raise IndexError("Must provide exactly 1 error for each neuron in output layer.")
		for neuron, error in zip(self, errors):
			neuron.error += error

	def output(self) -> list[Number]:
		return [neuron.computeOutput() for neuron in self]

class FeedforwardNeuralNetwork:
	def __init__(self, *layers: set[Union[FeedforwardHiddenLayer, FeedforwardInputLayer, FeedforwardOutputLayer]]) -> None:
		self.layers = list(layers)

		for index in range(len(self) - 1):
			self[index].connect(self[index + 1])

		if type(self[INPUT]) != FeedforwardInputLayer:
			raise SyntaxError("Bad input layer type.")

		if type(self[OUTPUT]) != FeedforwardOutputLayer:
			raise SyntaxError("Bad output layer type.")

		if not all([type(self[index]) == FeedforwardHiddenLayer for index in range(1, len(self) - 1)]):
			raise SyntaxError("Bad hidden layer type.")

	# Basic object stuff
	def __len__(self) -> int:
		return len(self.layers)

	def __iter__(self) -> Iterable:
		return iter(self.layers)

	def __reversed__(self) -> Iterable:
		return reversed(self.layers)

	# Input/propagation control
	def receiveInputs(self, *inputs: set[Number]) -> None:
		self[INPUT].receiveInputs(*inputs)

	def receiveErrors(self, *errors: set[Number]) -> None:
		self[OUTPUT].receiveErrors(*errors)

	def feedforward(self) -> None:
		for layer in self:
			layer.feedforward()

	def backpropagate(self) -> None:
		for layer in reversed(self):
			layer.backpropagate()

	def output(self) -> list[Number]:
		return self[OUTPUT].output()

	def reset(self) -> None:
		for layer in self:
			layer.reset()

	# High-level functions
	def predict(self, *inputs: set[Number]) -> list[Number]:
		self.reset()

		self.receiveInputs(*inputs)
		self.feedforward()

		return self.output()

	def learn(self, inputs: list[Number], outputs: list[Number]) -> Number:
		prediction = self.predict(*inputs)

		if len(output) != len(outputs):
			raise SyntaxError("Bad number of desired outputs.")

		errors = [predicted - expected for predicted, expected in zip(prediction, outputs)]
		self.receiveErrors(*errors)
		self.backpropagate()

		return sum(errors)

	def train(self, dataset: TrainingDataset, trainingFraction: float=0.8) -> list[Number]:
		training, testing = dataset.split(trainingFraction)