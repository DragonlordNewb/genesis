from random import random, randint, choice
from math import exp
from typing import Callable, Union, Iterable
from x1 import ent

Scalar = Union[int, float]

class VectorArithmeticError(ValueError):
	pass

badLength = VectorArithmeticError("Vectors must have same length to combine.")

class Vector:
	def __init__(self, *items: set[Scalar]):
		self.items = list(items)

	def __repr__(self) -> str:
		return "<" + ",".join([str(item) for item in self]) + ">"

	def __len__(self) -> int:
		return len(self.items)

	def __iter__(self) -> Iterable:
		return iter(self.items)

	def __add__(self, other: object) -> object:
		if len(self) != len(other):
			raise badLength
		return Vector(*[a + b for a, b in zip(self, other)])

	def __iadd__(self, other: object) -> None:
		if len(self) != len(other):
			raise badLength
		for index in range(len(self)):
			self.items[index] += other.items[index]

	def __sub__(self, other: object) -> object:
		if len(self) != len(other):
			raise badLength
		return Vector(*[a - b for a, b in zip(self, other)])

	def __isub__(self, other: object) -> None:
		if len(self) != len(other):
			raise badLength
		for index in range(len(self)):
			self.items[index] -= other.items[index]

	def __mul__(self, other: object) -> Scalar: 
		# dot product
		return sum([a * b for a, b in zip(self, other)])

	def __abs__(self) -> Scalar:
		return sqrt(sum([x ** 2 for x in self]))

	def scale(self, factor: Scalar) -> None:
		for item in self:
			item *= factor

	@classmethod
	def zeroes(cls, count: int) -> object:
		return cls(*[0 for _ in range(count)])

	@classmethod
	def unit(cls, value: Scalar, count: int) -> object:
		return cls(*[value for _ in range(len(count))])

Number = Union[Scalar, Vector]

customdist = lambda A, m, s: lambda x: A * exp(-1 * (((x - m) / s) ** 2))
# at x = m +- s, y = 1/e, which is mathematically convenient; plus it cuts out a lot of 
# computationally-expensive calculations like 1/sqrt(2pi).

def sgn(x: Number) -> -1 or 0 or 1:
	if x > 0:
		return 1
	if x < 0:
		return -1
	return 0

def plusOrMinus(i: int=1) -> Number:
	result = 0
	while not result:
		result = randint(-1, 1) * i
	return result

def generateNormal(mu: Number, sigma: Number):
	u1 = random()
	u2 = random()
	z1 = sqrt(-2 * log(u1)) * cos(2 * pi * u2)
	return mu + sigma * z1  # transform to desired mean and variance

def sigmoid(x: Number) -> Number:
	return 1 / (1 + exp(-1 * x))

# === INERTIAL OPTIMIZERS === #

class UnivariateInertialOptimizer:
	def __init__(self, curve: Union[Callable, None]=None, iv: Scalar=0, resistance: Scalar=0.05) -> None:
		if curve:
			self.curve = curve
		elif not hasattr(self, "curve"):
			raise SyntaxError("Must supply a \"curve\" function at initialization or by subclassing for a UnivariateInertialOptimizer.")

		self.inertia = 0
		self.value = iv
		self.emotion = 0
		self.resistance = resistance

	def update(self) -> Scalar:
		initialValue = self.emotion
		self.value += self.inertia
		self.emotion = self.curve(self.value)
		finalValue = self.emotion
		dy = finalValue - initialValue
		dx = self.inertia
		differential = (dy / dx) * sgn(dy)
		self.inertia += differential
		self.inertia *= 1 - self.resistance
		return differential

	def stimulate(self, factor: Number=1) -> None:
		self.inertia += plusOrMinus(random()) * factor

	@staticmethod
	def optimize(curve: Callable, triggerLimit: Scalar=0.1, triggerSensitivity: Scalar=10, iv: Scalar=0, resistance: Scalar=0.1, maximumIterations=1000) -> Scalar:
		optimizer = UnivariateInertialOptimizer(curve=curve, iv=iv, resistance=resistance)
		while abs(optimizer.inertia) <= triggerLimit:
			optimizer.stimulate()
			optimizer.update()

		iteration = 0
		while iteration < maximumIterations:
			while abs(optimizer.inertia) >= triggerLimit:
				optimizer.update()
			
			i = triggerSensitivity
			while True:
				optimizer.update()
				if optimizer.inertia >= triggerLimit:
					break

				i -= 1
				if i == 0:
					return optimizer.value

			iteration += 1

		return optimizer.value

class MultivariateInertialOptimizer:
	def __init__(self, *curves: set[Callable], resistance: float=0.05) -> None:
		self.curves = list(curves)
		self.inertia = Vector.zeroes(len(self))
		self.value = Vector.zeroes(len(self))
		self.emotion = Vector.zeroes(len(self))

		self.resistance = resistance

	def __len__(self) -> int:
		return len(self.curves)

	def __iter__(self) -> Iterable:
		return iter(self.curves)

	def transform(self, v: Vector) -> Vector:
		items = v.items
		outputItems = []
		for item, curve in zip(items, self):
			outputItems.append(curve(item))
		return Vector(*outputItems)

	def update(self) -> Scalar:
		initialvalue = abs(self.emotion)
		self.value += self.inertia
		self.emotion = self.transform(self.value)
		finalValue = abs(self.emotion)
		dy: Vector = finalValue - initialValue
		dx: Vector = self.inertia
		differential = abs(dy) / abs(dx)
		differentialVector = Vector.unit(differential, len(self))
		self.inertia += differential
		self.inertia.scale(1 - self.resistance)
		return differential

	def stimulate(self, factor: Scalar) -> None:
		self.inertia += Vector(*[plusOrMinus(random()) * factor for _ in range(len(self))])

	@staticmethod
	def optimize(*curves: set[Callable], triggerLimit: Scalar=0.1, triggerSensitivity: Scalar=10, iv: Scalar=0, resistance: Scalar=0.1, maximumIterations=1000) -> Vector:
		optimizer = UnivariateInertialOptimizer(*curves, resistance=resistance)
		while abs(optimizer.inertia) <= triggerLimit:
			optimizer.stimulate()
			optimizer.update()

		iteration = 0
		while iteration < maximumIterations:
			while abs(optimizer.inertia) >= triggerLimit:
				optimizer.update()
			
			i = triggerSensitivity
			while True:
				optimizer.update()
				if optimizer.inertia >= triggerLimit:
					break

				i -= 1
				if i == 0:
					return optimizer.value

			iteration += 1

		return optimizer.value

# === MARKOV CHAINS === #

class Evaluator:
	def __init__(self, condition: Union[str, float]) -> None:
		self.condition = condition
		self.factor = 1

	def __repr__(self) -> str:
		return "<Evaluator: " + self.condition + ">"

	def evaluate(self) -> bool:
		if type(self.condition) == str:
			return eval(self.condition)
		elif type(self.condition) == float:
			return random() < self.condition * self.factor

	@classmethod
	def fromString(cls, src: str) -> object:
		spl = src.lower().split("?")
		if spl[0] == "eval":
			return cls(spl[1])
		elif spl[0] == "prob":
			if spl[1] == "percent":
				return cls(float(spl[2]) / 100)
			elif spl[1] == "fraction":
				return cls(float(spl[2]))
			return cls(float(spl[1]))

class MarkovLink:
	def __init__(self, key: str, *mapping: set[tuple[Evaluator, str]], value: Scalar=0):
		self.key = key
		self.mapping = list(mapping)
		self.value = value

		self.receivingEvaluators = []

	def __repr__(self) -> str:
		return "<"

	def __iter__(self) -> Iterable:
		return iter(self.mapping)

	def evaluate(self) -> str:
		for evaluator, key in self:
			if evaluator.evaluate():
				return key
		return self.key # watch out for infinite loops because i do not care

	def feedback(self, value: Scalar) -> None:
		for evaluator in self.receivingEvaluators:
			evaluator.factor += value

class MarkovChain:
	def __init__(self, entryKey: str, *links: set[MarkovLink]) -> None:
		self.links = list(links)
		self.currentLink = self[entryKey]

	def __iter__(self) -> Iterable:
		return iter(self.links)

	def __getitem__(self, key: str) -> MarkovLink:
		for link in self:
			if link.key == key:
				return link
		raise KeyError("No link with key \"" + key + "\".")

class SimpleMarkovChain(MarkovChain):
	def next(self) -> MarkovLink:
		self.currentLink = self[self.currentLink.evaluate()]
		return self.currentLink

class CogniscentMarkovChain(MarkovChain):
	def next(self) -> MarkovLink:
		nextLink = self[self.currentLink.evaluate()]
		self.currentLink.feedback(nextLink.value)
		self.currentLink = nextLink
		return self.currentLink

# === NEURAL NETWORKING === #

from x1.neuro import *

# === LANGUAGE MODELLING === #

# === GENETIC ALGORITHMS === #

class Agent:
	def __init__(self, *entities: set[ent.Entity], function: Callable=None) -> None:
		if function:
			self.function = function

		if not hasattr(self, "function"):
			raise SyntaxError("Must supply a \"function\" callable to Agent by subclassing or adding the keyword argument at initialization.")

		self.entities = list(entities)

		self.parent = None

	def __repr__(self) -> str:
		return "<Agent running " + repr(self.function) + " with args " + ", ".join([repr(ent) for ent in self])

	def __len__(self) -> int:
		return len(self.entities)

	def __del__(self) -> None:
		if self.parent:
			self.parent.remove(self)

	def duplicate(self) -> object:
		agent = Agent(*self.entities, function=self.function)
		agent.parent = self.parent

	def evaluate(self, **kwargs: dict[str: any]) -> any:
		return self.function(*self.entities, **kwargs)

	def vary(self, count: int=1, magnitude: Number=1) -> None:
		while count > 0:
			choice(self.entities).vary(magnitude)
			count -= 1

	def variants(self, variantCount: int=1, mutationCount: int=1, magnitude: Number=1) -> None:
		variants = []
		for _ in range(count):
			variants.append(self.duplicate())
		
		for variant in variants:
			variant.vary(mutationCount, magnitude)

		return variants

class SingleParentGeneticIntelligence:
	def __init__(self, *precursors: set[Agent], evaluator: Callable=None, expansionRate: Number=2, extinctionRate: Number=0.5, mutationRate: Number=1, mutationMagnitude: Number=1) -> None:
		self.generations = [list(precursors)]
		self.expansionRate = expansionRate
		self.extinctionRate = extinctionRate
		self.mutationRate = mutationRate
		self.mutationMagnitude = mutationMagnitude

		if evaluator:
			self.evaluator = evaluator

		if not hasattr(self, "evaluator"):
			raise SyntaxError("Must supply a \"evaluator\" callable to GeneticIntelligence by subclassing or adding the keyword argument at initialization.")

		for agent in self.agents():
			agent.parent = self

	def __repr__(self) -> str:
		return "<GeneticIntelligence>"

	def __len__(self) -> int:
		return len(self[-1])

	def __iter__(self) -> Iterable:
		return iter(self.generations)

	def __getitem__(self, index: int) -> list[Agent]:
		return self.generations[index]

	def agents(self) -> Iterable:
		output = []
		for generation in self:
			for agent in generation:
				output.append(agent)
		return iter(output)

	def remove(self, agent: Agent) -> None:
		for index, generation in enumerate(self):
			if agent in generation:
				self.generations[index] = [x for x in self[index] if x != agent]

	def latest(self) -> list[Agent]:
		return self[-1]

	def produceNewGeneration(self, expansionRate: Number=None, extinctionRate: Number=None, mutationRate: Number=None, mutationMagnitude: Number=None) -> None:
		expansionRate = expansionRate or self.expansionRate
		extinctionRate = extinctionRate or self.extinctionRate
		mutationRate = mutationRate or self.mutationRate
		mutationMagnitude = mutationMagnitude or self.mutationMagnitude
		if not (expansionRate and extinctionRate and mutationRate and mutationMagnitude):
			raise SyntaxError("Must supply expansionRate, extinctionRate, etc variables at function call or init")

		parents = self.latest()
		children = []
		for parent in parents:
			for variant in parent.variants(self.expansionRate, mutationRate, mutationMagnitude):
				children.append(variant)

		childrenByScore = {self.evaluator(child.function): child for child in children}
		minimum = extinctionRate * max(childrenByScore.keys())
		survivors = [childrenByScore[key] for key in childrenByScore.keys() if key >= minimum]
		for child in childrenByScore.values():
			if child not in survivors:
				del child
		
		self.generations.append(survivors)

	def predict(self, **kwargs: dict[str, any]) -> any:
		agentsByScore = {self.evaluator(agent.function): agent for agent in self.latest()}
		best = agentsByScore[max(agenstByScore.keys())]
		return best.evaluate(**kwargs)