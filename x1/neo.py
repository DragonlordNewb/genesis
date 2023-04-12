from x1 import ent
from typing import Callable, Iterable, Union
from warnings import warn

Number = Union[int, float]

class DirectiveOverrideWarning(Warning):
	pass

class Actor(ent.Entity):
	def __init__(self, parent, *components: set[ent.Property]) -> None:
		ent.Entity.__init__(self, *components)
		self.parent = parent
		self.children = None

	def child(self, keep: list[str]=None, new: list[ent.Property]=[]) -> None:
		properties = []
		for key in keep:
			properties.append(self[key])
		for prop in new:
			properties.append(prop)
		return Actor(self, *properties)

class InvalidActionError(Exception):
	pass

class Action:
	def __init__(self, **deltas: dict[str, Callable]) -> None:
		self.deltas = deltas
		self.sentiment = 0

	def __len__(self) -> int:
		return len(self.deltas.keys())

	def __iter__(self) -> Iterable:
		self.n = -1
		return self

	def __next__(self) -> tuple[str, Callable]:
		self.n += 1
		if self.n >= len(self):
			raise StopIteration

		k = self.deltas.keys()[self.n]
		return (k, self[k])

	def __getitem__(self, key: str) -> Callable:
		return self.deltas[key]

	def take(self, target: Actor) -> None:
		for key, _ in self:
			if key not in target.signature:
				raise InvalidActionError("Action " + self.name + " can't be taken on Actor.")
		for key, operator in self:
			target[key].value = operator(target[key].value)

	def combine(self, *otherActions: set[object]) -> object:
		actions = [self] + list(otherActions)
		return CompoundAction(*actions)

class CompoundAction(Action):
	def __init__(self, *actions: set[Action]) -> None:
		self.actions = list(actions)

	def __iter__(self) -> Iterable:
		return iter(self.actions)

	def take(self, target: Actor) -> None:
		for action in self:
			action.take(target)

class Directive:
	def __init__(self, target: Actor, **properties: dict[str, any]) -> None:
		self.target = target
		self._desiredPropertiesDict = properties
		self.desiredProperties = [
			ent.Property.make(key, self._desiredPropertiesDict[key]) 
			for key in self._desiredPropertiesDict.keys()
		]
		self.ideal = target.child(new=self.desiredProperties)
		self.best = ~ideal

		self.completed = False

		self.actionsTaken: list[tuple[Action, Actor]] = []

	def __mod__(self, other: object) -> Number:
		return self.ideal % other.ideal

	def completion(self) -> Number:
		if self.completed:
			return 1
		comp = (self.target % self.ideal) / self.best

	def complete(self) -> None:
		self.completed = True

	def addAction(self, action: Action, target: Actor) -> None:
		self.actionsTaken.append((action, target))

class InfiniteAbstractionCore:
	def __init__(self, *basicActions: set[Action], directive: Directive=None) -> None:
		self.actions = list(basicActions)
		self.directive = directive
		self.completedDirectives: list[Directive]

	def assign(self, directive: Directive) -> None:
		if not directive.completed:
			warn("Changing directives without completion may cause abberant behavior.", DirectiveOverrideWarning)
		self.directive = directive
		
	def completion(self) -> Number:
		value = self.directive.completion()
		if value == 1:
			self.directive.complete()
			if self.directive not in self.completedDirectives:
				self.completedDirectives.append(self.directive)
		return value

	def generateNewAction(self, *actions: set[Action]) -> CompoundAction:
		ca = CompoundAction(*actions)
		if ca not in self.actions:
			self.actions.append(ca)
		return ca

	def take(self, action: Action, target: Actor) -> Number:
		before = self.completion()
		self.directive.addAction(action, target)
		action.take(target)
		after = self.completion()
		delta = after - before
		action.sentiment += delta
		return delta
		