from x1 import ent
from typing import Callable, Iterable

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

class InfiniteAbstractionCore:
	def __init__(self, *basicActions: set[Action]) -> None:
		self.actions = list(basicActions)

	def generateNewAction(self, *actions: set[Action]) -> CompoundAction:
		ca = CompoundAction(*actions)
		if ca not in self.actions:
			self.actions.append(ca)
		return ca