from typing import Any
from typing import Iterable
from typing import Union
from typing import Callable

def levenshtein(a, b):
	if len(b) == 0:
		return len(a)
	if len(a) == 0:
		return len(b)
	if len(a) == len(b):
		return levenshtein(a[1:], b[1:])
	return 1 + min([
		levenshtein(a, b[1:]),
		levenshtein(a[1:], b),
		levenshtein(a[1:], b[1:])
	])

normalizedGaussian = lambda mu, sigma: lambda x: exp(-0.5 * (((x - mu) / sigma) ** 2))

class _Similar:
	def __mod__(self, other) -> float:
		return self.similarity(other)
	
	def similarity(self, other) -> float:
		return 1 / (abs(self.difference(other)) + 1)

	def difference(self, *args):
		raise NotImplementedError

class Property(_Similar):
	acceptedTypes: list[type] = []
	acceptors: list[Callable] = []

	def __init__(self, value: Any) -> None:
		if (type(value) not in self.acceptedTypes) and not any([acceptor(value) for acceptor in self.acceptors]):
			raise TypeError
		self.value = value

	def __repr__(self) -> str:
		return "<" + type(self).__name__ + "=" + repr(self.vale) + ">"

	def __str__(self) -> str:
		return str(self.value)
	
	def __iter__(self) -> Iterable:
		return iter(self.value)

	def __int__(self) -> int:
		return int(self.value)

	def __len__(self) -> int:
		return len(self.value)
	
	def __getitem__(self, index: Union[int, slice]) -> Any:
		return self.value[index]
	
class ScalarProperty(Property):
	acceptedTypes = [int, float]
	acceptors = []

	def similarity(self, other) -> float:
		return normalizedGaussian(self.value, 3)(other.value)
	
class LevenshteinProperty(Property):
	acceptedTypes = [str, list, tuple]
	acceptors = [
		lambda x: hasattr(x, "__iter__") and hasattr(x, "__getitem__")
	]

	def difference(self, other) -> int:
		return levenshtein(self, other)

class StringProperty(LevenshteinProperty):
	acceptedTypes = [str]

class ListProperty(LevenshteinProperty):
	acceptedTypes = [list, tuple]

class Entity(_Similar):
	def __init__(self, **properties: dict[str, Property]) -> None:
		self.properties = properties

	def __len__(self) -> int:
		return len(self.properties.keys())

	def __getitem__(self, name: str) -> Property:
		return self.properties[name]

	def __lshift__(self, other: object) -> object:
		keys = self.properties.keys()
		otherKeys = [key for key in other.properties.keys() if key not in keys]
		props = []
		for key in keys:
			if key in other.properties.keys():
				props.append(other[key])
			else:
				props.append(self[key])
		for key in otherKeys:
			props.append(other[key])
		return type(self)(*props)

	def __lt__(self, other: object) -> object:
		keys = self.properties.keys()
		otherKeys = [key for key in other.properties.keys() if key not in keys]
		props = []
		for key in keys:
			props.append(self[key])
		for key in otherKeys:
			props.append(other[key])
		return type(self)(*props)

	def __xor__(self, other: object) -> object:
		keys = [key for key in self.properties.keys() if key not in other.properties.keys()]
		otherKeys = [key for key in other.properties.keys() if key not in self.properties.keys()]
		props = []
		for key in keys:
			props.append(self[key])
		for key in otherKeys:
			props.append(other[key])
		return type(self)(*props)

	def __mod__(self, other: object) -> float:
		return self.similarity(other)

	def keys(self):
		return self.properties.keys()

	def similarity(self, other: object) -> float:
		output = 0
		for key in self.keys():
			if key in other.keys():
				output += self[key] % other[key]

		return output / len(self)

class Classification:
	def __init__(self, *entities: set[Entity], strictness: float=0.9) -> None:
		self.entities = list(entities)
		self.strictness = strictness

	def __iter__(self) -> Iterable:
		return iter(self.entities)

	def __contains__(self, other: Entity) -> bool:
		return len(self.matches(other)) > 0

	def matches(self, entity: Entity) -> list[tuple[Entity, float]]:
		return [(ent, entity % ent) for ent in self if entity % ent >= self.strictness]

	def find(self, entity: Entity) -> tuple[Entity, float, bool, int]:
		entitiesBySimilarity = {entity % ent: ent for ent in self}
		maximumSimilarity = max(entitiesBySimilarity.keys())
		ent = entitiesBySimilarity[maximumSimilarity]
		matches = sum([1 for ent in self if entity % ent >= self.strictness])
		matched = maximumSimilarity >= self.strictness
		return ent, maximumSimilarity, matched, matches