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

class Property:
	acceptedTypes: list[type] = []
	acceptors: list[Callable] = []

	def __init__(self, value: Any) -> None:
		if (type(value) not in self.acceptedTypes) and not any([acceptor(value) for acceptor in self.acceptors]):
			raise TypeError
		self.value = value

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

	def __mod__(self, other) -> float:
		return self.similarity(other)
	
	def similarity(self, other) -> float:
		return 1 / (abs(self.difference(other)) + 1)
	
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