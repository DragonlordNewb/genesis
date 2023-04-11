from typing import Union, Iterable
from math import exp

Number = Union[int, float]

normalizedGaussian = lambda mu, sigma: lambda x: exp(-0.5 * (((x - mu) / sigma) ** 2))

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

def ngrams(data: list[any], n: int) -> list[list[any]]:
	# evil little script to generate ngrams
    return [list(x) for x in zip(*(data[i:] for i in range(n)))]

def orderedSubsets(data: list[any]) -> list[list[any]]:
	output = []
	for n in range(len(data)):
		for ngram in ngrams(data, n + 1):
			output.append(ngram)
	return output

def mean(data: list[Number]) -> Number:
	return sum(data) / len(data)

def stdev(data: list[Number]) -> Number:
	mu = mean(data)
	z = sum(pow(x - mu, 2) for x in data) / len(data)
	return pow(z, 1/2)

class Property:
	def __init__(self, name: str, value: any, leniency: Number=5, weight: Number=1) -> None:
		self.name = name
		self.value = value
		self.leniency = leniency
		self.weight = weight

	def __repr__(self) -> str:
		return "<Property: " + self.name + "=" + repr(self.value) + ">"

	def __mod__(self, other: object) -> Number:
		return self.similarity(other) * self.weight

	def __invert__(self) -> Number:
		return self.similarity(self)

class ScalarProperty(Property):
	def similarity(self, other: Property) -> Number:
		return normalizedGaussian(self.value, self.leniency)(other.value)

class StringProperty(Property):
	def similarity(self, other: Property) -> Number:
		return levenshtein(self.value, other.value)

class Timestamp(ScalarProperty):
	def __init__(self, t: Number, leniency: Number=5, weight: Number=100) -> None:
		ScalarProperty.__init__(self, "timestamp", t, leniency, weight)

class Entity:
	def __init__(self, *properties: set[Property]) -> None:
		self.properties = properties
		self.properties.sort(key=lambda prop: prop.name)
		self.signature = set([prop.name for prop in self])
		if "timestamp" in self:
			self[timestamp].weight = len(self) / 2

	def __repr__(self) -> str:
		"<Entity: \n  " + "\n  ".join([repr(prop) for prop in self]) + ">"

	def __len__(self) -> int:
		return len(self.properties)

	def __iter__(self) -> object:
		self.n = -1
		return self
	
	def __next__(self) -> Property:
		self.n += 1
		if self.n >= len(self):
			raise StopIteration
		return self.properties[self.n]

	def __getitem__(self, key: str) -> Property:
		for prop in self:
			if prop.name == key:
				return prop
		raise KeyError("Entity has no property \"" + str(key) + "\".")

	def __contains__(self, key: str) -> bool:
		for prop in self:
			if prop.name == key:
				return True
		return False

	def __invert__(self) -> None:
		return self.similarity(self)

	def __mod__(self, other: object) -> Number:
		return self.similarity(other) / ~self

	def similarity(self, other: object) -> Number:
		if self.signature != other.signature:
			raise TypeError("Entities must have the same signature to be compared.")
		return sum([prop1 % prop2 for prop1, prop2 in zip(self, other)])

class Pattern:
	def __init__(self, *entities: set[Entity]) -> None:
		self.entities = entities
		self.signature = self.entities[0].signature
		for entity in self:
			if entity.signature != self.signature:
				raise TypeError("Signatures must be all the same during Pattern instantiation.")
	
	def __repr__(self) -> str:
		"<Pattern: \n  " + "\n  ".join([repr(entity) for entity in self]) + ">"

	def __len__(self) -> int:
		return len(self.entities)

	def __iter__(self) -> object:
		self.n = -1
		return self
	
	def __next__(self) -> Property:
		self.n += 1
		if self.n >= len(self):
			raise StopIteration
		return self[self.n]

	def __getitem__(self, index: int) -> Property:
		return self.entities[index]

	def __contains__(self, key: str) -> bool:
		return key in self.signature

	def __mod__(self, other: object) -> Number:
		return self.similarity(other) / ~self

	def __invert__(self, other: object) -> Number:
		return self.similarity(self)

	def similarity(self, other: object) -> Number:
		if self.signature != other.signature:
			raise TypeError("Entities must have the same signature to be compared.")
		return sum([entity1 % entity2 for entity1, entity2 in zip(self, other)])

	@classmethod
	def fromSequencedData(cls, *entities: set[Entity], threshold: float=0.8) -> list[tuple[object, float]]:
		# returns a list of (pattern, confidence) pairs

		entities = list(entities)

		potentialPatterns: dict[list[Entity], float] = {}

		for potentialPatternSequence in orderedSubsets(entities):
			potentialPatterns[potentialPatternSequence] = 0.0
			ngramList = ngrams(entities, len(potentialPatternSequence))
			for potentialMatch in ngramList:
				potentialPatterns[potentialPatternSequence] += sum(a % b for a, b in zip(potentialPatternSequence, potentialMatch)) / len(potentialPatternSequence)
			potentialPatterns[potentialPatternSequence] /= len(ngramList)
			potentialPatterns[potentialPatternSequence] *= len(potentialPatternSequence)

		output = []
		for potentialPattern, confidence in zip(potentialPatterns.keys(), potentialPatterns.values()):
			if confidence >= threshold:
				output.append((Pattern(*potentialPattern), confidence))

		return output

	@classmethod
	def fromUnsequencedData(cls, axis: str, *entities: set[Entity], threshold=0.8) -> list[tuple[object, float]]:
		entities = list(entities)
		entities.sort(key=lambda ent: ent[axis].value)
		return Pattern.fromSequencedData(*entities, threshold=threshold)

class Chronology(Pattern):
	def __init__(self, *entities: set[Entity]) -> None:
		if any(["timestamp" not in entity for entity in entities]):
			raise SyntaxError("Chronology entities must be timestamped. Use a Pattern if the entities cannot be timestamped.")
		Pattern.__init__(self, *entities)
		self.iv = entities

	def similarity(self, other: object) -> Number:
		if self.signature != other.signature:
			raise TypeError("Entities must have the same signature to be compared.")

		sdup = self.duplicate()
		odup = other.duplicate()

		sOffset = sdup[0]["timestamp"].value
		oOffset = odup[0]["timestamp"].value

		for entity in sdup:
			entity["timestamp"].value -= sOffset

		for entity in odup:
			entity["timestamp"].value -= oOffset

		return sdup.properSimilarity(odup)

	def properSimilarity(self, other: object) -> Number:
		if self.signature != other.signature:
			raise TypeError("Entities must have the same signature to be compared.")
		return sum([entity1 % entity2 for entity1, entity2 in zip(self, other)])

	def duplicate(self) -> object:
		return Chronology(*self.iv)

class Classification:
	def __init__(self, *objects: set[Union[Entity, Pattern, Chronology]], strictness: float=0.8, selfImprove: bool=False) -> None:
		self.entities = [obj for obj in objects if type(obj) == Entity]
		self.patterns = [obj for obj in objects if type(obj) == Pattern]
		self.chronologies = [obj for obj in objects if type(obj) == Chronology]
		self.objects = list(objects)

		self.selfImprove = selfImprove
		self.strictness = strictness

	def __repr__(self) -> str:
		return "<Classification with " + str(len(self)) + " objects>"

	def __len__(self) -> int:
		return len(self.objects)

	def __contains__(self, obj: Union[Entity, Pattern, Chronology]) -> bool:
		for other in self._getObjectList(obj):
			if obj % other >= self.strictness:
				if self.selfImprove and obj not in self.objects:
					self += obj
				return True
		return False

	def __getitem__(self, obj: Union[Entity, Pattern, Chronology]) -> tuple[Union[Entity, Pattern, Chronology], float]:
		objectsByConfidence = {}
		for other in self._getObjectList(obj):
			objectsByConfidence[obj % other] = other

		confidence = max(objectsByConfidence.keys())
		bestMatch = objectsByConfidence[confidence]
		if self.selfImprove and confidence >= self.strictness:
			self += obj
		return bestMatch, confidence

	def __iadd__(self, obj: Union[Entity, Pattern, Chronology]) -> None:
		if obj not in self.objects:
			if type(obj) == Entity:
				self.entities.append(obj)
			elif type(obj) == Pattern:
				self.patterns.append(obj)
			elif type(obj) == Chronology:
				self.chronologies.append(obj)
			else:
				raise TypeError("Can't add object of type \"" + type(obj).__name__ + "\" to Classification.")
			self.objects.append(obj)

	def _getObjectList(self, obj: Union[Entity, Pattern, Chronology]) -> list[Union[Entity, Pattern, Chronology]]:
		if type(obj) == Entity:
			return self.entities
		elif type(obj) == Pattern:
			return self.patterns
		elif type(obj) == Chronology:
			return self.chronologies
		else:
			raise TypeError("Can't use object of type \"" + type(obj).__name__ + "\" with Classification.")

class Classifier:
	def __init__(self, *classifications: set[Classification], strictness: float=0.8, selfImprove: bool=False) -> None:
		self.classifications = list(classifications)
		
		self.strictness = strictness
		self.selfImprove = selfImprove

		for classif in self:
			classif.selfImprove = self.selfImprove

	def __repr__(self) -> str:
		return "<Classifier: \n  " + "\n  ".join([repr(classif) for classif in self])

	def __len__(self) -> int:
		return len(self.classifications)

	def __iter__(self) -> object:
		return iter(self.classifications)

	def __contains__(self, obj: Union[Entity, Pattern, Chronology]) -> bool:
		for classif in self:
			if obj in classif:
				return True
		return False

	def __getitem__(self, obj: Union[Entity, Pattern, Chronology]) -> tuple[Classification, Union[Entity, Pattern, Chronology], float]:
		objectsByConfidence = {}
		for classif in self:
			bestMatch, confidence = classif[obj]
			objectsByConfidence[confidence] = (bestMatch, classif)

		confidence = max(objectsByConfidence.keys())
		bestMatch, classification = objectsByConfidence[confidence]
		if self.selfImprove and confidence >= self.strictness:
			classification += obj
		return classification, bestMatch, confidence

	def __lshift__(self, entity: Entity) -> Classification:
		for classif in self:
			if entity in classif:
				classif += entity
				return classif
		
		self.classifications.append(Classification(entity, strictness=self.strictness, selfImprove=self.selfImprove))
		return self.classifications[-1]

	@classmethod
	def fromData(cls, *data: set[Union[Entity, Pattern, Chronology]], strictness: float=0.8, selfImprove: bool=False) -> object:
		"""
		Add your data and parameters here, and Genesis will automatically generate classifications for you.
		"""
		data = list(data)
		masterType = type(data[0])
		for item in data:
			if type(item) != masterType:
				raise TypeError("Can't use multiple types in Classifier.fromDataPack().")

		classifications = []

		currentEntitySet = []
		currentDataset = [item for item in data]
		currentEntity = currentDataset.pop()
		while len(currentDataset) > 0:
			entitiesBySimilarity = {}
			for entity in currentDataset:
				entitiesBySimilarity[currentEntity % entity] = entity 

			maximumSimilarity = max(entitiesBySimilarity.keys())
			if maximumSimilarity >= strictness:
				currentEntitySet.append(currentEntity)
			else:
				classifications.append(Classification(*currentEntitySet, strictness=strictness, selfImprove=selfImprove))
				currentEntitySet = []

			currentEntity = entitiesBySimilarity[maximumSimilarity]

		return cls(*classifications, strictness=strictness, selfImprove=selfImprove)

class Differential:
	def __init__(self, *values: set[tuple[Number, Number]]) -> None:
		self.x = [x for x, y in values]
		self.y = [y for x, y in values]

	def __len__(self) -> int:
		return len(self.x)

	def __iter__(self) -> Iterable:
		return iter(zip(self.x, self.y))

	def __getitem__(self, index: int) -> tuple[Number, Number]:
		return (self.x[index], self.y[index])

	def differentiate(self):
		slopes = []
		for index in range(len(self) - 1):
			dx = self.x[index + 1] - self.x[index]
			dy = self.y[index + 1] - self.y[index]
			slope = dy / dx
			slopes.append((self.x[index], slope))
		return Differential(*slopes)

	def total(self) -> Number:
		return sum(self.y)

	def differentiable(self) -> bool:
		return len(self) > 1

class SingleAxisAnalyzer:
	def __init__(self, *entities: set[Entity], axis: str="timestamp", maximumDepth: int=100, minimumTotal: Number=0.1) -> None:
		self.entities = list(entities)
		self.signature = set(self.entities[0].signature)
		for entity in self:
			if entity.signature != self.signature:
				raise TypeError("Signatures must be all the same during SingleAxisExtrapolator instantiation.")

		self.differentials = {}

		self.axis = axis

		self.sortedEntities = sorted([entity for entity in self], key=lambda ent: ent[self.axis].value)
		x = [entity[self.axis].value for entity in self.sortedEntities]

		for axis in [key for key in self.signature if key != self.axis]:
			differentiationChain = []
			y = [entity[axis].value for entity in self.sortedEntities]
			differentiationChain.append(*zip(x, y))
			while len(differentiationChain) < maximumDifferentationDepth and differentiationChain[-1].total() > minimumTotal and differentiationChain[-1].differentiable():
				differentiationChain.append(differentiationChain[-1].differentiate())

			self.differentials[axis] = [diff for diff in differentiationChain]

		self.xmin = self.sortedEntities[0][self.axis].value
		self.xmax = self.sortedEntities[-1][self.axis].value

	def __contains__(self, axis: str) -> bool:
		return axis in self.differentials.keys()

	def extrapolate(self, x: Number, _forced: bool=False) -> Entity:
		if x > self.xmax:
			dx = x - self.xmax

			values = {}
			for axis in [key for key in self.signature if key != self.axis]:
				slope = 0
				for differential in self.differentials[axis]:
					slope += differential[y[-1]] * dx

				values[axis] = (slope * dx) + self.sortedEntities[0][axis].value

			return Entity(*[ScalarProperty(name=axis, value=value) for axis, value in zip(values.keys(), values.values())])
		elif x < self.xmin:
			dx = x - self.xmin
			
			values = {}
			for axis in [key for key in self.signature if key != self.axis]:
				slope = 0
				for differential in self.differentials[axis]:
					slope += differential[y[0]] * dx

				values[axis] = (slope * dx) + self.sortedEntities[-1][axis].value

			return Entity(*[ScalarProperty(name=axis, value=value) for axis, value in zip(values.keys(), values.values())])

		elif not _forced:
			return self.interpolate(x, _forced=True)
		else:
			raise RuntimeError("Internal error, check your arguments.")

	def interpolate(self, x: Number, _forced=False) -> Entity:
		if x == self.xmin:
			return self.sortedEntities[0]
		elif x == self.xmax:
			return self.sortedEntities[-1]
		elif self.xmin < x < self.xmax:
			index = 0
			while self.sortedEntities[index][self.axis].value < x:
				index += 1
			index -= 1
		elif not _forced:
			return self.extrapolate(x, _forced=True)
		else:
			raise RuntimeError("Internal error, check your arguments.")

class PartialEntityPatcher:
	def __init__(self, entities: set[Entity], strictness: float=-1) -> None:
		self.entities = list(entities)
		self.signature = self.entities[0].signature
		for entity in self:
			if entity.signature != self.signature:
				raise TypeError("Signatures must be all the same during PartialEntityPatcher instantiation.")

		self.strictness = strictness
		self.selfImprove = self.strictness >= 0

	def __len__(self) -> int:
		return len(self.entities)

	def __mod__(self, entity: Entity) -> float:
		return sum([entity % otherEntity for otherEntity in self]) / len(self)

	def __iter__(self) -> Iterable:
		return iter(self.entities)

	def __getitem__(self, entity: Entity) -> tuple[Entity, float]:
		entitiesBySimilarity = {}

		selfSimilarity = ~entity

		for otherEntity in self:
			similarity = 0
			for key in otherEntity.signature:
				similarity += entity[key] % otherEntity[key]
			
			entitiesBySimilarity[similarity] = otherEntity

		confidence = max(entitiesBySimilarity.keys())
		result = entitiesBySimilarity(confidence)
		confidence /= selfSimilarity

		if self.selfImprove:
			if confidence >= self.confidence:
				self.entities.append(entity)

		return (result, confidence)

	def patch(self, entity: Entity) -> Entity:
		if self.isPatched(entity):
			return entity

		propertyList = []
		bestMatch, _ =	self[entity]
		for key in self.signature:
			if key in entity.signature:
				propertyList.append(entity[key])
			else:
				propertyList.append(bestMatch[key])

		return Entity(*propertyList)

	def isPatched(self, entity: Entity) -> bool:
		for key in self.signature:
			if key not in entity.signature:
				return False
		return True

class PartialPatternPatcher:
	def __init__(self, patterns: set[Pattern], strictness: float=-1) -> None:
		self.patterns = list(entities)
		self.signature = self.patterns[0].signature
		for patterns in self:
			if patterns.signature != self.signature:
				raise TypeError("Signatures must be all the same during PartialPatternPatcher instantiation.")

		self.strictness = strictness
		self.selfImprove = self.strictness >= 0

		self.patterns = [(pattern, PartialEntityPatcher(*pattern.entities, strictness=self.strictness)) for pattern in self]

	def __iter__(self) -> Iterable:
		return iter(self.patterns)

	def __getitem__(self, pattern: Pattern) -> tuple[Pattern, float]:
		patternsBySimilarity = {}

		selfSimilarity = ~pattern

		for otherPattern, _ in self:
			similarity = 0
			for entity in pattern:
				similarity += max([entity % otherEntity for otherEntity in otherPattern])

			patternsBySimilarity[similarity] = otherPattern

		confidence = max(patternsBySimilarity.keys())
		result = patternsBySimilarity(confidence)
		confidence /= selfSimilarity

		if self.selfImprove:
			if confidence >= self.confidence:
				self.entities.append(entity)

		return (result, confidence)

	def patch(self, pattern: Pattern) -> Pattern:
		entitiesList = []
		for entity in pattern:
			if not entity.isPatched():
				patchersBySimilarity = {}
				for _, patcher in self:
					patchersBySimilarity[patcher % entity] = patcher

				bestSimilarity = max(patchersBySimilarity.keys())
				patcher = patchersBySimilarity[bestSimilarity]

				entitiesList.append(patcher.patch(entity))

			entitiesList.append(entity)

		pattern = Pattern(*entitiesList)

		bestMatch = self[pattern]

		pass # haven't quite figured this bit out yet, please fix Future Lux!!!!

	def isPatched(self, pattern: Pattern) -> bool:
		for key in self.signature:
			if key not in entity.signature:
				return False
		return True