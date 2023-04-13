from nltk.chunk import ne_chunk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.sentiment import vader
from nltk import pos_tag, download
from typing import Union, Iterable

download("punkt")
download("words")
download("maxent_ne_chunker")
download("averaged_perceptron_tagger")
download("vader_lexicon")

sia = vader.SentimentIntensityAnalyzer()

class SyntaxTree:
	def __init__(self, label: str, structure) -> None:
		self.structure = structure
		self.label = label
		
	# recursive method fails on extremely complex sentences
	@classmethod
	def fromIterable(cls, structure: Iterable) -> object:
		output = []
		for item in structure:
			if type(item) == str:
				output.append(str)
			output.append(SyntaxTree.fromIterable(item))
		return SyntaxTree("S", output)
		
	@classmethod
	def fromString(cls, string: str) -> object:
		return SyntaxTree.fromIterable(ne_chunk(string))

class Sentence:
	def __init__(self, string: str) -> None:
		self.string = string
		self.words = word_tokenize(str(self))
		self.pos = pos_tag(self.words)
		
	def __str__(self) -> str:
		return self.string
	
	@classmethod
	def fromString(cls, string: str) -> list[object]:
		return [cls(string=sent) for sent in sent_tokenize(string)]