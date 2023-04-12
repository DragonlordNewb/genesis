import blessed
import sys
import time
import threading

term = blessed.Terminal()

class Spinner:
	busy = False
	delay = 0.1

	@staticmethod
	def spinning_cursor():
		while 1: 
			for cursor in '|/-\\': yield cursor

	def __init__(self, text="", delay=0.05):
		self.text = text
		self.spinner_generator = self.spinning_cursor()
		if delay and float(delay): self.delay = delay

	def spinner_task(self):
		with term.cbreak(), term.hidden_cursor():
			while self.busy:
				sys.stdout.write(self.text + " " + next(self.spinner_generator))
				sys.stdout.flush()
				time.sleep(self.delay)
				sys.stdout.write("\r")
				sys.stdout.flush()

	def __enter__(self):
		self.busy = True
		threading.Thread(target=self.spinner_task).start()

	def __exit__(self, exception, value, tb):
		self.busy = False
		time.sleep(self.delay)
		print(self.text + " - done.")
		if exception is not None:
			return False
		
Pinwheel = Spinner
		
class Ellipsis:
	busy = False
	delay = 0.2

	@staticmethod
	def spinning_cursor():
		while 1: 
			for cursor in ["   ", ".  ", ".. ", "..."]: yield cursor

	def __init__(self, text="", delay=0.1):
		self.text = text
		self.spinner_generator = self.spinning_cursor()
		if delay and float(delay): self.delay = delay

	def spinner_task(self):
		with term.cbreak(), term.hidden_cursor():
			while self.busy:
				sys.stdout.write(self.text + " " + next(self.spinner_generator))
				sys.stdout.flush()
				time.sleep(self.delay)
				sys.stdout.write("\r")
				sys.stdout.flush()

	def __enter__(self):
		self.busy = True
		threading.Thread(target=self.spinner_task).start()

	def __exit__(self, exception, value, tb):
		self.busy = False
		time.sleep(self.delay)
		print(self.text + " ...done.")
		if exception is not None:
			return False
		
class Indent:
	def __init__(self, text="  "):
		self.lastStdoutWrite = sys.stdout.write
		self.text = text

	def __enter__(self):
		def wrt(string):
			self.lastStdoutWrite(self.text + string)
		sys.stdout.write = wrt

	def __exit__(self, exception, value, tb):
		sys.stdout.write = self.lastStdoutWrite
		if exception is not None:
			return False

def clear():
	print(term.clear())

cls = clear