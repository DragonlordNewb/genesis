import x1

nn = x1.neural.FeedforwardNeuralNetwork(3, 5, 5, 5, 5, 5, 5, 5, 5, 5, 3)

input_data = [
	[0.1, 0.2, 0.3],
	[0.4, 0.5, 0.6],
	[0.7, 0.8, 0.9],
	[0.2, 0.4, 0.6],
	[0.3, 0.5, 0.7]
]

output_data = [
	[0.7, 0.2, 0.1],
	[0.3, 0.6, 0.1],
	[0.1, 0.4, 0.5],
	[0.6, 0.3, 0.1],
	[0.5, 0.2, 0.3]
]

print("before", nn.predict(input_data[0]))

nn.train(
	inputSets=input_data, outputSets=output_data,
	epochs=10
)

print("after", nn.predict(input_data[0]))