import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import plotly.express as px
import plotly.graph_objects as go
import random
import streamlit as st

def select_problem_set():
	axes_range = 1
	SQUARE = "Basic Exponential: x^2 - 0.5"
	SINE_WAVE = "Sine Wave: sin(x)"
	SINUSOIDAL = "Complex Sinusoidal: 2 * (sin(x) + cos(3x))"
	INVERTED_SINE = "Inverted Sine Wave: 1 / sin(x)"

	func = st.selectbox("Curve Function", [
		SQUARE,
		SINE_WAVE,
		SINUSOIDAL,
		INVERTED_SINE,
	], 0)

	dataFunc = lambda x: x
	if func == SQUARE:
		axes_range = 1
		dataFunc = lambda x: x**2 - 0.5
	elif func == SINE_WAVE:
		axes_range = 3
		dataFunc = lambda x: np.sin(x)
	elif func == SINUSOIDAL:
		axes_range = 5
		dataFunc = lambda x: 2 * (np.sin(x) + np.cos(3 * x))
	elif func == INVERTED_SINE:
		axes_range = 5
		dataFunc = lambda x: 1 / np.sin(x)
	
	return dataFunc, axes_range

def generate_data(dataFunc, axes_range, number_samples, noise=0.0):
	data_arr = []
	for _ in range(number_samples):
		x = random.uniform(-axes_range, axes_range)
		y = random.uniform(-axes_range, axes_range)
		above = "Above" if dataFunc(x) < y else "Below"
		noise_x = random.uniform(-noise * axes_range, noise * axes_range)
		noise_y = random.uniform(-noise * axes_range, noise * axes_range)
		data_arr.append([x + noise_x, y + noise_y, above])
	data = np.array(data_arr)
	return pd.DataFrame(data, columns=["X", "Y", "Above/Below"])

def get_data_graph(data, dataFunc, axes_range, colorRow="Above/Below"):
	fig = px.scatter(
		data, 
		x="X", 
		y="Y", 
		range_x=[-axes_range, axes_range],
		range_y=[-axes_range, axes_range],
		color=colorRow
	)

	x_line = np.linspace(-axes_range, axes_range, 400)
	y_line = dataFunc(x_line)
	fig.add_scatter(
		x=x_line, 
		y=y_line, 
		mode='lines', 
		name='Equation Line',
		line=go.scatter.Line(color="red")
	)

	return fig

def build_model(learning_rate, number_hidden_layers, hidden_layer_nodes, hidden_activation_function, output_activation_function):
	layers = [tf.keras.layers.InputLayer(input_shape=(2,))]
	for i in range(number_hidden_layers):
		layers.append(tf.keras.layers.Dense(units=hidden_layer_nodes, activation=hidden_activation_function, kernel_initializer='random_uniform', bias_initializer='random_uniform'))
	layers.append(tf.keras.layers.Dense(units=1, activation=output_activation_function, kernel_initializer='random_uniform', bias_initializer='random_uniform'))

	model = tf.keras.Sequential(layers)

	model.compile(
		optimizer=tf.keras.optimizers.experimental.RMSprop(learning_rate=learning_rate),
		loss="mean_squared_error",
		metrics=[tf.keras.metrics.RootMeanSquaredError()]
	)
	return model

def train(model, data, batch_size, validation_split):
	history = model.fit(
		x=data[["X", "Y"]].astype(float),
		y=np.where(data[["Above/Below"]] == "Above", 1.0, 0.0),
		epochs=1,
		batch_size=batch_size,
		validation_split=validation_split,
		verbose=1
	)
	return history.history['loss'][-1], history.history['val_loss'][-1]

def predict(model, data):
	return model.predict(data[["X", "Y"]].astype(float))

def graph_activation_function(activation_function):
	x = np.linspace(-10, 10, 100)
	if activation_function == "sigmoid":
		y = 1 / (1 + np.exp(-x))
	elif activation_function == "relu":
		y = np.maximum(0, x)
	elif activation_function == "tanh":
		y = np.tanh(x)
	fig = px.line(x=x, y=y, width=300, height=150)
	fig.update_layout(xaxis_title="", yaxis_title="", margin=dict(l=0, r=0, t=0, b=0))
	st.plotly_chart(fig)

def main():
	st.title("Using a Neural Network to Classify Points Above or Below a Curve")
	
	st.write("*Created by* [*Freddy Ouellette*](https://freddyouellette.com) — [*GitHub*](https://github.com/freddyouellette/neural-network-curves)")
	
	st.header("Select the Problem Set:")
	dataFunc, axes_range = select_problem_set()
	number_samples = st.slider("Number of samples", 10, 10000, 5000)
	sample_noise = st.slider("Sample noise", 0.0, 1.0, 0.01, format="%f")
	
	data = generate_data(dataFunc, axes_range, number_samples, sample_noise)
	confirmation_data = generate_data(dataFunc, axes_range, number_samples, 0.0)

	st.subheader("What We Want:")
	st.plotly_chart(get_data_graph(data, dataFunc, axes_range))

	st.header("Build the Neural Network:")
	learning_rate = st.number_input("Learning rate", None, 1.0, 0.1, format="%f")
	optcol1, optcol2 = st.columns(2)
	with optcol1:
		number_hidden_layers = st.number_input("Number of hidden layers", 1, 10, 2)
		hidden_activation_function = st.selectbox("Hidden layers activation function", ["sigmoid", "relu", "tanh"])
		graph_activation_function(hidden_activation_function)
	with optcol2:
		hidden_layer_nodes = st.number_input("Hidden layer nodes", 1, 100, 5)
		output_activation_function = st.selectbox("Output layer activation function", ["sigmoid", "relu", "tanh"])
		graph_activation_function(output_activation_function)
	batch_size = st.number_input("Batch size", 1, 1000, 5)
	epochs = st.number_input("Epochs", 1, 1000, 10)
	validation_split = st.number_input("Validation split", 0.01, 0.99, 0.1)

	model = build_model(learning_rate, number_hidden_layers, hidden_layer_nodes, hidden_activation_function, output_activation_function)

	if st.button('Train the Neural Network', use_container_width=True):
		col1, col2 = st.columns(2)
		with col1:
			training_label = st.subheader("Training...")
			nn_info = st.empty()
		with col2: loss_chart = st.empty()
		predictions_chart = st.empty()
		
		with col1:
			nn_info.write(
				"* Epoch: %s / %s\n" % (1, epochs) +
				"* Loss: ...\n"
				"* Learning Rate: %s\n" % learning_rate +
				"* Batch Size: %s\n" % batch_size +
				"* Number of Hidden Layers: %s\n" % number_hidden_layers +
				"* Number of Nodes Per Hidden Layer: %s\n" % hidden_layer_nodes +
				"* Validation Split: %s\n" % validation_split
			)
		
		loss_data = []

		for epoch in range(epochs):
			loss, val_loss = train(model, data, batch_size, validation_split)
			
			loss_data.append([epoch + 1, loss, val_loss])
			
			with col2:
				loss_chart.plotly_chart(
					px.line(
						pd.DataFrame(loss_data, columns=["Epoch", "Loss", "Validation Loss"]),
						x="Epoch",
						y=["Loss", "Validation Loss"],
						# title="Loss",
						width=400,
						height=300,
					)
				)
			
			predictions = predict(model, confirmation_data)

			confirmation_data["Predictions"] = predictions
			
			with col1:
				nn_info.write(
					"* Epoch: %s / %s\n" % (epoch + 1, epochs) +
					"* Loss: %s\n" % loss +
					"* Learning Rate: %s\n" % learning_rate +
					"* Batch Size: %s\n" % batch_size +
					"* Number of Hidden Layers: %s\n" % number_hidden_layers +
					"* Number of Nodes Per Hidden Layer: %s\n" % hidden_layer_nodes +
					"* Validation Split: %s\n" % validation_split
				)
			predictions_chart.plotly_chart(get_data_graph(confirmation_data, dataFunc, axes_range, "Predictions"))
		with col1: training_label.header("Training Complete!")

if __name__ == "__main__":
	main()