import math
import numpy as np
import tensorflow as tf
import os 

class JakeModel:
	seed = 128
	rng = np.random.RandomState(seed)
	
	def __init__(self, get_train_data_fn, num_notes=128):
		self.num_notes = num_notes
		_data_1hot = get_train_data_fn() # A list of 2d np arrays
		self.data = self.convert_to_tuples(_data_1hot) # List of list of tuples

		# Maps N to model, and x, and y
		self.model = {}
		self.x = {}
		self.y = {}
		self.output_layer = {}

	def __call__(self, notes, N=10):
		return test(notes, N)

	def test(self, test_points_1hot, N):
		""" Given notes (list of 1hot notes), predicts the next note """
		tuples_notes = self.convert_to_tuples(test_points_1hot)
		test_data = self.format(tuples_notes)
		sess = self.model[N]
		x = self.x[N]
		f = sess.run(self.output_layer[N], feed_dict={x: test_data})
		print f


	def format(self, list_list_tup):
		""" Given a list of list of tuples, returns a list of lists representing 
		the list of input vectors (list of test points) """
		n_test_points = len(list_list_tup)
		N = len(list_list_tup[0])
		tp_size = len(list_list_tup[0][0])

		formatted = np.empty([n_test_points, N * tp_size])
		for t in xrange(n_test_points):
			test_point = list_list_tup[t]
			for tupi in xrange(len(test_point)):
				tup = test_point[tupi]
				for e in xrange(len(tup)):
					ele = tup[e]
					formatted[t, tupi * N + e] = ele
		print "input: ", list_list_tup
		print "formatted: ", formatted
		return formatted

	def convert_to_tuples(self, data_1hot):
		""" Converts list of 2d arrays to list of list of tuples """
		n_songs = len(data_1hot)
		result = [None] * n_songs
		for s in xrange(n_songs):
			notes = data_1hot[s]
			n_notes = len(notes)
			datum = [None] * n_notes
			for i in xrange(n_notes):
				note = notes[i]
				# Finds indices where the value is 1 instead of 0
				datum[i] = (np.argmax(note[:self.num_notes]), np.argmax(note[self.num_notes:]))
			result[s] = datum
		return result

	def make_input_domains(self, song_data, N):
		domains = []
		for s in range(len(song_data)):
			for n in range(N, len(song_data[s])):
				domains.append((s, n))
		return domains

	def get_batch(self, batch_size, data_tuples, input_domains, N):
		""" Randomly generates data batch """
		# Want to randomly choose note-indices from songs, i.e. 
		#  choose a list of (song-ind, note-ind) tuples from the
		#  input_domains list
		batch_mask = self.rng.choice(len(input_domains), batch_size)

		# each x is like [note, dur, note, dur, ...]
		# batch_x is [x, x, ...] so it's a batch_size by 2N arr
		batch_x = np.empty([batch_size, 2 * N])
		batch_y = np.empty([batch_size, self.num_notes])
		for bi in range(len(batch_mask)):
			bm = batch_mask[bi]
			s, n = input_domains[bm]
			note, duration = data_tuples[s][n]

			# assign this row to the batch_x[bm]
			for i in range(0, N):
				di = i + n - N
				batch_x[bi, 2 * i], batch_x[bi, 2 * i + 1] = data_tuples[s][di]
			batch_y[bi, note] = 1
		return batch_x, batch_y

	def save(self, N, path):
		if os.path.exists(path + ".index"):
			raise ValueError("Model already exists here: {}".format(path))

		sess = self.model[N]
		saver = tf.train.Saver()
		save_path = saver.save(sess, path)
		print "Model for N={} saved to {}.".format(N, path)

	def load(self, N, path):
		sess = tf.Session()
		saver = tf.train.Saver()
		saver.restore(sess, path)
		self.models[N] = sess

	def train(self, min_N=2, max_N=10):
		for n in range(min_N, max_N + 1):
			self.train_with(n)

	def train_with(self, N):
		input_domains = self.make_input_domains(self.data, N)

		print "data tuples: " , self.data
		print "input_domains:", input_domains

		# number of neurons in each layer
		input_num_units = 2 * N
		hidden_num_units = 4 * N
		output_num_units = self.num_notes # We'll do the softmax manually

		# define placeholders
		x = tf.placeholder(tf.float32, [None, input_num_units])
		self.x[N] = x
		y = tf.placeholder(tf.float32, [None, output_num_units])
		self.y[N] = y

		# set metavars
		epochs = 3
		batch_size = 2
		num_batches = int(5 * len(input_domains) / batch_size)
		learning_rate = 0.01

		weights = {
			'hidden': tf.Variable(tf.random_normal([input_num_units, hidden_num_units], seed=self.seed)),
			'output': tf.Variable(tf.random_normal([hidden_num_units, output_num_units], seed=self.seed))
		}

		biases = {
			'hidden': tf.Variable(tf.random_normal([hidden_num_units], seed=self.seed)),
			'output': tf.Variable(tf.random_normal([output_num_units], seed=self.seed))
		}

		# Wire it up
		hidden_layer = tf.add(tf.matmul(x, weights['hidden']), biases['hidden'])
		hidden_layer = tf.nn.relu(hidden_layer)
		output_layer = tf.matmul(hidden_layer, weights['output']) + biases['output']
		self.output_layer[N] = output_layer

		# Loss fn
		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output_layer, y))

		# Optimizer
		optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

		# Initialize
		init = tf.global_variables_initializer()

		# Start session
		sess = tf.Session()
		sess.run(init)
			
		# train
		for epoch in xrange(epochs):
			print "Starting epoch {}: ".format(epoch + 1)
			avg_cost = 0
			for i in xrange(num_batches):
				print "{}..".format(i + 1), 
				batch_x, batch_y = self.get_batch(batch_size, self.data, input_domains, N)
				_, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
				
				avg_cost += c
			avg_cost = 1.0 * avg_cost / num_batches
			print "\nEpoch: {}, cost = {:.5f}".format(epoch + 1, avg_cost)
		
		print "\nTraining complete!"

		self.model[N] = sess

		# correct_prediction = tf.equal(tf.argmax(output_layer, 1), tf.argmax(y, 1))
		# accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
		# test_x, test_y = self.get_batch(len(input_domains), self.data, input_domains, N)
		# acc = sess.run(accuracy, feed_dict={x: test_x, y: test_y})
		# print "Test data final accuracy: {}".format(acc)


		# find predictions on test set
		# pred_temp = tf.equal(tf.argmax(output_layer, 1), tf.argmax(y, 1))
		# accuracy = tf.reduce_mean(tf.cast(pred_temp, "float"))
		# print "Validation Accuracy:", accuracy.eval({x: val_x.reshape(-1, 784), y: dense_to_one_hot(val_y.values)})
		
		# predict = tf.argmax(output_layer, 1)
		# pred = predict.eval({x: test_x.reshape(-1, 784)})

		# self.model[N] = model

data = [np.array([[0,1,0,0,1,0,0], [0,0,1,0,1,0,0], [0,0,0,1,0,1,0], [0,1,0,0,1,0,0]]),
		np.array([[1,0,0,0,0,0,1], [0,1,0,0,1,0,0], [0,0,0,1,0,1,0], [0,1,0,0,1,0,0], [0,0,0,1,0,0,1]])]
def fn():
	return data
m = JakeModel(fn, 4)
m.train_with(2)
m.test([[[0,1,0,0,1,0,0], [0,0,1,0,1,0,0]], [[0,1,0,0,1,0,0], [0,0,0,1,0,1,0]]], 2)