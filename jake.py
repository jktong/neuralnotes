import math
import numpy as np
import tensorflow as tf
import os

class ScalarModel:
	MIN_C = 2
	MAX_C = 10

	def __init__(self, get_train_data_fn, C=None, num_notes=128):
		self.num_notes = num_notes

		# Maps C to ScalarContextModel objects
		self.models = {} 
		if C == None:
			for c in range(MIN_C, MAX_C + 1):
				self.models[c] = ScalarContextModel(get_train_data_fn, c, num_notes)
				self.models[c].train()
		else:
			self.models[C] = ScalarContextModel(get_train_data_fn, C, num_notes)
			self.models[C].train()

	def predict(self, input_data):
		""" Returns predictions given the input_data. """
		c = len(input_data[0])
		if c not in self.models:
			raise ValueError("No model trained with C={}".format(c))

		return self.models[c].predict(input_data)

	def save(self, the_dir):
		for c, m in self.models.items():
			try:
				p = os.path.join(the_dir, "{}.model".format(c))
				m.save(p)
			except Exception as e:
				print "Error saving model for C={}:\n\t{}".format(c, e)

	def load(self, the_dir):
		for c in range(MIN_C, MAX_C + 1):
			try:
				m = ScalarContextModel(get_train_data_fn, c, self.num_notes)
				p = os.path.join(the_dir, "{}.model".format(c))
				m.load(p)
			except Exception as e:
				print "Error loading model for C={}:\n\t{}".format(c, e)


class ScalarContextModel:
	seed = 128 #lol 
	rng = np.random.RandomState(seed)

	epochs = 8
	batch_size = 2
	learning_rate = 0.01
	
	def __init__(self, get_train_data_fn, C, num_notes=128):
		""" get_train_data_fn(C) returns a 3D numpy array, which is 
		an array of samples. Each sample is a 2D array, that is, an 
		array of C+1 notes, where each note is an array of length 2,
		with elements [note_pitch, note_duration] """
		self.C = C
		self.num_notes = num_notes
		self.data = get_train_data_fn(C) # 3D tensor

		self.num_batches = int(5 * len(self.data) / self.batch_size)

		# Store model, x, y, and output_layer
		self.model = None
		self.x, self.y = None, None
		self.output_layer = None

	@staticmethod
	def log(msg, t=0):
		print "  " * t + msg, 

	@staticmethod
	def logln(msg, t=0):
		print "  " * t + msg

	def predict(self, input_data):
		""" Returns a 2D (#input_samples by num_notes) array of predictions,
		one per sample given in the input. For each sample, a row of probability 
		distributions is given.  """
		assert len(input_data[0]) == C

		sess, x, output_layer = self.model, self.x, self.output_layer
		l = len(input_data)
		self.log("Predicting results from {} test sample{}.".format(l, "s" if l > 1 else ""), 0)
		results = sess.run(output_layer, feed_dict={x: self.reshape(test_data)})
		return results

	def reshape(self, sample):
		""" Reshapes a sample from Cx2 to 2Cx1 """
		return sample.reshape(len(sample) * 2)

	def get_batch(self, batch_size):
		""" Randomly generates data batch of the batch_size and returns it. """
		# Want to randomly choose note-indices from songs, i.e. 
		#  choose a list of (song-ind, note-ind) tuples from the
		#  input_domains list
		batch_mask = self.rng.choice(len(self.data), batch_size)

		# each x is like [note, dur, note, dur, ...]
		# batch_x is [x, x, ...] so it's a batch_size by 2N arr
		batch_x = np.empty([batch_size, self.C * 2])
		batch_y = np.zeros([batch_size, self.num_notes])
		for bi in range(len(batch_mask)):
			bm = batch_mask[bi]
			datum = self.data[bm]
			classification = datum[-1]
			note, duration = classification[0], classification[1]

			# assign this row to the batch_x[bm]
			batch_x[bi] = self.reshape(datum[0:-1])
			batch_y[bi, note] = 1

		return batch_x, batch_y

	def save(self, path):
		if os.path.exists(path + ".index"):
			raise ValueError("Model already exists here: {}".format(path))

		sess = self.model
		saver = tf.train.Saver()
		save_path = saver.save(sess, path)
		self.log("Model for C={} saved to {}.".format(self.C, path), 0)

	def load(self, path):
		sess = tf.Session()
		saver = tf.train.Saver()
		saver.restore(sess, path)
		self.model = sess

	def train(self):
		# number of neurons in each layer
		input_num_units = self.C * 2
		hidden_num_units = 4 * self.C
		output_num_units = self.num_notes # We'll do the softmax manually

		# define placeholders
		x = tf.placeholder(tf.float32, [None, input_num_units])
		self.x = x
		y = tf.placeholder(tf.float32, [None, output_num_units])
		self.y = y

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
		self.output_layer = output_layer

		# Loss fn
		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output_layer, y))

		# Optimizer
		optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)

		# Initialize
		init = tf.global_variables_initializer()

		# Start session
		sess = tf.Session()
		self.model = sess
		sess.run(init)
			
		# train
		for epoch in xrange(self.epochs):
			self.logln("Training epoch {}.".format(epoch + 1))
			total_cost = 0
			for i in xrange(self.num_batches):
				self.log("", 1)
				self.log("Batch {}/{}\r".format(i + 1, self.num_batches), 0)
				batch_x, batch_y = self.get_batch(self.batch_size)
				_, error = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
				
				total_cost += error
			avg_cost = 1.0 * total_cost / self.num_batches
			self.logln("\nEpoch {} training complete. Average error = {:.5f}".format(epoch + 1, avg_cost), 1)
		
		self.logln("\nTraining complete for C={}.".format(self.C), 1)

data = np.array([
	[[1, 0], [2, 0], [3, 1]],
	[[2, 0], [3, 1], [1, 0]],
	[[0, 2], [1, 0], [3, 1]],
	[[1, 0], [3, 1], [1, 0]],
	[[3, 1], [1, 0], [3, 2]]])

def fn(C):
	return data

m = ScalarModel(fn, 2, 4)
m.save("/home/jb16/git/neuralnotes/models")
# m.test([[[0,1,0,0,1,0,0], [0,0,1,0,1,0,0]], [[0,1,0,0,1,0,0], [0,0,0,1,0,1,0]]], 2)