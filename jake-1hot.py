import math
import numpy as np
import tensorflow as tf
import os
import time
import random
from get_data import all_samples_all_contexts_padded

class ScalarModel:
	MIN_C = 2
	MAX_C = 10

	def __init__(self, get_train_data_fn, C=None, num_notes=128, train=True):
		self.num_notes = num_notes
		self.get_train_data_fn = get_train_data_fn

		# Maps C to ScalarContextModel objects
		self.models = {} 
		if C == None:
			for c in range(MIN_C, MAX_C + 1):
				self._create(c, train)
		else:
			self._create(C, train)

	def _create(self, C, train=True):
		self.models[C] = ScalarContextModel(self.get_train_data_fn, C, self.num_notes)
		if train:
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
				p = os.path.join(the_dir, "manual-save-C-{}.model".format(c))
				m.save(p)
			except Exception as e:
				print "Error saving model for C={}:\n\t{}".format(c, e)

	def load(self, the_dir):
		for c in range(MIN_C, MAX_C + 1):
			try:
				m = ScalarContextModel(get_train_data_fn, c, self.num_notes)
				p = os.path.join(the_dir, "manual-save-C-{}.model".format(c))
				m.load(p)
			except Exception as e:
				print "Error loading model for C={}:\n\t{}".format(c, e)

	def resume(self, timestamp, C, epoch):
		self.models[c].resume(timestamp, epoch)


class ScalarContextModel:
	MODELS_DIR = "models"

	seed = random.random()
	rng = np.random.RandomState()

	epochs = 20
	batch_size = 300
	learning_rate = 0.01
	
	def __init__(self, get_train_data_fn, C, num_notes=128):
		""" get_train_data_fn(C) returns a 3D numpy array, which is 
		an array of samples. Each sample is a 2D array, that is, an 
		array of C+1 notes, where each note is an array of length 2,
		with elements [note_pitch, note_duration] """
		self.C = C
		self.num_notes = num_notes
		self.data = get_train_data_fn([C]) # 3D tensor

		seen_notes = set()
		note_counts = [0] * num_notes
		# See what is the domain of all y-notes:
		for sample in self.data:
			seen_notes.add(sample[-1][0])
			note_counts[int(sample[-1][0])] += 1
		# self.logln("{} total seen note classifications: {}".format(len(seen_notes), seen_notes))
		# print "{} data points.".format(len(self.data))
		# print "Distribution of notes:"
		# print note_counts
		# ordered_note_counts = sorted([(note, note_counts[note]) for note in range(len(note_counts))], key=lambda t: -t[1])
		# print "Ordered (note-pitch, note-count) pairs:"
		# print ordered_note_counts
		# return


		self.num_batches = max(10, int(5 * len(self.data) / self.batch_size))

		# Store model, x, y, and output_layer
		self.model = None
		self.x, self.y = None, None
		self.output_layer = None
		self.acc = None
		self.accuracy_history = {}

		# Initialize architecture
		self.init()

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
		""" Reshapes a sample from C x K to CK x 1, where K is the number of 
		elements (129 + 22 I think) in the vector. """
		return sample.reshape(len(sample) * len(sample[0]))

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
		self.logln("Model for C={} saved to {}.".format(self.C, path), 0)

	def load(self, path):
		sess = tf.Session()
		saver = tf.train.Saver()
		saver.restore(sess, path)
		self.model = sess

	def init(self):
		# The recommended equation for hidden layer size is:
		#  samples / (alpha*(input_layer + output_layer)), alpha 2 to 10ish
		# So for us, we have:
		#  156000 / (alpha * (10 + 128)) ~= (1/alpha) * 1040 
		# for alpha = 5, this is 208
		# number of neurons in each layer
		input_num_units = self.C * 2
		hidden1_num_units = 600
		hidden2_num_units = 200
		hidden3_num_units = 400
		output_num_units = self.num_notes # We'll do the softmax manually

		# define placeholders
		x = tf.placeholder(tf.float32, [None, input_num_units])
		self.x = x
		y = tf.placeholder(tf.float32, [None, output_num_units])
		self.y = y

		weights = {
			'hidden1': tf.Variable(tf.random_normal([input_num_units, hidden1_num_units], seed=self.seed)),
			'hidden2': tf.Variable(tf.random_normal([hidden1_num_units, hidden2_num_units], seed=self.seed)),
			'hidden3': tf.Variable(tf.random_normal([hidden2_num_units, hidden3_num_units], seed=self.seed)),
			'output': tf.Variable(tf.random_normal([hidden1_num_units, output_num_units], seed=self.seed))
		}
		self.weights = weights

		biases = {
			'hidden1': tf.Variable(tf.random_normal([hidden1_num_units], seed=self.seed)),
			'hidden2': tf.Variable(tf.random_normal([hidden2_num_units], seed=self.seed)),
			'hidden3': tf.Variable(tf.random_normal([hidden3_num_units], seed=self.seed)),
			'output': tf.Variable(tf.random_normal([output_num_units], seed=self.seed))
		}
		self.biases = biases

		# Wire it up
		hidden_layer1 = tf.add(tf.matmul(x, weights['hidden1']), biases['hidden1'])
		hidden_layer1 = tf.nn.relu(hidden_layer1)
		# hidden_layer2 = tf.add(tf.matmul(hidden_layer1, weights['hidden2']), biases['hidden2'])
		# hidden_layer2 = tf.nn.relu(hidden_layer2)
		# hidden_layer3 = tf.add(tf.matmul(hidden_layer2, weights['hidden3']), biases['hidden3'])
		# hidden_layer3 = tf.nn.relu(hidden_layer3)
		output_layer = tf.matmul(hidden_layer1, weights['output']) + biases['output']
		self.output_layer = output_layer

	def resume(self, timestamp, epoch):
		timestamp = "{}".format(timestamp)
		d = os.path.join(self.MODELS_DIR, "{}-C-{}".format(timestamp))
		path = os.path.join(d, "auto-epoch-{}.model".format(epoch))
		self.load(path)
		self.train(start_epoch=epoch, timestamp=timestamp)

	def trend(self):
		return sorted([acc for epoch, acc in sorted(self.accuracy_history.items())])

	def train(self, start_epoch=1, timestamp=None):
		# If starting from scratch, timestamp should be None
		if timestamp == None:
			timestamp = "{}".format(int(time.time()))

		# Loss fn
		cost_fn = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.output_layer, self.y))

		# Evaluate accuracy
		self.acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.output_layer, 1), tf.argmax(self.y, 1)), tf.float32))

		# Optimizer
		optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost_fn)

		# Initialize
		init = tf.initialize_all_variables()

		# Start session
		sess = tf.Session()
		self.model = sess
		sess.run(init)
			
		# train
		for epoch in xrange(start_epoch, self.epochs + 1):
			self.logln("Training epoch {}.".format(epoch))
			total_cost = 0
			max_acc = 0
			for i in xrange(self.num_batches):
				self.log("", 1)
				self.log("Batch {}/{}\r".format(i + 1, self.num_batches), 0)
				batch_x, batch_y = self.get_batch(self.batch_size)
				_, acc_result, error = sess.run([optimizer, self.acc, cost_fn], feed_dict={self.x: batch_x, self.y: batch_y})
				max_acc = max(acc_result, max_acc)
				
				total_cost += error
			avg_cost = 1.0 * total_cost / self.num_batches
			all_x, all_y = self.get_batch(len(self.data))
			# all_x, all_y = self.get_batch(2)
			a = self.acc.eval(session=sess, feed_dict={self.x: all_x, self.y: all_y})
			self.accuracy_history[epoch] = a
			self.logln("\nEpoch {} training complete. Average loss = {:.5f}, max batch acc = {:.5f}, accuracy = {:.5f},".format(epoch, avg_cost, max_acc, a), 1)
			save_dir = os.path.join(self.MODELS_DIR, "{}-C-{}".format(timestamp, self.C))
			if not os.path.exists(save_dir):
				os.makedirs(save_dir)
			save_path = os.path.join(save_dir, "auto-epoch-{}.model".format(epoch))
			try:
				self.log("Saving state...")
				self.save(save_path)
			except:
				self.logln("Failed to save state to " + save_path)

			# print "data to test is:"
			# print all_x
			# print "data classification to test is:"
			# print all_y
			results = sess.run(self.output_layer, feed_dict={self.x: all_x})
			# print "first two results are equal: ", results[0] == results[1]
			# argmaxes = [np.argmax(row) for row in results]
			# print "results is {} by {}:".format(len(results), len(results[0]))
			# print results
			# print "argmaxes is size {}".format(len(argmaxes))
			# print argmaxes

			# print "Run separately the first and 2nd data point"
			# print sess.run(self.output_layer, feed_dict={self.x: all_x[0:1]})
			# print sess.run(self.output_layer, feed_dict={self.x: all_x[1:2]})

			# xx = sess.run(self.weights["hidden1"])
			# print "hidden1"
			# print xx



		
		self.logln("\nTraining complete for C={}.".format(self.C), 1)

data = np.array([
	[[1, 0], [2, 0], [3, 1]],
	[[2, 0], [3, 1], [1, 0]],
	[[0, 2], [1, 0], [3, 1]],
	[[1, 0], [3, 1], [1, 0]],
	[[3, 1], [1, 0], [3, 2]]])

def fn(C):
	return data

def fake_data(C, num_samples=1000):
	data = np.random.randint(0, 128, size=(num_samples, C + 1, 2))
	return data

# m = ScalarModel(fn, 2, 4)
# m = ScalarModel(fake_data, C=2)
# m.save("/home/jb16/git/neuralnotes/models")
# m.test([[[0,1,0,0,1,0,0], [0,0,1,0,1,0,0]], [[0,1,0,0,1,0,0], [0,0,0,1,0,1,0]]], 2)

m = ScalarModel(all_samples_all_contexts_padded, C=8)