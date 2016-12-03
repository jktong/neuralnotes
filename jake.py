import math
import numpy as np
import tensorflow as tf

num_notes = 128
frac_training = .75
N = 5

# random seed
seed = 128
rng = np.random.RandomState(seed)

def convert_to_tuples(data_1hot):
	n_songs = len(data_1hot)
	result = np.empty([n_songs])
	for s in xrange(n_songs):
		n_notes = len(n_songs[s])
		datum = np.empty([n_notes])
		for i in xrange(n_notes):
			note = n_notes[i]
			datum[i] = (note[:num_notes].index(1), note[num_notes:].index(1))
		result[s] = datum
	return result

def flatten(data):
	""" Takes in list of tuples and flattens to list """
	flattened = np.empty([2 * len(data)])
	for f in xrange(len(data)):
		flattened[f], flattened[f + 1] = data[f]

	return flattened

def make_input_domains(song_data):
	domains = []
	for s in range(len(song_data)):
		for n in range(N, len(song_data[s])):
			domains.append((s, n))
	return domains

def get_batch(batch_size, data_tuples, input_domains):
	""" Randomly generates data batch """
	# Want to randomly choose note-indices from songs, i.e. 
	#  choose a list of (song-ind, note-ind) tuples from the
	#  input_domains list
	batch_mask = rng.choice(len(input_domains), batch_size)

	batch_x = np.empty([batch_size])
	batch_y = np.empty([batch_size])
	for bm in batch_mask:
		s, n = input_domains[bm]
		note, duration = data_tuples[s][n]

		x = np.empty([2 * N])
		for i in range(n - N, n):
			x[i], x[i + 1] = data_tuples[s][n]
		batch_x[bm] = x
		batch_y[bm] = note

    return batch_x, batch_y


# 3d matrix of song data
# songs[0, 1, 2] = first song, second note (row), 3rd element
def train_model(get_data_fn):
	_data_1hot = get_data_fn() # A list of 2d np arrays
	tuples_data = convert_to_tuples(_data_1hot) # List of list of tuples

	train_N = math.floor(frac_training * len(songs_list))
	training = songs_list[:train_N]
	input_domains = make_input_domains(training_data)
	test = songs_list[train_N:]

	# number of neurons in each layer
	input_num_units = 2 * N
	hidden_num_units = 4 * N
	output_num_units = N

	# define placeholders
	x = tf.placeholder(tf.float32, [None, input_num_units])
	y = tf.placeholder(tf.float32, [None, output_num_units])

	# set metavars
	epochs = 2
	batch_size = 128
	num_batches = int(5 * len(input_domains) / batch_size)
	learning_rate = 0.01

	weights = {
	    'hidden': tf.Variable(tf.random_normal([input_num_units, hidden_num_units], seed=seed)),
	    'output': tf.Variable(tf.random_normal([hidden_num_units, output_num_units], seed=seed))
	}

	biases = {
	    'hidden': tf.Variable(tf.random_normal([hidden_num_units], seed=seed)),
	    'output': tf.Variable(tf.random_normal([output_num_units], seed=seed))
	}

	# Wire it up
	hidden_layer = tf.add(tf.matmul(x, weights['hidden']), biases['hidden'])
	hidden_layer = tf.nn.relu(hidden_layer)
	output_layer = tf.matmul(hidden_layer, weights['output']) + biases['output']

	# Loss fn
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output_layer, y))

	# Optimizer
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

	# Initialize
	init = tf.initialize_all_variables()

	with tf.Session() as sess:
	    # create initialized variables
	    sess.run(init)
	    
	    # train
	    for epoch in xrange(epochs):
	        avg_cost = 0
	        for i in xrange(num_batches):
	       		print "Starting epoch {} batch {}...".format(epoch + 1, i + 1)
	            batch_x, batch_y = get_batch(batch_size, tuples_data, input_domains)
	            _, c = sess.run([optimizer, cost], feed_dict = {x: batch_x, y: batch_y})
	            
	            avg_cost += c
	        avg_cost = 1.0 * avg_cost / num_batches
	        print "Epoch: {}, cost = {:.5f}".format(epoch + 1, avg_cost)
	    
	    print "\nTraining complete!"
	    
	    # find predictions on test set
	    pred_temp = tf.equal(tf.argmax(output_layer, 1), tf.argmax(y, 1))
	    accuracy = tf.reduce_mean(tf.cast(pred_temp, "float"))
	    print "Validation Accuracy:", accuracy.eval({x: val_x.reshape(-1, 784), y: dense_to_one_hot(val_y.values)})
	    
	    predict = tf.argmax(output_layer, 1)
	    pred = predict.eval({x: test_x.reshape(-1, 784)})

train_model()