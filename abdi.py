import tensorflow as tf
import numpy as np

class Model(object):
    def __init__(self, name,
                 num_notes=100,
                 num_lengths=25,
                 notes_dim=50,
                 lengths_dim=10):
        self.name = name
        self.num_notes = num_notes
        self.num_lengths = num_lengths
        self.notes_dim = notes_dim
        self.lengths_dim = lengths_dim
        self.max_N = 10

        self.build()

    def run_rnn(self, tensor, cell, scope):
        outputs, _ = tf.nn.dynamic_rnn(cell, tensor, dtype=tf.float32, scope=scope, time_major=False)
        return outputs[:,-1,:]

    def to_embeddings(self, tensor, embeddings):
        steps = tf.unpack(tensor, axis=1)
        new_steps = [tf.matmul(step, embeddings) for step in steps]
        tensor = tf.pack(new_steps, axis=1)
        return tensor

    def build(self):
        # data: [S, N, D]
        # S = # of songs/examples
        # N+1 = # of notes (N is context notes)
        # D = length of 2-hot vector

        # format data
        self.data = tf.placeholder(tf.float32, shape=[None, self.max_N+1, self.num_notes + self.num_lengths])
        X = self.data[:,:-1,:] # shape: (S, N, D)
        notes = X[:,:,:self.num_notes] # shape: (S, N, num_notes)
        lengths = X[:,:,self.num_notes:] # shape: (S, N, num_lengths)
        y = self.data[:,-1,:self.num_notes] # shape: (S, num_notes)
        
        # define parameters
        with tf.variable_scope(self.name) as scope:
            # embedding weights for notes and lengths
            self.notes_emb = tf.get_variable("notesEmb", shape=[self.num_notes, self.notes_dim],
                                             initializer=tf.random_normal_initializer())
            self.lengths_emb = tf.get_variable("lengthsEmb", shape=[self.num_lengths, self.lengths_dim],
                                               initializer=tf.random_normal_initializer())

            # single rnn
            self.cell = tf.nn.rnn_cell.BasicRNNCell(self.notes_dim+self.lengths_dim)

            # feed-forward weights
            self.final_weights = tf.get_variable('finalW', shape=[self.notes_dim+self.lengths_dim,
                                                                  self.num_notes],
                                                 initializer=tf.random_normal_initializer())
            self.final_bias = tf.get_variable('finalB', shape=[self.num_notes],
                                              initializer=tf.random_normal_initializer())

        # computation graph
        notes = self.to_embeddings(notes, self.notes_emb) # shape: (S, N, notes_dim)
        lengths = self.to_embeddings(lengths, self.lengths_emb) # shape: (S, N, lengths_dim)
        feature_vecs = tf.concat(2, [notes, lengths]) # shape: (S, N, notes_dim+lengths_dim)
        feature_vec = self.run_rnn(feature_vecs, self.cell, scope) # shape: (S, notes_dim+lengths_dim)
        self.logits = tf.matmul(feature_vec, self.final_weights) + self.final_bias # shape: (S, notes_dim)

        # loss functions
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.logits, y)) # shape: scalar

        # session/saver management
        self.sess = tf.get_default_session()
        if self.sess is None:
            self.sess = tf.InteractiveSession()
        self.saver = tf.train.Saver()
        
        return self

    def init(self):
        tf.initialize_all_variables().run()

        return self

    def load(self, model_path):
        self.saver.restore(self.sess, model_path)

    def save(self, model_path):
        self.saver.save(self.sess, model_path)

    def train(self, data,
              learning_rate=0.001,
              epochs=100):
        train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss)
        for i in xrange(epochs):
            train_op.run(feed_dict={self.data: data})
            print 'epoch {} loss:'.format(i), self.loss.eval(feed_dict={self.data: data})

        return self

    def predict(self, data):
        pass
        

if __name__ == '__main__':
    # testing that things work
    m = Model('mymodel')

    # uncomment one
    #m.init() # run this if you are training the model for the first time
    #m.load('./models/test1') # else load a model already on disk

    # fake data (data2 doesn't work yet, needs paddings)
    data1 = np.load('./data/fake_S100_N10_100_25.npy')
    data2 = np.load('./data/fake_S100_N5_100_25.npy')

    # train and save
    m.train(data1).save('./models/model_two')
