import argparse
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

    def run_rnn(self, tensor, seq_lengths, cell, scope):
        outputs, _ = tf.nn.dynamic_rnn(cell, tensor,
                                       sequence_length=seq_lengths,
                                       dtype=tf.float32,
                                       scope=scope,
                                       time_major=False)
        return outputs[:,-1,:]

    def to_embeddings(self, tensor, embeddings):
        steps = tf.unpack(tensor, axis=1)
        new_steps = [tf.matmul(step, embeddings) for step in steps]
        tensor = tf.pack(new_steps, axis=1)
        return tensor

    def train_step(self, data):
        loss, acc, _ = self.sess.run([self.loss, self.acc, self.train_op], feed_dict={self.data: data})
        return loss, acc

    def train(self, data,
              learning_rate=0.001,
              epochs=100):
        self.train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss)
        for i in xrange(epochs):
            loss, acc = self.train_step(data)
            print '[epoch {}]\tloss: {}\tacc: {}'.format(i, loss, acc)

        return self

    def predict(self, data):
        return tf.nn.softmax(self.logits).eval(feed_dict={self.data: data})

    def build(self):
        # data: [S, N, D]
        # S = # of songs/examples
        # N+1 = # of notes (N is # of context notes)
        # D = length of 2-hot vector

        # format data
        self.data = tf.placeholder(tf.float32, shape=[None, self.max_N+1, self.num_notes + self.num_lengths])
        #self.seq_
        X = self.data[:,:-1,:] # shape: (S, N, D)
        notes = X[:,:,:self.num_notes] # shape: (S, N, num_notes)
        lengths = X[:,:,self.num_notes:] # shape: (S, N, num_lengths)
        y = self.data[:,-1,:self.num_notes-1] # shape: (S, num_notes)
        
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
                                                                  self.num_notes-1],
                                                 initializer=tf.random_normal_initializer())
            self.final_bias = tf.get_variable('finalB', shape=[self.num_notes-1],
                                              initializer=tf.random_normal_initializer())

        # computation graph
        notes = self.to_embeddings(notes, self.notes_emb) # shape: (S, N, notes_dim)
        lengths = self.to_embeddings(lengths, self.lengths_emb) # shape: (S, N, lengths_dim)
        feature_vecs = tf.concat(2, [notes, lengths]) # shape: (S, N, notes_dim+lengths_dim)
        seq_lengths = self.max_N - tf.reduce_sum(X, reduction_indices=1)[:,-1] # shape: (S,)
        feature_vec = self.run_rnn(feature_vecs, seq_lengths, self.cell, scope) # shape: (S, notes_dim+lengths_dim)
        self.logits = tf.matmul(feature_vec, self.final_weights) + self.final_bias # shape: (S, notes_dim)

        # loss and accuracy
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.logits, y)) # shape: scalar
        self.acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.logits, 1), tf.argmax(y, 1)), tf.float32)) # shape: scalar

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
        return self

    def save(self, model_path):
        self.saver.save(self.sess, model_path)
        return self


def main(args):
    # testing that things work
    m = Model(args.name)

    if args.load_path:
        m.load(args.load_path)
    else:
        m.init()

    # padding function for testing
    def pad(d, num_notes, num_lengths):
        X = d[:,:-1,:]
        Y = d[:,-1:,:]
        S, N, D = X.shape
        assert num_notes + num_lengths == D
        if N < 10:
            diff = 10 - N
            X_pad = np.zeros((S, diff, D))
            X_pad[:,:,[num_notes-1,-1]] = 1
            new_d = np.concatenate((X, X_pad, Y,), axis=1)
            return new_d
        else:
            return d
            
    #data = pad(np.load('./data/fake_S1000_N10_100_25.npy'), 100, 25)
    data = pad(np.load('./data/fake_S1000_N3_100_25.npy'), 100, 25)
    m.train(data)
    m.save(args.save_path)
    print m.predict(data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name',
                        help='Name for model.',
                        default='main')
    parser.add_argument('-l', '--load_path',
                        help='Path to model on disk. Will initialize model params randomly if not given.')
    parser.add_argument('-s', '--save_path',
                        help='Save path for model. Required.',
                        required=True)
    args = parser.parse_args()
    main(args)
