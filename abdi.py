import argparse
import tensorflow as tf
import numpy as np
import ujson as json

#from get_data import get_iterator_per_song_per_context

class Model(object):
    def __init__(self, name,
                 num_notes=129,
                 num_lengths=24,
                 notes_dim=50,
                 lengths_dim=10,
                 hidden_dim=256,
                 max_N=10):
        self.name = name
        self.num_notes = num_notes
        self.num_lengths = num_lengths
        self.notes_dim = notes_dim
        self.lengths_dim = lengths_dim
        self.hidden_dim = hidden_dim
        self.max_N = max_N

        self.build()

    def run_rnn(self, tensor, seq_lengths, cell, scope):
        outputs, _ = tf.nn.dynamic_rnn(cell, tensor,
#                                       sequence_length=seq_lengths,
                                       dtype=tf.float32,
                                       scope=scope,
                                       time_major=False)
        return outputs[:,-1,:]

    def to_embeddings(self, tensor, embeddings):
        steps = tf.unpack(tensor, axis=1)
        new_steps = [tf.matmul(step, embeddings) for step in steps]
        tensor = tf.pack(new_steps, axis=1)
        return tensor

    def train_step(self, batch):
        loss, acc, _ = self.sess.run([self.loss, self.acc, self.train_op], feed_dict={self.data: batch})
        return loss, acc

    def train(self, it,
              epochs=2500,
              learning_rate=0.001):
        self.history = []
        self.train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss)
        data = it()
        try:
            for i in xrange(epochs):
                try:
                    batch = data.next()
                except StopIteration:
                    data = it()
                    batch = data.next()
                loss, acc = self.train_step(batch)
                #self.history.append((loss, acc))
                print '[epoch {}]\tloss: {}\tacc: {}'.format(i, loss, acc)
                del batch
        except KeyboardInterrupt:
            pass

        return self

    def build(self):
        tf.set_random_seed(42)
        
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
            self.cell = tf.nn.rnn_cell.BasicLSTMCell(self.notes_dim+self.lengths_dim)

            # feed-forward weights
            self.hidden_weights = tf.get_variable('hiddenW', shape=[self.notes_dim+self.lengths_dim, self.hidden_dim],
                                                  initializer=tf.random_normal_initializer())
            self.hidden_bias = tf.get_variable('hiddenB', shape=[self.hidden_dim],
                                               initializer=tf.random_normal_initializer())
            self.final_weights = tf.get_variable('finalW', shape=[self.hidden_dim, self.num_notes-1],
                                                 initializer=tf.random_normal_initializer())
            self.final_bias = tf.get_variable('finalB', shape=[self.num_notes-1],
                                              initializer=tf.random_normal_initializer())

        # computation graph
        notes = self.to_embeddings(notes, self.notes_emb) # shape: (S, N, notes_dim)
        lengths = self.to_embeddings(lengths, self.lengths_emb) # shape: (S, N, lengths_dim)
        feature_vecs = tf.concat(2, [notes, lengths]) # shape: (S, N, notes_dim+lengths_dim)
        seq_lengths = self.max_N - tf.reduce_sum(X, reduction_indices=1)[:,-1] # shape: (S,)
        feature_vec = self.run_rnn(feature_vecs, seq_lengths, self.cell, scope) # shape: (S, notes_dim+lengths_dim)
        hidden_vec = tf.nn.relu(tf.add(tf.matmul(feature_vec, self.hidden_weights), self.hidden_bias)) # shape: (S, hidden_dim)
        self.logits = tf.add(tf.matmul(hidden_vec, self.final_weights), self.final_bias) # shape: (S, notes_dim)

        # loss and accuracy
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.logits, y)) # shape: scalar
        self.acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.logits, 1), tf.argmax(y, 1)), tf.float32)) # shape: scalar

        # session/saver management
        self.sess = tf.get_default_session()
        if self.sess is None:
            self.sess = tf.InteractiveSession()
        self.saver = tf.train.Saver()
        
        return self

    def predict(self, data):
        return tf.nn.softmax(self.logits).eval(feed_dict={self.data: data})

    def accuracy(self, data):
        return self.acc.eval(feed_dict={self.data: data})

    def init(self):
        tf.initialize_all_variables().run()
        return self

    def load(self, model_path):
        self.saver.restore(self.sess, model_path)
        return self

    def save(self, model_path):
        self.saver.save(self.sess, model_path)
        return self

    def save_history(self, hist_path):
        with open(hist_path, 'w') as f:
            json.dump(self.history, f)


def main(args):
    contexts = [2,4,8,16,32]
    max_N = max(contexts)
    it = lambda: get_iterator_per_song_per_context(contexts, max_N, pad_end=False, mode='train')

    m = Model(args.name, max_N=max_N)
    if args.load_path:
        m.load(args.load_path)
    else:
        m.init()

    m.train(it, epochs=args.epochs, learning_rate=args.learning_rate)
    m.save(args.save_path)

    if args.history_path:
        m.save_history(args.history_path)

    print vars(args)

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
    parser.add_argument('-i', '--history_path',
                        help='Save path for loss/accuracy history.')
    parser.add_argument('-e', '--epochs',
                        help='Number of epochs.',
                        type=int,
                        default=10000)
    parser.add_argument('-r', '--learning_rate',
                        help='Learning rate.',
                        type=float,
                        default=0.001)
    args = parser.parse_args()
    main(args)
