import h5py
import csv
import numpy as np


#f = h5py.File('./data/musicnet.h5')

def load_metadata():
    metadata = {}
    with open('./data/musicnet_metadata.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            metadata[row['id']] = row
    return metadata

def load_data(f, _id):
    '''
    All 330 ids can be found at the bottom of:
    http://homes.cs.washington.edu/~thickstn/stats.html
    '''
    _id = u'id_{}'.format(_id)
    group = f[_id]
    return group

np.random.seed(42)
def make_fake_data(S, N, num_notes, num_lengths):
    data = np.zeros((S, N+1, num_notes+num_lengths))
    for i in xrange(S):
        for j in xrange(N+1):
            idx1 = np.random.randint(num_notes)
            idx2 = np.random.randint(num_lengths) + num_notes
            data[i,j,[idx1,idx2]] = 1
    np.save('./data/fake_S{}_N{}_{}_{}.npy'.format(S, N, num_notes, num_lengths), data)
    
