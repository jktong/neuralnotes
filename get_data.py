#!/usr/bin/env python
from config import *
import csv
import numpy as np
import os.path
import ujson as json
from intervaltree import Interval,IntervalTree

import pyaudio
import wave
import sys

C0 = 220*2**(3/12.0)*(2**-4)
BITRATE = 44100 # samples/second
CHUNK = 1024
WINDOW = np.blackman(CHUNK)

def midinote_to_freq(midinote):
    return C0*2**(midinote/12.0)

def pair_to_melody_matrix(pair, pad=False):
    note_range = 128 + int(pad)

def melody_matrix_to_pair(melody_matrix, pad=False):
    note_range = 128 + int(pad)
    N = melody_matrix.shape[0]
    midi_noteval_pair = np.zeros((N, 2))
    midi_noteval_pair[:,0] = melody_matrix.argmax(axis=1)
    midi_noteval_pair[:,1] = melody_matrix[:,note_range:].argmax(axis=1)
    return midi_noteval_pair

def play_melody(melody, pad=False):
    if melody.shape[1] == 2:
        # treat as midi-noteval pairs
        midinotes = melody[:,0].astype(int)
        notevalues = melody[:,1].astype(int)
    else:
        note_range = 128 + int(pad)
        midi_noteval_pair = melody_matrix_to_pair(melody, pad=pad)
        midinotes = midi_noteval_pair[:,0].astype(int)
        notevalues = midi_noteval_pair[:,1].astype(int)
    frequencies = midinote_to_freq(midinotes)
    wavedata = ''
    count = 0
    print frequencies
    for i, freq in enumerate(frequencies):
        note_len = NOTEVALUE_LENS[NOTEVALUE_LIST[notevalues[i]]]*3
        numframes = int(BITRATE * note_len)
        #restframes = numframes % BITRATE
        
        for x in xrange(numframes):
            wavedata += chr(int(np.sin(x/((BITRATE/freq)/np.pi))*127+128))
            
        #for x in xrange(restframes):
        #    wavedata += chr(128)

    # isntantiate PyAudio
    p = pyaudio.PyAudio()
    
    # open stream
    stream = p.open(format=p.get_format_from_width(1),
                    channels=1,
                    rate=BITRATE,
                    output=True)

    stream.write(wavedata)
    stream.stop_stream()
    stream.close()
    p.terminate()

all_data = np.load(open(MUSICNET_DATA,'rb'))

def get_pieces_by_ensemble(ensemble):
    print 'Collecting {} pieces...'.format(ensemble)
    pieces = []
    with open(MUSICNET_METADATA, 'rb') as csvfile:
        filereader = csv.reader(csvfile)
        for row in filereader:
            if row[4] == ensemble:
                pieces.append(row[0])
    print '{} pieces found!'.format(len(pieces))
    return pieces

def get_piece_melody(note_tree, pad=False):
    notes = []
    begin = note_tree.begin()
    end = note_tree.end()
    next_intervals = note_tree.search(begin, end)
    while len(next_intervals) > 0:
        highest_note = sorted(next_intervals, 
                              key=lambda x:(x[0],-x[2][1]))[0]
        start,next_begin,(_,note,_,_,note_value) = highest_note
        notes.append((note, note_value))
        next_intervals = note_tree.search(next_begin, end, strict=True)
    melody_matrix = notes_to_matrix(notes, pad=pad)
    return melody_matrix

def notes_to_matrix(notes, pad=False):
    note_range = 128 + int(pad)
    note_value_range = len(NOTEVALUE_INDEX) + int(pad)
    melody_matrix = np.zeros((len(notes),note_range + note_value_range))
    for i, note in enumerate(notes):
        melody_matrix[i][note[0]] = 1
        melody_matrix[i][note_range + 
                         NOTEVALUE_INDEX.get(note[1],
                                             NOTEVALUE_INDEX['Quarter'])] = 1
    return melody_matrix


def write_melody_matrices(pad=False):
    print 'Padding set to {}!'.format(pad)
    outfile_name = 'Data/musicnet_pianosolos{}.npz'.format('_padded' if pad else '')
    print 'Writing melody matrices to {}...'.format(outfile_name)
    piano_solos = get_pieces_by_ensemble('Solo Piano')
    melodies = {}
    count = 0
    for piece in piano_solos:
        _, note_tree = all_data[piece]
        melodies[piece] = get_piece_melody(note_tree, pad)
        count += 1
        print '{}/{} pieces done...'.format(count, len(piano_solos))
    with open(outfile_name, 'w') as outfile:
        np.savez(outfile, **melodies)

def write_midi_noteval_pairs():
    outfile_name = 'Data/musicnet_pianosolos_midinoteval.npz'
    print 'Writing midi-noteval pairs to {}...'.format(outfile_name)
    try:
        piano_solos = np.load(open(MUSICNET_PIANOSOLOS),'rb')
        melodies = {piece : melody_matrix_to_pair(piano_solos[piece])
                    for piece in piano_solos.files}
    except IOError:
        return
        piano_solos = get_pieces_by_ensemble('Solo Piano')
        melodies = {}
        count = 0
        for piece in piano_solos:
            _, note_tree = all_data[piece]
            melodies[piece] = get_piece_melody(note_tree, False) # need different func
            count += 1
        print '{}/{} pieces done...'.format(count, len(piano_solos))
        #TODO
        pass
    with open(outfile_name, 'w') as outfile:
        np.savez(outfile, **melodies)

def write_melodies(format='midi-noteval'):
    if format == 'midi-noteval':
        write_midi_noteval_pairs()
    else:
        print 'Pad? (y/n)'
        pad = raw_input().stripe() == 'y'
        write_melody_matrices(pad)

def samples_per_song_for_context(c):
    """
    Params:
    - c is the number of 'context' notes prior to the one we want to predict.
    Return a dictionary with file indices as keys and samples for each song
    as values.
    Each sample contains c+1 rows, each row being a midi, notevalue pair
    The first c rows are the notes from the context and the last row is the
    note to be predicted.
    Each song has n-c samples, where n is the number notes in the song, because
    the sliding window needs to be of size c+1.
    """
    sample_dict = {}
    pieces = np.load(open(MUSICNET_PIANOSOLOS_MIDINOTEVAL, 'rb'))
    for file in pieces.files:
        song = pieces[file]
        n = song.shape[0]
        samples = np.zeros((n-c,c+1,2))
        for i in xrange(n-c):
            samples[i] = song[i:i+c+1]
        sample_dict[file] = samples
    return sample_dict

def all_samples_for_context(c):
    """
    Return a single matrix containing all samples generated by going over
    all windows of size c+1 in each melody.
    Each sample is of shape (c+1, 2)
    """
    sample_dict = samples_per_song_for_context(c)
    N = sum([samples.shape[0] for samples in sample_dict.values()])
    all_samples = np.zeros((N,c+1,2))
    sample_index = 0
    for samples in sample_dict.values():
        n = samples.shape[0]
        all_samples[sample_index:sample_index + n] = samples
        sample_index += n
    return all_samples

def pairs_to_melody_matrix(midi_noteval_pairs, max, pad=True):
    note_range = 128 + int(pad)
    note_value_range = len(NOTEVALUE_INDEX) + int(pad)
    c = midi_noteval_pairs.shape[0]-1
    max = max if pad else c # if we're not padding max is the same as c
    melody_matrix = np.zeros((max + 1, note_range + note_value_range))
    for i in xrange(max+1):
        if i < c+1:
            midi_note = midi_noteval_pairs[i][0]
            note_value = midi_noteval_pairs[i][1]
            row = i
            if i == c:
                row = -1
        else:
            midi_note = note_range - 1
            note_value = note_value_range - 1
            row = i - 1
        melody_matrix[row, midi_note] = 1
        melody_matrix[row, note_value] = 1
    return melody_matrix

def samples_to_padded_melody_matrices(samples, c, max):
    """
    Params:
    - samples is a matrix of midi_noteval samples of shape (c+1,2)
    Return a new matrix with each sample replaced by a padded melody matrix
    of size (max+1,152)
    """
    note_range = 129
    note_value_range = len(NOTEVALUE_INDEX) + 1
    N = samples.shape[0]
    new_samples = np.zeros((N, max+1, note_range + note_value_range))
    for i in xrange(N):
        new_samples[i] = pairs_to_melody_matrix(samples[i], max, True)
    return new_samples

def all_samples_all_contexts_padded(min, max):
    note_range = 129
    note_value_range = len(NOTEVALUE_INDEX) + 1
    samples_list = []
    for c in xrange(min, max+1):
        samples_list.append(all_samples_for_context(c))
    N = sum([samples.shape[0] for samples in samples_list])
    all_samples = np.zeros((N, max+1, note_range + note_value_range))
    sample_index = 0
    for samples in samples_list:
        n = samples.shape[0]
        padded_samples = samples_to_padded_melody_matrices(samples, c, max)
        all_samples[sample_index:sample_index + n] = padded_samples
    return all_samples

if __name__=='__main__':
    write_melodies(format='midi-noteval')
