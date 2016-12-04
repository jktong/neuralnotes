#!/usr/bin/env python
from config import *
import csv
import numpy as np
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

def play_melody(melody_matrix):
    midinotes = melody_matrix.argmax(axis=1)
    notevalues = melody_matrix[:,128:].argmax(axis=1)
    frequencies = midinote_to_freq(midinotes)
    wavedata = ''
    count = 0
    for i, freq in enumerate(frequencies):
        note_len = NOTEVALUE_LENS[NOTEVALUE_LIST[notevalues[i]]]*2
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

def get_piece_melody(note_tree):
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
    melody_matrix = notes_to_matrix(notes)
    return melody_matrix

def notes_to_matrix(notes):
    melody_matrix = np.zeros((len(notes),128 + len(NOTEVALUE_INDEX)))
    for i, note in enumerate(notes):
        melody_matrix[i][note[0]] = 1
        melody_matrix[i][128 + 
                         NOTEVALUE_INDEX.get(note[1],
                                             NOTEVALUE_INDEX['Quarter'])] = 1
    return melody_matrix


if __name__=='__main__':
    piano_solos = get_pieces_by_ensemble('Solo Piano')
    melodies = {}
    count = 0
    for piece in piano_solos:
        _, note_tree = all_data[piece]
        melodies[piece] = get_piece_melody(note_tree)
        count += 1
        print '{}/{} pieces done...'.format(count, len(piano_solos))
    with open('Data/musicnet_pianosolos.npz', 'w') as outfile:
        np.savez(outfile, **melodies)
