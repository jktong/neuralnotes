#!/usr/bin/python

import pyaudio
import wave
import sys
import numpy as np

CHUNK = 1024
WINDOW = np.blackman(CHUNK)

TET_NOTES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
ENHARMONICS = {'Cb':'B',
               'Db':'C#',
               'Eb':'D#',
               'E#':'F',
               'Fb':'E',
               'Gb':'F#',
               'Ab':'G#',
               'Bb':'A#',
               'B#':'C'}
# Take the pitch for A4 (220Hz),
# transpose up three semitones to C4,
# transpose down four octaves to C0
C0 = 220*2**(3/12.0)*(2**-4)
BITRATE = 44100

def pitch_to_freq(pitch):
    ''' Return the frequency of a pitch denoted according to
    Scientific Pitch Notation.
    https://en.wikipedia.org/wiki/Scientific_pitch_notation
    '''
    octave = pitch[-1]
    note = pitch[:2]
    if note in ENHARMONICS:
        note = ENHARMONICS[note]
    freq = C0*2**TET_NOTES.index(note)*2**octave
    return freq

def play_pitch(pitch, seconds):
    freq = pitch
    if type(pitch) == str:
        freq = pitch_to_freq(pitch)
    p = pyaudio.PyAudio()
    stream = p.open(format = p.get_format_from_width(1),
                    channels = 1,
                    rate = BITRATE,
                    output = True)
    num_frames = int(BITRATE * seconds)
    rest_frames = num_frames % BITRATE
    wavedata = ''
    np.arange(num_frames) / (BITRATE/freq)

def play_note(note_freq=261.63, note_len=1):
    #See http://www.phy.mtu.edu/~suits/notefreqs.html
    # note_freq is in Hz; 261.63=C4-note
    # note_len is in seconds

    #See http://en.wikipedia.org/wiki/Bit_rate#Audio
    BITRATE = 44100 #number of frames per second/frameset.

    NUMBEROFFRAMES = int(BITRATE * note_len)
    RESTFRAMES = NUMBEROFFRAMES % BITRATE
    WAVEDATA = ''

    for x in xrange(NUMBEROFFRAMES):
        #print np.sin(x/((BITRATE/note_freq)/np.pi))*127+128
        WAVEDATA = WAVEDATA+chr(int(np.sin(x/((BITRATE/note_freq)/np.pi))*127+128))

    #fill remainder of frameset with silence
    for x in xrange(RESTFRAMES):
        WAVEDATA = WAVEDATA+chr(128)

    p = pyaudio.PyAudio()
    stream = p.open(format = p.get_format_from_width(1), 
                    channels = 1, 
                    rate = BITRATE, 
                    output = True)
    stream.write(WAVEDATA)
    stream.stop_stream()
    stream.close()
    p.terminate()

def play_notes(note_freqs):
    BITRATE = 44100 #number of frames per second/frameset.

    NUMBEROFFRAMES = int(BITRATE * note_len)
    RESTFRAMES = NUMBEROFFRAMES % BITRATE
    WAVEDATA = ''

    for x in xrange(NUMBEROFFRAMES):
        #print np.sin(x/((BITRATE/note_freq)/np.pi))*127+128
        WAVEDATA = WAVEDATA+chr(int(np.sin(x/((BITRATE/note_freq)/np.pi))*127+128))

    #fill remainder of frameset with silence
    for x in xrange(RESTFRAMES):
        WAVEDATA = WAVEDATA+chr(128)

    p = pyaudio.PyAudio()
    stream = p.open(format = p.get_format_from_width(1), 
                    channels = 1, 
                    rate = BITRATE, 
                    output = True)
    stream.write(WAVEDATA)
    stream.stop_stream()
    stream.close()
    p.terminate()

if __name__=='__main__':
    if len(sys.argv) < 2:
        print ('Plays a wave file.\n\nUsage: {} filename.wav'.format(sys.argv[0]))
        sys.exit(-1)

    wf = wave.open(sys.argv[1], 'rb')
    sampwidth = wf.getsampwidth()
    RATE = wf.getframerate()
    
    # isntantiate PyAudio
    p = pyaudio.PyAudio()
    
    # open stream
    stream = p.open(format=p.get_format_from_width(sampwidth),
                    channels=wf.getnchannels(),
                    rate=RATE,
                    output=True)
    
    # read data
    data = wf.readframes(CHUNK)
    
    frequencies = []
    while len(data) == CHUNK*sampwidth:
        # play data
        stream.write(data)
        # unpack data and times by the hamming window
        indata = np.array(wave.struct.unpack('%dh'%(len(data)/sampwidth), data)) * WINDOW
        # take the fft and square each value
        fftData = abs(np.fft.rfft(indata))**2
        # find the maximum
        which = fftData[1:].argmax() + 1
        # use quadratic interpolation around the max
        if which != len(fftData) - 1:
            y0, y1, y2 = np.log(fftData[which-1:which+2:])
            x1 = (y2 - y0) * .5 / (2 * y1 - y2 - y0)
            # find the frequency and output it
            freq = (which + x1) * RATE / CHUNK
            frequencies.append(freq)
            #print 'The freq is {} Hz.'.format(freq)
        else:
            freq = which * RATE/ CHNUNK
            frequencies.append(freq)
            #print 'The freq is {} Hz.'.format(freq)
            
        data = wf.readframes(CHUNK)
    if data:
        stream.write(data)
    '''
    # play stream
    while len(data) > 0:
        stream.write(data)
        data = wf.readframes(CHUNK)
    '''

    print frequencies
    
    #play_notes(note_len=3)
        
    # stop stream
    stream.stop_stream()
    stream.close()

    p.terminate()
    
