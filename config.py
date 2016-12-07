MUSICNET_DATA='Data/musicnet.npz'
MUSICNET_METADATA='Data/musicnet_metadata.csv'
MUSICNET_PIANOSOLOS='Data/musicnet_pianosolos.npz'
MUSICNET_PIANOSOLOS_PADDED='Data/musicnet_pianosolos_padded.npz'
MUSICNET_PIANOSOLOS_MIDINOTEVAL='Data/musicnet_pianosolos_midinoteval.npz'

NOTEVALUE_LENS={'Triplet Sixty Fourth':1.0/64/3,
                'Sixty Fourth':1.0/64,
                'Dotted Sixty Fourth':1.5/64,
                'Triplet Thirty Second':1.0/32/3,
                'Thirty Second':1.0/32,
                'Dotted Thirty Second':1.5/32,
                'Triplet Sixteenth':1.0/16/3,
                'Sixteenth':1.0/16,
                'Dotted Sixteenth':1.5/16, 
                'Triplet':1.0/8/3,
                'Eighth':1.0/8,
                'Dotted Eighth':1.5/8,
                'Triplet Quarter':1.0/4/3,
                'Quarter':1.0/4,
                'Tied Quarter-Sixty Fourth':17.0/64,
                'Tied Quarter-Thirty Second':9.0/32,
                'Tied Quarter-Sixteenth':5.0/16,
                'Tied Quarter-Eighth':3.0/8,
                'Dotted Quarter':1.5/4,
                'Half':1.0/2, 
                'Dotted Half':1.5/2,
                'Whole':1.0,
                'Dotted Whole':1.5}

NOTEVALUE_LIST=sorted(NOTEVALUE_LENS.keys(), key=lambda x: NOTEVALUE_LENS[x])

NOTEVALUE_INDEX={nv:i for i, nv in enumerate(NOTEVALUE_LIST)}

RANDOM_STATE = np.random.RandomState(42)
