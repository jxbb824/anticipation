"""
Top-level functions for preprocessing data to be used for training.
"""

from tqdm import tqdm

import numpy as np

from anticipation import ops
from anticipation.config import *
from anticipation.vocab import *
from anticipation.convert import compound_to_events, midi_to_interarrival


def extract_spans(all_events, rate):
    events = []
    controls = []
    span = True
    next_span = end_span = TIME_OFFSET+0
    for time, dur, note in zip(all_events[0::3],all_events[1::3],all_events[2::3]):
        assert(note not in [SEPARATOR, REST]) # shouldn't be in the sequence yet

        # end of an anticipated span; decide when to do it again (next_span)
        if span and time >= end_span:
            span = False
            next_span = time+int(TIME_RESOLUTION*np.random.exponential(1./rate))

        # anticipate a 3-second span
        if (not span) and time >= next_span:
            span = True
            end_span = time + DELTA*TIME_RESOLUTION

        if span:
            # mark this event as a control
            controls.extend([CONTROL_OFFSET+time, CONTROL_OFFSET+dur, CONTROL_OFFSET+note])
        else:
            events.extend([time, dur, note])

    return events, controls


ANTICIPATION_RATES = 10
def extract_random(all_events, rate):
    events = []
    controls = []
    for time, dur, note in zip(all_events[0::3],all_events[1::3],all_events[2::3]):
        assert(note not in [SEPARATOR, REST]) # shouldn't be in the sequence yet

        if np.random.random() < rate/float(ANTICIPATION_RATES):
            # mark this event as a control
            controls.extend([CONTROL_OFFSET+time, CONTROL_OFFSET+dur, CONTROL_OFFSET+note])
        else:
            events.extend([time, dur, note])

    return events, controls


def extract_instruments(all_events, instruments):
    events = []
    controls = []
    for time, dur, note in zip(all_events[0::3],all_events[1::3],all_events[2::3]):
        assert note < CONTROL_OFFSET         # shouldn't be in the sequence yet
        assert note not in [SEPARATOR, REST] # these shouldn't either

        instr = (note-NOTE_OFFSET)//2**7
        if instr in instruments:
            # mark this event as a control
            controls.extend([CONTROL_OFFSET+time, CONTROL_OFFSET+dur, CONTROL_OFFSET+note])
        else:
            events.extend([time, dur, note])

    return events, controls


def maybe_tokenize(compound_tokens):
    # skip sequences with very few events
    if len(compound_tokens) < COMPOUND_SIZE*MIN_TRACK_EVENTS:
        return None, None, 1 # short track

    events, truncations = compound_to_events(compound_tokens, stats=True)
    end_time = ops.max_time(events, seconds=False)

    # don't want to deal with extremely short tracks
    if end_time < TIME_RESOLUTION*MIN_TRACK_TIME_IN_SECONDS:
        return None, None, 1 # short track

    # don't want to deal with extremely long tracks
    if end_time > TIME_RESOLUTION*MAX_TRACK_TIME_IN_SECONDS:
        return None, None, 2 # long track

    # skip sequences more instruments than MIDI channels (16)
    if len(ops.get_instruments(events)) > MAX_TRACK_INSTR:
        return None, None, 3 # too many instruments

    return events, truncations, 0


def tokenize_ia(datafiles, output, augment_factor, idx=0, debug=False):
    assert augment_factor == 1 # can't augment interarrival-tokenized data

    all_truncations = 0
    seqcount = rest_count = 0
    stats = 4*[0] # (short, long, too many instruments, inexpressible)
    np.random.seed(0)

    with open(output, 'w') as outfile:
        concatenated_tokens = []
        for j, filename in tqdm(list(enumerate(datafiles)), desc=f'#{idx}', position=idx+1, leave=True):
            with open(filename, 'r') as f:
                _, _, status = maybe_tokenize([int(token) for token in f.read().split()])

            if status > 0:
                stats[status-1] += 1
                continue

            filename = filename[:-len('.compound.txt')] # get the original MIDI

            # already parsed; shouldn't raise an exception
            tokens, truncations = midi_to_interarrival(filename, stats=True)
            tokens[0:0] = [MIDI_SEPARATOR]
            concatenated_tokens.extend(tokens)
            all_truncations += truncations

            # write out full sequences to file
            while len(concatenated_tokens) >= CONTEXT_SIZE:
                seq = concatenated_tokens[0:CONTEXT_SIZE]
                concatenated_tokens = concatenated_tokens[CONTEXT_SIZE:]
                outfile.write(' '.join([str(tok) for tok in seq]) + '\n')
                seqcount += 1

    if debug:
        fmt = 'Processed {} sequences (discarded {} tracks, discarded {} seqs, added {} rest tokens)'
        print(fmt.format(seqcount, stats[0]+stats[1]+stats[2], stats[3], rest_count))

    return (seqcount, rest_count, stats[0], stats[1], stats[2], stats[3], all_truncations)


def tokenize(datafiles, output, augment_factor, idx=0, debug=False):
    tokens = []
    all_truncations = 0
    seqcount = rest_count = 0
    stats = 4*[0] # (short, long, too many instruments, inexpressible)
    np.random.seed(0)

    with open(output, 'w') as outfile:
        concatenated_tokens = []
        current_music_id = None  # Track current music ID
        
        for j, filename in tqdm(list(enumerate(datafiles)), desc=f'#{idx}', position=idx+1, leave=True):
            # Extract music ID from filename
            music_id = filename.split('/')[-1][:-len('.mid.compound.txt')].rsplit('__', 1)[0]
            
            # If music ID changes, clear concatenated_tokens to avoid merging
            if current_music_id is not None and current_music_id != music_id:
                concatenated_tokens = []  # Discard remaining tokens from previous piece
            
            current_music_id = music_id
            
            with open(filename, 'r') as f:
                all_events, truncations, status = maybe_tokenize([int(token) for token in f.read().split()])

            if status > 0:
                stats[status-1] += 1
                continue

            instruments = list(ops.get_instruments(all_events).keys())
            end_time = ops.max_time(all_events, seconds=False)

            # different random augmentations
            for k in range(augment_factor):
                if k % 10 == 0:
                    # no augmentation
                    events = all_events.copy()
                    controls = []
                elif k % 10 == 1:
                    # span augmentation
                    lmbda = .05
                    events, controls = extract_spans(all_events, lmbda)
                elif k % 10 < 6:
                    # random augmentation
                    r = np.random.randint(1,ANTICIPATION_RATES)
                    events, controls = extract_random(all_events, r)
                else:
                    if len(instruments) > 1:
                        # instrument augmentation: at least one, but not all instruments
                        u = 1+np.random.randint(len(instruments)-1)
                        subset = np.random.choice(instruments, u, replace=False)
                        events, controls = extract_instruments(all_events, subset)
                    else:
                        # no augmentation
                        events = all_events.copy()
                        controls = []

                if len(concatenated_tokens) == 0:
                    z = ANTICIPATE if k % 10 != 0 else AUTOREGRESS

                all_truncations += truncations
                events = ops.pad(events, end_time)
                rest_count += sum(1 if tok == REST else 0 for tok in events[2::3])
                tokens, controls = ops.anticipate(events, controls)
                assert len(controls) == 0 # should have consumed all controls (because of padding)
                
                # Only add SEPARATOR for non-no-augmentation cases
                if k % 10 != 0:
                    tokens[0:0] = [SEPARATOR, SEPARATOR, SEPARATOR]

                if len(concatenated_tokens) > 0:
                    time_offset = ops.max_time(concatenated_tokens, seconds=False) + 10
                    tokens = ops.translate(tokens, time_offset, seconds=False)

                concatenated_tokens.extend(tokens)

                # write out full sequences to file
                while len(concatenated_tokens) >= EVENT_SIZE*M:
                    seq = concatenated_tokens[0:EVENT_SIZE*M]
                    concatenated_tokens = concatenated_tokens[EVENT_SIZE*M:]

                    # relativize time to the context
                    seq = ops.translate(seq, -ops.min_time(seq, seconds=False), seconds=False)
                    assert ops.min_time(seq, seconds=False) == 0
                    if ops.max_time(seq, seconds=False) >= MAX_TIME:
                        stats[3] += 1
                        continue

                    # Add global control
                    seq.insert(0, z)

                    # Add music ID at the end of the sequence
                    outfile.write(' '.join([str(tok) for tok in seq]) + ' ' + music_id + '\n')
                    seqcount += 1

                    # grab the current augmentation controls if we didn't already
                    z = ANTICIPATE if k % 10 != 0 else AUTOREGRESS

    if debug:
        fmt = 'Processed {} sequences (discarded {} tracks, discarded {} seqs, added {} rest tokens)'
        print(fmt.format(seqcount, stats[0]+stats[1]+stats[2], stats[3], rest_count))

    return (seqcount, rest_count, stats[0], stats[1], stats[2], stats[3], all_truncations)
