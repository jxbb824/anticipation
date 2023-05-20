# Anticipatory Music Infilling Models

Implementation of the anticipatory infilling models described in Anticipatory Autoregressive Models for Controllable Music Generation.

by [__John Thickstun__](https://johnthickstun.com/), [__David Hall__](http://dlwh.org/), [__Chris Donahue__](https://chrisdonahue.com/), and [__Percy Liang__](https://cs.stanford.edu/~pliang/).

-------------------------------------------------------------------------------------

TODO: description

Begin by installing the anticipation package.

```
pip install .
```

## Software Dependencies

TODO: requirements.txt

## Preprocessing the LakhMidi dataset

The LakhMidi dataset contains over 170,000 MIDI files and the following preprocessing scripts require considerable computation, system memory, and storage resources to execute. See the [Resource Management](###resource-management) section for discussion of how to manage the demands of preprocessing within the constraints of a particular system.

Download the full [LakhMidi dataset](https://colinraffel.com/projects/lmd/) from [here](http://hog.ee.columbia.edu/craffel/lmd/lmd_full.tar.gz). Extract the data to `DATAPATH`.

```
tar -xvvf lmd_full.tar.gz -C $DATAPATH
```

Preprocess the binary MIDI files to an intermediate text representation. 
```
python scripts/midi-preprocess.py $DATAPATH/lmd_full
```
Tokenize batches of intermediate LakhMIDI data files according to the vocabulary defined in `src/settings/vocab.py`. The top-level script depends upon the directory structure of the LakhMIDI dataset. Parallelism is again controlled by `PREPROC_WORKERS`.

```
python scripts/tokenize-lakh.py $DATAPATH/lmd_full
```

Define the train/validation/test splits. LakhMidi files are named according to their (hexadecimal) MD5 checksum: our convention is to use files starting with `f` as the test set, files starting with `e` as validation, and the rest of the dataset for training.
```
mv $DATAPATH/lmd_full/tokenized-events-e.txt $DATAPATH/valid.txt
mv $DATAPATH/lmd_full/tokenized-events-f.txt $DATAPATH/test.txt
cat $DATAPATH/lmd_full/tokenized-events-*.txt > $DATAPATH/lmd_full/train-ordered.txt
```

Finally, we must shuffle our training data. For this shuffle procedure, your have a machine with enough RAM to read the whole training data file.
```
shuf $DATAPATH/lmd_full/train-ordered.txt > $DATAPATH/train.txt
```

The final preprocessed train/valid/test splits are available at `DATAPATH`.

### Resource Management

**Compute**. Preprocessing scripts are designed to run with multiprocessing parallelism: you can configure the number of workers using `PREPROC_WORKERS` in `src/settings/constants.py`.

**Memory**. The most memory intensive operation is the final shuffle operation, which requires the entire final training dataset to be loaded into memory. Alternative memory-efficient solutions do exist for shuffling (or approximately shuffling) the lines of a file, which you may wish to explore if memory is a constraint. Warning: we have observed that approximate shuffling using a *local* shuffle of the training data is not sufficient to achieve good model performance and should be avoided.

**Disk**. The base preprocessed LakhMidi dataset--prepared for autoregressive modeling without anticipation--is about 10Gb. The size of the dataset when prepared for anticipatory autoregressive modeling is a multiple of the base dataset size, controlled by AUGMENT\_FACTOR in `src/settings/constants.py`. See the [Cleanup](###cleanup) for cleaning up temporary files on disk after preprocessing.

### Cleanup

Each stage of preprocessing generates temporary files with intermediate outputs. After completing preprocessing, these temporary files can be cleaned up as follows.

Delete the intermediate data files generated by the `midi-preprocess` script.
```
rm $DATAPATH/lmd_full/*/*.txt
```

Delete the intermediate tokenized events generated by the `tokenize-lakh` script.
```
rm $DATAPATH/lmd_full/tokenized-events-*.txt
```

Delete the ordered training data file, which is superseded by the shuffled version.
```
rm $DATAPATH/lmd_full/train-ordered.txt
```

## Reproducing the Human Evaluation Procedure

See the [humaneval](humaneval) directory for a description of how to reproduce the human evaluations described in the paper.
