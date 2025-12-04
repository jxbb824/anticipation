Note: This repository extends the upstream Anticipatory Music Transformer project; the guide below is intended for new users to set up and experiment with the code.

- Install: `pip install -r requirements.txt && pip install -e .`; ensure the `dattri` package is installed for LoGra/TracIn/TRAK attribution scripts.
- Fine-tune: `python finetune.py --train_file <train.txt> --valid_file <valid.txt> --output_dir <out_dir> [--exclude_indices ...]`. For TracIn checkpoints use `finetune_tracin.py`.
- Generate samples: `python generate_samples.py --model_path <model_dir> --output_file <samples.txt>`. Prompt-conditioned generation: `python generate_prompted_samples.py --model_path <model_dir> --source_file <prompts.txt> --output_file <samples.txt>`.
- Attribution scores: `score.py` (LoGra), `score_trak.py` (TRAK), and `score_tracin.py` (TracIn) accept explicit train/valid/model paths and emit `.pt` score matrices.
- Similarity matrices: `calculate_audio_similarity.py` (CLAP) and `calculate_audio_similarity_mert.py` (MERT) take train/test audio dirs; `calculate_melody_similarity_pmi.py` computes PMI melody similarity from train/test token files.
- Analysis/plots: `analyze_rank_similarity_multi.py` draws similarity-vs-rank curves from score/similarity tensors; `batch_causal_similarity_exp.py` runs the HALS removal experiment end-to-end given train/valid/generated paths and a similarity tensor.

# Anticipatory Music Transformer

Implementation of the methods described in [Anticipatory Music Transformer](https://arxiv.org/abs/2306.08620).

by [__John Thickstun__](https://johnthickstun.com/), [__David Hall__](http://dlwh.org/), [__Chris Donahue__](https://chrisdonahue.com/), and [__Percy Liang__](https://cs.stanford.edu/~pliang/).

-------------------------------------------------------------------------------------

This repository provides the code for creating anticipatory training datasets, and for sampling from models trained with anticipation. It does _not_ contain code for training these models: you may use the preprocessed datasets constructed here as input to your favorite codebase for training autoregressive transformer models. We used the [Levanter](https://github.com/stanford-crfm/levanter) codebase to train models, and include instructions [here](train) for training an Anticipatory Music Transformer with Levanter.

For additional detail about this work, see the [paper](https://arxiv.org/abs/2306.08620). You may also be interested in this [blog post](https://crfm.stanford.edu/2023/06/16/anticipatory-music-transformer.html).

Pretrained models are hosted by the Center for Research on Foundation Models (CRFM) on the [HuggingFace Hub](https://huggingface.co/stanford-crfm). 

This project is licensed under the terms of the Apache License, Version 2.0.

Begin by installing the anticipation package (from the root of this repository).

```
pip install .
```

## Software Dependencies

Run the following command to install dependencies.

```
pip install -r requirements.txt
```

## Generating Music with an Anticipatory Music Transformer

See the [Colab](https://colab.research.google.com/drive/1HCQDtGFwROpHRqcmZbV0byqbxDb74YGu?usp=sharing) notebook for interactive examples of music generation using the Anticipatory Music Transformer.

Load a pretrained model using the HuggingFace Transformers package, e.g.:

```
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained('stanford-crfm/music-medium-800k').cuda()
```

Sample from this model using the custom `generate` function implementated by the anticipation package. You can convert generated event tokens to midi using the `events_to_midi` function:

```
from anticipation.sample import generate
from anticipation.convert import events_to_midi

length = 10 # time in seconds
events = generate(model, start_time=0, end_time=length, top_p=.98)
mid = events_to_midi(events)
mid.save('generated.mid')
```

Load your own MIDI and tokenize it using the `midi_to_events` function.

```
from anticipation.convert import midi_to_events

events = midi_to_events('examples/strawberry.mid')
```

To isolate a segment of a longer stream of events, use the `ops` library to clip the stream and translate the clipped segment to time zero. To isolate a melodic line, use `extract_instruments`:

```
from anticipation import ops
from anticipation.tokenize import extract_instruments

events = ops.clip(events, 41, 41+20)
events = ops.translate(events, -ops.min_time(events, seconds=False))

events, melody = extract_instruments(events, [53])
```

To generate an accompaniment to an isolated melody, call the `generate` function using the melody as control inputs. Recombine the generated accompaniment with the melody controls using `ops.combine`:

```
history = ops.clip(events, 0, 5, clip_duration=False)
accompaniment = generate(model, 5, 20, inputs=history, controls=melody, top_p=.98)
completed_events = ops.combine(accompaniment, melody)
mid = events_to_midi(completed_events)
mid.save('generated.mid')
```

See the [Colab](https://colab.research.google.com/drive/1HCQDtGFwROpHRqcmZbV0byqbxDb74YGu?usp=sharing) notebook for additional examples of infilling control using the Anticipatory Music Transformer.

## Training an Anticipatory Music Transformer

See the [train](train) directory for instructions on preprocessing the Lakh MIDI dataset and using [Levanter](https://github.com/stanford-crfm/levanter) to train an Anticipatory Music Transformer.

## Reproducing the Human Evaluation Procedure

See the [humaneval](humaneval) directory for instructions on reproducing data used for the human evaluation results reported in the paper.

-------------------------------------------------------------------------------------

To reference this work, please cite

```bib
@article{thickstun2023anticipatory,
  title={Anticipatory Music Transformer},
  author={Thickstun, John and Hall, David and Donahue, Chris and Liang, Percy},
  journal={arXiv preprint arXiv:2306.08620},
  year={2023}
}
```
