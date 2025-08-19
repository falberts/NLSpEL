
# NLSpEL

This code is adapted from the following paper:

Shavarani, H. S., & Sarkar, A. (2023). SpEL: Structured prediction for entity linking. arXiv preprint arXiv:2310.14684.

The original source code is avaible [here](https://github.com/shavarani/SpEL).


## Installation

First, install dependencies, preferably in a venv:

```bash
 python -m pip install -r requirements.txt
```

Some packages might require older Python installations. To ensure compatability, you can use Python 3.10 and enter the following lines in the terminal:

```bash
sudo apt-get update
sudo apt-get install python3.10-distutils
python3.10 -m pip install -r requirements.txt
```

Currently, the checkpoints can be downloaded manually from Google Drive:

[nlspel-step-1-ddp.pt]() (link to be added)

[nlspel-step-2.pt]() (link to be added)

After downloading, make sure to add these files to the .checkpoints folder (or create this folder manually in NLSpEL/SpEL/)

Afterwards, model can be used like this:

```python
from transformers import AutoTokenizer
from spel.model import SpELAnnotator, dl_sa
from spel.configuration import device
from spel.utils import get_subword_to_word_mapping
from spel.span_annotation import WordAnnotation, PhraseAnnotation

finetuned_after_step = 2
sentence = "Grace Kelly by Mika reached the top of the UK Singles Chart in 2007."
tokenizer = AutoTokenizer.from_pretrained("pdelobelle/robbert-v2-dutch-ner")

# Load the model:
BERT_MODEL_NAME = "nlspel-step-2.pt"

spel = SpELAnnotator()
spel.init_model_from_scratch(BERT_MODEL_NAME=BERT_MODEL_NAME, device=device)
finetuned_after_step=finetuned_after_step)

# Run NLSpEL:
inputs = tokenizer(sentence, return_tensors="pt")
token_offsets = list(zip(inputs.encodings[0].tokens,inputs.encodings[0].offsets))
subword_annotations = spel.annotate_subword_ids(inputs.input_ids, k_for_top_k_to_keep=10, token_offsets=token_offsets)

# Create word level annotations:
tokens_offsets = token_offsets[1:-1]
subword_annotations = subword_annotations[1:]
for sa in subword_annotations:
    sa.idx2tag = dl_sa.mentions_itos
word_annotations = [WordAnnotation(subword_annotations[m[0]:m[1]], tokens_offsets[m[0]:m[1]])
                    for m in get_subword_to_word_mapping(inputs.tokens(), sentence)]

# Create phrase level annotations:
phrase_annotations = []
for w in word_annotations:
    if not w.annotations:
        continue
    if phrase_annotations and phrase_annotations[-1].resolved_annotation == w.resolved_annotation:
        phrase_annotations[-1].add(w)
    else:
        phrase_annotations.append(PhraseAnnotation(w))

# Print out the created annotations:
for phrase_annotation in phrase_annotations:
   print(dl_sa.mentions_itos[phrase_annotation.resolved_annotation])
```

To see run the evaluation, call:
```bash
python3.10 evaluate_local.py
```

Inside this file, edit the BERT_MODEL_NAME variable to either nlspel-step-1-ddp.pt or nlspel-step-2.pt.

___

TODO:

* Add code used for pre-processing.
* Fix finetune_step2.py (currently broken, returns 0 an NaN scores. Should be a quick fix as it worked earlier last week.)