"""Recover readable HW6 artifacts from saved notebook outputs; never invent run timestamps."""
import base64
import json
from pathlib import Path

notebook = json.loads(Path('HW6_STL10_Representation_Learning.ipynb').read_text(encoding='utf-8'))
log = [
    'Homework 6 - Console output recovered from the executed notebook\n',
    'Source: HW6_STL10_Representation_Learning.ipynb\n',
    'The original Colab RUN_LOG.txt was not downloaded. The text below preserves\n',
    'saved console output in cell order. Original wall-clock timestamps are not\n',
    'available in these outputs and have not been reconstructed or invented.\n',
    'This recovered log does not fully meet the timestamp requirement in Section 0.3.\n',
    'Epoch durations are retained exactly as printed.\n\n',
]
figures = Path('figures')
figures.mkdir(exist_ok=True)
image_names = iter(['nearest_neighbors.png', 'training_losses.png'])
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] != 'code' or not cell.get('outputs'):
        continue
    log.append(f'--- Notebook cell {i+1}; execution count {cell.get("execution_count")} ---\n')
    for output in cell['outputs']:
        if output['output_type'] == 'stream':
            log.append(''.join(output.get('text', [])))
        elif output['output_type'] == 'error':
            log.append('\n'.join(output.get('traceback', [])) + '\n')
        elif 'image/png' in output.get('data', {}):
            path = figures / next(image_names)
            path.write_bytes(base64.b64decode(''.join(output['data']['image/png'])))
            log.append(f'[Saved figure: {path.as_posix()}]\n')
    log.append('\n')
Path('RUN_LOG.txt').write_text(''.join(log), encoding='utf-8')

documents = {
'METRICS.md': '''# Homework 6 results

SID4: 6812 | SEED: 6812 | SLICE: 812 | HP_ID: 2 | CLS_A: 2 | CLS_B: 9

These results are copied from the saved, executed notebook. All models used
ResNet-18 without pretrained weights and the same 500 labeled training images.
Test accuracy was measured on all 8,000 STL-10 test images.

| Model | Test accuracy | Mean precision@5 for the three queries |
|---|---:|---:|
| Supervised | 43.60% | 60.00% |
| Rotation SSL | 46.30% | 33.33% |
| SimCLR | 51.79% | 53.33% |

SimCLR had the highest test accuracy, 8.19 percentage points above supervised
learning and 5.49 points above rotation SSL. Supervised learning had the highest
retrieval score for the three selected queries.

| Training stage | Epochs | First epoch loss | Last epoch loss |
|---|---:|---:|---:|
| Supervised | 15 | 2.2351 | 0.1029 |
| Rotation pretraining | 15 | 0.9463 | 0.2238 |
| Rotation linear classifier | 20 | 3.5381 | 1.2308 |
| SimCLR pretraining | 20 | 4.4767 | 2.2030 |
| SimCLR linear classifier | 20 | 2.7842 | 0.7257 |

The losses belong to different tasks and should not be compared directly.
Rotation used 100,000 unlabeled images. SimCLR used 20,000 and temperature 0.2.
Both linear classifiers used frozen encoders. Training ran on a Tesla T4 with
Python 3.13.15, PyTorch 2.11.0+cu130 and torchvision 0.26.0+cu130.

The retrieval counts below were checked against the saved neighbor plot.

| Query | Supervised correct neighbors | Rotation SSL | SimCLR |
|---|---:|---:|---:|
| Airplane, test index 3 | 5/5 | 2/5 | 2/5 |
| Bird, test index 5 | 0/5 | 0/5 | 1/5 |
| Car, test index 24 | 4/5 | 3/5 | 5/5 |

Precision@5 is the fraction of retrieved neighbors with the same class as the
query, averaged over these three queries. The query itself was excluded.
''',
'HW6_findings.md': '''# Homework 6 findings

SID4: 6812 | SEED: 6812 | SLICE: 812 | HP_ID: 2 | CLS_A: 2 | CLS_B: 9

## Which model worked best?

SimCLR gave the best test accuracy at 51.79%. Rotation SSL reached 46.30%, and
the supervised model reached 43.60%. Both self-supervised models benefited from
unlabeled training in this run, although the experiment does not isolate why.
The architecture and labeled subset were shared, but the amount of pretraining
data and computation differed.

The supervised model's training loss fell from 2.2351 to 0.1029. Its test accuracy
was much lower than that low training loss might suggest. This is consistent with
overfitting: a ResNet-18 can fit a small set of 500 images without learning features
that generalize well. Stronger label-preserving augmentation and weight-decay
tuning could help. Any tuning should use validation data within the label budget,
not the test set.

Rotation pretraining improved accuracy by 2.70 percentage points over supervised
learning. Predicting orientation can encourage useful shape and spatial features,
but it does not require the model to distinguish object classes. Backgrounds and
horizons can also help solve the rotation task. This may explain why its features
helped classification somewhat while still producing several poor neighbors.
Adding crop augmentation to rotation training could discourage some shortcuts.
Longer pretraining is another option, although a lower rotation loss alone would
not guarantee better classification.

SimCLR encourages different views of one image to have similar representations.
That can make features less sensitive to changes in crop and appearance. This is
a plausible explanation for its better classification result, not proof of the
cause. More unlabeled images, a larger contrastive batch and longer pretraining
could improve it further. Crop strength also needs care because a small crop can
remove the object. These changes would need new experiments to evaluate.

## What the nearest neighbors showed

The same airplane, bird and car queries were used for all three models. Neighbors
were ranked by cosine similarity between encoder embeddings, with each query
excluded from its own results.

The supervised model retrieved five airplanes for the airplane query and four
cars plus one truck for the car query. However, none of its five neighbors for
the bird query were birds: it returned dogs, cats and a monkey. Its representation
therefore separated some examples well but failed on this bird example.

Rotation SSL returned two airplanes and three ships for the airplane query.
Several share broad horizontal shapes and sky or water backgrounds, suggesting
that scene structure may influence the similarity. Its bird query returned three
deer, a ship and a monkey. The car query returned three cars, a truck and a ship.
High cosine scores did not consistently mean the objects belonged to the same class.

SimCLR retrieved five cars for the car query, including cars with different colors
and viewpoints. That is a useful example of semantic grouping. Its airplane query
still mixed airplanes with a ship and cars. For the bird query, only one neighbor
was a bird; the others were a cat, deer and monkeys. Animal classes remained difficult
for this query even with the strongest classifier.

Supervised learning had the best mean query precision@5, at 60.00%, compared with
53.33% for SimCLR and 33.33% for rotation SSL. This does not contradict the test
accuracy ranking. Retrieval was assessed on only three queries, while classification
used all 8,000 test images. Local neighbor structure and linear classification also
measure different properties of the learned features.

![Nearest neighbors](figures/nearest_neighbors.png)

## Training and experiment limits

All three pretraining/supervised loss curves decreased over the run. The linear
classifier losses also decreased overall. Loss values across different objectives
are not directly comparable.

![Training losses](figures/training_losses.png)

This was one run with seed 6812, so it does not establish how stable the ranking is
across seeds. Rotation used 100,000 unlabeled images and SimCLR used 20,000; the
comparison was not matched for data or compute.

The four SimCLR augmentation types match the instructor's demo: random resized
crop, horizontal flip, color jitter and grayscale. This run used crop scale
(0.2, 1.0), while the demo used (0.6, 1.0). Color jitter was applied with probability
0.8 here rather than every time. The saved results correspond to these settings.
''',
'AI_USE.md': '''# AI use — Homework 6

I used ChatGPT/Codex to help write the notebook, set up the three training methods,
check the contrastive loss and frozen encoders, and draft the supporting documents.
The assistant also helped interpret the saved results and compare the augmentations
with the instructor's demo. I ran the notebook in Colab and provided the executed
notebook for review. The initial implementation and these documents were generated
with substantial AI assistance; I am not claiming that I wrote them independently.

One mistake was the first version of RUN_LOG.txt. It started with:

```text
Homework 6 — Implementation validation

Local environment inspected:
Python 3.13; PyTorch 2.6.0+cu124; torchvision 0.21.0+cu124.
CUDA available; NVIDIA GeForce GTX 1650, 4 GB VRAM.
```

This was a summary of local code checks, not the console log from the actual
training run. The problem became clear when the standing requirements were compared
with the file: it did not contain the required timestamped training output.

The assistant added logging around each notebook stage so that printed output and
errors would be saved with UTC timestamps. That fixes the logging behavior for runs.
However, I did not download the original log from Colab. The current RUN_LOG.txt
was recovered from the notebook's saved output, so it contains the training progress
but not the original timestamps. That limitation is stated in the file.

The assistant checked the contrastive loss against a separate scalar calculation,
checked cosine retrieval with known vectors, and checked that encoder weights and
BatchNorm statistics stayed unchanged during linear training. The saved full run
finished all four parts without a recorded exception. These checks do not replace
my responsibility to understand and explain the notebook.

This appendix is an assisted draft. Before submitting, I need to review it and
describe my own understanding and contributions in my own words.
''',
'CHECKPOINTS.md': '''# Homework 6 checkpoints

The saved notebook ran all training stages in Colab and wrote checkpoints under
`/content/hw6_outputs/`. Those checkpoint files are not present in this local folder.
They cannot be recovered from the notebook's printed outputs or plots.

Expected training checkpoints:

- `supervised.pt`: supervised encoder and classification head.
- `rotation.pt`: rotation encoder and pretext head.
- `rotation_linear.pt`: frozen-encoder linear classifier.
- `simclr.pt`: contrastive encoder and projection head.
- `simclr_linear.pt`: frozen-encoder linear classifier.

The original run used the notebook cells in order. Equivalent script command:

```text
python hw6.py
```

On Windows: `py -3 hw6.py`. Use the saved notebook configuration and matching
software/hardware for reproducibility. The recorded runtime was a Tesla T4,
Python 3.13.15, PyTorch 2.11.0+cu130 and torchvision 0.26.0+cu130.

If the Colab session is still available, download these files or copy them to
persistent storage and add accessible links here. No checkpoint link is available yet.
''',
'README.md': '''# Homework 6 — STL-10 representation learning

SID4: 6812 | SEED: 6812 | SLICE: 812 | HP_ID: 2 | CLS_A: 2 | CLS_B: 9

This assignment compares supervised learning, rotation self-supervision and SimCLR
using ResNet-18 without pretrained weights. All three methods use the same 500
labeled images for classification training. The executed notebook includes the
training output, test results and nearest-neighbor plots.

| Model | Test accuracy |
|---|---:|
| Supervised | 43.60% |
| Rotation SSL | 46.30% |
| SimCLR | 51.79% |

## Files

- `HW6_STL10_Representation_Learning.ipynb`: executed submission notebook.
- `METRICS.md`: results and training-loss summary.
- `HW6_findings.md`: comparison, error analysis and improvement ideas.
- `RUN_LOG.txt`: console output recovered from the notebook; original timestamps unavailable.
- `AI_USE.md`: assistance disclosure and a specific correction; review in your own words.
- `CHECKPOINTS.md`: training command and missing-checkpoint status.
- `figures/`: original plots extracted from the executed notebook.
- `hw6.py`, `requirements.txt`: source and dependencies.
- `Demo_6_Self_Supervised_Learning.ipynb`: instructor reference, not the submission notebook.

## Running

Open the HW6 notebook in a GPU runtime and run its cells in order. The script
equivalent is `python hw6.py` or `py -3 hw6.py` on Windows. The notebook downloads
STL-10 and writes artifacts to `hw6_outputs/`. Save those artifacts before ending
the session. Checkpoints are saved every epoch; automatic resume is not implemented.

The recorded run used a Tesla T4. A smaller GPU may need a smaller SimCLR batch,
which changes the experiment. Python, NumPy and PyTorch seeds are set to 6812,
and deterministic algorithms are enabled. Matching results still depends on using
the same software and hardware conditions.

The demo's four SimCLR augmentation types match this implementation. Crop scale
and the probability of applying color jitter differ; see HW6_findings.md.

Do not run `build_notebook.py` over the executed submission: it regenerates the
notebook and clears its outputs. `validate_hw6.py` is for the unexecuted development
notebook and includes an empty-output assertion; it is not a submission-run command.

## Submission status

The notebook has completed all four parts. The original timestamped Colab log and
model checkpoints have not been recovered. RUN_LOG.txt makes the missing timestamps
explicit; it is not a replacement for the original timestamped log if that can still
be downloaded. See CHECKPOINTS.md for the remaining model-artifact requirement.

Submit through the private `data266-6812` repository, give the instructor and graders
access, and tag the completed submission `hw6`. The configured repository is
https://github.com/AkhileshTandur/data266-6812. Privacy, grader access, final commits
and the submission tag have not been verified here. Preserve the real incremental
development history. HW6 requires no extra six-image gallery or second parameter sweep.
'''
}
for name, content in documents.items():
    Path(name).write_text(content, encoding='utf-8')
print('Recovered run output and both figures; wrote metrics, findings, AI disclosure, checkpoint notes and README.')
