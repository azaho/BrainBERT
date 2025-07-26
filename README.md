# BrainBERT Seizure Embeddings Analysis

BrainBERT is an modeling approach for learning self-supervised representations of intracranial electrode data. See [paper](https://arxiv.org/abs/2302.14367) for details.

This branch on the repository describes steps required to evaluate BrainBERT embeddings on seizures from the MGH dataset (not available online).

## Reproducing the evaluation
1. Create a virtual environment (optional):
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

2. Install the requirements (both for BrainBERT and the embeddings analysis):
```bash
pip install -r requirements.txt
```

3. Configure MGH dataset path in `.env`, by following the structure from `.env.example`.

4. Download the pretrained BrainBERT weights from [here](https://drive.google.com/file/d/14ZBOafR7RJ4A6TsurOXjFVMXiVH6Kd_Q/view?usp=sharing) and make sure to put them in the `pretrained_weights/` directory.

5. Run the notebook `mgh2024_brainbert_embeddings.ipynb` to get analysis results for a pretrained (or randomly initialized) BrainBERT, and also for the raw voltage data.