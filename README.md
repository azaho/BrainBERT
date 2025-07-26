# BrainBERT Neuroprobe Evaluation

BrainBERT is an modeling approach for learning self-supervised representations of intracranial electrode data. See [paper](https://arxiv.org/abs/2302.14367) for details.

[Neuroprobe](https://neuroprobe.dev) is a benchmark for understanding how the brain processes information across multiple tasks. Visit the [Neuroprobe GitHub page](https://github.com/azaho/neuroprobe/)

This branch on the repository describes steps required to evaluate BrainBERT on Neuroprobe.

## Reproducing the evaluation
1. Create a virtual environment (optional):
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

2. Install the requirements (both for BrainBERT and Neuroprobe):
```bash
pip install -r requirements.txt
```

3. Configure BrainTreebank dataset path in `neuroprobe/config.py`. NOTE: This dataset copy must already have line noise removed and electrodes Laplacian re-referenced, according to the methods and code of the original [BrainBERT paper](https://arxiv.org/abs/2302.14367).
```python
# In neuroprobe/config.py
ROOT_DIR = "braintreebank"  # Root directory for the extracted braintreebank data
```

4. Download the pretrained BrainBERT weights from [here](https://drive.google.com/file/d/14ZBOafR7RJ4A6TsurOXjFVMXiVH6Kd_Q/view?usp=sharing) and make sure to put them in the `pretrained_weights/` directory.

5. Run the file `run_neuroprobe_eval_frozen_population.py` to get regression results for any given subject/trial pair using:
```bash
python run_neuroprobe_eval_frozen_population.py --only_1second --subject_id SUBJECT_ID --trial_id TRIAL_ID --eval_name TASK_NAME --split_type SPLIT_TYPE
```
Optionally, include a tag `--randomly_initialized_model`, to run regressins on an untrained BrainBERT model. To learn more, visit the original [Neuroprobe GitHub page](https://github.com/azaho/neuroprobe/). Alternatively, run the bash script `run_neuroprobe_eval_frozen_population.sh` to run all combinations of tasks in parallel (the script is set up to run using SLURM. Make sure to edit the parameters according to your compute cluster.)

6. The results will now be saved in the `eval_results_SPLIT_TYPE` diretory!