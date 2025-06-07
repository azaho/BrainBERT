from btbench.braintreebank_subject import BrainTreebankSubject
import btbench.btbench_train_test_splits as btbench_train_test_splits
import btbench.btbench_config as btbench_config

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import torch, numpy as np
import argparse, json, os, time, psutil
import gc

parser = argparse.ArgumentParser()
parser.add_argument('--eval_name', type=str, default='onset', help='Evaluation name(s) (e.g. onset, gpt2_surprisal). If multiple, separate with commas.')
parser.add_argument('--subject', type=int, required=True, help='Subject ID')
parser.add_argument('--trial', type=int, required=True, help='Trial ID')
parser.add_argument('--verbose', action='store_true', help='Whether to print progress')
parser.add_argument('--save_dir', type=str, default='eval_results', help='Directory to save results')
parser.add_argument('--splits_type', type=str, choices=['SS_SM', 'SS_DM', 'DS_DM'], default='SS_SM', help='Type of splits to use (SS_SM or DM_SM)')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--only_1second', action='store_true', help='Whether to only evaluate on 1 second after word onset')
parser.add_argument('--lite', action='store_true', help='Whether to use the lite eval for BTBench (which is the default)')
parser.add_argument('--time_granularity', type=int, default=1, help='Time granularity of the BrainBERT features (how many bins within a window)')
args = parser.parse_args()

eval_names = args.eval_name.split(',') if ',' in args.eval_name else [args.eval_name]
subject_id = args.subject
trial_id = args.trial 
verbose = bool(args.verbose)
save_dir = args.save_dir
splits_type = args.splits_type
seed = args.seed
only_1second = bool(args.only_1second)
lite = bool(args.lite)
time_granularity = args.time_granularity

# Set random seeds for reproducibility
np.random.seed(seed)
torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)

bins_start_before_word_onset_seconds = 0.5 if not only_1second else 0
bins_end_after_word_onset_seconds = 2.5 if not only_1second else 1
bin_size_seconds = 0.125

if not only_1second:
    # Loop over all time bins
    bin_starts = np.arange(-bins_start_before_word_onset_seconds, bins_end_after_word_onset_seconds, bin_size_seconds)
    bin_ends = bin_starts + bin_size_seconds
    # Add a time bin for the whole window and for 1 second after the word onset
    bin_starts = [0, -bins_start_before_word_onset_seconds][::-1] + list(bin_starts)
    bin_ends = [1, bins_end_after_word_onset_seconds][::-1] + list(bin_ends)
else:
    bin_starts = [0]
    bin_ends = [1]


max_log_priority = -1 if not verbose else 4
def log(message, priority=0, indent=0):
    if priority > max_log_priority: return

    current_time = time.strftime("%H:%M:%S")
    gpu_memory_reserved = torch.cuda.memory_reserved() / 1024**3 if torch.cuda.is_available() else 0
    process = psutil.Process()
    ram_usage = process.memory_info().rss / 1024**3
    print(f"[{current_time} gpu {gpu_memory_reserved:04.1f}G ram {ram_usage:05.1f}G] {' '*4*indent}{message}")


log("Loading model...", priority=0)
import models
import torch
from omegaconf import OmegaConf
# def build_model(cfg):
#     ckpt_path = cfg.upstream_ckpt
#     init_state = torch.load(ckpt_path, weights_only=False)
#     upstream_cfg = init_state["model_cfg"]
#     upstream = models.build_model(upstream_cfg)
#     return upstream
# def load_model_weights(model, states, multi_gpu):
#     if multi_gpu:
#         model.module.load_weights(states)
#     else:
#         model.load_weights(states)
# ckpt_path = "pretrained_weights/stft_large_pretrained.pth"
# cfg = OmegaConf.create({"upstream_ckpt": ckpt_path})
# model = build_model(cfg)
# model.to('cuda')
# init_state = torch.load(ckpt_path, weights_only=False)
# if not random_init:
#     load_model_weights(model, init_state['model'], False)
# del init_state
# torch.cuda.empty_cache()     # optional: releases unused GPU cached blocks back to the OS



from scipy import signal, stats
import numpy as np
def get_stft(x, fs, clip_fs=-1, normalizing=None, boundary=None, clip=0, **kwargs):
    assert len(x.shape) == 3, "x must be of shape (batch_size, n_channels, n_samples)"

    # x is of shape (batch_size, n_channels, n_samples)
    f, t, Zxx = signal.stft(x, fs, boundary=boundary, window="boxcar", **kwargs)
    # Zxx is of shape (batch_size, n_channels, n_freqs, n_times)
   
    Zxx = Zxx[:, :, :clip_fs]
    f = f[:clip_fs]

    Zxx = np.abs(Zxx)
    if normalizing=="zscore":
        if clip > 0: Zxx = Zxx[:, :, :, clip:-clip]
        Zxx = stats.zscore(Zxx, axis=-1)
        t = t[clip:-clip]
    elif normalizing=="db":
        if clip > 0: Zxx = Zxx[:, :, :, clip:-clip]
        Zxx = np.log2(Zxx)
        t = t[clip:-clip]

    if np.isnan(Zxx).any():
        import pdb; pdb.set_trace()

    return f, t, Zxx # shape: (batch_size, n_channels, n_freqs, n_times)
def get_brainbert_features(x, fs=2048, max_batch_size=12):
    """
    x: np.ndarray of shape (batch_size, n_channels, n_samples)
    returns: np.ndarray of shape (batch_size, n_channels, n_timebins, d_model)
    """
    assert len(x.shape) == 3, "x must be of shape (batch_size, n_channels, n_samples)"
    
    batch_size, n_channels, n_samples = x.shape

    out_array = None     # will hold the full output
    idx_start = 0        # pointer into out_array

    with torch.no_grad():
        for i in range(0, batch_size, max_batch_size):
            log(f"Processing batch {i//max_batch_size+1} of {(batch_size-1)//max_batch_size+1}", priority=3, indent=2)

            log(f"Getting STFT for batch {i//max_batch_size+1} of {(batch_size-1)//max_batch_size+1}", priority=4, indent=3)
            # slice this batch
            batch_x = x[i:i+max_batch_size]
            # STFT -> (batch_i, n_channels, n_freqs, n_times)
            f, t, linear = get_stft(
                batch_x, fs, clip_fs=40,
                nperseg=400, noverlap=350,
                normalizing="zscore", return_onesided=True
            )

            batch_i, n_channels, n_freqs, n_times = linear.shape

            batch_out = torch.FloatTensor(linear).transpose(-1, -2) # (batch_i, n_channels, n_times, n_freqs)

            log(f"Time-bin pooling", priority=4, indent=3)
            # time‐bin pooling
            if time_granularity > 0:
                timebin_size = n_times // time_granularity
                n_timebins = n_times // timebin_size
                batch_out = batch_out[:, :, :(n_timebins * timebin_size), :]
                batch_out = batch_out.reshape(batch_i, n_channels, n_timebins, timebin_size, n_freqs)
                batch_out = batch_out.mean(dim=3)  # (batch_i, n_channels, n_timebins, n_freqs)
            else:
                n_timebins = n_times

            log(f"Converting to numpy", priority=4, indent=3)
            batch_np = batch_out.cpu().numpy()

            # On first batch, allocate the big output array
            if out_array is None:
                out_array = np.empty((batch_size, n_channels, n_timebins, n_freqs), dtype=batch_np.dtype)

            # fill slice
            idx_end = idx_start + batch_i
            out_array[idx_start:idx_end] = batch_np
            idx_start = idx_end

            # cleanup - add f and t to the list of variables to delete
            del batch_out, batch_np, linear, batch_x, f, t
            # torch.cuda.empty_cache()

    return out_array

def get_region_labels(subject):
    """
    subject: BrainTreebankSubject
    returns: np.ndarray of shape (n_channels,)
    """
    return subject.get_all_electrode_metadata()['DesikanKilliany'].to_numpy()

def combine_regions(X_train, X_test, regions_train, regions_test):
    """
    X_train: np.ndarray of shape (n_samples, n_channels_train, n_timebins, d_model) or (n_samples, n_channels_train, n_timesamples)
    X_test: np.ndarray of shape (n_samples, n_channels_test, n_timebins, d_model) or (n_samples, n_channels_test, n_timesamples)
    regions_train: np.ndarray of shape (n_channels_train,)
    regions_test: np.ndarray of shape (n_channels_test,)
    """
    # Find the intersection of regions between train and test
    unique_regions_train = np.unique(regions_train)
    unique_regions_test = np.unique(regions_test)
    common_regions = np.intersect1d(unique_regions_train, unique_regions_test)
    
    if X_train.ndim == 3:
        # Add a dummy dimension to X_train and X_test for d_model=1
        X_train = X_train[:, :, :, np.newaxis]
        X_test = X_test[:, :, :, np.newaxis]

    n_samples_train, _, n_timebins, d_model = X_train.shape
    n_samples_test = X_test.shape[0]
    n_regions_intersect = len(common_regions)
    
    # Create new arrays to store region-averaged data
    X_train_regions = np.zeros((n_samples_train, n_regions_intersect, n_timebins, d_model), dtype=X_train.dtype)
    X_test_regions = np.zeros((n_samples_test, n_regions_intersect, n_timebins, d_model), dtype=X_test.dtype)
    
    # For each common region, average across all channels with that region label
    for i, region in enumerate(common_regions):
        # Find channels corresponding to this region
        train_mask = regions_train == region
        test_mask = regions_test == region
        
        # Average across channels with the same region
        X_train_regions[:, i, :, :] = X_train[:, train_mask, :, :].mean(axis=1)
        X_test_regions[:, i, :, :] = X_test[:, test_mask, :, :].mean(axis=1)
    
    return X_train_regions, X_test_regions, common_regions


# use cache=True to load this trial's neural data into RAM, if you have enough memory!
# It will make the loading process faster.
subject = BrainTreebankSubject(subject_id, allow_corrupted=False, cache=True, dtype=torch.float32)
all_electrode_labels = btbench_config.BTBENCH_LITE_ELECTRODES[subject.subject_identifier] if lite else subject.electrode_labels

for eval_name in eval_names:
    file_save_dir = f"{save_dir}/brainbert_frozen_mean_granularity_{time_granularity}"
    file_save_path = f"{file_save_dir}/population_{subject.subject_identifier}_{trial_id}_{eval_name}.json"
    # Check if the file already exists, and if so, skip this evaluation
    if os.path.exists(file_save_path):
        log(f"File {file_save_path} already exists. Skipping {eval_name} for subject {subject.subject_identifier}, trial {trial_id}.")
        continue

    results_population = {
        "time_bins": [],
    }

    # Load all electrodes at once
    subject.clear_neural_data_cache()
    subject.set_electrode_subset(all_electrode_labels)  # Use all electrodes
    if verbose:
        log("Subject loaded", priority=0)

    # train_datasets and test_datasets are arrays of length k_folds, each element is a BrainTreebankSubjectTrialBenchmarkDataset for the train/test split
    if splits_type == "SS_SM":
        train_datasets, test_datasets = btbench_train_test_splits.generate_splits_SS_SM(subject, trial_id, eval_name, k_folds=5, dtype=torch.float32, 
                                                                                        output_indices=False, 
                                                                                        start_neural_data_before_word_onset=int(bins_start_before_word_onset_seconds*btbench_config.SAMPLING_RATE), 
                                                                                        end_neural_data_after_word_onset=int(bins_end_after_word_onset_seconds*btbench_config.SAMPLING_RATE),
                                                                                        lite=lite, allow_partial_cache=True)
    elif splits_type == "SS_DM":
        train_datasets, test_datasets = btbench_train_test_splits.generate_splits_SS_DM(subject, trial_id, eval_name, max_other_trials=3, dtype=torch.float32, 
                                                                                        output_indices=False, 
                                                                                        start_neural_data_before_word_onset=int(bins_start_before_word_onset_seconds*btbench_config.SAMPLING_RATE), 
                                                                                        end_neural_data_after_word_onset=int(bins_end_after_word_onset_seconds*btbench_config.SAMPLING_RATE),
                                                                                        lite=lite, allow_partial_cache=True)
        train_datasets = [train_datasets]
        test_datasets = [test_datasets]
    elif splits_type == "DS_DM":
        if verbose: log("Loading the training subject...", priority=0)
        train_subject_id = btbench_config.DS_DM_TRAIN_SUBJECT_ID
        train_subject = BrainTreebankSubject(train_subject_id, allow_corrupted=False, cache=True, dtype=torch.float32)
        train_subject_electrodes = btbench_config.BTBENCH_LITE_ELECTRODES[train_subject.subject_identifier] if lite else train_subject.electrode_labels
        train_subject.set_electrode_subset(train_subject_electrodes)
        all_subjects = {
            subject_id: subject,
            train_subject_id: train_subject,
        }
        if verbose: log("Subject loaded.", priority=0)
        train_datasets, test_datasets = btbench_train_test_splits.generate_splits_DS_DM(all_subjects, subject_id, trial_id, eval_name, dtype=torch.float32, 
                                                                                        output_indices=False, 
                                                                                        start_neural_data_before_word_onset=int(bins_start_before_word_onset_seconds*btbench_config.SAMPLING_RATE), 
                                                                                        end_neural_data_after_word_onset=int(bins_end_after_word_onset_seconds*btbench_config.SAMPLING_RATE),
                                                                                        lite=lite, allow_partial_cache=True)
        train_datasets = [train_datasets]
        test_datasets = [test_datasets]

    for bin_start, bin_end in zip(bin_starts, bin_ends):
        data_idx_from = int((bin_start+bins_start_before_word_onset_seconds)*btbench_config.SAMPLING_RATE)
        data_idx_to = int((bin_end+bins_start_before_word_onset_seconds)*btbench_config.SAMPLING_RATE)

        bin_results = {
            "time_bin_start": float(bin_start),
            "time_bin_end": float(bin_end),
            "folds": []
        }

        # Loop over all folds
        for fold_idx in range(len(train_datasets)):
            train_dataset = train_datasets[fold_idx]
            test_dataset = test_datasets[fold_idx]

            log(f"Fold {fold_idx+1}, Bin {bin_start}-{bin_end}")
            log("Preparing data...", priority=2, indent=1)

            # Convert PyTorch dataset to numpy arrays for scikit-learn
            X_train = np.array([item[0][:, data_idx_from:data_idx_to] for item in train_dataset])
            y_train = np.array([item[1] for item in train_dataset])
            X_test = np.array([item[0][:, data_idx_from:data_idx_to] for item in test_dataset])
            y_test = np.array([item[1] for item in test_dataset])

            log("Computing BrainBERT features...", priority=2, indent=1)
            X_train = get_brainbert_features(X_train)
            X_test = get_brainbert_features(X_test)

            if splits_type == "DS_DM":
                if verbose: log("Combining regions...", priority=2, indent=1)
                regions_train = get_region_labels(train_subject)
                regions_test = get_region_labels(subject)
                X_train, X_test, common_regions = combine_regions(X_train, X_test, regions_train, regions_test)

            # Flatten the data after preprocessing in-place
            X_train = X_train.reshape(X_train.shape[0], -1)
            X_test = X_test.reshape(X_test.shape[0], -1)

            log(f"Standardizing data...", priority=2, indent=1)

            # Standardize the data in-place
            scaler = StandardScaler(copy=False)
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            log(f"Training model...", priority=2, indent=1)

            log(f"Fitting model...", priority=3, indent=2)
            # Train logistic regression
            clf = LogisticRegression(random_state=seed, max_iter=10000, tol=1e-3)
            clf.fit(X_train, y_train)
            gc.collect()

            log(f"Evaluating model...", priority=3, indent=2)
            # Evaluate model
            train_accuracy = clf.score(X_train, y_train)
            test_accuracy = clf.score(X_test, y_test)
            gc.collect()

            log(f"Getting predictions...", priority=3, indent=2)
            # Get predictions - for multiclass classification
            train_probs = clf.predict_proba(X_train)
            test_probs = clf.predict_proba(X_test)
            gc.collect()

            log(f"Filtering test samples...", priority=3, indent=2)
            # Filter test samples to only include classes that were in training
            valid_class_mask = np.isin(y_test, clf.classes_)
            y_test_filtered = y_test[valid_class_mask]
            test_probs_filtered = test_probs[valid_class_mask]

            log(f"Converting y_test to one-hot encoding...", priority=3, indent=2)
            # Convert y_test to one-hot encoding
            y_test_onehot = np.zeros((len(y_test_filtered), len(clf.classes_)))
            for i, label in enumerate(y_test_filtered):
                class_idx = np.where(clf.classes_ == label)[0][0]
                y_test_onehot[i, class_idx] = 1

            y_train_onehot = np.zeros((len(y_train), len(clf.classes_)))
            for i, label in enumerate(y_train):
                class_idx = np.where(clf.classes_ == label)[0][0]
                y_train_onehot[i, class_idx] = 1

            log(f"Calculating ROC AUC...", priority=3, indent=2)
            # For multiclass ROC AUC, we need to calculate the score for each class
            n_classes = len(clf.classes_)
            if n_classes > 2:
                train_roc = roc_auc_score(y_train_onehot, train_probs, multi_class='ovr', average='macro')
                test_roc = roc_auc_score(y_test_onehot, test_probs_filtered, multi_class='ovr', average='macro')
            else:
                train_roc = roc_auc_score(y_train_onehot, train_probs)
                test_roc = roc_auc_score(y_test_onehot, test_probs_filtered)

            log(f"Saving results...", priority=3, indent=2)
            fold_result = {
                "train_accuracy": float(train_accuracy),
                "train_roc_auc": float(train_roc),
                "test_accuracy": float(test_accuracy),
                "test_roc_auc": float(test_roc)
            }
            bin_results["folds"].append(fold_result)
            if verbose: 
                log(f"Population, Fold {fold_idx+1}, Bin {bin_start}-{bin_end}: Train accuracy: {train_accuracy:.3f}, Test accuracy: {test_accuracy:.3f}, Train ROC AUC: {train_roc:.3f}, Test ROC AUC: {test_roc:.3f}", priority=0, indent=0)

        if bin_start == -bins_start_before_word_onset_seconds and bin_end == bins_end_after_word_onset_seconds:
            results_population["whole_window"] = bin_results # whole window results
        elif bin_start == 0 and bin_end == 1:
            results_population["one_second_after_onset"] = bin_results # one second after onset results
        else:
            results_population["time_bins"].append(bin_results) # time bin results

        del X_train, X_test, y_train, y_test
        gc.collect()

    results = {
        "model_name": "BrainBERT (frozen) _ logistic regression",
        "author": "Andrii Zahorodnii",
        "description": f"BrainBERT features (frozen) + logistic regression, all electrodes' features concatenated, mean across all timesteps within a timebin (splitting the window into {time_granularity} timebins).",
        "organization": "MIT",
        "organization_url": "https://azaho.org/",
        "timestamp": time.time(),

        "evaluation_results": {
            f"{subject.subject_identifier}_{trial_id}": {
                "population": results_population
            }
        },

        "only_1second": only_1second,
        "seed": seed
    }

    os.makedirs(file_save_dir, exist_ok=True) # Create save directory if it doesn't exist
    with open(file_save_path, "w") as f:
        json.dump(results, f, indent=4)
    if verbose:
        log(f"Results saved to {file_save_path}", priority=0)