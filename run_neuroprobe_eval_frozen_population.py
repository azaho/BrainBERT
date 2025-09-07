import neuroprobe.train_test_splits as neuroprobe_train_test_splits
import neuroprobe.config as neuroprobe_config
from neuroprobe.braintreebank_subject import BrainTreebankSubject

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import torch, numpy as np
import argparse, json, os, time
import gc, psutil

max_log_priority = 4
def log(message, priority=0, indent=0):
    if priority > max_log_priority: return

    current_time = time.strftime("%H:%M:%S")
    gpu_memory_reserved = torch.cuda.memory_reserved() / 1024**3 if torch.cuda.is_available() else 0
    process = psutil.Process()
    ram_usage = process.memory_info().rss / 1024**3
    print(f"[{current_time} gpu {gpu_memory_reserved:.1f}G ram {ram_usage:.1f}G] {' '*4*indent}{message}")


### DEFINING PARAMETERS ###

splits_options = [
    'SS_SM', # same subject, same trial
    'SS_DM', # same subject, different trial    
    'DS_DM', # different subject, different trial
]

parser = argparse.ArgumentParser()
parser.add_argument('--eval_name', type=str, default='onset', help='Evaluation name(s) (e.g. onset, gpt2_surprisal). If multiple, separate with commas.')
parser.add_argument('--split_type', type=str, choices=splits_options, default='SS_SM', help=f'Type of splits to use ({", ".join(splits_options)})')
parser.add_argument('--subject_id', type=int, required=True, help='Subject ID')
parser.add_argument('--trial_id', type=int, required=True, help='Trial ID')

parser.add_argument('--silent', action='store_true', help='Whether to suppress progress messages')
parser.add_argument('--overwrite', action='store_true', help='Whether to overwrite existing results')
parser.add_argument('--save_dir', type=str, default=None, help='Directory to save results')
parser.add_argument('--seed', type=int, default=42, help='Random seed')

parser.add_argument('--only_1second', action='store_true', help='Whether to only evaluate on 1 second after word onset')
parser.add_argument('--full', action='store_true', help='Whether to use the full eval for Neuroprobe (NOTE: Lite is the default!)')
parser.add_argument('--nano', action='store_true', help='Whether to use Neuroprobe Nano for faster evaluation')

parser.add_argument('--randomly_initialized_model', type=int, default=0, help='Whether to use a randomly initialized model (default is to use the pretrained BrainBERT model)')
parser.add_argument('--batch_size', type=int, default=50, help='Batch size for feature computation')

parser.add_argument('--feature_type', type=str, default='keepall', help='How to extract features from the model. Options: \'meanE\' (mean across electrodes), \'meanT\' (mean across timebins), \'cls\' (only take the first token of the electrode dimension), any combinations of these (you can use _ to concatenate them) or \'keepall\' (keep all tokens)')
args = parser.parse_args()

eval_names = args.eval_name.split(',')
splits_type = args.split_type.upper()
subject_id = args.subject_id
trial_id = args.trial_id

verbose = not bool(args.silent)
overwrite = bool(args.overwrite)
save_dir = args.save_dir if args.save_dir is not None else f"eval_results_{splits_type}"
seed = args.seed

only_1second = bool(args.only_1second)
lite = not bool(args.full)
nano = bool(args.nano)
assert (not nano) or (splits_type != "SS_DM"), "Nano only works with SS_SM or DS_DM splits; does not work with SS_DM."
assert (not nano) or lite, "--nano and --full cannot be used together. Neuroprobe Full and Neuroprobe Nano are different evaluations."

batch_size = args.batch_size
random_init = bool(args.randomly_initialized_model)
feature_type = args.feature_type

# Set random seeds for reproducibility
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

### LOAD SUBJECT ###

# use cache=True to load this trial's neural data into RAM, if you have enough memory!
# It will make the loading process faster.
subject = BrainTreebankSubject(subject_id, allow_corrupted=False, cache=True, dtype=torch.float32)

if nano:
    all_electrode_labels = neuroprobe_config.NEUROPROBE_NANO_ELECTRODES[subject.subject_identifier]
elif lite:
    all_electrode_labels = neuroprobe_config.NEUROPROBE_LITE_ELECTRODES[subject.subject_identifier]
else:
    all_electrode_labels = subject.electrode_labels
subject.set_electrode_subset(all_electrode_labels)  # Use all electrodes

### LOAD MODEL ###

import models
import torch
from omegaconf import OmegaConf
def build_model(cfg):
    ckpt_path = cfg.upstream_ckpt
    init_state = torch.load(ckpt_path, weights_only=False)
    upstream_cfg = init_state["model_cfg"]
    upstream = models.build_model(upstream_cfg)
    return upstream
def load_model_weights(model, states, multi_gpu):
    if multi_gpu:
        model.module.load_weights(states)
    else:
        model.load_weights(states)
log(f"Loading the model...", priority=0)
ckpt_path = "pretrained_weights/stft_large_pretrained.pth"
cfg = OmegaConf.create({"upstream_ckpt": ckpt_path})
model = build_model(cfg)
model.to('cuda')
init_state = torch.load(ckpt_path, weights_only=False)
if not random_init:
    load_model_weights(model, init_state['model'], False)
del init_state
torch.cuda.empty_cache()     # optional: releases unused GPU cached blocks back to the OS

### SETUP FEATURE GENERATION FUNCTION ###

from scipy import signal, stats
import numpy as np
def get_stft(x, fs, clip_fs=-1, normalizing=None, boundary=None, clip=0, **kwargs):
    assert len(x.shape) == 3, "x must be of shape (batch_size, n_channels, n_samples)"

    # x is of shape (batch_size, n_channels, n_samples)
    f, t, Zxx = signal.stft(x, fs, boundary=boundary, **kwargs)
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

def get_brainbert_features(x, fs=2048):
    """
    x: np.ndarray of shape (batch_size, n_channels, n_samples)
    returns: np.ndarray of shape (batch_size, n_channels, n_timebins, d_model)
    """
    assert len(x.shape) == 3, "x must be of shape (batch_size, n_channels, n_samples)"
    n_channels = x.shape[1]

    with torch.no_grad():
        log(f"Getting STFT...", priority=4, indent=3)
        # slice this batch
        # STFT -> (batch_i, n_channels, n_freqs, n_times)
        f, t, linear = get_stft(
            x, fs, clip_fs=40,
            nperseg=400, noverlap=350,
            normalizing="zscore", return_onesided=True
        )
        batch_i, n_channels, n_freqs, n_times = linear.shape
        
        # to tensor shape: (batch_i * n_channels, n_times, n_freqs)
        batch_inputs = torch.FloatTensor(linear).transpose(-1, -2).to('cuda')
        batch_inputs = batch_inputs.reshape(batch_i * n_channels, n_times, n_freqs)

        batch_mask = torch.zeros(batch_inputs.shape[:2], dtype=torch.bool, device='cuda')

        # model forward -> (batch_i*n_channels, n_times, d_model)
        batch_out = model.forward(batch_inputs, batch_mask, intermediate_rep=True)
        
        d_model = batch_out.shape[-1]
        batch_out = batch_out.reshape(batch_i, n_channels, n_times, d_model)
        
        # Make a copy to ensure no references to GPU memory persist
        result = batch_out.clone()
        log(f"Forwarding batch...", priority=4, indent=3)

    # Clean up intermediate variables immediately
    del batch_inputs, batch_mask, linear, f, t, batch_out
    torch.cuda.empty_cache()
    gc.collect()
    
    return result

def load_dataset(dataset):
    
    X = None

    for item_start_i in range(0, len(dataset), batch_size):
        if verbose:
            log(f"Loading batch {item_start_i//batch_size+1} of {len(dataset)//batch_size}", priority=1, indent=2)

        item_end_i = min(item_start_i + batch_size, len(dataset))
        batch_input = torch.cat([dataset[i][0][:, data_idx_from:data_idx_to].unsqueeze(0) for i in range(item_start_i, item_end_i)], dim=0)
        batch = {
            'data': batch_input, # shape (batch_size, n_electrodes, n_samples),
            'electrode_labels': [all_electrode_labels],
            'metadata': {
                'subject_identifier': subject.subject_identifier,
                'trial_id': trial_id,
                'sampling_rate': 2048,
            },
        }

        with torch.no_grad():
            features = get_brainbert_features(batch['data'])

            if 'meanT' in feature_type:
                features = features.mean(dim=2, keepdim=True) # shape: (batch_size, n_electrodes + 1, 1, d_model)
            if 'meanE' in feature_type:
                features = features.mean(dim=1, keepdim=True) # shape: (batch_size, 1, n_timebins, d_model)
            if 'cls' in feature_type:
                features = features[:, 0:1, :, :] # shape: (batch_size, 1, n_timebins, d_model) -- take just the cls token
                
            # Convert to numpy and explicitly force a copy to avoid memory references
            features_np = features.detach().cpu().float().numpy().copy()
            # Clear the original tensor immediately
            del features
            torch.cuda.empty_cache()

        if X is None:
            X = np.zeros((len(dataset), *features_np.shape[1:]), dtype=features_np.dtype)
            X.fill(0) # Force the array to be physically allocated in memory
        
        # Force a copy when assigning to avoid memory reference issues
        X[item_start_i:item_end_i] = features_np.copy()

        if verbose and item_start_i == 0:
            log(f"Input shape: {batch['data'].shape}", priority=1, indent=3)
            log(f"Features shape: {features_np.shape}", priority=1, indent=3)
        
        # Clean up all batch-related variables
        del features_np, batch_input, batch
        gc.collect()
        
    y = [dataset[i][1] for i in range(len(dataset))]
    return X, np.array(y)

### CALCULATE TIME BINS ###

bins_start_before_word_onset_seconds = 0.5 if not only_1second else 0
bins_end_after_word_onset_seconds = 1.5 if not only_1second else 1
bin_size_seconds = 0.25
bin_step_size_seconds = 0.125

bin_starts = []
bin_ends = []
if not only_1second:
    for bin_start in np.arange(-bins_start_before_word_onset_seconds, bins_end_after_word_onset_seconds-bin_size_seconds, bin_step_size_seconds):
        bin_end = bin_start + bin_size_seconds
        if bin_end > bins_end_after_word_onset_seconds: break

        bin_starts.append(bin_start)
        bin_ends.append(bin_end)
    bin_starts += [-bins_start_before_word_onset_seconds]
    bin_ends += [bins_end_after_word_onset_seconds]
bin_starts += [0]
bin_ends += [1]

############## REGION AVERAGING (FOR DS/DM SPLITS) ###############

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
    
    d_model_dimension_unsqueezed = False
    if X_train.ndim == 3:
        # Add a dummy dimension to X_train and X_test for d_model=1
        X_train = X_train[:, :, :, np.newaxis]
        X_test = X_test[:, :, :, np.newaxis]
        d_model_dimension_unsqueezed = True

    n_samples_train, _, n_timebins, d_model = X_train.shape
    n_samples_test = X_test.shape[0]
    n_regions_intersect = len(common_regions)
    
    # Create new arrays to store region-averaged data with explicit dtype to save memory
    dtype = X_train.dtype
    X_train_regions = np.zeros((n_samples_train, n_regions_intersect, n_timebins, d_model), dtype=dtype)
    X_test_regions = np.zeros((n_samples_test, n_regions_intersect, n_timebins, d_model), dtype=dtype)
    
    # For each common region, average across all channels with that region label
    for i, region in enumerate(common_regions):
        # Find channels corresponding to this region
        train_mask = regions_train == region
        test_mask = regions_test == region
        
        # Average across channels with the same region - use np.mean to avoid extra copies
        X_train_regions[:, i, :, :] = np.mean(X_train[:, train_mask, :, :], axis=1)
        X_test_regions[:, i, :, :] = np.mean(X_test[:, test_mask, :, :], axis=1)

    if d_model_dimension_unsqueezed: # remove the dummy dimension
        X_train_regions = X_train_regions[:, :, :, 0]
        X_test_regions = X_test_regions[:, :, :, 0]
    
    return X_train_regions, X_test_regions, common_regions

neural_data_loaded = False
for eval_name in eval_names:
    start_time = time.time()

    random_init_suffix = "_randomly_initialized" if random_init else ""
    file_save_dir = f"{save_dir}/brainbert{random_init_suffix}_{feature_type}"
    os.makedirs(file_save_dir, exist_ok=True) # Create save directory if it doesn't exist

    file_save_path = f"{file_save_dir}/population_{subject.subject_identifier}_{trial_id}_{eval_name}.json"
    if os.path.exists(file_save_path) and not overwrite:
        if verbose:
            log(f"Skipping {file_save_path} because it already exists", priority=0)
        continue

    # Load neural data if it hasn't been loaded yet; NOTE: this is done here to avoid unnecessary loading of neural data if the file is going to be skipped.
    if not neural_data_loaded:
        if verbose:
            log(f"Loading the test subject...", priority=0)
        subject.load_neural_data(trial_id)
        subject_load_time = time.time() - start_time
        if verbose:
            log(f"Subject loaded in {subject_load_time:.2f} seconds", priority=0)
        neural_data_loaded = True

    results_population = {
        "time_bins": [],
    }

    # train_datasets and test_datasets are arrays of length k_folds, each element is a BrainTreebankSubjectTrialBenchmarkDataset for the train/test split
    if splits_type == "SS_SM":
        train_datasets, test_datasets = neuroprobe_train_test_splits.generate_splits_SS_SM(subject, trial_id, eval_name, dtype=torch.float32, 
                                                                                        output_indices=False, 
                                                                                        start_neural_data_before_word_onset=int(bins_start_before_word_onset_seconds*neuroprobe_config.SAMPLING_RATE), 
                                                                                        end_neural_data_after_word_onset=int(bins_end_after_word_onset_seconds*neuroprobe_config.SAMPLING_RATE),
                                                                                        lite=lite, nano=nano)
    elif splits_type == "SS_DM":
        train_datasets, test_datasets = neuroprobe_train_test_splits.generate_splits_SS_DM(subject, trial_id, eval_name, dtype=torch.float32, 
                                                                                        output_indices=False, 
                                                                                        start_neural_data_before_word_onset=int(bins_start_before_word_onset_seconds*neuroprobe_config.SAMPLING_RATE), 
                                                                                        end_neural_data_after_word_onset=int(bins_end_after_word_onset_seconds*neuroprobe_config.SAMPLING_RATE),
                                                                                        lite=lite)
        train_datasets = [train_datasets]
        test_datasets = [test_datasets]
    elif splits_type == "DS_DM":
        if verbose: log("Loading the training subject...", priority=0)
        train_subject_id = neuroprobe_config.DS_DM_TRAIN_SUBJECT_ID
        train_subject = BrainTreebankSubject(train_subject_id, allow_corrupted=False, cache=True, dtype=torch.float32)
        train_subject_electrodes = neuroprobe_config.NEUROPROBE_LITE_ELECTRODES[train_subject.subject_identifier] if lite else train_subject.electrode_labels
        train_subject.set_electrode_subset(train_subject_electrodes)
        all_subjects = {
            subject_id: subject,
            train_subject_id: train_subject,
        }
        if verbose: log("Subject loaded.", priority=0)
        train_datasets, test_datasets = neuroprobe_train_test_splits.generate_splits_DS_DM(all_subjects, subject_id, trial_id, eval_name, dtype=torch.float32, 
                                                                                        output_indices=False, 
                                                                                        start_neural_data_before_word_onset=int(bins_start_before_word_onset_seconds*neuroprobe_config.SAMPLING_RATE), 
                                                                                        end_neural_data_after_word_onset=int(bins_end_after_word_onset_seconds*neuroprobe_config.SAMPLING_RATE),
                                                                                        lite=lite, nano=nano)
        train_datasets = [train_datasets]
        test_datasets = [test_datasets]


    for bin_start, bin_end in zip(bin_starts, bin_ends):
        data_idx_from = int((bin_start+bins_start_before_word_onset_seconds)*neuroprobe_config.SAMPLING_RATE)
        data_idx_to = int((bin_end+bins_start_before_word_onset_seconds)*neuroprobe_config.SAMPLING_RATE)

        bin_results = {
            "time_bin_start": float(bin_start),
            "time_bin_end": float(bin_end),
            "folds": []
        }

        # Loop over all folds
        for fold_idx in range(len(train_datasets)):
            train_dataset = train_datasets[fold_idx]
            test_dataset = test_datasets[fold_idx]

            if verbose:
                log(f"Fold {fold_idx+1}, Bin {bin_start}-{bin_end}")
                log("Preparing and preprocessing data & Generating features...", priority=1, indent=1)

            start_time = time.time()
            X_train, y_train = load_dataset(train_dataset)
            gc.collect()  # Collect after creating large arrays
            X_test, y_test = load_dataset(test_dataset)
            gc.collect()  # Collect after creating large arrays

            if splits_type == "DS_DM":
                if verbose: log("Combining regions...", priority=1, indent=1)
                regions_train = get_region_labels(train_subject)
                regions_test = get_region_labels(subject)
                X_train_new, X_test_new, common_regions = combine_regions(X_train, X_test, regions_train, regions_test)
                # Explicitly delete the old arrays to free memory
                del X_train, X_test
                X_train, X_test = X_train_new, X_test_new
                del X_train_new, X_test_new
                gc.collect()
            features_processing_time = time.time() - start_time
            if verbose:
                log(f"Features processing time: {features_processing_time:.2f} seconds", priority=0)

            start_time = time.time()

            # Flatten the data after preprocessing in-place
            original_X_train_shape = X_train.shape
            original_X_test_shape = X_test.shape
            X_train = X_train.reshape(X_train.shape[0], -1)
            X_test = X_test.reshape(X_test.shape[0], -1)

            if verbose:
                log(f"Standardizing data...", priority=1, indent=1)

            # Standardize the data in-place
            scaler = StandardScaler(copy=False)
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            # Delete original arrays and replace with scaled versions
            del X_train, X_test
            X_train, X_test = X_train_scaled, X_test_scaled
            del X_train_scaled, X_test_scaled
            gc.collect()  # Collect after standardization

            if verbose:
                log(f"Training model...", priority=1, indent=1)

            # Train logistic regression
            clf = LogisticRegression(random_state=seed, max_iter=10000, tol=1e-3)
            clf.fit(X_train, y_train)

            torch.cuda.empty_cache()
            gc.collect()

            # Evaluate model
            train_accuracy = clf.score(X_train, y_train)
            test_accuracy = clf.score(X_test, y_test)

            # Get predictions - for multiclass classification
            train_probs = clf.predict_proba(X_train)
            test_probs = clf.predict_proba(X_test)
            gc.collect()  # Collect after predictions

            # Filter test samples to only include classes that were in training
            valid_class_mask = np.isin(y_test, clf.classes_)
            y_test_filtered = y_test[valid_class_mask]
            test_probs_filtered = test_probs[valid_class_mask]

            # Convert y_test to one-hot encoding
            y_test_onehot = np.zeros((len(y_test_filtered), len(clf.classes_)))
            for i, label in enumerate(y_test_filtered):
                class_idx = np.where(clf.classes_ == label)[0][0]
                y_test_onehot[i, class_idx] = 1

            y_train_onehot = np.zeros((len(y_train), len(clf.classes_)))
            for i, label in enumerate(y_train):
                class_idx = np.where(clf.classes_ == label)[0][0]
                y_train_onehot[i, class_idx] = 1

            # For multiclass ROC AUC, we need to calculate the score for each class
            n_classes = len(clf.classes_)
            if n_classes > 2:
                train_roc = roc_auc_score(y_train_onehot, train_probs, multi_class='ovr', average='macro')
                test_roc = roc_auc_score(y_test_onehot, test_probs_filtered, multi_class='ovr', average='macro')
            else:
                train_roc = roc_auc_score(y_train_onehot, train_probs)
                test_roc = roc_auc_score(y_test_onehot, test_probs_filtered)

            regression_run_time = time.time() - start_time
            if verbose:
                log(f"Regression run in {regression_run_time:.2f} seconds", priority=0)

            fold_result = {
                "train_accuracy": float(train_accuracy),
                "train_roc_auc": float(train_roc),
                "test_accuracy": float(test_accuracy),
                "test_roc_auc": float(test_roc),
                "timing": {
                    "features_processing_time": features_processing_time,
                    "regression_run_time": regression_run_time,
                }
            }
            bin_results["folds"].append(fold_result)
            
            # Clean up variables no longer needed
            del X_train, y_train, X_test, y_test, train_probs, test_probs
            del y_test_filtered, test_probs_filtered, y_test_onehot, y_train_onehot
            del clf, scaler
            gc.collect()  # Collect after cleanup

            if verbose: 
                log(f"Population, Fold {fold_idx+1}, Bin {bin_start}-{bin_end}: Train accuracy: {train_accuracy:.3f}, Test accuracy: {test_accuracy:.3f}, Train ROC AUC: {train_roc:.3f}, Test ROC AUC: {test_roc:.3f}", priority=0, indent=0)

        # Clean up datasets after processing all folds for this time bin
        torch.cuda.empty_cache()
        gc.collect()

        if bin_start == -bins_start_before_word_onset_seconds and bin_end == bins_end_after_word_onset_seconds and not only_1second:
            results_population["whole_window"] = bin_results # whole window results
        elif bin_start == 0 and bin_end == 1:
            results_population["one_second_after_onset"] = bin_results # one second after onset results
        else:
            results_population["time_bins"].append(bin_results) # time bin results
    

    results = {
        "model_name": "BrainBERT (frozen" + ("; random init)" if random_init else ")"),
        "author": "Christopher Wang, Vighnesh Subramaniam, Adam Uri Yaari, Gabriel Kreiman, Boris Katz, Ignacio Cases, Andrei Barbu",
        "description": f"BrainBERT frozen evaluation using all electrodes ({feature_type}).",
        "organization": "MIT",
        "organization_url": "https://github.com/czlwang/BrainBERT",
        "timestamp": time.time(),

        "evaluation_results": {
            f"{subject.subject_identifier}_{trial_id}": {
                "population": results_population
            }
        },

        "config": {
            "feature_type": feature_type,
            "only_1second": only_1second,
            "seed": seed,
            "subject_id": subject_id,
            "trial_id": trial_id,
            "splits_type": splits_type,
            "randomly_initialized_model": random_init,
        },

        "timing": {
            "subject_load_time": subject_load_time,
        }
    }

    with open(file_save_path, "w") as f:
        json.dump(results, f, indent=4)
    if verbose:
        log(f"Results saved to {file_save_path}", priority=0)

    # Clean up at end of each eval_name loop
    del train_datasets, test_datasets
    gc.collect()
    torch.cuda.empty_cache()  # Ensure GPU memory is also cleaned up
    if verbose:
        log(f"Completed evaluation {eval_name}, memory cleaned up", priority=0)