import numpy as np
import torch
from mgh2024_subject import MGH2024Subject


class MGHSeizureDataset(torch.utils.data.Dataset):
    def __init__(self, subject, session_id, 
                 WINDOW_SIZE=3, SEIZURE_REGION_SIZE=20, ANNOTATION_PADDING=60*10,
                 output_index=False):
        self.subject = subject
        self.subject_id = subject.subject_id
        self.session_id = session_id
        self.output_index = output_index

        subject.load_neural_data(session_id)

        self.WINDOW_SIZE = WINDOW_SIZE

        annotations_clean = self.subject.get_annotations(session_id=session_id, window_from=0, window_to=None, remove_persyst=True)
        annotations_all = self.subject.get_annotations(session_id=session_id, window_from=0, window_to=None, remove_persyst=False, remove_cashlab=False, remove_software_changes=False)

        annotation_onsets = annotations_all[0] # array of seconds
        session_start = 0 # seconds
        session_end = self.subject.session_metadata[session_id]['session_length'] # seconds

        seizure_keywords = ["seizure", "ictal", "onset"]#"interictal", "onset", "start", "end", "offset"]
        seizure_onsets = np.array([annotation_onset for annotation_onset, annotation_description in zip(*annotations_clean) if any(x in annotation_description.lower() for x in seizure_keywords)])

        # For each window, check if it's far enough from all annotations
        far_from_all_annotations = []
        seizure_windows = []
        for window_start in np.arange(session_start, session_end, WINDOW_SIZE):
            window_end = window_start + WINDOW_SIZE
            # Check distance to all annotations
            min_distance_any_annotation = min(min(abs(annotation_onsets - window_start)), min(abs(annotation_onsets - window_end)))
            min_distance_seizure_annotations = min(min(abs(seizure_onsets - window_start)), min(abs(seizure_onsets - window_end)))
            
            # If window is far enough from all annotations, add it to far_from_all_annotations
            if min_distance_any_annotation >= ANNOTATION_PADDING:
                far_from_all_annotations.append(window_start)
            # If window is within seizure region size, add it to seizure_windows
            if min_distance_seizure_annotations <= SEIZURE_REGION_SIZE:
                seizure_windows.append(window_start)

        self.far_from_all_annotations = np.array(far_from_all_annotations)
        self.seizure_windows = np.array(seizure_windows)
        
        # Rebalance classes by randomly sampling from far_from_all_annotations to match seizure_windows length
        if len(self.far_from_all_annotations) > len(self.seizure_windows):
            self.far_from_all_annotations = np.random.choice(self.far_from_all_annotations, size=len(self.seizure_windows), replace=False)
    
    def __getitem__(self, idx):
        label = idx % 2
        if label == 0:
            index = self.far_from_all_annotations[idx // 2]
        else:
            index = self.seizure_windows[idx // 2]

        window_start = int(index * self.subject.get_sampling_rate(self.session_id))
        window_end = window_start + int(self.WINDOW_SIZE * self.subject.get_sampling_rate(self.session_id))
        if self.output_index:
            return (window_start, window_end), label

        neural_data = self.subject.get_all_electrode_data(session_id=self.session_id, window_from=window_start, window_to=window_end)
        return neural_data, label

    def __len__(self):
        return len(self.seizure_windows) + len(self.far_from_all_annotations)