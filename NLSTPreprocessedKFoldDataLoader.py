from collections import defaultdict
import pandas
from sklearn.model_selection import StratifiedKFold, train_test_split
from scipy.ndimage import map_coordinates, zoom
from torch.utils.data import Dataset
from torch.utils.data import DataLoader as TorchDataLoader, WeightedRandomSampler
import numpy
import random
import torch
import os
import matplotlib.pyplot as plt
from collections import defaultdict
import random
import base64
import io
from PIL import Image
from ct_image_augmenter import CTImageAugmenter, CTImageAugmenter3D
import numpy as np

from visualizationuploader import VisualizationUploader


class NLSTPreprocessedKFoldDataLoader:
    def __init__(
            self,
            config,
            lung_metadataframe,
            load_data_name=False
    ):
        self.config = config

        self.dataloaders = None
        self.dataloaders_by_subset = None
        self.data_names_by_subset = None
        self.data_splits = None
        self.load_data_name = None
        self.torch_generator = None
        self.dataloaders = defaultdict(list)
        self.dataloaders_by_subset = defaultdict(list)
        self.data_names_by_subset = defaultdict(list)
        self.data_splits = defaultdict(lambda: defaultdict(list))
        self.load_data_name = load_data_name
        self.torch_generator = torch.Generator()
        csv_path = self.config.directories.rootdir_tab
        self.features_df = pandas.read_csv(csv_path)
        self.lung_metadataframe = lung_metadataframe

        self.torch_generator.manual_seed(self.config.seed_value)
        self._set_data_splits(self.lung_metadataframe)
        self._set_dataloaders()

    def get_dataloaders(self):
        return self.dataloaders
    
    def get_data_names(self):
        folds = self.config.number_of_k_folds
        if folds == 0:
            folds = 1
        data_names = {subset_type: [
            self.data_splits[subset_type]['file_names'][datafold_id]
            for datafold_id in range(folds)
        ] for subset_type in ["train", "validation", "test"]}
        return data_names

    def _get_torch_dataloader(
        self,
        file_names,
        labels,
        subset_type,
        torch_dataloader_kwargs
    ):
        if self.config.weighted_random_sampler:
            print(f"\nUsing WeightedRandomSampler for {subset_type} subset")
            dataset = NLSTPreprocessedDataLoader(
                config=self.config,
                file_names=file_names,
                labels=labels,
                load_data_name=self.load_data_name,
                subset_type=subset_type,
                lung_metadataframe=self.lung_metadataframe,
                features_df=self.features_df 
            )

            if subset_type == "train":
                # Convert labels to numpy for processing
                labels_np = numpy.array(labels)
                class_counts = numpy.bincount(labels_np)
                class_weights = 1. / class_counts
                # Assign weight to each sample
                sample_weights = class_weights[labels_np]
                sampler = WeightedRandomSampler(
                    weights=sample_weights,
                    num_samples=len(sample_weights),
                    replacement=True
                )
                shuffle = False  # Disable shuffle when using sampler
                torch_dataloader = TorchDataLoader(
                    dataset=dataset,
                    sampler=sampler,
                    generator=self.torch_generator,
                    worker_init_fn=self._get_torch_dataloader_worker_init_fn,
                    **torch_dataloader_kwargs
                )
            else:
                torch_dataloader = TorchDataLoader(
                    dataset=dataset,
                    shuffle=False,  # Validation/test can be shuffled normally or not
                    generator=self.torch_generator,
                    worker_init_fn=self._get_torch_dataloader_worker_init_fn,
                    **torch_dataloader_kwargs
                )
        else:
            print(f"\nUsing regular DataLoader for {subset_type} subset")
            torch_dataloader = TorchDataLoader(
            dataset=NLSTPreprocessedDataLoader(
                config=self.config,
                file_names=file_names,
                labels=labels,
                load_data_name=self.load_data_name,
                subset_type=subset_type,
                lung_metadataframe=self.lung_metadataframe,
                features_df=self.features_df
            ),
            generator=self.torch_generator,
            shuffle=True if subset_type == "train" else False,
            worker_init_fn=self._get_torch_dataloader_worker_init_fn,
            **torch_dataloader_kwargs
        )
                
        return torch_dataloader


    def _get_torch_dataloader_worker_init_fn(self, worker_id):
        numpy.random.seed(self.config.seed_value + worker_id)
        random.seed(self.config.seed_value + worker_id)

    def _set_dataloaders(self):
        folds = self.config.number_of_k_folds
        if folds == 0: # Works for no cross validation and cross validation 
            # When n_folds is 0 it works as if it was just one data split, adding another diemnsion to the datalaoder dictionary
            folds = 1
        for subset_type in ["train", "validation", "test"]:
            for datafold_id in range(folds):
                self.dataloaders[subset_type].append(
                    self._get_torch_dataloader(
                        file_names=self.data_splits[subset_type] \
                            ['file_names'][datafold_id],
                        labels=self.data_splits[subset_type] \
                            ['labels'][datafold_id],
                        subset_type=subset_type,
                        torch_dataloader_kwargs=
                            self.config.torch_dataloader_kwargs
                    )
                )
            print(f"\n✅ Set {subset_type} dataloaders with {len(self.dataloaders[subset_type])} folds")
            # Print the number of samples in batches for each fold
            for fold_id, dataloader in enumerate(self.dataloaders[subset_type], 1):
                num_samples = len(dataloader.dataset)
                print(f"  - Fold {fold_id}: {num_samples} samples")
                
                # Print the distribution of labels in the fold
                labels = dataloader.dataset.labels
                unique_labels, counts = numpy.unique(labels, return_counts=True)
                label_distribution = dict(zip(unique_labels, counts))
                print(f"    Label distribution: {label_distribution}")

                # Print the distribution of the batches
                for batch in dataloader:
                    # Print the distribution of 0 and 1 labels in the batch
                    labels = batch[1]
                    unique_labels, counts = numpy.unique(labels, return_counts=True)
                    label_distribution = dict(zip(unique_labels, counts))
                    print(f"    Batch label distribution: {label_distribution}")


    def _set_data_splits(self, lung_metadataframe):

        # fixed_splits_path = r'C:\Users\HP\OneDrive - Universidade do Porto\Documentos\UNIVERSIDADE\Tese\clinical_models\data\lung_metadata_with_splits.csv'
        # if os.path.exists(fixed_splits_path):
        #     fixed_test_df = pandas.read_csv(fixed_splits_path)
        #     fixed_test_pids = fixed_test_df[fixed_test_df['split_fold_1'] == 'test']['pid'].tolist()


        # metadata_with_splits = lung_metadataframe.copy()

        # ## Use the test from smaller dataset as test here
        # fixed_test_set = metadata_with_splits[metadata_with_splits['pid'].isin(fixed_test_pids)]
        # remaining_df = metadata_with_splits[~metadata_with_splits['pid'].isin(fixed_test_pids)]


        # # Stratified split on remaining
        # strat_test_size = self.config.test_fraction_of_entire_dataset * metadata_with_splits.shape[0] - fixed_test_set.shape[0]
        # if strat_test_size > 0:
        #     train_val_df, strat_test_df = train_test_split(
        #         remaining_df,
        #         test_size=strat_test_size,
        #         random_state=self.config.seed_value,
        #         stratify=remaining_df['label']
        #     )
        #     test_df = pandas.concat([fixed_test_set, strat_test_df], ignore_index=True)
        # else:
        #     train_val_df = remaining_df
        #     test_df = fixed_test_set

        metadata_with_splits = lung_metadataframe.copy()

        # === Step 1: Fixed stratified test split ===
        train_val_df, test_df = train_test_split(
            lung_metadataframe,
            test_size=self.config.test_fraction_of_entire_dataset,
            random_state=self.config.seed_value,
            stratify=lung_metadataframe['label']
        )

        if not self.config.number_of_k_folds:
            self.config.number_of_k_folds = 1 #Doesnt work

        # Assign test split once (same for all folds)
        for fold_id in range(1, self.config.number_of_k_folds + 1):
            metadata_with_splits.loc[test_df.index, f'split_fold_{fold_id}'] = 'test'

        # === Step 2: Stratified K-Fold on train_val_df only ===
        skf = StratifiedKFold(
            n_splits=self.config.number_of_k_folds,
            shuffle=True,
            random_state=self.config.seed_value
        )
        # Missing a generator, no??

        for fold_id, (train_idx, val_idx) in enumerate(
            skf.split(train_val_df, train_val_df['label']), 1
        ):
            train_split = train_val_df.iloc[train_idx]
            val_split = train_val_df.iloc[val_idx]

            # Assign to DataFrame
            metadata_with_splits.loc[train_split.index, f'split_fold_{fold_id}'] = 'train'
            metadata_with_splits.loc[val_split.index, f'split_fold_{fold_id}'] = 'val'

            # Save in internal structure if needed
            self.data_splits['train']['file_names'].append(train_split['path'].tolist())
            self.data_splits['train']['labels'].append(train_split['label'].tolist())
            self.data_splits['validation']['file_names'].append(val_split['path'].tolist())
            self.data_splits['validation']['labels'].append(val_split['label'].tolist())
            self.data_splits['test']['file_names'].append(test_df['path'].tolist())
            self.data_splits['test']['labels'].append(test_df['label'].tolist())

        # === Save annotated DataFrame to CSV ===
        metadata_with_splits.to_csv(
            '/nas-ctm01/homes/fmferreira/AI4LUNGS/clinical_metadata_with_splits.csv',
            index=False
        )
        print("\n✅ Saved split assignments to 'clinical_metadata_with_splits.csv'")


        # # === One fixed split ===
        # train_df, val_df = train_test_split(
        #     train_val_df,
        #     test_size=2,   # <-- only 2 samples for "val"
        #     random_state=self.config.seed_value,
        #     stratify=train_val_df['label']
        # )

        # fold_id = 1  # since we only want one split
        # metadata_with_splits.loc[train_df.index, f'split_fold_{fold_id}'] = 'train'
        # metadata_with_splits.loc[val_df.index, f'split_fold_{fold_id}'] = 'val'

        # # Optional: still assign test = val if pipeline expects it
        # metadata_with_splits.loc[val_df.index, f'split_fold_{fold_id}'] = 'val'

        # self.data_splits['train']['file_names'].append(train_df['path'].tolist())
        # self.data_splits['train']['labels'].append(train_df['label'].tolist())
        # self.data_splits['validation']['file_names'].append(val_df['path'].tolist())
        # self.data_splits['validation']['labels'].append(val_df['label'].tolist())
        # self.data_splits['test']['file_names'].append(test_df['path'].tolist())
        # self.data_splits['test']['labels'].append(test_df['label'].tolist())

        # # Save
        # metadata_with_splits.to_csv(
        #     '/nas-ctm01/homes/mipaiva/small_scripts/ignore_metadata_with_splits.csv',
        #     index=False
        # )
        # print("\n✅ Saved single split with 2 val samples.")

def array_to_base64(npy_img):
        # Ensure the input is a NumPy array
        if isinstance(npy_img, torch.Tensor):
            npy_img = npy_img.cpu().numpy()
            
        # --- 1. Handle Multi-Dimensional Input (3D/2.5D) ---
        if npy_img.ndim == 3:
            # Assuming shape is [Z, H, W] (Slices, Height, Width). Select the center slice.
            center_slice_index = npy_img.shape[0] // 2
            npy_img = npy_img[center_slice_index, :, :]
        
        elif npy_img.ndim == 4:
            # Assuming batch dimension or similar, reduce it to 2D
            npy_img = npy_img[0, npy_img.shape[1] // 2, :, :]

        # --- 2. Scale 0-1 float to 0-255 uint8 (Confirmed 0-1 range) ---
        image_8bit = (npy_img * 255).astype(numpy.uint8) # Use numpy instead of np for consistency
        
        # --- 3. Convert to Base64 (PNG encoding) ---
        pil_img = Image.fromarray(image_8bit) 
        buffered = io.BytesIO()
        
        pil_img.save(buffered, format="PNG") 
        
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

class NLSTPreprocessedDataLoader(Dataset):
    def __init__(
            self,
            config,
            file_names,
            labels,
            load_data_name,
            subset_type,
            lung_metadataframe,
            features_df
    ):
        self.config = config
        self.load_data_name = load_data_name
        self.subset_type = subset_type
        self.lung_metadataframe = lung_metadataframe
        self.augmented_to_original_data_ratio = config.data_augmentation.augmented_to_original_data_ratio
        self.use_image = getattr(config, 'image', False)
        self.use_text = getattr(config, 'text', False)
        self.use_tabular = getattr(config, 'tabular', False)
        self.features_df = features_df
        self.apply_data_augmentations = config.data_augmentation.apply

        self.text_column = config.stage_type
        if 'roi' in config:
            if self.config.roi in ['lung', 'masked']:
                self.roi = config.roi
            else:
                self.roi = 'ws'
        else:
            self.roi = None
                
        self.visualization = config.visualize_imgur
        if self.visualization:
            self.visualization_uploader = VisualizationUploader(
                client_id='f5a89997db63c60',
                album_id='WC0PErb6jxLRWHt'
            )

        if self.apply_data_augmentations and subset_type == "train":
            print("Data Aug")
            label_to_files = defaultdict(list)
            for file, label in zip(file_names, labels):
                label_to_files[label].append(file)

            # Identify majority and minority class
            class_counts = {k: len(v) for k, v in label_to_files.items()}
            max_class = max(class_counts, key=class_counts.get)
            min_class = min(class_counts, key=class_counts.get)

            diff = class_counts[max_class] - class_counts[min_class]

            # Sample (with replacement) from the minority class to balance
            additional_files = random.choices(label_to_files[min_class], k=diff)
            self.file_names = file_names + additional_files
            self.labels = labels + [min_class] * diff
            

            # Track which indices are duplicates/augmented
            original_count = len(file_names)
            self.augmented_indices = set(range(original_count, len(self.file_names)))
            print(f"Original shape: {original_count}, New Shape: {len(self.file_names)}")
        else:
            self.file_names = file_names
            self.labels = labels
            self.augmented_indices = set()

        if config.data_augmentation.apply and subset_type == "train":
            if config.dimension == 3:
                self.data_augmenter = CTImageAugmenter3D(
                    parameters=config.data_augmentation.parameters
                )
            else:
                # Use CTImageAugmenter for 2D and 2.5D
                self.data_augmenter = CTImageAugmenter(
                    parameters=config.data_augmentation.parameters
                )
        

    def __len__(self):
        return len(self.file_names)
    

    def _get_tabular_data(self, data_index):
    # 1. Get the PID for the current sample
        dataframe_row = self.lung_metadataframe.loc[
            self.lung_metadataframe['path'] == self.file_names[data_index]
        ]
        current_pid = dataframe_row['pid'].values[0]

        # 2. Grab the features for this PID from the stage-specific table
        # We assume the stage-specific CSV has 'pid' and then only features
        features_row = self.features_df.loc[self.features_df['pid'] == current_pid]
        
        # 3. Drop 'pid' and convert everything else to a tensor
        # No more worry about fup_days or study_yr!
        tabular_features = features_row.drop(columns=['pid'])
        tabular_features = features_row.drop(columns = ['rndgroup_1','rndgroup_2'])
        
        final_array = tabular_features.values[0].astype(np.float32)
        return torch.from_numpy(final_array)
    

    def _get_clinical_text(self, data_index):
        """Retrieves the raw sentence from the metadata."""
        dataframe_row = self.lung_metadataframe.loc[
            self.lung_metadataframe['path'] == self.file_names[data_index]
        ]
        # Return the sentence as a string
        return str(dataframe_row[self.text_column].values[0])

    def __getitem__(self, data_index):
        dataframe_row = self.lung_metadataframe.loc[
        self.lung_metadataframe['path'] == self.file_names[data_index]]
        current_pid = int(dataframe_row['pid'].values[0])
        
        try:

            # We use a dictionary to make the model inputs interchangeable
            inputs = {}
            inputs['pid'] = current_pid


            if self.use_image:
                        inputs['image'] = self._get_data(data_index)
            
            # --- Text Branch ---
            if self.use_text:
                inputs['text'] = self._get_clinical_text(data_index)

            # --- Tabular Branch ---
            if self.use_tabular:
                inputs['tabular'] = self._get_tabular_data(data_index)

            label = self._get_label(data_index) 
            time = self._get_time(data_index)
            # Standard output structure
            output = [inputs, label, time]

            if getattr(self.config, "use_stage_label", False):
                output.append(self._get_stage_label(data_index))
            elif getattr(self.config, "use_stagebin_label", False):
                output.append(self._get_stage_label(data_index, stage='binary_stage'))

            if self.load_data_name:
                output.insert(0, self.file_names[data_index])

            return tuple(output)
        
        except Exception as e:
            print(f"[ERROR] Error in __getitem__ at index {data_index}: {e}")
            print(f"File path: {self.file_names[data_index]}")
            print(f"Label: {self.labels[data_index]}")
            raise e

    def _get_stage_label(self, data_index, stage='stage'):
        # Assuming stage labels are stored in self.lung_metadataframe['stage']
        dataframe_row = self.lung_metadataframe.loc[
            self.lung_metadataframe['path'] == self.file_names[data_index]
        ]
        stage_value = dataframe_row[stage].values[0]
        return torch.tensor([int(stage_value)])

    def get_slice_range_3d(self, total_slices, slice_idx, n_slices):
        """
        Calculates a robust slice range around slice_idx, always returning n_slices if possible.
        If slice_idx is None, it defaults to the center of the volume.
        """
        if slice_idx is None:
            slice_idx = total_slices // 2

        half = n_slices // 2

        # Initial guess
        start = slice_idx - half
        end = slice_idx + half + (0 if n_slices % 2 == 0 else 1)

        # Clamp to volume bounds
        if start < 0:
            end += abs(start)
            start = 0
        if end > total_slices:
            excess = end - total_slices
            start = max(0, start - excess)
            end = total_slices

        # Final adjustment to ensure exactly n_slices
        current_len = end - start
        if current_len < n_slices:
            if start > 0:
                missing = n_slices - current_len
                shift = min(missing, start)
                start -= shift
                current_len = end - start
            if current_len < n_slices and end < total_slices:
                missing = n_slices - current_len
                shift = min(missing, total_slices - end)
                end += shift

        return start, end
    
    


    def _get_data(self, data_index):
        dataframe_row = self.lung_metadataframe.loc[
            self.lung_metadataframe['path'] == self.file_names[data_index]
        ]
        pid = dataframe_row['pid'].values[0]
        study_yr = dataframe_row['study_yr'].values[0]
        reversed = dataframe_row['reversed'].values[0]
        if dataframe_row['sct_slice_num'].values[0] is None:
            slice_idx = None
            print(f"[WARNING] No slice index found for {self.file_names[data_index]}. Using None.")
        else:
            slice_idx = int(dataframe_row['sct_slice_num'].values[0])


        if getattr(self.config, "random", False):  # If config.random exists and is True
            if self.config.dimension == 2:
                image = numpy.random.rand(512, 512).astype(numpy.float32)
            elif self.config.dimension == 2.5:
                image = numpy.random.rand(10, 512, 512).astype(numpy.float32)
            elif self.config.dimension == 3:
                image = numpy.random.rand(512, 512, 32).astype(numpy.float32)
            else:
                raise ValueError(f"[ERROR] Unknown dimension {self.config.dimension}")
        else:
            if self.config.dimension == 2:
                data_path = '/nas-ctm01/datasets/public/medical_datasets/lung_ct_datasets/nlst/preprocessed_data/protocol_5/2d'
                data_path = os.path.join(data_path, self.roi) if self.roi else data_path
                image = self._get_slice(data_index, data_path, pid, study_yr)

            elif self.config.dimension == 3:
                data_path = '/nas-ctm01/datasets/public/medical_datasets/lung_ct_datasets/nlst/preprocessed_data/protocol_5/general_shift_crop'
                data_path = os.path.join(data_path, self.roi) if self.roi else data_path
                image = self._get_scan(data_index, data_path, pid, study_yr, slice_idx, reversed)
            elif self.config.dimension == 2.5: # TODO refix the other dimensions
                data_path = '/nas-ctm01/datasets/public/medical_datasets/lung_ct_datasets/nlst/preprocessed_data/protocol_5/25d'
                image = self._get_2_5(data_index, data_path, pid, study_yr)
            else:
                raise ValueError(f"[ERROR] Unknown dimension {self.config.dimension}")
             
        if image is None:
            raise ValueError(f"[ERROR] Image is None at index {data_index}. File info: {self.lung_metadataframe.loc[self.lung_metadataframe['path'] == self.file_names[data_index]]}")

        # TODO: Do the same for lung roi and 2.5D and resample

        image = image.astype(numpy.float32)
        # Apply augmentation only to duplicated/repeated images
        if (self.apply_data_augmentations and 
            data_index in self.augmented_indices and 
            self.subset_type == "train"):
            
            image = self.data_augmenter(image)
            # Squeeze the image to remove single-dimensional entries
            if image.ndim == 3 and self.config.dimension != 3:
                image = numpy.squeeze(image)
            elif image.ndim == 4 and self.config.dimension == 3:
                print(f"Image shape before squeeze: {image.shape}")
                image = numpy.squeeze(image, axis=-1)
                print(f"Image shape after augmentation: {image.shape}")

        if self.visualization:
            self.visualization_uploader.upload_image(
                image=image,  # Assuming the last slice is the one to visualize
                file_name= f"slice_{pid}_{study_yr}.png",
                dataset_name="NLSTPreprocessed"
            )
        
        image_tensor = torch.from_numpy(image).float()

        if self.config.dimension == 2:
            # Shape: (H, W) -> (1, H, W) -> (3, H, W)
            if image_tensor.ndim == 2:
                image_tensor = image_tensor.unsqueeze(0).repeat(3, 1, 1)
        
        elif self.config.dimension == 2.5 or self.config.dimension == 3:
            # If your FM expects 3 channels but you have 9 slices, 
            # you may need to project them or select 3. 
            # For now, let's ensure it has a channel dim:
            if image_tensor.ndim == 3:
                image_tensor = image_tensor.unsqueeze(0) # (1, D, H, W)

        return image_tensor
    
    def _get_time(self, data_index):
        # Retrieve the row from the main metadata dataframe
        dataframe_row = self.lung_metadataframe.loc[
            self.lung_metadataframe['path'] == self.file_names[data_index]
        ]
        
        time_value = dataframe_row['fup_days'].values[0]

        # Convert the value to a torch tensor (float32, 1D tensor)
        time = torch.tensor([
            float(time_value)
        ])
        return time 
    


    def _get_slice(self, data_index, data_path, pid, study_yr):
        try:
            slice_image = numpy.load(
                os.path.join(
                    data_path,
                    f"{pid}_{study_yr}.npy"
                )
            )

            if self.config.resize:
                slice_image = numpy.resize(slice_image, (224, 224))
            

            return slice_image
        except Exception as e:
            print(f"Error loading slice {data_index}: {e}")
            print(f"File path: {self.file_names[data_index]}")
            return None
    

    def get_slice_range(total_slices, slice_idx, n_slices):
        """
        Calculates a robust slice range around slice_idx, always returning n_slices if possible.
        If slice_idx is None, it defaults to the center of the volume.
        """
        if slice_idx is None:
            slice_idx = total_slices // 2

        half = n_slices // 2

        # Initial guess
        start = slice_idx - half
        end = slice_idx + half + (0 if n_slices % 2 == 0 else 1)

        # Clamp to volume bounds
        if start < 0:
            end += abs(start)
            start = 0
        if end > total_slices:
            excess = end - total_slices
            start = max(0, start - excess)
            end = total_slices

        # Final adjustment to ensure exactly n_slices
        current_len = end - start
        if current_len < n_slices:
            if start > 0:
                missing = n_slices - current_len
                shift = min(missing, start)
                start -= shift
                current_len = end - start
            if current_len < n_slices and end < total_slices:
                missing = n_slices - current_len
                shift = min(missing, total_slices - end)
                end += shift

        return start, end

    def resample_volume_centered_full(self, dicom_image, slice_idx, n_slices=9, coverage='full'):
        """
        Resample the *entire* 3D image along the slice axis to n_slices,
        centered around the nodule slice.
        
        Parameters:
            dicom_image: np.ndarray [num_slices, H, W]
            slice_idx: int - center slice of nodule
            n_slices: int - desired number of output slices
            coverage: 'full' for whole scan, or int specifying number of original slices to span
        """
        total_slices = dicom_image.shape[0]
        if slice_idx is None:
            slice_idx = total_slices // 2

        # Define the range we want to cover
        if coverage == 'full':
            start = 0
            end = total_slices
        else:
            half_span = coverage // 2
            start = max(0, slice_idx - half_span)
            end = min(total_slices, slice_idx + half_span)

        # Extract the region of interest
        sub_volume = dicom_image[start:end]

        # Compute zoom factor along z-axis only
        zoom_factor = (n_slices / sub_volume.shape[0], 1, 1)

        # Interpolate entire volume
        resampled_volume = zoom(sub_volume, zoom_factor, order=1)  # order=1 = linear interp
        #print(f"Resampled volume shape: {resampled_volume.shape}")
        return resampled_volume

    def _get_scan(self, data_index, data_path, pid, study_yr, slice_idx, reversed):
        dicom_image = numpy.load(
                os.path.join(
                    data_path,
                    f"{pid}_{self.roi}.npy"
                )
            )
        
        if 'resample_z' in self.config:
            if self.config.resample_z:
                # Load the full 3D volume and resample it
                n_slices = 9
                dicom_image = self.resample_volume_centered_full(
                        dicom_image, slice_idx=slice_idx, n_slices=n_slices, coverage=160
                )
        else:
            

            n_slices = 9

            # Compute start and end slice indices from the center of the nodule
            # Based on the metadataframe
            start, end = self.get_slice_range_3d(
                total_slices=dicom_image.shape[0],
                slice_idx= slice_idx,
                n_slices=n_slices
            )

            # Extract the central volume
            dicom_image = dicom_image[start:end, :, :]

            # if reversed:
            #     dicom_image = numpy.flip(dicom_image, axis=0)

            if self.config.resize:
                dicom_image = numpy.resize(dicom_image, (dicom_image.shape[0], 224, 224))

        return dicom_image
    
    def _get_2_5(self, data_index, data_path, pid, study_yr):
        # if different n_slices, read from 3D and change the number TODO
        
        dicom_image = numpy.load(
                os.path.join(
                    data_path,
                    f"{pid}_{study_yr}.npy"
                )
            )

        if self.config.resize:
            dicom_image = numpy.resize(dicom_image, (dicom_image.shape[0], 224, 224))

        return dicom_image


    def _get_label(self, data_index):
        labels = torch.tensor([
            float(self.labels[data_index])
        ])

        return labels
    