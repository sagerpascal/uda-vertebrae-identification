import glob

import numpy as np
import torch
from torch.utils.data import Dataset
from skimage.segmentation import felzenszwalb

def check_src_tgt_ok(src_dataset_name, tgt_dataset_name):
    if not (src_dataset_name == "biomedia" and tgt_dataset_name == "covid19-ct"):
        raise AssertionError("you must use spine-labeled / covid-ct_harvard pair")


def get_n_class(mode):
    if mode == "detection":
        return 2  # non-spine, spine
    elif mode == "identification":
        return 1  # continous value
    else:
        raise AttributeError(f"Unknown mode: {mode}")


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)


class BaseDataset(Dataset):

    def __init__(self, sample_dir, has_labels, categorise=False, is_3D=True, n_classes=2, preprocessing=None,
                 load_detection_sample=False, weak_seg_mask=False, sample_dir_2=None):
        self.sample_dir = sample_dir
        self.categorise = categorise
        self.has_labels = has_labels
        self.is_3D = is_3D
        self.n_classes = n_classes
        self.preprocessing = preprocessing
        self.load_detection_sample = load_detection_sample
        self.weak_seg_mask = weak_seg_mask
        self.files, self.labels, self.detections, self.paths = self.get_samples_list(sample_dir, sample_dir_2)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        id_ = self.files[item]
        label_id = self.labels[id_]
        detection_id = self.detections[id_]
        path = self.paths[id_]

        sample = np.load(path + '/' + id_ + '-sample.npy')
        if self.is_3D:
            sample = np.expand_dims(sample, axis=0)

        if self.has_labels:
            labeling = np.load(path + '/' + label_id + '.npy')
            labeling = labeling.astype('int64')
            # labeling = labeling[idx]
        else:
            labeling = None

        if self.load_detection_sample:
            detection = np.load(path + '/' + detection_id + '.npy')
            detection = detection.astype('int64')
            # detection = detection[idx]
        else:
            detection = None

        if self.preprocessing is not None:
            sample, labeling, detection = self.do_preprocessing(sample, labeling, detection)

        if self.weak_seg_mask:
            weak_mask = self.get_segmentation_mask(sample, detection)
        else:
            weak_mask = None

        sample = self.to_tensor_(sample)
        labeling = self.to_tensor_(labeling)
        detection = self.to_tensor_(detection)
        weak_mask = self.to_tensor_(weak_mask).long()

        return {'sample': sample,
                'labeling': labeling,
                'detection': detection,
                'weak_mask': weak_mask,
                }

    def to_tensor_(self, array):
        if array is not None:
            return torch.from_numpy(array)
        else:
            return torch.tensor(-1)

    def do_preprocessing(self, sample, labelling, detection):
        sample = np.transpose(sample, (1, 2, 0))

        if labelling is not None:
            labelling = np.expand_dims(labelling, 2)
        if detection is not None:
            detection = np.expand_dims(detection, 2)

        if labelling is not None and detection is not None:
            res = self.preprocessing(image=sample, mask=labelling, detection=detection)
            sample, labelling, detection = res['image'], res['mask'], res['detection']
        elif labelling is not None:
            res = self.preprocessing(image=sample, mask=labelling)
            sample, labelling = res['image'], res['mask']
        elif detection is not None:
            res = self.preprocessing(image=sample, detection=detection)
            sample, detection = res['image'], res['detection']
        else:
            res = self.preprocessing(image=sample)
            sample = res['image']

        sample = np.transpose(sample, (2, 0, 1))

        if labelling is not None:
            labelling = np.squeeze(labelling, axis=2)
        if detection is not None:
            detection = np.squeeze(detection, axis=2)

        return sample, labelling, detection

    def get_samples_list(self, sample_dir, sample_dir_2):
        ext_len = len("-sample.npy")
        files = []
        labels = {}
        detections = {}
        paths = {}
        sample_paths = glob.glob(sample_dir + "/**/*sample.npy", recursive=True)
        if sample_dir_2 is not None:
            sample_paths_2 = glob.glob(sample_dir_2 + "/**/*sample.npy", recursive=True)
            sample_paths += sample_paths_2
        for sample_path in sample_paths:
            sample_path_without_ext = sample_path[:-ext_len]
            label = sample_path_without_ext.rsplit('/', 1)[1]

            files.append(label)
            paths[label] = sample_path.rsplit("/", 1)[0]

            # read file and assign to labels
            labels[label] = label + "-labelling"
            detections[label] = label + "-detection"

        print(f"Number of samples: {len(files)}")

        return files, labels, detections, paths

    def get_segmentation_mask(self, sample, detection):
        img = sample[4]
        if np.any(detection != 0):
            spine_area = np.where(detection != 0)
            dim0_min, dim0_max = np.min(spine_area[0]), np.max(spine_area[0])
            dim1_min, dim1_max = np.min(spine_area[1]), np.max(spine_area[1])
            tgt_img_cropped = img[dim0_min:dim0_max + 1, dim1_min:dim1_max + 1]
            detection_cropped = detection[dim0_min:dim0_max + 1, dim1_min:dim1_max + 1]
            spine_cropped = tgt_img_cropped * detection_cropped

            spine_cropped_enhanced = (spine_cropped > 1.42) * spine_cropped
            segments_felzenszwalb = felzenszwalb(spine_cropped_enhanced, scale=60, sigma=0.8, min_size=150)

            for i in range(np.max(segments_felzenszwalb) + 1):
                positions = np.where(segments_felzenszwalb == i)
                length = np.max(positions[1]) - np.min(positions[1])
                if length > 50:
                    segments_felzenszwalb[segments_felzenszwalb == i] = 0

            unique = np.unique(segments_felzenszwalb)
            positions = {}
            for u in unique:
                if u == 0:
                    continue
                p = np.where(segments_felzenszwalb == u)
                positions[u] = (np.min(p[1]), np.max(p[1]))

            for u1, p1 in positions.items():
                for u2, p2 in positions.items():
                    if u1 == u2:
                        continue
                    if ((p1[0] - 4) <= p2[0] and (p1[1] + 4) >= p2[1]):
                        segments_felzenszwalb[segments_felzenszwalb == u2] = u1

            segments_felzenszwalb = np.pad(segments_felzenszwalb, (
                (dim0_min, img.shape[0] - dim0_max - 1), (dim1_min, img.shape[1] - dim1_max - 1)), mode="constant",
                                           constant_values=0)

            return segments_felzenszwalb

        else:
            return np.zeros_like(img)


def get_dataset(dataset_name, split, type, mode, use_data_augmentation, with_detection, with_weak_mask=False,
                use_train_labels_target=None):
    assert mode == "detection" or mode == "identification"
    assert type == "source" or type == "target" or type == "target-labeled"

    if split == "train" or split == "training":
        suffix = "training"
    elif (split == "test" or split == "testing") and type == "target-labeled":
        suffix = "testing_labeled"
    elif split == "test" or split == "testing":
        suffix = "testing"
    else:
        raise AttributeError("Unknown dataset split")

    assert not use_train_labels_target or (
            use_train_labels_target and mode == "identification" and type == "target" and suffix == "training")

    if dataset_name == "biomedia":
        sample_dir = f"../data/biomedia/samples/{mode}/{suffix}"
        sample_dir_2 = f"../data/biomedia/samples/{mode}/{suffix}_labeled"
    elif dataset_name == "covid19-ct":
        sample_dir = f"../data/covid19-ct/samples/{mode}/{suffix}"
        sample_dir_2 = f"../data/covid19-ct/samples/{mode}/{suffix}_labeled"
    else:
        raise AttributeError("Unknown dataset name")

    if not use_train_labels_target:
        sample_dir_2 = None

    preprocessing = get_augmentations() if use_data_augmentation else None

    has_labels = type == "source" or type == "target-labeled"
    if mode == "detection":
        return BaseDataset(sample_dir, has_labels=has_labels, is_3D=True, n_classes=2,
                           preprocessing=preprocessing, sample_dir_2=sample_dir_2)
    elif mode == "identification":
        return BaseDataset(sample_dir, has_labels=has_labels, is_3D=False, n_classes=1,
                           load_detection_sample=with_detection, preprocessing=preprocessing, sample_dir_2=sample_dir_2,
                           weak_seg_mask=with_detection and with_weak_mask)


def get_augmentations():
    import albumentations as A
    transform = [
        # A.Blur(p=0.5),
        # A.Downscale(p=0.5),
        A.Rotate(limit=(20, 20), border_mode=0),
        A.RandomCrop(height=70, width=280, p=0.5),
        A.PadIfNeeded(min_height=80, min_width=320, always_apply=True, border_mode=0),
    ]
    return A.Compose(transform, additional_targets={"detection": "mask"})


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    dataset = BaseDataset("../../data/samples/identification/training", has_labels=True, is_3D=False, n_classes=1)
    dataset_aug = BaseDataset("../../data/samples/identification/training", has_labels=True, is_3D=False, n_classes=1,
                              preprocessing=get_augmentations())
    for i in range(30):
        files = dataset[i]
        files_aug = dataset_aug[i]

        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
        ax[0, 0].imshow(files['sample'])
        ax[0, 1].imshow(files['sample'])
        ax[0, 1].imshow(files['labeling'], cmap='jet', alpha=0.5)
        ax[1, 0].imshow(files_aug['sample'])
        ax[1, 1].imshow(files_aug['sample'])
        ax[1, 1].imshow(files_aug['labeling'], cmap='jet', alpha=0.5)
        plt.tight_layout()
        plt.show()
