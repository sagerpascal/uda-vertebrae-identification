import os
import sys
os.chdir(sys.path[0])

import argparse
import glob

import numpy as np
from tqdm import tqdm

from utility_functions import opening_files
from utility_functions.sampling_helper_functions import densely_label, pre_compute_disks

parser = argparse.ArgumentParser()
parser.add_argument("--training_dataset_dir", default=None, help="Path to training data (input)")
parser.add_argument("--testing_dataset_dir", default=None, help="Path to testing data (input)")
parser.add_argument("--training_sample_dir", default=None, help="Output path to store training samples")
parser.add_argument("--testing_sample_dir", default=None, help="Output path to store testing samples")
parser.add_argument("--volume_format", default=".dcm", help="Format of the CT-scan volume (either .nii.gz or .dcm)")
parser.add_argument("--label_format", default=".nii.gz", help="Format of the labels (either .lml or .nii.gz)")
parser.add_argument('--spacing', default=[1., 1., 1.], nargs='+', type=float, help="Spacing between the pixels in xyz")
parser.add_argument('--sample_size', default=[80., 80., 96.], nargs='+', type=float, help="Size of the samples")
parser.add_argument('--no_of_samples', default=10, type=int, help="Number of samples to create per volume")
parser.add_argument('--no_of_zero_samples', default=2, type=int,
                    help="Number of samples which can contain no vertebrae")
parser.add_argument("--without_label", default=False, action="store_true", help='whether dataset has labels or not')
args = parser.parse_args()

assert args.without_label or (str(args.volume_format) == ".nii.gz" and str(args.label_format) == ".lml" or str(
    args.volume_format) == ".dcm" and str(args.label_format) == ".nii.gz")

invalid_samples = []

def generate_samples(dataset_dir, sample_dir, spacing, sample_size, no_of_samples, no_of_zero_samples, volume_format,
                     label_format, with_label=True):
    # numpy these so they can be divided later on
    sample_size = np.array(sample_size)

    if volume_format == '.nii.gz':
        # only one file per volume
        paths = glob.glob(dataset_dir + "/**/*" + volume_format, recursive=True)
    else:
        # multiple files per volume
        paths = glob.glob(dataset_dir + "/**/")

    ext_len = len(volume_format)
    np.random.seed(1)

    sample_size_np = np.array(sample_size, int)
    print("Generating " + str(no_of_samples * len(paths)) + " detection samples of size " + str(sample_size_np[0]) +
          " x " + str(sample_size_np[1]) + " x " + str(sample_size_np[2]) + " for " + str(len(paths)) + " scans")

    for cnt, data_path in tqdm(enumerate(paths), total=len(paths)):

        if volume_format == '.nii.gz' and (label_format == ".lml" or not with_label):
            # get path to corresponding metadata
            data_path_without_ext = data_path[:-ext_len]
            metadata_path = data_path_without_ext + label_format

            try:
                # get image, resample it and scale centroids accordingly
                volume, *_ = opening_files.read_volume_nii_format(data_path, spacing=spacing)
                if with_label:
                    labels, centroids = opening_files.extract_centroid_info_from_lml(metadata_path)

            except Exception as e:
                print(e)
                invalid_samples.append(data_path)
                continue

            name = (data_path.rsplit('/', 1)[-1])[:-ext_len]

        elif volume_format == '.dcm' and (label_format == ".nii.gz" or not with_label):
            if with_label:
                meta_file_dir = glob.glob(data_path + "*" + label_format)
                if len(meta_file_dir) != 1:
                    raise AttributeError(f"More/Less than one annotation file for volume {data_path}")
                meta_file_dir = meta_file_dir[0]
                labels, centroids = opening_files.extract_centroid_info_from_nii(meta_file_dir, spacing=spacing)

            try:
                volume, *_ = opening_files.read_volume_dcm_series(data_path, spacing=spacing, series_prefix="mediastinum")
            except Exception as e:
                print(e)
                invalid_samples.append(data_path)
                continue

            name = data_path.rsplit('/')[-2]

        else:
            raise AttributeError("Volume-Format and Label-Format Combination not supported")

        if with_label:
            centroid_indexes = np.round(centroids / np.array(spacing)).astype(int)

            # densely populate
            # calculate the disk size (dependent on spacing) -> return dict with mapping vertebrae -> affected indexes
            disk_indices = pre_compute_disks(spacing)
            dense_labelling = densely_label(volume.shape,  # Convert 2D disks to 3D labels
                                            disk_indices,
                                            labels,
                                            centroid_indexes,
                                            use_labels=False)
            # dense_labelling = spherical_densely_label(volume.shape, 14.0, labels, centroid_indexes, use_labels=False)

        sample_size_in_pixels = (sample_size / np.array(spacing)).astype(int)  # e.g. [64 64 80]

        # crop or pad depending on what is necessary
        if volume.shape[0] < sample_size_in_pixels[0]:
            dif = sample_size_in_pixels[0] - volume.shape[0]
            volume = np.pad(volume, ((0, dif), (0, 0), (0, 0)),
                            mode="constant", constant_values=-5)
            if with_label:
                dense_labelling = np.pad(dense_labelling, ((0, dif), (0, 0), (0, 0)),
                                         mode="constant")

        if volume.shape[1] < sample_size_in_pixels[1]:
            dif = sample_size_in_pixels[1] - volume.shape[1]
            volume = np.pad(volume, ((0, 0), (0, dif), (0, 0)),
                            mode="constant", constant_values=-5)
            if with_label:
                dense_labelling = np.pad(dense_labelling, ((0, 0), (0, dif), (0, 0)),
                                         mode="constant")

        if volume.shape[2] < sample_size_in_pixels[2]:
            dif = sample_size_in_pixels[2] - volume.shape[2]
            volume = np.pad(volume, ((0, 0), (0, 0), (0, dif)),
                            mode="constant", constant_values=-5)
            if with_label:
                dense_labelling = np.pad(dense_labelling, ((0, 0), (0, 0), (0, dif)),
                                         mode="constant")

        # # TODO Plot volume for debugging
        # fig = plt.figure(figsize=(7, 7))
        # ax = plt.axes()
        # ax.imshow(volume[volume.shape[0] // 2, :, :], cmap=plt.cm.bone)
        # plt.show()

        random_area = volume.shape - sample_size_in_pixels

        i = 0
        j = 0
        while i < no_of_samples:
            if with_label or i == 0:
                # random area if we have labels or one random area if we don't
                random_factor = np.random.rand(3)
            else:
                # without labels: Use are where it is more likely to have labels
                lower, higher = 0.4, 0.6  # In this range of the axis
                random_factor = np.random.rand(3) * np.array([higher - lower, higher - lower, 1]) + np.array(
                    [lower, lower, 0])

            random_position = np.round(random_area * random_factor).astype(int)
            corner_a = random_position
            corner_b = random_position + sample_size_in_pixels

            sample = volume[corner_a[0]:corner_b[0], corner_a[1]:corner_b[1], corner_a[2]:corner_b[2]]
            if with_label:
                labelling = dense_labelling[corner_a[0]:corner_b[0], corner_a[1]:corner_b[1], corner_a[2]:corner_b[2]]

                # if a centroid is contained
                unique_labels = np.unique(labelling).shape[0]

                if unique_labels > 1 or j < no_of_zero_samples:
                    if unique_labels == 1:
                        j += 1
                    i += 1

            # if we don't have labels -> accept sample (because we don't know if the label is visible or not)
            else:
                i += 1

            # save file
            name_plus_id = name + "-" + str(i)
            path = '/'.join([sample_dir, name_plus_id])
            sample_path = path + "-sample"
            np.save(sample_path, sample)
            if with_label:
                labelling_path = path + "-labelling"
                np.save(labelling_path, labelling)


if args.training_dataset_dir is not None:
    generate_samples(dataset_dir=str(args.training_dataset_dir),
                     sample_dir=str(args.training_sample_dir),
                     spacing=tuple(args.spacing),
                     sample_size=tuple(args.sample_size),
                     no_of_samples=int(args.no_of_samples),
                     no_of_zero_samples=int(args.no_of_zero_samples),
                     volume_format=str(args.volume_format),
                     label_format=str(args.label_format),
                     with_label=not bool(args.without_label),
                     )

if args.testing_dataset_dir is not None:
    generate_samples(dataset_dir=str(args.testing_dataset_dir),
                     sample_dir=str(args.testing_sample_dir),
                     spacing=tuple(args.spacing),
                     sample_size=tuple(args.sample_size),
                     no_of_samples=int(args.no_of_samples),
                     no_of_zero_samples=int(args.no_of_zero_samples),
                     volume_format=str(args.volume_format),
                     label_format=str(args.label_format),
                     with_label=True,  # always needs labels!
                     )
