import argparse
import glob

import elasticdeform
import numpy as np
from tqdm import tqdm

from utility_functions import opening_files
from utility_functions.sampling_helper_functions import densely_label, pre_compute_disks

parser = argparse.ArgumentParser()
parser.add_argument("--training_dataset_dir", default="/workspace/data/covid-ct_harvard/training_dataset",
                    help="Path to training data (input)")
parser.add_argument("--testing_dataset_dir", default="/workspace/data/covid-ct_harvard/testing_dataset",
                    help="Path to testing data (input)")
parser.add_argument("--training_sample_dir", default="/workspace/data/covid-ct_harvard/samples/identification/training",
                    help="Output path to store training samples")
parser.add_argument("--testing_sample_dir", default="/workspace/data/covid-ct_harvard/samples/identification/testing",
                    help="Output path to store testing samples")
parser.add_argument("--volume_format", default=".dcm", help="Format of the CT-scan volume (either .nii.gz or .dcm)")
parser.add_argument("--label_format", default=".nii.gz", help="Format of the labels (either .lml or .nii.gz)")
parser.add_argument('--spacing', default=[1., 1., 1.], nargs='+', type=float, help="Spacing between the pixels in xyz")
parser.add_argument('--sample_size', default=[80, 320], nargs='+', type=int, help="Size of the samples")
parser.add_argument('--no_of_samples', default=300, type=int, help="Number of samples to create per volume")
parser.add_argument('--sample_channels', default=8, type=int, help="Number of channels (width in z-axis)")
parser.add_argument('--no_of_vertebrae_in_each', default=1, type=int, help="Min. number of vertebrae per sample")
parser.add_argument("--without_label", default=False, action="store_true", help='whether dataset has labels or not')
parser.add_argument("--with_detection", default=False, action="store_true",
                    help='whether the dataset has detection samples')
args = parser.parse_args()

assert args.without_label or (str(args.volume_format) == ".nii.gz" and str(args.label_format) == ".lml" or str(
    args.volume_format) == ".dcm" and str(args.label_format) == ".nii.gz")


def generate_slice_samples(dataset_dir, sample_dir, sample_size, spacing, no_of_samples, no_of_vertebrae_in_each,
                           sample_channels, volume_format, label_format, with_label=True, with_detection=False):
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
    print("Generating " + str(no_of_samples * len(paths)) + " identification samples of size " + str(
        sample_channels) + " x "
          + str(sample_size_np[0]) + " x " + str(sample_size_np[1]) + " for " + str(len(paths)) + " scans")

    for cnt, data_path in tqdm(enumerate(paths), total=len(paths)):

        if volume_format == '.nii.gz' and (label_format == ".lml" or not with_label):
            # get path to corresponding metadata
            data_path_without_ext = data_path[:-ext_len]
            metadata_path = data_path_without_ext + ".lml"

            volume, *_ = opening_files.read_volume_nii_format(data_path, spacing=spacing)
            if with_label:
                labels, centroids = opening_files.extract_centroid_info_from_lml(metadata_path)
            name = (data_path.rsplit('/', 1)[-1])[:-ext_len]

        elif volume_format == '.dcm' and (label_format == ".nii.gz" or not with_label):
            if with_label:
                meta_file_dir = glob.glob(data_path + "*" + ".nii.gz")
                if len(meta_file_dir) != 1:
                    raise AttributeError(f"More/Less than one annotation file for volume {data_path}")
                meta_file_dir = meta_file_dir[0]
                labels, centroids = opening_files.extract_centroid_info_from_nii(meta_file_dir, spacing=spacing)

            volume, *_ = opening_files.read_volume_dcm_series(data_path, spacing=spacing, series_prefix="mediastinum")
            name = data_path.rsplit('/')[-2]

        else:
            raise AttributeError("Volume-Format and Label-Format Combination not supported")

        if with_detection:
            detection = np.load(data_path + '/detection.npy')

        if with_label:
            centroid_indexes = np.round(centroids / np.array(spacing)).astype(int)

            disk_indices = pre_compute_disks(spacing)
            dense_labelling = densely_label(volume.shape, disk_indices, labels, centroid_indexes, use_labels=True)
            # dense_labelling = spherical_densely_label(volume.shape, 14.0, labels, centroid_indexes, use_labels=True)

            # dense_labelling_squashed = np.any(dense_labelling, axis=(1, 2))
            # lower_i = np.min(np.where(dense_labelling_squashed == 1))
            # upper_i = np.max(np.where(dense_labelling_squashed == 1))
            lower_i = np.min(centroid_indexes[:, 0])
            lower_i = np.max([lower_i - 15, 0]).astype(int)
            upper_i = np.max(centroid_indexes[:, 0])
            upper_i = np.min([upper_i + 15, volume.shape[0] - 1]).astype(int)

            #import matplotlib.pyplot as plt
            #fig, axes = plt.subplots(nrows=2, figsize=(20, 15), dpi=300)
            #axes[0].imshow(volume[volume.shape[0] // 2, :, :], cmap='bone')
            #axes[0].imshow(dense_labelling[dense_labelling.shape[0] // 2, :, :], cmap='gist_ncar', alpha=0.3)
            #axes[1].imshow(volume[:, volume.shape[0] // 2, :], cmap='bone')
            #axes[1].imshow(dense_labelling[:, dense_labelling.shape[1] // 2, :], cmap='gist_ncar', alpha=0.3)
            #plt.tight_layout()
            #plt.show()

            cuts = []
            while len(cuts) < no_of_samples:
                # cut = np.random.randint(lower_i + 4, high=upper_i - 4)
                cut = np.random.randint(lower_i, high=upper_i)
                sample_labels_slice = dense_labelling[cut - 4: cut + 4, :, :]
                # sample_labels_slice = dense_labelling[cut, :, :]
                # (not sure) Ignore slice if at border of vertebrae (at least one label must exist in both directions for +4, -4 frames
                if np.unique(sample_labels_slice).shape[0] > no_of_vertebrae_in_each:
                    cuts.append(cut)

        elif with_detection:
            lower_i = np.min(np.where(detection > 0)[0])
            upper_i = np.max(np.where(detection > 0)[0])

            cuts = []
            while len(cuts) < no_of_samples:
                cut = np.random.randint(lower_i, high=upper_i)
                cuts.append(cut)

        else:
            mu, sigma = round(volume.shape[0] / 2), 10
            # TODO: without labels -> just hope we are lucky and they are in the middle :S
            cuts = list(np.random.normal(mu, sigma, no_of_samples).astype(int))

        count = 0
        for i in cuts:

            volume_slice = volume[i - 4:i + 4, :, :]
            # volume_slice = volume[i, :, :]

            if with_detection:
                detection_slice = detection[i, :, :]

            if with_label:
                sample_labels_slice = dense_labelling[i, :, :]

            if volume_slice.shape[0] != sample_channels:
                print("Number of sample channels is incorrect!")
                break

            # get vertebrae identification map
            # detection_slice = (sample_labels_slice > 0).astype(int)

            '''
            [volume_slice, sample_labels_slice] = elasticdeform.deform_random_grid(
                [volume_slice, sample_labels_slice], sigma=7, points=3, order=0)
            '''

            if with_label:
                if with_detection:
                    [volume_slice, sample_labels_slice, detection_slice] = elasticdeform.deform_random_grid(
                        [volume_slice, np.expand_dims(sample_labels_slice, axis=0),
                         np.expand_dims(detection_slice, axis=0)], sigma=7, points=3, order=0, axis=(1, 2))
                else:
                    [volume_slice, sample_labels_slice] = elasticdeform.deform_random_grid(
                        [volume_slice, np.expand_dims(sample_labels_slice, axis=0)], sigma=7, points=3, order=0,
                        axis=(1, 2))

                sample_labels_slice = np.squeeze(sample_labels_slice, axis=0)

            else:
                if with_detection:
                    [volume_slice, detection_slice] = elasticdeform.deform_random_grid(
                        [volume_slice, np.expand_dims(detection_slice, axis=0)], sigma=7, points=3, order=0,
                        axis=(1, 2))
                else:
                    volume_slice = elasticdeform.deform_random_grid(volume_slice, sigma=7, points=3, order=0,
                                                                    axis=(1, 2))

            if with_detection:
                detection_slice = np.squeeze(detection_slice, axis=0)

            # crop or pad depending on what is necessary
            if volume_slice.shape[1] < sample_size[0]:
                dif = sample_size[0] - volume_slice.shape[1]
                volume_slice = np.pad(volume_slice, ((0, 0), (0, dif), (0, 0)),
                                      mode="constant", constant_values=-5)

                if with_label:
                    sample_labels_slice = np.pad(sample_labels_slice, ((0, dif), (0, 0)), mode="constant")

                if with_detection:
                    detection_slice = np.pad(detection_slice, ((0, dif), (0, 0)), mode="constant")

            if volume_slice.shape[2] < sample_size[1]:
                dif = sample_size[1] - volume_slice.shape[2]
                volume_slice = np.pad(volume_slice, ((0, 0), (0, 0), (0, dif)),
                                      mode="constant", constant_values=-5)
                # detection_slice = np.pad(detection_slice, ((0, 0), (0, dif)),
                #                         mode="constant")
                if with_label:
                    sample_labels_slice = np.pad(sample_labels_slice, ((0, 0), (0, dif)), mode="constant")

                if with_detection:
                    detection_slice = np.pad(detection_slice, ((0, 0), (0, dif)), mode="constant")

            '''
            if volume_slice.shape[0] < sample_size[0]:
                dif = sample_size[0] - volume_slice.shape[0]
                volume_slice = np.pad(volume_slice, ((0, dif), (0, 0)),
                                      mode="constant", constant_values=0)
                # detection_slice = np.pad(detection_slice, ((0, dif), (0, 0)),
                #                         mode="constant")
                sample_labels_slice = np.pad(sample_labels_slice, ((0, dif), (0, 0)),
                                             mode="constant")

            if volume_slice.shape[1] < sample_size[1]:
                dif = sample_size[1] - volume_slice.shape[1]
                volume_slice = np.pad(volume_slice, ((0, 0), (0, dif)),
                                      mode="constant", constant_values=0)
                # detection_slice = np.pad(detection_slice, ((0, 0), (0, dif)),
                #                         mode="constant")
                sample_labels_slice = np.pad(sample_labels_slice, ((0, 0), (0, dif)),
                                             mode="constant")
            '''

            # volume_slice = np.expand_dims(volume_slice, axis=2)
            # detection_slice = np.expand_dims(detection_slice, axis=2)
            # combines_slice = np.concatenate((volume_slice, detection_slice), axis=2)
            j = 0
            while True:
                random_area = volume_slice.shape[1:3] - sample_size
                # random_area = volume_slice.shape - sample_size
                random_factor = np.random.rand(2)
                random_position = np.round(random_area * random_factor).astype(int)
                corner_a = random_position
                corner_b = corner_a + sample_size

                cropped_combines_slice = volume_slice[:, corner_a[0]:corner_b[0], corner_a[1]:corner_b[1]]

                if with_label:
                    # cropped_combines_slice = volume_slice[corner_a[0]:corner_b[0], corner_a[1]:corner_b[1]]
                    cropped_sample_labels_slice = sample_labels_slice[corner_a[0]:corner_b[0], corner_a[1]:corner_b[1]]
                    if with_detection:
                        cropped_sample_detection_slice = detection_slice[corner_a[0]:corner_b[0],
                                                         corner_a[1]:corner_b[1]]

                    care_about_labels = np.count_nonzero(cropped_sample_labels_slice)
                    j += 1
                    if care_about_labels > 500 or j > 100:
                        break

                elif with_detection:
                    cropped_sample_detection_slice = detection_slice[corner_a[0]:corner_b[0], corner_a[1]:corner_b[1]]

                    care_about_labels = np.count_nonzero(cropped_sample_detection_slice)
                    j += 1
                    if care_about_labels > 500 or j > 100:
                        break

                else:
                    # If without label -> just accept it without checking it :S
                    break

            # save file
            count += 1
            name_plus_id = name + "-" + str(count)
            path = '/'.join([sample_dir, name_plus_id])
            sample_path = path + "-sample"
            np.save(sample_path, cropped_combines_slice)  # 8x80x320

            if with_label:
                labelling_path = path + "-labelling"
                np.save(labelling_path, cropped_sample_labels_slice)  # 80x320

            if with_detection:
                detection_path = path + "-detection"
                np.save(detection_path, cropped_sample_detection_slice)  # 80x320


generate_slice_samples(dataset_dir=str(args.training_dataset_dir),
                       sample_dir=str(args.training_sample_dir),
                       sample_size=tuple(args.sample_size),
                       no_of_samples=int(args.no_of_samples),
                       spacing=tuple(args.spacing),
                       sample_channels=int(args.sample_channels),
                       no_of_vertebrae_in_each=int(args.no_of_vertebrae_in_each),
                       volume_format=str(args.volume_format),
                       label_format=str(args.label_format),
                       with_label=not bool(args.without_label),
                       with_detection=bool(args.with_detection),
                       )

generate_slice_samples(dataset_dir=str(args.testing_dataset_dir),
                       sample_dir=str(args.testing_sample_dir),
                       sample_size=tuple(args.sample_size),
                       no_of_samples=int(args.no_of_samples),
                       spacing=tuple(args.spacing),
                       sample_channels=int(args.sample_channels),
                       no_of_vertebrae_in_each=int(args.no_of_vertebrae_in_each),
                       volume_format=str(args.volume_format),
                       label_format=str(args.label_format),
                       with_label=not bool(args.without_label),
                       with_detection=bool(args.with_detection),
                       )
