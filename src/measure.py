# The aim of this script is to provide measurements for any part of the pipeline
import argparse
import glob
import pandas as pd
from collections import OrderedDict

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage.measurements import label

from metrics import Accuracy as AccuracyCustom
from metrics import DetectionMetricWrapper, Fscore, IoU, AverageValueMeter, Dice
from metrics import Recall as RecallCustom
from models.model_util import get_models
from utility_functions import opening_files
from utility_functions.labels import LABELS_NO_L6, VERTEBRAE_SIZES
from utility_functions.sampling_helper_functions import pre_compute_disks, densely_label
from utility_functions.writing_files import create_lml_file

parser = argparse.ArgumentParser()
parser.add_argument("--plot_path", default="plots_debug", help="Where to save the created plots")
parser.add_argument('--spacing', default=[1., 1., 1.], nargs='+', type=float, help="Spacing between the pixels in xyz")
parser.add_argument('--detection_input_size', default=[64, 64, 80], nargs='+', type=int,
                    help="Input size detection model")
parser.add_argument('--detection_input_shift', default=[32, 32, 40], nargs='+', type=int,
                    help="Shift for overlapping detections")
parser.add_argument("--n_plots", default=-1, type=int, help="Number of sample plots to create")
parser.add_argument("--testing_dataset_dir", default="../testing_dataset", help="Path to testing data (input)")
parser.add_argument("--volume_format", default=".dcm", help="Format of the CT-scan volume (either .nii.gz or .dcm)")
parser.add_argument("--label_format", default=".nii.gz", help="Format of the labels (either .lml, .nii, .json)")
parser.add_argument("--resume_detection", type=str, default=None, metavar="PTH.TAR", help="model(pth) path")
parser.add_argument("--resume_identification", type=str, default=None, metavar="PTH.TAR", help="model(pth) path")
parser.add_argument('--without_label', action="store_true", help="Whether to use Labels")
parser.add_argument('--save_detections', action="store_true", help="Whether to save the detection as numpy file")
parser.add_argument('--save_predictions', action="store_true", help="Whether to save the predicted centroids as .lml file")
parser.add_argument("--is_data_parallel", action="store_true", help='whether you use torch.nn.DataParallel')
parser.add_argument("--ignore_small_masks_detection", action="store_true",
                    help='whether to ignore small masks (mostly error masks)')
args = parser.parse_args()

assert str(args.volume_format) == ".nii.gz" and (str(args.label_format) == ".lml" or str(args.label_format) == ".json")\
       or str(args.volume_format) == ".dcm" and str(args.label_format) == ".nii.gz"


def load_checkpoint(path, use_parallel):
    prefix = "module."
    checkpoint = torch.load(path, map_location=torch.device('cpu') )

    if use_parallel is False and list(checkpoint['g_state_dict'].keys())[0].startswith(prefix):
        checkpoint['g_state_dict'] = OrderedDict(
            [(k[len(prefix):], v) if k.startswith(prefix) else (k, v) for k, v in checkpoint['g_state_dict'].items()])
        checkpoint['f1_state_dict'] = OrderedDict(
            [(k[len(prefix):], v) if k.startswith(prefix) else (k, v) for k, v in checkpoint['f1_state_dict'].items()])

    return checkpoint


class DetectionModelWrapper:

    def __init__(self, mode, n_class):
        self.model, self.model_head = get_models(mode=mode,
                                                 n_class=n_class,
                                                 is_data_parallel=args.is_data_parallel)

        self.checkpoint = load_checkpoint(args.resume_detection, args.is_data_parallel)
        self.model.load_state_dict(self.checkpoint['g_state_dict'])
        self.model_head.load_state_dict(self.checkpoint['f1_state_dict'])

        if torch.cuda.is_available():
            self.model.cuda()
            self.model_head.cuda()

    def predict(self, batch):
        with torch.no_grad():
            batch_t = torch.as_tensor(batch)
            if torch.cuda.is_available():
                batch_t = batch_t.cuda()
            batch_t = batch_t.permute(0, 4, 1, 2, 3)

            outputs = self.model(batch_t)
            outputs1 = self.model_head(outputs)

            result = F.softmax(outputs1, dim=1)
            result = result.permute(0, 2, 3, 4, 1)

            return result.cpu().numpy()


class IdentificationModelWrapper:

    def __init__(self, mode, n_class):
        self.model, self.model_head = get_models(mode=mode,
                                                 n_class=n_class,
                                                 is_data_parallel=args.is_data_parallel)

        self.checkpoint = load_checkpoint(args.resume_identification, args.is_data_parallel)
        self.model.load_state_dict(self.checkpoint['g_state_dict'])
        self.model_head.load_state_dict(self.checkpoint['f1_state_dict'])

        if torch.cuda.is_available():
            self.model.cuda()
            self.model_head.cuda()

    def predict(self, batch):
        with torch.no_grad():
            batch_t = torch.as_tensor(batch)
            if torch.cuda.is_available():
                batch_t = batch_t.cuda()

            batch_t = batch_t.permute(0, 3, 1, 2)

            outputs = self.model(batch_t)
            result = self.model_head(outputs)
            result = result.permute(0, 2, 3, 1)

            return result.cpu().numpy()


def load_spine_model(mtype):
    if mtype == "detection":
        return DetectionModelWrapper(
            mode=mtype,
            n_class=2,
        )
    elif mtype == "identification":
        return IdentificationModelWrapper(
            mode=mtype,
            n_class=1,
        )

    else:
        raise AttributeError("Unknown model type " + mtype)


def apply_detection_model(volume, model, X_size, y_size, ignore_small_masks_detection, img_name=None):
    # E.g if X_size = 30 x 30 x 30 and y_size is 20 x 20 x 20
    #  Then cropping is ((5, 5), (5, 5), (5, 5)) pad the whole thing by cropping
    # Then pad an additional amount to make it divisible by Y_size + cropping
    # Then iterate through in y_size + cropping steps
    # Then uncrop at the end

    border = ((X_size - y_size) / 2.0).astype(int)
    border_paddings = np.array(list(zip(border, border))).astype(int)
    volume_padded = np.pad(volume, border_paddings, mode="constant")

    # pad to make it divisible to patch size
    divisible_area = volume_padded.shape - X_size
    paddings = np.mod(y_size - np.mod(divisible_area.shape, y_size), y_size)
    paddings = np.array(list(zip(np.zeros(3), paddings))).astype(int)
    volume_padded = np.pad(volume_padded, paddings, mode="constant")

    output = np.zeros(volume_padded.shape)

    print(X_size, y_size, volume.shape, output.shape)
    for x in range(0, volume_padded.shape[0] - X_size[0] + 1, y_size[0]):
        for y in range(0, volume_padded.shape[1] - X_size[1] + 1, y_size[1]):
            for z in range(0, volume_padded.shape[2] - X_size[2] + 1, y_size[2]):
                corner_a = [x, y, z]
                corner_b = corner_a + X_size
                corner_c = corner_a + border
                corner_d = corner_c + y_size
                patch = volume_padded[corner_a[0]:corner_b[0], corner_a[1]:corner_b[1], corner_a[2]:corner_b[2]]
                patch = patch.reshape(1, *X_size, 1)
                result = model.predict(patch)  # patch: [1, 64, 64, 80, 1]    result: [1, 64, 64, 80, 2]
                result = np.squeeze(result, axis=0)
                decat_result = np.argmax(result, axis=3)
                cropped_decat_result = decat_result[border[0]:-border[0], border[1]:-border[1], border[2]:-border[2]]
                output[corner_c[0]:corner_d[0], corner_c[1]:corner_d[1], corner_c[2]:corner_d[2]] = cropped_decat_result
                # output[corner_c[0]:corner_d[0], corner_c[1]:corner_d[1], corner_c[2]:corner_d[2]] = decat_result
                # print(x, y, z, np.bincount(decat_result.reshape(-1).astype(int)))

    output = output[border[0]:border[0] + volume.shape[0],
             border[1]:border[1] + volume.shape[1],
             border[2]:border[2] + volume.shape[2]]

    if ignore_small_masks_detection:
        # only keep the biggest connected component
        structure = np.ones((3, 3, 3), dtype=int)
        labeled, ncomponents = label(output, structure)
        unique, counts = np.unique(labeled, return_counts=True)
        output_without_small_masks = np.zeros(labeled.shape)
        output_without_small_masks[labeled == unique[np.argsort(counts)[-2]]] = 1

        if img_name is not None:
            flatten_output = np.sum(output, axis=0)
            flatten_output[flatten_output > 1] = 1
            flatten = np.sum(output_without_small_masks, axis=0)
            flatten[flatten > 1] = 1

            fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 20), dpi=300)

            axes[0, 0].imshow(volume[volume.shape[0] // 2, :, :], cmap='bone')
            axes[0, 0].imshow(output[output.shape[0] // 2, :, :], cmap=cm.winter, alpha=0.3)
            axes[0, 0].set_title("Center Slice")
            axes[0, 1].imshow(volume[volume.shape[0] // 2, :, :], cmap='bone')
            axes[0, 1].imshow(output_without_small_masks[output_without_small_masks.shape[0] // 2, :, :],
                              cmap=cm.winter,
                              alpha=0.3)
            axes[0, 1].set_title("Center Slice")

            axes[1, 0].imshow(volume[volume.shape[0] // 2, :, :], cmap='bone')
            axes[1, 0].imshow(flatten_output, cmap=cm.winter, alpha=0.3)
            axes[1, 0].set_title("All Slices")
            axes[1, 1].imshow(volume[volume.shape[0] // 2, :, :], cmap='bone')
            axes[1, 1].imshow(flatten, cmap=cm.winter, alpha=0.3)
            axes[1, 1].set_title("All Slices")

            flatten_output = np.sum(output, axis=1)
            flatten_output[flatten_output > 1] = 1
            flatten = np.sum(output_without_small_masks, axis=1)
            flatten[flatten > 1] = 1

            axes[2, 0].imshow(np.rot90(volume[:, volume.shape[0] // 2, :]), cmap='bone')
            axes[2, 0].imshow(np.rot90(flatten_output), cmap=cm.winter, alpha=0.3)
            axes[2, 0].set_title("All Slices")
            axes[2, 1].imshow(np.rot90(volume[:, volume.shape[0] // 2, :]), cmap='bone')
            axes[2, 1].imshow(np.rot90(flatten), cmap=cm.winter, alpha=0.3)
            axes[2, 1].set_title("All Slices")

            fig.tight_layout()
            fig.savefig(img_name)
            plt.close(fig)
            plt.close()
            # fig.show()

        output = output_without_small_masks

    return output


def apply_identification_model(volume, i_min, i_max, model):
    paddings = np.mod(16 - np.mod(volume.shape[1:3], 16), 16)
    paddings = np.array(list(zip(np.zeros(3), [0] + list(paddings)))).astype(int)
    volume_padded = np.pad(volume, paddings, mode="constant")
    output = np.zeros(volume_padded.shape)
    i_min = max(i_min, 4)
    i_max = min(i_max, volume_padded.shape[0] - 4)

    for i in range(i_min, i_max, 1):
        volume_slice_padded = volume_padded[i - 4:i + 4, :, :]
        volume_slice_padded = np.transpose(volume_slice_padded, (1, 2, 0))
        patch = volume_slice_padded.reshape(1, *volume_slice_padded.shape)
        result = model.predict(patch)
        result = np.squeeze(result, axis=0)
        result = np.squeeze(result, axis=-1)
        result = np.round(result)
        output[i, :, :] = result

    output = output[:volume.shape[0], :volume.shape[1], :volume.shape[2]]
    return output


def test_scan(detection_model, detection_X_shape, detection_y_shape,
              identification_model, volume, ignore_small_masks_detection, img_name=None):
    # first stage is to put the volume through the detection model to find where vertebrae are
    print("apply detection")
    detections = apply_detection_model(volume, detection_model, detection_X_shape, detection_y_shape,
                                       ignore_small_masks_detection, img_name=img_name)
    print("finished detection")

    # get the largest island
    largest_island_np = np.transpose(np.nonzero(detections))
    i_min = np.min(largest_island_np[:, 0])
    i_max = np.max(largest_island_np[:, 0])

    # second stage is to pass slices of this to the identification network
    print("apply identification")
    identifications = apply_identification_model(volume, i_min, i_max, identification_model)
    print("finished identification")

    # crop parts of slices
    identifications_croped = identifications * detections
    print("finished multiplying")

    fig, axes = plt.subplots(ncols=5, nrows=5, figsize=(20, 15), dpi=300)
    slice_idx = identifications_croped.shape[0] // 2 - 12
    v_min = 0
    v_max = np.max(identifications_croped)
    for i in range(5):
        for j in range(5):
            axes[i, j].imshow(volume[slice_idx, :, :], cmap='bone')
            mask = axes[i, j].imshow(identifications_croped[slice_idx, :, :], cmap='gist_ncar', alpha=0.4, vmin=v_min,
                                     vmax=v_max)
            axes[i, j].set_title(f"Sagital  {slice_idx}")
            slice_idx += 1
            fig.colorbar(mask, ax=axes[i, j])

    plt.tight_layout()
    if img_name is not None:
        fig.savefig(img_name.split(".png")[0] + "_sagital-slices.png")
        plt.close(fig)
        plt.close()

    def fix_sagital_slices(array):
        array = array.copy()
        column_medians = np.ma.median(np.ma.masked_where(array == 0, array), axis=0).filled(0)
        medians = np.repeat(column_medians[np.newaxis, :], array.shape[0], axis=0)
        non_zero_mask = array != 0
        array[non_zero_mask] = medians[non_zero_mask]
        return array

    def fix_sagital_slices_3d(array):
        array = array.copy()
        column_medians = np.ma.median(np.ma.masked_where(array == 0, array), axis=1).filled(0)
        medians = np.repeat(column_medians[:, np.newaxis, :], array.shape[1], axis=1)
        non_zero_mask = array != 0
        array[non_zero_mask] = medians[non_zero_mask]
        return array

    identifications_croped_fixed = fix_sagital_slices_3d(identifications_croped)

    fig, axes = plt.subplots(ncols=5, nrows=5, figsize=(20, 15), dpi=300)
    slice_idx = identifications_croped.shape[0] // 2 - 12
    v_min = 0
    v_max = np.max(identifications_croped)
    for i in range(5):
        for j in range(5):
            fixed_slice = fix_sagital_slices(identifications_croped[slice_idx, :, :])
            assert np.all(fixed_slice == identifications_croped_fixed[slice_idx, :, :])

            axes[i, j].imshow(volume[slice_idx, :, :], cmap='bone')
            mask = axes[i, j].imshow(fixed_slice, cmap='gist_ncar', alpha=0.4, vmin=v_min, vmax=v_max)
            axes[i, j].set_title(f"Sagital  {slice_idx}")
            slice_idx += 1
            fig.colorbar(mask, ax=axes[i, j])

    plt.tight_layout()
    if img_name is not None:
        fig.savefig(img_name.split(".png")[0] + "_sagital-slices-fixed.png")
        plt.close(fig)
        plt.close()

    fig, axes = plt.subplots(ncols=3, nrows=4, figsize=(20, 15), dpi=300)

    def plot_median(array, axis):
        arr_masked = np.ma.masked_where(array == 0, array)
        return np.ma.median(arr_masked, axis=axis).filled(0).T

    def plot_center_slice(array, axis):
        if axis == 0:
            return array[array.shape[0] // 2, :, :].T
        elif axis == 1:
            return array[:, array.shape[1] // 2, :].T
        elif axis == 2:
            return array[:, :, array.shape[2] // 2].T

    axes[0, 0].imshow(plot_center_slice(volume, axis=0), cmap='bone')
    mask_00 = axes[0, 0].imshow(detections[detections.shape[0] // 2, :, :].T, cmap='gist_ncar', alpha=0.4)
    axes[0, 0].set_title("Sagital Center Slice (detections)")

    axes[0, 1].imshow(plot_center_slice(volume, axis=0), cmap='bone')
    mask_01 = axes[0, 1].imshow(plot_median(detections, 0), cmap='gist_ncar', alpha=0.3)
    axes[0, 1].set_title("Sagital Median (detections)")

    axes[0, 2].imshow(plot_center_slice(volume, axis=1), cmap='bone')
    mask_02 = axes[0, 2].imshow(plot_median(detections, 1), cmap='gist_ncar', alpha=0.3)
    axes[0, 2].set_title("Coronal Median (detections)")

    axes[1, 0].imshow(plot_center_slice(volume, axis=0), cmap='bone')
    mask_10 = axes[1, 0].imshow(identifications[identifications.shape[0] // 2, :, :].T, cmap='gist_ncar', alpha=0.3)
    axes[1, 0].set_title("Sagital Center Slice (identifications)")

    axes[1, 1].imshow(plot_center_slice(volume, axis=0), cmap='bone')
    mask_11 = axes[1, 1].imshow(plot_median(identifications, 0), cmap='gist_ncar', alpha=0.3)
    axes[1, 1].set_title("Sagital Median (identifications)")

    axes[1, 2].imshow(plot_center_slice(volume, axis=1), cmap='bone')
    mask_12 = axes[1, 2].imshow(plot_median(identifications, 1), cmap='gist_ncar', alpha=0.3)
    axes[1, 2].set_title("Coronal Median (identifications)")

    axes[2, 0].imshow(plot_center_slice(volume, axis=0), cmap='bone')
    mask_20 = axes[2, 0].imshow(identifications_croped[identifications_croped.shape[0] // 2, :, :].T, cmap='gist_ncar',
                                alpha=0.3)
    axes[2, 0].set_title("Sagital Center Slice (identifications * detections)")

    axes[2, 1].imshow(plot_center_slice(volume, axis=0), cmap='bone')
    mask_21 = axes[2, 1].imshow(plot_median(identifications_croped, 0), cmap='gist_ncar', alpha=0.3)
    axes[2, 1].set_title("Sagital Median (identifications * detections)")

    axes[2, 2].imshow(plot_center_slice(volume, axis=1), cmap='bone')
    mask_22 = axes[2, 2].imshow(plot_median(identifications_croped, 1), cmap='gist_ncar', alpha=0.3)
    axes[2, 2].set_title("Coronal Median (identifications * detections)")

    axes[3, 0].imshow(plot_center_slice(volume, axis=0), cmap='bone')
    mask_30 = axes[3, 0].imshow(identifications_croped_fixed[identifications_croped_fixed.shape[0] // 2, :, :].T,
                                cmap='gist_ncar', alpha=0.3)
    axes[3, 0].set_title("Sagital Center Slice (fixedd)")

    axes[3, 1].imshow(plot_center_slice(volume, axis=0), cmap='bone')
    mask_31 = axes[3, 1].imshow(plot_median(identifications_croped_fixed, 0), cmap='gist_ncar', alpha=0.3)
    axes[3, 1].set_title("Sagital Median (fixed)")

    axes[3, 2].imshow(plot_center_slice(volume, axis=1), cmap='bone')
    mask_32 = axes[3, 2].imshow(plot_median(identifications_croped_fixed, 1), cmap='gist_ncar', alpha=0.3)
    axes[3, 2].set_title("Coronal Median (fixed)")

    fig.colorbar(mask_00, ax=axes[0, 0])
    fig.colorbar(mask_01, ax=axes[0, 1])
    fig.colorbar(mask_02, ax=axes[0, 2])
    fig.colorbar(mask_10, ax=axes[1, 0])
    fig.colorbar(mask_11, ax=axes[1, 1])
    fig.colorbar(mask_12, ax=axes[1, 2])
    fig.colorbar(mask_20, ax=axes[2, 0])
    fig.colorbar(mask_21, ax=axes[2, 1])
    fig.colorbar(mask_22, ax=axes[2, 2])
    fig.colorbar(mask_30, ax=axes[3, 0])
    fig.colorbar(mask_31, ax=axes[3, 1])
    fig.colorbar(mask_32, ax=axes[3, 2])
    fig.tight_layout()
    # fig.show()
    if img_name is not None:
        fig.savefig(img_name)
        plt.close(fig)
        plt.close()

    # aggregate the predictions
    print("start aggregating")
    identifications_croped = np.round(identifications_croped).astype(int)
    histogram = {}
    for key in range(1, len(LABELS_NO_L6)):
        histogram[key] = np.argwhere(identifications_croped == key)
    '''
    for i in range(identifications_croped.shape[0]):
        for j in range(identifications_croped.shape[1]):
            for k in range(identifications_croped.shape[2]):
                key = identifications_croped[i, j, k]
                if key != 0:
                    if key in histogram:
                        histogram[key] = histogram[key] + [[i, j, k]]
                    else:
                        histogram[key] = [[i, j, k]]
    '''
    print("finish aggregating")

    print("start averages")
    # find averages
    labels = []
    centroid_estimates = []
    for key in sorted(histogram.keys()):
        if 0 <= key < len(LABELS_NO_L6):
            arr = histogram[key]
            # print(LABELS_NO_L6[key], arr.shape[0])
            if arr.shape[0] > max(VERTEBRAE_SIZES[LABELS_NO_L6[key]] ** 3 * 0.4, 3000):
                print(LABELS_NO_L6[key], arr.shape[0])
                centroid_estimate = np.median(arr, axis=0)
                # ms = MeanShift(bin_seeding=True, min_bin_freq=300)
                # ms.fit(arr)
                # centroid_estimate = ms.cluster_centers_[0]
                centroid_estimate = np.around(centroid_estimate, decimals=2)
                labels.append(LABELS_NO_L6[key])
                centroid_estimates.append(list(centroid_estimate))
    print("finish averages")

    return labels, centroid_estimates, detections, identifications_croped


def complete_detection_picture(dataset_dir, plot_path, start, end, volume_format,
                               label_format, ignore_small_masks_detection, detection_input_size, detection_input_shift,
                               spacing=(1.0, 1.0, 1.0), with_label=True, with_metrics=True, save_detections=False):
    if volume_format == '.nii.gz':
        # only one file per volume
        scan_paths = glob.glob(dataset_dir + "/**/*" + volume_format, recursive=True)
    else:
        # multiple files per volume
        scan_paths = glob.glob(dataset_dir + "/**/")

    if end != -1:
        scan_paths = scan_paths[start:end]

    no_of_scan_paths = len(scan_paths)
    i = 1

    if with_metrics:
        n_class = 2
        metrics = {"IoU": DetectionMetricWrapper(IoU(), n_class),
                   "IoU ignored 0": DetectionMetricWrapper(IoU(ignore_channels=[0]), n_class),
                   "IoU ignored 1": DetectionMetricWrapper(IoU(ignore_channels=[1]), n_class),
                   "Recall": DetectionMetricWrapper(RecallCustom(), n_class),
                   "Recall ignored 0": DetectionMetricWrapper(RecallCustom(ignore_channels=[0]), n_class),
                   "Recall ignored 1": DetectionMetricWrapper(RecallCustom(ignore_channels=[1]), n_class),
                   "F-Score": DetectionMetricWrapper(Fscore(), n_class),
                   "F-Score ignored 0": DetectionMetricWrapper(Fscore(ignore_channels=[0]), n_class),
                   "F-Score ignored 1": DetectionMetricWrapper(Fscore(ignore_channels=[1]), n_class),
                   "Accuracy": DetectionMetricWrapper(AccuracyCustom(), n_class),
                   "Accuracy ignored 0": DetectionMetricWrapper(AccuracyCustom(ignore_channels=[0]), n_class),
                   "Accuracy ignored 1": DetectionMetricWrapper(AccuracyCustom(ignore_channels=[1]), n_class),
                   "Dice": DetectionMetricWrapper(Dice(), n_class),
                   "Dice ignored 0": DetectionMetricWrapper(Dice(ignore_channels=[0]), n_class),
                   "Dice ignored 1": DetectionMetricWrapper(Dice(ignore_channels=[1]), n_class),
                   }
        metric_meters = {k: AverageValueMeter() for k in metrics.keys()}

    detection_model = load_spine_model("detection")

    for col, scan_path in enumerate(scan_paths):

        if "msk" in scan_path:
            # a mask of the verse data set
            continue
        
        print("Processing", scan_path)

        if volume_format == '.nii.gz':
            if with_label:
                scan_path_without_ext = scan_path[:-len(volume_format)]
                centroid_path = scan_path_without_ext + label_format
                if label_format == ".lml":
                    labels, centroids = opening_files.extract_centroid_info_from_lml(centroid_path)
                else:
                    labels, centroids = opening_files.extract_centroid_info_from_json(centroid_path)
            scan_name = (scan_path.rsplit('/', 1)[-1])[:-len(volume_format)]

        elif volume_format == '.dcm' and label_format == ".nii.gz":
            if with_label:
                meta_file_dir = glob.glob(scan_path + "*" + label_format)
                if len(meta_file_dir) != 1:
                    raise AttributeError(f"More/Less than one annotation file for volume {scan_path}")
                meta_file_dir = meta_file_dir[0]
                labels, centroids = opening_files.extract_centroid_info_from_nii(meta_file_dir, spacing=spacing)
            scan_name = scan_path.rsplit('/')[-2]

        else:
            raise AttributeError("Volume-Format and Label-Format Combination not supported")

        fig, axes = plt.subplots(figsize=(20, 10), dpi=300)
        axes.set_title(scan_name, fontsize=10, pad=10)
        print(i)

        if volume_format == '.nii.gz':
            volume, *_ = opening_files.read_volume_nii_format(scan_path, spacing=spacing)
        elif volume_format == '.dcm':
            result = opening_files.read_volume_dcm_series(scan_path, spacing=spacing)
            if result is None:
                volume, *_ = opening_files.read_volume_dcm_series(scan_path, spacing=spacing,
                                                                  series_prefix='mediastinum')
            else:
                volume, *_ = result

        img_name = plot_path + f"/effect_postprocessing_{i}.png"
        detections = apply_detection_model(volume, detection_model, detection_input_size,
                                           detection_input_shift, ignore_small_masks_detection, img_name)

        # save detections as weak binary label
        if save_detections:
            np.save(scan_path + "detection", detections)


        if with_label:
            centroid_indexes = centroids / np.array(spacing)
            cut = np.round(np.mean(centroid_indexes[:, 0])).astype(int)
        else:
            # just take the center slice if we don't know where the vertebrae are
            cut = volume.shape[0] // 2

        if with_metrics and with_label:
            disk_indices = pre_compute_disks(spacing)
            dense_labelling = densely_label(volume.shape,  # Convert 2D disks to 3D labels
                                            disk_indices,
                                            labels,
                                            centroid_indexes,
                                            use_labels=False)

            fig, ax = plt.subplots(ncols=3, figsize=(8, 8))
            ax[0].imshow(volume[volume.shape[0] // 2, :, :])
            ax[1].imshow(volume[volume.shape[0] // 2, :, :])
            ax[1].imshow(detections[detections.shape[0] // 2, :, :], cmap='jet', alpha=0.5)
            ax[2].imshow(volume[volume.shape[0] // 2, :, :])
            ax[2].imshow(dense_labelling[detections.shape[0] // 2, :, :], cmap='jet', alpha=0.5)
            plt.tight_layout()
            plt.show()

            dense_labelling_t = torch.from_numpy((dense_labelling)).long().unsqueeze(0)
            detections_t = torch.from_numpy((detections)).float().unsqueeze(0).unsqueeze(0).repeat(1, 2, 1, 1, 1)
            detections_t[:, 0] = 1 - detections_t[:, 1]
            assert torch.all(torch.sum(detections_t, axis=1) == 1)

            for metric_name, metric_fn in metrics.items():
                val = metric_fn(detections_t, dense_labelling_t)
                metric_meters[metric_name].add(val.item())

                print(f"{metric_name}: {metric_meters[metric_name].mean}")

        volume_slice = volume[cut, :, :]
        detections_slice = detections[cut, :, :]

        masked_data = np.ma.masked_where(detections_slice == 0, detections_slice)

        axes.imshow(volume_slice.T, cmap='gray')
        axes.imshow(masked_data.T, cmap=cm.autumn, alpha=0.4)
        fig.tight_layout()
        fig.savefig(plot_path + f'/detection-complete-{col}.png')
        plt.close(fig)
        plt.close()

        i += 1

    print("RESULTS:")
    for metric_name, metric_fn in metrics.items():
        print(f"{metric_name}: {metric_meters[metric_name].mean}")


def complete_identification_picture(scans_dir, plot_path, start, end, ignore_small_masks_detection, with_label,
                                    detection_input_size, detection_input_shift, volume_format, label_format,
                                    save_predictions, spacing=(1.0, 1.0, 1.0), weights=np.array([0.1, 0.9])):
    if volume_format == '.nii.gz':
        # only one file per volume
        scan_paths = glob.glob(scans_dir + "/**/*" + volume_format, recursive=True)
    else:
        # multiple files per volume
        scan_paths = glob.glob(scans_dir + "/**/")

    if end != -1:
        scan_paths = scan_paths[start:end]

    detection_model = load_spine_model("detection")
    identification_model = load_spine_model("identification")
    i = 1

    for col, scan_path in enumerate(scan_paths):
        fig, axes = plt.subplots(figsize=(8, 8), dpi=300)
        print(i, scan_path)

        if volume_format == '.nii.gz':
            if with_label:
                scan_path_without_ext = scan_path[:-len(volume_format)]
                centroid_path = scan_path_without_ext + label_format
                if label_format == ".lml":
                    labels, centroids = opening_files.extract_centroid_info_from_lml(centroid_path)
                else:
                    labels, centroids = opening_files.extract_centroid_info_from_json(centroid_path)
            scan_name = (scan_path.rsplit('/', 1)[-1])[:-len(volume_format)]
            volume, *_ = opening_files.read_volume_nii_format(scan_path, spacing=spacing)

        elif volume_format == '.dcm' and label_format == ".nii.gz":
            if with_label:
                meta_file_dir = glob.glob(scan_path + "*" + label_format)
                if len(meta_file_dir) != 1:
                    raise AttributeError(f"More/Less than one annotation file for volume {scan_path}")
                meta_file_dir = meta_file_dir[0]
                labels, centroids = opening_files.extract_centroid_info_from_nii(meta_file_dir, spacing=spacing)
            scan_name = scan_path.rsplit('/')[-2]
            volume, *_ = opening_files.read_volume_dcm_series(scan_path, spacing=spacing)

        else:
            raise AttributeError("Volume-Format and Label-Format Combination not supported")

        if with_label:
            centroid_indexes = centroids / np.array(spacing)
            cut = np.round(np.mean(centroid_indexes[:, 0])).astype(int)
        else:
            # without label -> just take the sagital center
            cut = round(volume.shape[0] / 2)

        axes.set_title(scan_name, fontsize=10, pad=10)

        pred_labels, pred_centroid_estimates, pred_detections, pred_identifications = test_scan(
            detection_model=detection_model,
            detection_X_shape=detection_input_size,
            detection_y_shape=detection_input_shift,
            identification_model=identification_model,
            volume=volume,
            ignore_small_masks_detection=ignore_small_masks_detection,
            img_name=plot_path + '/centroids_' + str(col) + '_debug.png'
        )

        volume_slice = volume[cut, :, :]
        # detections_slice = pred_detections[cut, :, :]
        identifications_slice = pred_identifications[cut, :, :]
        # identifications_slice = np.max(pred_identifications, axis=0)

        # masked_data = np.ma.masked_where(identifications_slice == 0, identifications_slice)
        # masked_data = np.ma.masked_where(detections_slice == 0, detections_slice)

        axes.imshow(volume_slice.T, cmap='gray', origin='lower')
        # axes[col].imshow(masked_data.T, vmin=1, vmax=27, cmap=cm.jet, alpha=0.4, origin='lower')

        if with_label:
            for label, centroid_idx in zip(labels, centroid_indexes):
                u, v = centroid_idx[1:3]
                axes.annotate(label, (u, v), color="white", size=6)
                axes.scatter(u, v, color="white", s=8)

            axes.plot(centroid_indexes[:, 1], centroid_indexes[:, 2], color="white")

        for pred_label, pred_centroid_idx in zip(pred_labels, pred_centroid_estimates):
            u, v = pred_centroid_idx[1:3]
            axes.annotate(pred_label, (u, v), color="red", size=6)
            axes.scatter(u, v, color="red", s=8)

        if save_predictions:
            create_lml_file(scan_path + "prediction.lml", pred_labels, pred_centroid_estimates)

        pred_centroid_estimates = np.array(pred_centroid_estimates)
        axes.plot(pred_centroid_estimates[:, 1], pred_centroid_estimates[:, 2], color="red")

        # get average distance
        if with_label:
            total_difference = 0.0
            no = 0.0
            for pred_label, pred_centroid_idx in zip(pred_labels, pred_centroid_estimates):
                if pred_label in labels:
                    label_idx = labels.index(pred_label)
                    print(pred_label, centroid_indexes[label_idx], pred_centroid_idx)
                    total_difference += np.linalg.norm(pred_centroid_idx - centroid_indexes[label_idx])
                    no += 1

            average_difference = total_difference / no
            print("average", average_difference)
            axes.set_xlabel("{:.2f}".format(average_difference) + "mm", fontsize=10)

        i += 1

        fig.tight_layout()
        # plt.show()
        fig.savefig(plot_path + '/centroids_' + str(col) + '.png')
        plt.close(fig)
        plt.close()


def get_stats(scans_dir, volume_format, label_format, ignore_small_masks_detection, detection_input_size,
              detection_input_shift, plot_path, spacing=(1.0, 1.0, 1.0)):
    if volume_format == '.nii.gz':
        # only one file per volume
        scan_paths = glob.glob(scans_dir + "/**/*" + volume_format, recursive=True)
    else:
        # multiple files per volume
        scan_paths = glob.glob(scans_dir + "/**/")

    detection_model = load_spine_model("detection")
    identification_model = load_spine_model("identification")

    all_correct = 0.0
    all_no = 0.0
    cervical_correct = 0.0
    cervical_no = 0.0
    thoracic_correct = 0.0
    thoracic_no = 0.0
    lumbar_correct = 0.0
    lumbar_no = 0.0

    all_difference = []
    cervical_difference = []
    thoracic_difference = []
    lumbar_difference = []

    differences_per_vertebrae = {}

    for i, scan_path in enumerate(scan_paths):
        print(i, scan_path)

        if volume_format == '.nii.gz':
            scan_path_without_ext = scan_path[:-len(volume_format)]
            centroid_path = scan_path_without_ext + label_format
            if label_format == ".lml":
                labels, centroids = opening_files.extract_centroid_info_from_lml(centroid_path)
            else:
                labels, centroids = opening_files.extract_centroid_info_from_json(centroid_path)
            volume, *_ = opening_files.read_volume_nii_format(scan_path, spacing=spacing)

        elif volume_format == '.dcm' and label_format == ".nii.gz":
            meta_file_dir = glob.glob(scan_path + "*" + label_format)
            if len(meta_file_dir) != 1:
                raise AttributeError(f"More/Less than one annotation file for volume {scan_path}")
            meta_file_dir = meta_file_dir[0]
            labels, centroids = opening_files.extract_centroid_info_from_nii(meta_file_dir, spacing=spacing)
            volume, *_ = opening_files.read_volume_dcm_series(scan_path, spacing=spacing)

        else:
            raise AttributeError("Volume-Format and Label-Format Combination not supported")

        centroid_indexes = centroids / np.array(spacing)

        pred_labels, pred_centroid_estimates, pred_detections, pred_identifications = test_scan(
            detection_model=detection_model,
            detection_X_shape=detection_input_size,
            detection_y_shape=detection_input_shift,
            identification_model=identification_model,
            volume=volume,
            ignore_small_masks_detection=ignore_small_masks_detection,
        )

        for label, centroid_idx in zip(labels, centroid_indexes):
            min_dist = 20
            min_label = ''
            for pred_label, pred_centroid_idx in zip(pred_labels, pred_centroid_estimates):
                dist = np.linalg.norm(pred_centroid_idx - centroid_idx)
                if dist <= min_dist:
                    min_dist = dist
                    min_label = pred_label

            all_no += 1
            if label[0] == 'C':
                cervical_no += 1
            elif label[0] == 'T':
                thoracic_no += 1
            elif label[0] == 'L':
                lumbar_no += 1

            if label == min_label:
                all_correct += 1
                if label[0] == 'C':
                    cervical_correct += 1
                elif label[0] == 'T':
                    thoracic_correct += 1
                elif label[0] == 'L':
                    lumbar_correct += 1

            print(label, min_label)

        # get average distance
        total_difference = 0.0
        no = 0.0
        for pred_label, pred_centroid_idx in zip(pred_labels, pred_centroid_estimates):
            if pred_label in labels:
                label_idx = labels.index(pred_label)
                print(pred_label, centroid_indexes[label_idx], pred_centroid_idx)
                difference = np.linalg.norm(pred_centroid_idx - centroid_indexes[label_idx])
                total_difference += difference
                no += 1

                # Add to specific vertebrae hash
                if pred_label in differences_per_vertebrae:
                    differences_per_vertebrae[pred_label].append(difference)
                else:
                    differences_per_vertebrae[pred_label] = [difference]

                # Add to total difference
                all_difference.append(difference)
                if pred_label[0] == 'C':
                    cervical_difference.append(difference)
                elif pred_label[0] == 'T':
                    thoracic_difference.append(difference)
                elif pred_label[0] == 'L':
                    lumbar_difference.append(difference)

        if no > 0:
            average_difference = total_difference / no
            print("average", average_difference, "\n")

    data = []
    labels_used = []
    for label in LABELS_NO_L6:
        if label in differences_per_vertebrae:
            labels_used.append(label)
            data.append(differences_per_vertebrae[label])

    plt.figure(figsize=(20, 10))
    plt.boxplot(data, labels=labels_used)
    plt.savefig(f'{plot_path}/boxplot.png')
    plt.close()

    all_rate = np.around(100.0 * all_correct / all_no, decimals=1)
    all_mean = np.around(np.mean(all_difference), decimals=2)
    all_std = np.around(np.std(all_difference), decimals=2)
    cervical_rate = np.around(100.0 * cervical_correct / cervical_no, decimals=1)
    cervical_mean = np.around(np.mean(cervical_difference), decimals=2)
    cervical_std = np.around(np.std(cervical_difference), decimals=2)
    thoracic_rate = np.around(100.0 * thoracic_correct / thoracic_no, decimals=1)
    thoracic_mean = np.around(np.mean(thoracic_difference), decimals=2)
    thoracic_std = np.around(np.std(thoracic_difference), decimals=2)
    lumbar_rate = np.around(100.0 * lumbar_correct / lumbar_no, decimals=1)
    lumbar_mean = np.around(np.mean(lumbar_difference), decimals=2)
    lumbar_std = np.around(np.std(lumbar_difference), decimals=2)

    print("All Id rate: " + str(all_rate) + "%  mean: " + str(all_mean) + "  std: " + str(all_std) + "\n")
    print("Cervical Id rate: " + str(cervical_rate) + "%  mean:" + str(cervical_mean) + "  std:" + str(
        cervical_std) + "\n")
    print("Thoracic Id rate: " + str(thoracic_rate) + "%  mean:" + str(thoracic_mean) + "  std:" + str(
        thoracic_std) + "\n")
    print("Lumbar Id rate: " + str(lumbar_rate) + "%  mean:" + str(lumbar_mean) + "  std:" + str(lumbar_std) + "\n")

    results = pd.DataFrame({'Region': ['All', 'Cervical', 'Thoracic', 'Lumbar'],
                            'ID rate': [all_rate, cervical_rate, thoracic_rate, lumbar_rate],
                            'Mean': [all_mean, cervical_mean, thoracic_mean, lumbar_mean],
                            'Std': [all_std, cervical_std, thoracic_std, lumbar_std]})

    results.to_csv(plot_path + "/results.csv", index=False)


def single_detection(scan_path, plot_path, volume_format, label_format, ignore_small_masks_detection,
                     detection_input_size, detection_input_shift, spacing=(1.0, 1.0, 1.0),
                     weights=np.array([0.1, 0.9])):
    if volume_format == '.nii.gz':
        scan_path_without_ext = scan_path[:-len(volume_format)]
        centroid_path = scan_path_without_ext + label_format
        if label_format == ".lml":
            labels, centroids = opening_files.extract_centroid_info_from_lml(centroid_path)
        else:
            labels, centroids = opening_files.extract_centroid_info_from_json(centroid_path)
        scan_name = (scan_path.rsplit('/', 1)[-1])[:-len(volume_format)]
        volume, *_ = opening_files.read_volume_nii_format(scan_path, spacing=spacing)

    elif volume_format == '.dcm' and label_format == ".nii.gz":
        meta_file_dir = glob.glob(scan_path + "*" + label_format)
        if len(meta_file_dir) != 1:
            raise AttributeError(f"More/Less than one annotation file for volume {scan_path}")
        meta_file_dir = meta_file_dir[0]
        labels, centroids = opening_files.extract_centroid_info_from_nii(meta_file_dir, spacing=spacing)
        scan_name = scan_path.rsplit('/')[-2]
        volume, *_ = opening_files.read_volume_dcm_series(scan_path, spacing=spacing)

    else:
        raise AttributeError("Volume-Format and Label-Format Combination not supported")

    centroid_indexes = centroids / np.array(spacing)

    cut = np.round(np.mean(centroid_indexes[:, 0])).astype(int)

    detection_model = load_spine_model("detection")
    detections = apply_detection_model(volume, detection_model, detection_input_size, detection_input_shift,
                                       ignore_small_masks_detection)

    volume_slice = volume[cut, :, :]
    detections_slice = detections[cut, :, :]
    masked_data = np.ma.masked_where(detections_slice == 0, detections_slice)

    fig, ax = plt.subplots(1)
    ax.imshow(volume_slice.T, cmap='gray')
    ax.imshow(masked_data.T, cmap=cm.jet, alpha=0.4, origin='lower')
    fig.savefig(plot_path + '/single.png')
    plt.close(fig)
    plt.close()


def single_identification(scan_path, plot_path, volume_format, label_format, ignore_small_masks_detection,
                          detection_input_size, detection_input_shift, spacing=(1.0, 1.0, 1.0),
                          weights=np.array([0.1, 0.9])):
    if volume_format == '.nii.gz':
        scan_path_without_ext = scan_path[:-len(volume_format)]
        centroid_path = scan_path_without_ext + label_format
        if label_format == ".lml":
            labels, centroids = opening_files.extract_centroid_info_from_lml(centroid_path)
        else:
            labels, centroids = opening_files.extract_centroid_info_from_json(centroid_path)
        scan_name = (scan_path.rsplit('/', 1)[-1])[:-len(volume_format)]
        volume, *_ = opening_files.read_volume_nii_format(scan_path, spacing=spacing)

    elif volume_format == '.dcm' and label_format == ".nii.gz":
        meta_file_dir = glob.glob(scan_path + "*" + label_format)
        if len(meta_file_dir) != 1:
            raise AttributeError(f"More/Less than one annotation file for volume {scan_path}")
        meta_file_dir = meta_file_dir[0]
        labels, centroids = opening_files.extract_centroid_info_from_nii(meta_file_dir, spacing=spacing)
        scan_name = scan_path.rsplit('/')[-2]
        volume, *_ = opening_files.read_volume_dcm_series(scan_path, spacing=spacing)

    else:
        raise AttributeError("Volume-Format and Label-Format Combination not supported")

    centroid_indexes = centroids / np.array(spacing)

    cut = np.round(np.mean(centroid_indexes[:, 0])).astype(int)

    detection_model = load_spine_model("detection")
    identification_model = load_spine_model("identification")

    detections = apply_detection_model(volume, detection_model, detection_input_size, detection_input_shift,
                                       ignore_small_masks_detection)
    identification = apply_identification_model(volume, cut - 1, cut + 1, identification_model)

    volume_slice = volume[cut, :, :]
    detection_slice = detections[cut, :, :]
    identification_slice = identification[cut, :, :]

    identification_slice *= detection_slice

    masked_data = np.ma.masked_where(identification_slice == 0, identification_slice)

    fig, ax = plt.subplots(1)

    ax.imshow(volume_slice.T, cmap='gray')
    ax.imshow(masked_data.T, cmap=cm.jet, vmin=1, vmax=27, alpha=0.5, origin='lower')
    fig.savefig(plot_path + '/single_identification.png')
    plt.close(fig)
    plt.close()


if __name__ == '__main__':
    if args.resume_detection is not None:
        complete_detection_picture(
            str(args.testing_dataset_dir),
            plot_path=str(args.plot_path),
            start=0,
            end=int(args.n_plots),
            detection_input_size=np.array(args.detection_input_size),
            detection_input_shift=np.array(args.detection_input_shift),
            spacing=tuple(args.spacing),
            volume_format=str(args.volume_format),
            label_format=str(args.label_format),
            with_label=not bool(args.without_label),
            ignore_small_masks_detection=args.ignore_small_masks_detection,
            save_detections=args.save_detections,
        )

    if args.resume_detection is not None and args.resume_identification is not None:
        get_stats(
            str(args.testing_dataset_dir),
            plot_path=str(args.plot_path),
            spacing=tuple(args.spacing),
            detection_input_size=np.array(args.detection_input_size),
            detection_input_shift=np.array(args.detection_input_shift),
            volume_format=str(args.volume_format),
            label_format=str(args.label_format),
            ignore_small_masks_detection=args.ignore_small_masks_detection,
        )

        complete_identification_picture(
            str(args.testing_dataset_dir),
            plot_path=str(args.plot_path),
            start=0,
            end=int(args.n_plots),
            with_label=not bool(args.without_label),
            detection_input_size=np.array(args.detection_input_size),
            detection_input_shift=np.array(args.detection_input_shift),
            spacing=tuple(args.spacing),
            volume_format=str(args.volume_format),
            label_format=str(args.label_format),
            ignore_small_masks_detection=args.ignore_small_masks_detection,
            save_predictions=args.save_predictions,
        )
