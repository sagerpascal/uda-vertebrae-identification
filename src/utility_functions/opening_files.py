import SimpleITK as sitk
import numpy as np

from utility_functions import processing
from utility_functions.labels import KSA_ID_LABEL_MAP


def read_volume_nii_format(dir, spacing):
    return _read_nii_file(dir, spacing, is_label=False)


def read_volume_dcm_series(dir, spacing, series_prefix='mediastinum'):
    try:
        image_np, mean, variance, metadata = read_volume_dcm_series_(dir, spacing, series_prefix='mediastinum 1.5')
    except TypeError:
        image_np, mean, variance, metadata = read_volume_dcm_series_(dir, spacing, series_prefix='mediastinum')
    return image_np, mean, variance, metadata


def read_volume_dcm_series_(dir, spacing, series_prefix='mediastinum'):
    series_reader = sitk.ImageSeriesReader()
    reader = sitk.ImageFileReader()

    series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(dir)
    if not series_IDs:
        print("ERROR: given directory \"" + dir + "\" does not contain a DICOM series.")
        return None

    for series_ID in series_IDs:
        file_names = series_reader.GetGDCMSeriesFileNames(dir, series_ID)

        # Analyze first image in series
        reader.SetFileName(file_names[0])
        reader.LoadPrivateTagsOn()
        reader.ReadImageInformation()
        series_decr = reader.GetMetaData('0008|103e')
        if not series_decr.lower().startswith(series_prefix.lower()):
            continue

        metadata = {
            'Patient ID': reader.GetMetaData('0010|0020'),
            'Patient Sex': reader.GetMetaData('0010|0040'),
            'Series Description': series_decr,
        }

        series_reader.SetFileNames(file_names)
        image = series_reader.Execute()
        sitk_dir = processing.resample_image(image, out_spacing=spacing)  # change image size, spacing, etc.
        image_np = sitk.GetArrayFromImage(sitk_dir)
        mean, variance = np.mean(image_np[image_np > 0]), np.std(image_np[image_np > 0])
        sitk_dir = processing.zero_mean_unit_var(sitk_dir)  # normalization
        image_np = sitk.GetArrayFromImage(sitk_dir).T
        return image_np, mean, variance, metadata  # 0=z, 1=x, 2=Y


def extract_centroid_info_from_nii(dir, spacing):
    label_np = _read_nii_file(dir, spacing, is_label=True)
    labels, centroids = [], []
    for label_id in np.unique(label_np):
        if label_id == 0:
            continue
        label_pos = np.where(label_np == label_id)

        labels.append(KSA_ID_LABEL_MAP[label_id])
        centroids.append(np.array(
            [(label_pos[0].min() + label_pos[0].max()) / 2., (label_pos[1].min() + label_pos[1].max()) / 2.,
             (label_pos[2].min() + label_pos[2].max()) / 2.]))

    return labels, centroids


def extract_centroid_info_from_lml(dir):
    centroids_file = open(dir, 'r')
    iter_centroids_file = iter(centroids_file)
    next(iter_centroids_file)

    labels = []
    centroids = []
    for centroid_line in iter_centroids_file:
        centroid_line_split = centroid_line.split()
        labels.append(centroid_line_split[1].split("_")[0])
        centroids.append(np.array(centroid_line_split[2:5]).astype(float))

    # labels is a list containing the vertebraes in the scan (e.g. ['T1', 'T2', 'T3', ...]
    # centroids is a list containing a np array with the coordiantes per vertebrae (e.g. [array([109.1, 27.8, 405.0]), ...]
    return labels, centroids  # Sagital view: 0=Z, 1=Y, 2=X


def _read_nii_file(dir, spacing, is_label):
    reader = sitk.ImageFileReader()
    reader.SetFileName(dir)
    reader.LoadPrivateTagsOn()
    reader.ReadImageInformation()
    sitk_dir = reader.Execute()
    sitk_dir = processing.resample_image(sitk_dir, is_label=is_label,
                                         out_spacing=spacing)  # change image size, spacing, etc.
    if not is_label:
        metadata = {
            'Patient ID': "",
            'Patient Sex': "",
            'Series Description': "",
        }
        image_np = sitk.GetArrayFromImage(sitk_dir)
        mean, variance = np.mean(image_np[image_np > 0]), np.std(image_np[image_np > 0])
        sitk_dir = processing.zero_mean_unit_var(sitk_dir)  # normalization
        return sitk.GetArrayFromImage(sitk_dir).T, mean, variance, metadata

    else:
        return sitk.GetArrayFromImage(sitk_dir).T
