import SimpleITK as sitk
import numpy as np
import json
from utility_functions import processing
from utility_functions.labels import KSA_ID_LABEL_MAP
import nibabel as nib
from preprocessing.reorient_verse import reorient_to, resample_nib


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

    return labels, centroids

def extract_centroid_info_from_json(dir):
    centroids_file = open(dir, 'r')
    data = json.load(centroids_file)

    labels = []
    centroids = []

    for entry in data:
        print(entry)
        if not "label" in entry:
            continue
        labels.append(KSA_ID_LABEL_MAP[entry['label']])
        centroids.append(np.array([entry['X'], entry['Y'], entry['Z']], dtype="float"))

    return labels, centroids


def _read_nii_file(dir, spacing, is_label):
    try:
        reader = sitk.ImageFileReader()
        reader.SetFileName(dir)
        reader.LoadPrivateTagsOn()
        reader.ReadImageInformation()
        sitk_dir = reader.Execute()
        sitk_dir = processing.resample_image(sitk_dir, is_label=is_label, out_spacing=spacing)
        image_np = sitk.GetArrayFromImage(sitk_dir)
        mean, variance = np.mean(image_np[image_np > 0]), np.std(image_np[image_np > 0])
        sitk_dir = processing.zero_mean_unit_var(sitk_dir)  # normalization
        img = sitk.GetArrayFromImage(sitk_dir)

    except Exception as e:
        print("Could not open file with SimpleITK", dir, "Now trying NiBabel...", e)
        img_nib = nib.load(dir)
        img_iso = resample_nib(img_nib, voxel_spacing=(1, 1, 1), order=3)
        img_iso = reorient_to(img_iso, axcodes_to=('L', 'P', 'S'))

        image_np = img_iso.get_fdata().astype(np.float32)
        mean, variance = np.mean(image_np[image_np > 0]), np.std(image_np[image_np > 0])

        if variance > 0:
            image_np = (image_np - mean) / variance

        img = image_np

    if not is_label:
        metadata = {
            'Patient ID': "",
            'Patient Sex': "",
            'Series Description': "",
        }

        return img.T, mean, variance, metadata

    else:
        return img.T
