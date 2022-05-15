import argparse
import random
import shutil
from distutils.dir_util import copy_tree
from pathlib import Path

import numpy as np
import pandas as pd
import patoolib
import pydicom
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str, default='../../data2/covid19-ct/subjects',
                    help="Path to the downloaded 'Subjext X.rar' files")
parser.add_argument("--tmp_path", type=str, default='../../data2/covid19-ct/dataset',
                    help="Path to unzip the data temporarily")

args = parser.parse_args()

labeled_train_subjects = [
    'Subject (1041)', 'Subject (134)', 'Subject (220)', 'Subject (281)', 'Subject (327)', 'Subject (376)',
    'Subject (581)', 'Subject (705)', 'Subject (859)', 'Subject (1103)', 'Subject (177)', 'Subject (264)',
    'Subject (303)', 'Subject (349)', 'Subject (410)', 'Subject (646)', 'Subject (781)', 'Subject (986)',
    'Subject (119)', 'Subject (18)', 'Subject (274)', 'Subject (306)', 'Subject (362)', 'Subject (502)',
    'Subject (662)', 'Subject (819)']

labeled_test_subjects = [
    'Subject (1)', 'Subject (10)', 'Subject (102)', 'Subject (11)', 'Subject (120)', 'Subject (14)', 'Subject (141)',
    'Subject (146)', 'Subject (148)', 'Subject (15)', 'Subject (36)', 'Subject (43)', 'Subject (56)', 'Subject (82)']


def _create_out_dir(dir_):
    dir_ = Path(dir_)
    if not dir_.exists():
        dir_.mkdir()


def _rm_tree(pth):
    pth = Path(pth)
    for child in pth.glob('*'):
        if child.is_file():
            child.unlink()
        else:
            _rm_tree(child)
    pth.rmdir()


def _unzip_files():
    for file in Path(str(args.dataset_path)).glob("*.rar"):
        outdir = (Path(str(args.tmp_path)) / file.name[:-4])
        if not outdir.exists():
            outdir.mkdir()
            try:
                patoolib.extract_archive(file, outdir=outdir)
            except patoolib.util.PatoolError:
                print("Extract rar-files failed - please install rar, unrar or 7z")

            folders_to_remove = []
            for fp in outdir.rglob("*"):
                if fp.is_dir():
                    folders_to_remove.append(fp)
                else:
                    if fp.suffix == "":
                        fp.rename(outdir / (fp.name + ".dcm"))
                    else:
                        fp.rename(outdir / fp.name)

            for folder in folders_to_remove:
                if folder.exists():
                    _rm_tree(folder)


def _analyze_files():
    files = {'subject': [], 'path': [], 'is_series': [], 'slice_location': [], 'slice_thickness': [],
             'series_descr': [], 'patient_id': [], 'patient_sex': []}

    n_files = len([y for y in Path(str(args.tmp_path)).rglob("*.dcm")])
    for file in tqdm(Path(str(args.tmp_path)).rglob("*.dcm"), desc="Analyzing Files", total=n_files):
        slice = pydicom.dcmread(file)
        is_series = hasattr(slice, 'SliceThickness') and slice.SliceThickness > 0 and hasattr(slice,
                                                                                              'SeriesDescription') and slice.SeriesDescription != 'nan'
        thickness = slice.SliceThickness if hasattr(slice, 'SliceThickness') else np.NaN
        slice_location = slice.SliceLocation if hasattr(slice, 'SliceLocation') else np.NaN
        series_descr = slice.SeriesDescription.lower() if is_series and hasattr(slice, 'SeriesDescription') else ""

        if series_descr != "":
            if series_descr.startswith("lung") or series_descr.startswith("-lung"):
                series_descr = "lung"
            elif series_descr.startswith("mediastinum"):
                series_descr = "mediastinum"
            elif series_descr.startswith("dose info"):
                series_descr = "dose info"
            elif series_descr.startswith("helical"):
                series_descr = "helical"
            else:
                print(f"Unknown Series description {series_descr}")

        if series_descr == "mediastinum":
            files['subject'].append(file.parent.name)
            files['path'].append(str(file))
            files['is_series'].append(is_series)
            files['slice_location'].append(slice_location)
            files['slice_thickness'].append(thickness)
            files['series_descr'].append(series_descr)
            files['patient_id'].append(slice.PatientID)
            files['patient_sex'].append(slice.PatientSex)

    df = pd.DataFrame.from_dict(files)
    df.to_csv(Path(str(args.tmp_path)) / 'df.csv')


def _get_series_split():
    df = pd.read_csv(Path(str(args.tmp_path)) / 'df.csv')

    result = pd.DataFrame()
    for subject in pd.unique(df['subject']):
        df_subject = df[df['subject'] == subject]
        min_slice_thickness = df_subject['slice_thickness'].min()
        if min_slice_thickness <= 2.5:
            df_subject = df_subject[df_subject['slice_thickness'] == min_slice_thickness]
            result = pd.concat([result, df_subject])

    train_labeled_df = result[result.subject.isin(labeled_train_subjects)]
    test_labeled_df = result[result.subject.isin(labeled_test_subjects)]

    result.drop(train_labeled_df.index, inplace=True)
    result.drop(test_labeled_df.index, inplace=True)

    subject_list = pd.unique(result['subject'])
    random.shuffle(subject_list)
    split_index = round(.8 * len(subject_list))
    train_subjects = subject_list[:split_index]
    test_subjects = subject_list[split_index:]

    train_df = result[result['subject'].isin(train_subjects)]
    test_df = result[result['subject'].isin(test_subjects)]

    train_df.to_csv(Path(str(args.tmp_path)) / 'df_train.csv')
    test_df.to_csv(Path(str(args.tmp_path)) / 'df_test.csv')
    train_labeled_df.to_csv(Path(str(args.tmp_path)) / 'df_train_labeled.csv')
    test_labeled_df.to_csv(Path(str(args.tmp_path)) / 'df_test_labeled.csv')


def _copy_files_in_split_folder():
    base_path = Path(str(args.tmp_path)) / ".."

    for subject in tqdm(pd.unique(pd.read_csv(Path(str(args.tmp_path)) / 'df_train.csv')['subject'])):
        shutil.move(str(base_path / 'dataset' / subject), str(base_path / 'training_dataset'))

    for subject in tqdm(pd.unique(pd.read_csv(Path(str(args.tmp_path)) / 'df_test.csv')['subject'])):
        shutil.move(str(base_path / 'dataset' / subject), str(base_path / 'testing_dataset'))

    for subject in tqdm(pd.unique(pd.read_csv(Path(str(args.tmp_path)) / 'df_train_labeled.csv')['subject'])):
        shutil.move(str(base_path / 'dataset' / subject), str(base_path / 'training_dataset_labeled'))

    for subject in tqdm(pd.unique(pd.read_csv(Path(str(args.tmp_path)) / 'df_test_labeled.csv')['subject'])):
        shutil.move(str(base_path / 'dataset' / subject), str(base_path / 'testing_dataset_labeled'))


# def _print_series_per_subject():
#     series_subject_map = {}
#     df = pd.read_csv(Path(str(args.tmp_path)) / 'df.csv')
#     for index, row in df.iterrows():
#         if row['is_series']:
#             series = f"{row['series_descr']} ({row['slice_thickness']} mm)"
#             if series not in series_subject_map:
#                 series_subject_map[series] = {row['subject']}
#             else:
#                 series_subject_map[series].add(row['subject'])
#
#     series_count = OrderedDict()
#     for serie, subjects in series_subject_map.items():
#         series_count[serie] = len(subjects)
#
#     df = pd.DataFrame({'Series': series_count.keys(), 'Count': series_count.values()})
#     ax = df.plot.bar(x='Series', y='Count', rot=90)
#     plt.tight_layout()
#     plt.show()


if __name__ == '__main__':
    _create_out_dir(str(args.tmp_path))
    _unzip_files()
    _analyze_files()
    # _print_series_per_subject()
    _get_series_split()
    _copy_files_in_split_folder()
    shutil.rmtree(str(args.tmp_path))
