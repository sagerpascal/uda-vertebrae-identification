from utility_functions.labels import KSA_LABEL_ID_MAP


def create_lml_file(filename, pred_labels, pred_centroid_estimates):
    f = open(filename, "w+")
    f.write("# 0\n")
    for pred_label, pred_centroid_idx in zip(pred_labels, pred_centroid_estimates):
        id = f"{KSA_LABEL_ID_MAP[pred_label] * 10}"
        label = f"{pred_label}_center"
        f.write(f"{id}\t{label}\t{pred_centroid_idx[0]}\t{pred_centroid_idx[1]}\t{pred_centroid_idx[2]}\n")

    f.close()


if __name__ == '__main__':
    pred_labels = ['C3', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'T11', 'T12', 'L1']
    pred_centroid_estimates = [[241.0, 257.0, 319.0], [238.0, 266.0, 306.0], [240.0, 275.0, 290.0],
                               [241.0, 284.0, 270.0], [242.0, 292.0, 252.0], [244.0, 299.0, 233.0],
                               [244.0, 305.0, 213.0], [244.0, 309.0, 190.0], [244.0, 310.0, 167.0],
                               [247.0, 309.0, 142.0], [248.0, 306.0, 120.0], [247.0, 304.0, 93.0],
                               [246.0, 301.0, 65.0], [245.0, 296.0, 33.0]]
    create_lml_file("test.lml", pred_labels, pred_centroid_estimates)
