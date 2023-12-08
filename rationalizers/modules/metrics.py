"""Adapted from https://github.com/bastings/interpretable_predictions"""


from rationalizers.utils import unroll


def evaluate_rationale(test_ids, annotations, lengths) -> dict:
    """
    Function that computes the token F1 Score (matching with annotations).

    :param y: Ground-truth labels.
    :param y_hat: Model label predictions.
    """
    correct, total, macro_prec_total, macro_rec_total, macro_n = 0, 0, 0, 0, 0

    test_ids = unroll(test_ids)
    annotations = unroll(annotations)
    lengths = unroll(lengths)
    print(len(test_ids), len(annotations), len(lengths))
    # print(annotations)

    for i in range(len(test_ids)):
        z_ex = test_ids[i][: lengths[i]]
        z_ex_nonzero = (z_ex > 0).float()
        z_ex_nonzero_sum = z_ex_nonzero.sum().item()

        # make this work for multiple aspects
        aspect_annotations = [ell.tolist() for ell in annotations[i][0]]
        if len(aspect_annotations) == 0:
            continue
        annotations_range = [[a[0], a[1]] for a in aspect_annotations]
        print(annotations_range[0], z_ex[0])
        matched = sum(
            1
            for i, zi in enumerate(z_ex)
            if zi > 0 and any(range[0] <= i < range[1] for range in annotations_range)
        )
        non_matched = sum(
            1
            for i, zi in enumerate(z_ex)
            if zi == 0 and any(range[0] <= i < range[1] for range in annotations_range)
        )
        precision = matched / (z_ex_nonzero_sum + 1e-9)
        recall = matched / (matched + non_matched + 1e-9)
        macro_prec_total += precision
        macro_rec_total += recall
        correct += matched
        total += z_ex_nonzero_sum
        if z_ex_nonzero_sum > 0:
            macro_n += 1

    precision = correct / (total + 1e-9)
    macro_precision = macro_prec_total / (float(macro_n) + 1e-9)
    macro_recall = macro_rec_total / (float(macro_n) + 1e-9)
    f1_score = (
        2 * macro_precision * macro_recall / (macro_precision + macro_recall + 1e-9)
    )

    report = {
        "macro_precision": macro_precision,
        "precision": precision,
        "macro_recall": macro_recall,
        "f1_score": f1_score,
    }

    return report
