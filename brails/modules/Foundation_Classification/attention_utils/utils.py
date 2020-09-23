import numpy as np
import seaborn as  sns
import matplotlib.pyplot as plt
import sklearn
import torch
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

def sliding_window(data, size, stepsize=1, padded=False, axis=-1, copy=True):
    """
    Calculate a sliding window over a signal
    Parameters
    ----------
    data : numpy array
        The array to be slided over.
    size : int
        The sliding window size
    stepsize : int
        The sliding window stepsize. Defaults to 1.
    axis : int
        The axis to slide over. Defaults to the last axis.
    copy : bool
        Return strided array as copy to avoid sideffects when manipulating the
        output array.
    Returns
    -------
    data : numpy array
        A matrix where row in last dimension consists of one instance
        of the sliding window.
    Notes
    -----
    - Be wary of setting `copy` to `False` as undesired sideffects with the
      output values may occurr.
    Examples
    --------
    >>> a = numpy.array([1, 2, 3, 4, 5])
    >>> sliding_window(a, size=3)
    array([[1, 2, 3],
           [2, 3, 4],
           [3, 4, 5]])
    >>> sliding_window(a, size=3, stepsize=2)
    array([[1, 2, 3],
           [3, 4, 5]])
    See Also
    --------
    pieces : Calculate number of pieces available by sliding
    """
    if axis >= data.ndim:
        raise ValueError(
            "Axis value out of range"
        )

    if stepsize < 1:
        raise ValueError(
            "Stepsize may not be zero or negative"
        )

    if size > data.shape[axis]:
        raise ValueError(
            "Sliding window size may not exceed size of selected axis"
        )

    shape = list(data.shape)
    shape[axis] = np.floor(data.shape[axis] / stepsize - size / stepsize + 1).astype(int)
    shape.append(size)

    strides = list(data.strides)
    strides[axis] *= stepsize
    strides.append(data.strides[axis])

    strided = np.lib.stride_tricks.as_strided(
        data, shape=shape, strides=strides
    )

    if copy:
        return strided.copy()
    else:
        return strided

def construct_confusion_matrix_image(classes, con_mat):
    con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)

    con_mat_df = pd.DataFrame(con_mat_norm,
                              index=classes,
                              columns=classes)

    figure = plt.figure(figsize=(8, 8))
    sns.heatmap(con_mat_df, annot=True, cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    figure.canvas.draw()

    # Now we can save it to a numpy array.
    data_confusion_matrix = np.fromstring(figure.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data_confusion_matrix = data_confusion_matrix.reshape(figure.canvas.get_width_height()[::-1] + (3,))
    data_confusion_matrix = torch.from_numpy(np.transpose(data_confusion_matrix, (2, 0, 1)))

    return data_confusion_matrix, figure

def evaluate(summary_writer, mode_name, y_gt, y_pred, avg_loss, classes, epoch):

    summary_writer.add_scalar('{}/Loss'.format(mode_name), avg_loss, epoch)

    report_items = precision_recall_fscore_support(y_gt, np.array(y_pred) > 0.5, average='binary')
    confusion_matrix = sklearn.metrics.confusion_matrix(y_gt, np.array(y_pred) > 0.5,labels=range(len(classes)))
    cm, cm_fig = construct_confusion_matrix_image(classes, confusion_matrix)

    summary_writer.add_scalar('{}/F1'.format(mode_name), report_items[2], epoch)
    summary_writer.add_scalar('{}/Precision'.format(mode_name), report_items[0], epoch)
    summary_writer.add_scalar('{}/Recall'.format(mode_name), report_items[1], epoch)

    summary_writer.add_image('{}/confusion_matrix'.format(mode_name), cm, epoch)
    if mode_name == 'Test':
        return report_items
    return report_items[2]
