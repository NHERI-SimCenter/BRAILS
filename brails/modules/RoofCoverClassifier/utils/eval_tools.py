import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import seaborn as sns

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

    return data_confusion_matrix,figure