"""Class object to use or create construction type classification models."""
#
# Copyright (c) 2022 The Regents of the University of California
#
# This file is part of BRAILS.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# You should have received a copy of the BSD 3-Clause License along with
# BRAILS. If not, see <http://www.opensource.org/licenses/>.
#
# Contributors:
# Barbaros Cetiner
#
# Last updated:
# 08-29-2024

import os
from brails.modules.ImageClassifier.ImageClassifier import ImageClassifier
import torch


class ConsTypeClassifier(ImageClassifier):
    """
    Class for facilitating classification of different construction types.

    ConsTypeClassifier is a specialized class for classifying different
    construction types (e.g., MAB, MAS, RCC, STL, WOD). It allows for making
    predictions using a pre-trained model and retraining the pre-trained model
    on new data. Inherits from the ImageClassifier class.

    This class loads a default pre-trained model if a custom model path is not
    provided during initialization. It can also be retrained using new
    datasets.
    """

    def __init__(self, model_path: str = None) -> None:
        """
        Initialize the ConsTypeClassifier.

        Args_
            model_path (str, optional): Path to the model file. If None, it
            will load the default construction type classifier model.
        """
        if model_path is None:
            os.makedirs('tmp/models', exist_ok=True)
            model_path = 'tmp/models/consTypeClassifier_v1.pth'
            if not os.path.isfile(model_path):
                print('Loading default construction type classifier model ' +
                      'file to tmp/models folder...')
                torch.hub.download_url_to_file('https://zenodo.org/record/' +
                                               '13525814/files/' +
                                               'constype_classifier_v1.pth',
                                               model_path, progress=False)
                print('Default construction type classifier model loaded')
            else:
                print('Default construction type classifier model at ' +
                      f"{model_path} loaded")
        else:
            print('Inferences will be performed using the custom model at ' +
                  f'{model_path}')

        self.model_path: str = model_path
        self.classes: list[str] = ['MAB', 'MAS',
                                   'RCC', 'STL', 'WOD']  # Construction types

    def predict(self, data_dir: str) -> None:
        """
        Perform construction type predictions on images in the specified path.

        Args__
            data_dir (str): Path to the directory containing images to be
                classified.
        """
        image_classifier = ImageClassifier()
        image_classifier.predict(self.model_path, data_dir, self.classes)
        self.preds = image_classifier.preds  # Store predictions

    def retrain(self,
                data_dir: str,
                batch_size: int = 8,
                nepochs: int = 100,
                plot_loss: bool = True) -> None:
        """
        Retrain the construction type classifier on new data.

        Args__
            data_dir (str): Path to the directory containing training data.
            batch_size (int, optional): Batch size for training. Default is 8.
            nepochs (int, optional): Number of epochs for training. Default is
                100.
            plot_loss (bool, optional): Whether to plot the loss during
                training. Default is True.
        """
        image_classifier = ImageClassifier()
        image_classifier.retrain(self.model_path, data_dir,
                                 batch_size, nepochs, plot_loss)


if __name__ == '__main__':
    pass
