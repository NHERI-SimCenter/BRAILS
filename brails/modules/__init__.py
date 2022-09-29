# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 The Regents of the University of California
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
# Yunhui Guo 
# Sascha Hornauer

from brails.modules.PytorchGenericModelClassifier.GenericImageClassifier import PytorchImageClassifier
from brails.modules.ImageClassifier.ImageClassifier import ImageClassifier
from brails.modules.ImageSegmenter.ImageSegmenter import ImageSegmenter
from brails.modules.PytorchRoofTypeClassifier.RoofTypeClassifier import PytorchRoofClassifier
from brails.modules.PytorchOccupancyClassClassifier.OccupancyClassifier import PytorchOccupancyClassifier
from brails.modules.ChimneyDetector.ChimneyDetector import ChimneyDetector
from brails.modules.FacadeParser.FacadeParser import FacadeParser
from brails.modules.FoundationClassifier.FoundationClassifier import FoundationHeightClassifier
from brails.modules.GarageDetector.GarageDetector import GarageDetector
from brails.modules.NumFloorDetector.NFloorDetector import NFloorDetector
from brails.modules.RoofCoverClassifier.RoofCoverClassifier import RoofCoverClassifier
from brails.modules.YearBuiltClassifier.YearBuiltClassifier import YearBuiltClassifier
