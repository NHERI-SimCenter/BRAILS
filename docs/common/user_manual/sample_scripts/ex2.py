from brails.modules import PytorchRoofClassifier, PytorchOccupancyClassifier
import glob

# Initialize a roof classifier:
roofModel = PytorchRoofClassifier()

# Use the roof classifier to infer the roof class for each image contained in tmp/images/satellite/ folder:
imgs = glob.glob('tmp/images/satellite/*')
predictions = roofModel.predictMultipleImages(imgs)

# Initialize an occupancy classifier:
occupancyModel = PytorchOccupancyClassifier()

# Use the occupancy classifier to infer the occupancy type for the building images located in tmp/images/street/ folder:
imgs = glob.glob('tmp/images/street/*')
predictions = occupancyModel.predictMultipleImages(imgs)
