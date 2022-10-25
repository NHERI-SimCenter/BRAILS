from brails.modules import PytorchRoofClassifier, PytorchOccupancyClassifier
import glob;

# initilize a roof classifier
roofModel = PytorchRoofClassifier()

# initilize an occupancy classifier
occupancyModel = PytorchOccupancyClassifier()

# use the roof classifier
imgs = glob.glob('tmp/images/satellite/*')
predictions = roofModel.predictMultipleImages(imgs)

# use the occupancy classifier
imgs = glob.glob('tmp/images/street/*')
predictions = occupancyModel.predictMultipleImages(imgs)
