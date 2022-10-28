from brails.modules import PytorchRoofClassifier, PytorchOccupancyClassifier
import glob

# Initialize a roof classifier:
roofModel = PytorchImageClassifier(modelName='transformer_rooftype_v1', imgDir='./roofType/')

# Retrain the BRAILS roof classifier model using the training dataset in roofType folder:
roofModel.fine_tuning(lr=0.001, batch_size=64, epochs=5)

# Use the new roof classifier to infer the roof class for each image contained in tmp/images/satellite/ folder:
imgs = glob.glob('tmp/images/satellite/*')
predictions = roofModel.predictMultipleImages(imgs)
