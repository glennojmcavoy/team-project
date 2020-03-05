from classifier.models import BasicTensorFlowModel
from preprocess.script import PreProcessImages
import numpy as np
from PIL import Image, ImageEnhance
import classifier.util as util

if __name__ == '__main__':
    # p = PreProcessImages("data/GTSRB/Final_Training/Images")
    # p.batchResize(keepAspectRatio=False, outputTargetSize=(40, 40), outputDirRoot="data/processed/resized/jpg/",outFormat="jpg")
    """
    Loading a saved model  and predicting an  image from the test data 
    """
    savedModel = BasicTensorFlowModel().loadSavedModel()
    toPredict = util.readImageForPrediction("data/processed/resized/test/00040/00001_00028.jpg")
    res = savedModel.predict(toPredict)
    print(util.getPredictedLabel(util.predictedLabelToMap(res)))

    """
    Basic perturbation off changing brightness of image to miss classify the image 
    """
    newImg = Image.open("data/processed/resized/test/00040/00001_00028.jpg")
    edit = ImageEnhance.Brightness(newImg)
    edited = edit.enhance(0.12)
    edited.show()
    edited.save("Test.jpg")
    toPredict = util.readImageForPrediction("Test.jpg")
    res = savedModel.predict(toPredict)
    print(util.getPredictedLabel(util.predictedLabelToMap(res)))
