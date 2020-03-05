import math
import os
import shutil
import sys
from typing import Tuple

import numpy as np
from PIL import Image


class PreProcessImages:
    def __init__(self, targetDirRoot: str, imageFileExt="ppm"):
        self.__targetDirRoot = targetDirRoot
        self.__imageFileExt = imageFileExt

    @staticmethod
    def splitDataIntoTrainAndTest(rootDir, outDirRoot, percentage=(80, 20)):
        trainP, testP = percentage
        assert trainP + testP == 100, "Tuple must add up to 100"
        dirs = os.walk(rootDir)
        dirsToCreate = ["test", "train"]
        for dToC in dirsToCreate:
            x = outDirRoot + dToC
            if not os.path.exists(x):
                os.makedirs(x)

        for root, childDirs, files in dirs:
            subdir = root.split("/")[-1]
            if root != rootDir:
                np.random.shuffle(files)
                toTrain = int(len(files) * (trainP / 100))
                testDir = outDirRoot + "test/" + subdir
                trainDir = outDirRoot + "train/" + subdir
                if not os.path.exists(testDir):
                    os.makedirs(testDir)
                if not os.path.exists(trainDir):
                    os.makedirs(trainDir)
                for i in range(0, len(files)):
                    trainOrTestDir = trainDir if i <= toTrain else testDir
                    currentFile = files.pop(0)
                    currentFilePath = rootDir + subdir + "/" + currentFile
                    toMove = trainOrTestDir + "/" + currentFile
                    os.rename(currentFilePath, toMove)
        shutil.rmtree(rootDir)
        return outDirRoot + "train/", outDirRoot + "test/"

    def batchResize(self, outputDirRoot=None, outputTargetSize=(40, 40), keepAspectRatio=False, outFormat: str = "ppm"):
        assert outputDirRoot is not None, "Must provide output path"
        x, y = outputTargetSize
        sys.stdout.write('Resizing images to ' + str(x) + ' x ' + str(y) + ' \n')

        dirs = os.walk(self.__targetDirRoot)
        for root, childDirs, files in dirs:
            if root != self.__targetDirRoot:
                currentDir = root.split("/")[-1]
                for count, file in enumerate(files):
                    if file.split(".")[-1] == self.__imageFileExt:
                        inPath = root + "/" + file
                        outDir = outputDirRoot + currentDir
                        if not os.path.exists(outDir):
                            os.makedirs(outDir)
                        newFileExt = file.split(".")[0] + "." + outFormat
                        outPath = outDir + "/" + newFileExt
                        self.__resizeImage(inPath, outPath, outputTargetSize, keepAspectRatio)
                        self.__printProgress(currentDir, count, len(files))
            sys.stdout.write('\n')

    @staticmethod
    def __resizeImage(inPath: str, outPath: str, outputTargetSize: Tuple, keepAspectRatio: bool):
        img = Image.open(inPath)
        if keepAspectRatio:
            img.thumbnail(outputTargetSize)
        else:
            img = img.resize(outputTargetSize)
        img.save(outPath)

    @staticmethod
    def __printProgress(currentDir: str, currentFileNumber: int, maxFiles: float):
        percent = int(math.ceil(((currentFileNumber / maxFiles) * 100)))
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Processing Directory %s [%-20s] %d%%" % (currentDir, '=' * percent, percent))
        sys.stdout.flush()
