import math
import os
import sys

from PIL import Image


class PreProcessImages:
    def __init__(self, targetDirRoot: str, outputDirRoot=None, outputTargetSize=(40, 40), keepAspectRatio=False, imageFileExt="ppm"):
        assert outputDirRoot is not None, "Must provide output path"
        self.__targetDirRoot = targetDirRoot
        self.__outputDirRoot = outputDirRoot
        self.__keepAspectRatio = keepAspectRatio
        self.__outputTargetSize = outputTargetSize
        self.__imageFileExt = imageFileExt

    def process(self):
        dirs = os.walk(self.__targetDirRoot)
        for root, childDirs, files in dirs:
            if root != self.__targetDirRoot:
                currentDir = root.split("/")[-1]
                for count, file in enumerate(files):
                    if file.split(".")[-1] == self.__imageFileExt:
                        inPath = root + "/" + file
                        outDir = self.__outputDirRoot + currentDir
                        if not os.path.exists(outDir):
                            os.makedirs(outDir)
                        outPath = outDir + "/" + file
                        self.__resizeImage(inPath, outPath)
                        self.__printProgress(currentDir, count, len(files))
            sys.stdout.write('\n')
        return self.__outputDirRoot

    def __resizeImage(self, inPath, outPath):
        img = Image.open(inPath)

        if self.__keepAspectRatio:
            img.thumbnail(self.__outputTargetSize)
        else:
            img = img.resize(self.__outputTargetSize)
        img.save(outPath)

    def __printProgress(self, currentDir, currentFileNumber, maxFiles):
        percent = int(math.ceil(((currentFileNumber / maxFiles) * 100)))
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Processing Directory %s [%-20s] %d%%" % (currentDir, '=' * percent, percent))
        sys.stdout.flush()
