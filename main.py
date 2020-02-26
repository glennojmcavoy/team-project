from preprocess.script import PreProcessImages

if __name__ == '__main__':
    p = PreProcessImages("data/GTSRB/Final_Training/Images", outputDirRoot="data/resized/", keepAspectRatio=False, outputTargetSize=(40, 40))
    p.process()
