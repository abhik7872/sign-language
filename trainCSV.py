from PIL import Image
import numpy as np
import sys
import os
import csv

train_dir = os.path.expanduser('archive/Gesture Image Pre-Processed Data')

def createFileList(data):
    files = []
    labels=[]
    # r=root, d=directories, f = files
    for r, d, f in os.walk(data):
        for file in f:
            if '.jpg' in file:
                label=r.split('\\')[-1]
                labels.append(label)
                files.append(os.path.join(r,file))
                
    return files

trainFile = createFileList(train_dir)

for file in trainFile:
    print(file)
    img_file = Image.open(file)
    # img_file.show()

    # get original image parameters...
    width, height = img_file.size
    format = img_file.format
    mode = img_file.mode

    # Make image Greyscale
    img_grey = img_file.convert('L')
    #img_grey.save('result.png')
    #img_grey.show()

    # Save Greyscale values
    value = np.asarray(img_grey.getdata(), dtype=np.int).reshape((img_grey.size[1], img_grey.size[0]))
    value = value.flatten()
    print(value)
    with open("train.csv", 'a') as f:
        writer = csv.writer(f)
        writer.writerow(value)