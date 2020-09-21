from PIL import Image 
import os 
import torch

import numpy as np
from torchvision import models 
from torch.autograd import Variable 



def convertTograyScale(imgArr):
    """
    Convert 3D Image to GrayScale Image

    Args : 
        imgArr (Numpy Array): RGB Image CxHXW

    returns:
        grayScalImg (Numpy Array) : Gray

    """

    grayScaleImg = np.sum(np.abs(imgArr), axis= 0)
    imgMax = np.percentile(grayScaleImg, 99)
    imgMin = np.min(grayScaleImg)

    #Limits the value between 0-1
    grayScaleImg = (np.clip((grayScaleImg - imgMin)/(imgMax - imgMin), 0, 1))

    grayScaleImg = np.expand_dims(grayScaleImg, axis = 0)

    return grayScaleImg

def formatNumpyoutput(npArr):

    #case1 : The numpy array has only two dimensions
    #Result : Add a dimension at the beginning

    if len(npArr) == 2:
        npArr = np.expand_dims(npArr, axis = 0)

    #case2 : The numpy array has only one channel
    #Result : Repeat the first channel and convert 1xHxW to 3xHxw

    if npArr.shape[0] == 1:
        npArr = np.repeat(npArr, 3, axis = 0)

    #case3 : The numpy array is of shape 3xHxW
    #Result: Convert it into HXWX3
    if npArr.shape[0] == 3:
        npArr =  npArr.transpose(1, 2, 0)


    #case4 : The numpy array is normalised between 0-1
    #Result: Multiply by 255 and change type to make it savable by PIL
    if np.max(npArr) <= 1:
        npArr = (npArr * 255).astype(np.uint8)

    return npArr

def saveImage(img, imgPath):
    
    if isinstance(img, (np.ndarray, np.generic)):
        img = formatNumpyoutput(img)
        img = Image.fromarray(img)
    img.save(imgPath)


def saveGradientImages(gradient, filename):

    resultsDir = os.getcwd() +"\\results\\"
    if not os.path.exists(resultsDir):
        os.makedirs(resultsDir)

    #Normalise
    gradient = gradient - gradient.min()
    gradient /=gradient.max()

    #Save Image
    imgSavePath = os.path.join(resultsDir , filename + ".jpg")
    saveImage(gradient, imgSavePath)



def preProcessImg(pilImg, resizeImg = True):

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if type(pilImg) != Image.Image:
        try:
            pilImg = Image.fromarray(pilImg)
        except Exception as e:
            print("Error in Loading Image")

    
    #Resize Image
    if resizeImg:
        pilImg = pilImg.resize((224, 224), Image.ANTIALIAS)

    arrayImg = np.float32(pilImg)

    # H W C --> C H W
    arrayImg = arrayImg.transpose(2, 0, 1) 

    #Normalise the Channels
    for channel, _ in enumerate(arrayImg):
        arrayImg[channel]/= 255
        arrayImg[channel]-= mean[channel]
        arrayImg[channel] /= std[channel]


    tensorImg = torch.from_numpy(arrayImg).float()

    #Add one more channel to the beginning. Tensor.shape = [1, 3, 224, 224]
    tensorImg.unsqueeze_(0)

    #Convert to Pytorch Variable
    varImg =  Variable(tensorImg, requires_grad = True)

    return varImg



def getExampleParams(example_index):

   # Pick one of the examples
    inputFolderPath = os.getcwd() + "\\input_images\\"
    example_list = (('snake.jpg', 56),
                    ('cat_dog.png', 243),
                    ('spider.png', 72))


    imgPath = inputFolderPath + example_list[example_index][0]
    targetClass = example_list[example_index][1]
    fileName = example_list[example_index][0].split(".")[0]
    
    #Read the original Image
    origImg = Image.open(imgPath).convert('RGB')

    #Normalise the Image
    normImg = preProcessImg(origImg)

    #Define a model
  
    pretrainedModel = models.alexnet(pretrained= True)

    return (origImg, normImg, targetClass, fileName, pretrainedModel)







