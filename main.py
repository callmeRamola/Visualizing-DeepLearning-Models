from utils import convertTograyScale, getExampleParams, saveGradientImages
import torch

#from SaliencyMap import SaliencyMap
from guidedBackprop import GuidedBackprop



if __name__ == '__main__':
    
    #Get params 
    targetExample = 2 #Snake

    (origImg, normImg, targetClass, fileName, pretrainedModel) =\
        getExampleParams(targetExample)

    guidedBackprop = GuidedBackprop(pretrainedModel)

    firstLayerGrads = guidedBackprop.visualize(normImg, targetClass)

    #Save Colored Gradients
    saveGradientImages(firstLayerGrads, fileName + "_ColoredGuiddedBackPropMap")

    grayScaleGrads = convertTograyScale(firstLayerGrads)
    
    #Save Gray Scale Images
    saveGradientImages(grayScaleGrads, fileName + "_GrayScaledGuiddedBackPropMap")

    print("GuidedBackprop generated. Check your results folder")


