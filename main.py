from utils import convertTograyScale, getExampleParams, saveGradientImages
import torch

from SaliencyMap import SaliencyMap



if __name__ == '__main__':
    
    #Get params 
    targetExample = 2 #Snake

    (origImg, normImg, targetClass, fileName, pretrainedModel) =\
        getExampleParams(targetExample)

    saliencyMap = SaliencyMap(pretrainedModel)

    firstLayerGrads = saliencyMap.visualize(normImg, targetClass)

    #Save Colored Gradients
    saveGradientImages(firstLayerGrads, fileName + "_ColoredSaliencyMap")

    grayScaleGrads = convertTograyScale(firstLayerGrads)
    
    #Save Gray Scale Images
    saveGradientImages(grayScaleGrads, fileName + "_GrayScaledSaliencMap")

    print("Saliency Map generated. Check your results folder")


