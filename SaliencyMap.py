from utils import convertTograyScale, getExampleParams, saveGradientImages
import torch
import torch.nn as nn


class SaliencyMap():
    """
    Saliency Map : Interpreting NN outputs

    Result : dScore(Input)/dInput : This indicates which pixels needs to be changed the
                                    least to affect the class scores the most

    Visualize the grad_out of the first layer
    """
    def __init__(self, model):
        self.model = model
        self.image_reconstruction = None
        
        self.model.eval()
        self.register_hooks()
        
    

    """
    This is the important part of the module. Focus here
    """
    def register_hooks(self):
        
        def first_layer_hook_fn(module, grad_out, grad_in):
            """Returns reconstructed activation map"""
            self.image_reconstruction = grad_out[0]
            
            
        #Name and corresponding Layer  
        modules = list(self.model.features._modules.items())  
        
        #Selecting the first layer and applying the hook
        first_layer = modules[0][1]
        first_layer.register_backward_hook(first_layer_hook_fn)
        

    """
      Output
    """
    def visualize(self, input_image, target_class):
        #last layer Output
        model_output = self.model(input_image)
        
        self.model.zero_grad()
        
        #Calculate the gradient w.r.t target class 
        #Set the other classes to zero. Eg. [0,0, 1]
        
        grad_target_map = torch.zeros(model_output.shape, dtype = torch.float)
        grad_target_map[0][target_class] = 1
        
        model_output.backward(grad_target_map)
        
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        result = self.image_reconstruction.data.numpy()[0] 
        return result
    
    
    
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
    
        
        