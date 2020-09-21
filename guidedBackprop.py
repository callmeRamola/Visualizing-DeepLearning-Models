import torch
import torch.nn as nn

class GuidedBackprop():
    """
    Visualize CNN activation Map using Backprop
    Result : An Image that represents what the network learnt for recognizing the given image
    Methods : First layer Input that maximizes the error between last layer output, for the given class and true label =1 
    !call visualize(image) to get image representation
    """
    
    def __init__(self, model):
        
        self.model = model
        self.image_reconstruction = None
        self.activation_maps = []
        
        
        self.model.eval()
        self.register_hooks()
        
        
    def register_hooks(self):
        
        def first_layer_hook_fn(module, grad_out, grad_in):
            """Returns reconstructed activation map"""
            self.image_reconstruction = grad_out[0]
            
        def forward_hook_fn(module, input, output):
            """Stores the forward pass activation maps"""
            self.activation_maps.append(output)
            
        def backward_hook_fn(module, grad_out, grad_in):
            """Outputs the grad of the model w.r.t layer (only positive)"""
            
            
            #Gradient of the forward output w.r.t forward input = error of the activation map
            #For Relu, grad of zero = 0, grad of identity = 1
            
            grad = self.activation_maps[-1] #coresponding to the forward pass output
            
            grad[grad > 0 ] = 1 
            
            #Set negative grads to zero
            positive_grad_out = torch.clamp(input = grad_out[0], min = 0.0)
            
            #backward grad_out = grad_out * grad of the forward output w.r.t forward input
            new_grad_out = positive_grad_out * grad
            
            #Delete the forward_output corresponding to the ReLu layer
            del self.activation_maps[-1]  
            
            return (new_grad_out,)
        
        """Change the module : Only Conv layers no flatted fc Linear layers """
        
        modules = list(self.model.features._modules.items())
        
        #register hooks to Relu Layers
        for name, module in modules:
            if isinstance(module, nn.ReLU):
                module.register_forward_hook(forward_hook_fn)
                module.register_backward_hook(backward_hook_fn)
                
        #Register hook to the first layer
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