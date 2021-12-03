import torch
import torchvision.transforms as transforms
import numpy as np
import sklearn.covariance
from PIL import Image
from npid_models import resnet
import pandas as pd
import torch.nn.functional as F
from torch.autograd.variable import Variable



def resize2d(img, size):
    with torch.no_grad():
        return (F.adaptive_avg_pool2d(Variable(img), size)).data


class Distance_Eval(object):
    
    def __init__(self, checkpoint, low_dim=128):
        
        self.image_decorator = None
        model = resnet.__dict__['resnet50'](pretrained=False,low_dim=low_dim)
        model = torch.nn.DataParallel(model).cuda()
        checkpoint = torch.load(checkpoint)
        model.load_state_dict(checkpoint['state_dict'])

        lemniscate = checkpoint['lemniscate']            
        
        calc = True
    
        self.net = model
    
        self.net.eval()
        trainFeatures = lemniscate.memory.t()
        trainFeatures = trainFeatures.cpu().numpy()
        
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
    
        if calc == True:
            self.mu, self.precision = self._fitGaussian(trainFeatures)
            np.save("precision", self.precision)
            np.save("mu", self.mu)
        else:
            self.mu = np.load("mu.npy")
            self.precision = np.load("precision.npy")
    
    def _fitGaussian(self, features):
        mu = np.mean(features, axis=1)
    
        cov = sklearn.covariance.EmpiricalCovariance(store_precision=True, assume_centered=False)
        cov.fit(features.transpose())
        prec = cov.precision_
    
        return mu, prec
    
    def _mahalanobis(self, x, mu, precision):
        x = x.squeeze()
        distance = np.matmul(x - mu, np.matmul(precision, x - mu))
        return distance
    
    def get_distance_and_features(self, pil_image):
  
        if self.transform is not None:
            # resize to 224
            pil_image.convert('RGB')
            pil_image = self.transform(pil_image)            
        
        with torch.no_grad():
            img = torch.FloatTensor(pil_image).unsqueeze(0)            
            # img = resize2d(img,(224,224))
            features = self.net(img)
            features = features.cpu().numpy()
            
        return self._mahalanobis(features, self.mu, self.precision), features
    



if __name__ == '__main__':
    print ("Test")
    # test_path = '/data/PixelWise/BDD10k_clear_daytime/train/train/4c840aee-0dd28451.jpg'
    # test_path = '/data/PixelWise/BDD10k_clear_daytime/train/train/7171644d-4e74752e.jpg'
    # test_path = '/data/PixelWise/BDD10k_night/train/train/2a051b12-a7404163.jpg'
    # test_path = '/data/PixelWise/BDD10k_clear_daytime/train/train/9e86ec05-56319156.jpg'
    # test_path = '/data/PixelWise/BDD10k_clear_daytime/train/train/851ec515-15faca1d.jpg'
    # test_path = '/data/PixelWise/BDD10k_dusk/train/train/2a2d54fa-fd3fc827.jpg'
    # test_path = sys.argv[1]
    
    # Checkpoint no pedestrians
    evaluator = Distance_Eval('/data/PixelWise/NoPed_BDD100k_e100ch.pth.tar')
    
    # Checkpoint day night
    # evaluator = Distance_Eval('/data/PixelWise/cdt10k_e100ch.pth.tar')
    
    test_path = '/data/PixelWise/BDD100k_NoPed/WithPed/769136f0-17e25e15.jpg'
    evaluator.image_check(test_path)
