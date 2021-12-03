# Author: Barbaros Cetiner

from lib.train_detector import Detector
import argparse

def get_args():
    parser = argparse.ArgumentParser('EfficientDet-based chimney detection model')
    parser.add_argument('-c', '--compound_coef', type=int, default=4,
                        help='Compund coefficient for the EfficientDet backbone, e.g., enter 7 for EfficientDet-D7') 
    parser.add_argument('-n', '--num_workers', type=int, default=0,
                        help='Number of workers of Dataloader')
    parser.add_argument('--head_only', type=boolean_string, default=False,
                        help='True if desired to finetune the regressor and the classifier only,'
                             'useful in early stage convergence or small/easy dataset')
    parser.add_argument('--num_gpus', type=int, default=1, 
                        help='Number of GPUs (Enter 0 for CPU-based training)')    
    parser.add_argument('--optim', type=str, default='adamw',
                        help='Select optimizer for training, '
                             'Use \'adamw\' until the last stage'
                             'then switch to \'sgd\'')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='The number of images per batch')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=25, help='Number of training epochs')
    parser.add_argument('--data_path', type=str, default='datasets/',
                        help='Path for the root folder of dataset')
    parser.add_argument('--val_interval', type=int, default=1,
                        help='Number of epoches between validating phases')
    parser.add_argument('--save_interval', type=int, default=5,
                        help='Number of epoches between model saving')
    parser.add_argument('--es_min_delta', type=float, default=0.0,
                        help='Early stopping parameter: Minimum change in loss to qualify as an improvement')
    parser.add_argument('--es_patience', type=int, default=0,
                        help='Early stopping parameter: Number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.')
    parser.add_argument('--customModel_path', type=str, 
                        default='models/efficientdet-d4_trained.pth',
                        help='Path for the custom pretrained model')

    args = parser.parse_args()
    return args

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean entry')
    return s == 'True'

def train(opt):
    # Create the Object Detector Object
    gtf = Detector()

    # Set the Training and Validation Datasets
    train_dir = "train"; val_dir = "valid"
    classes = ["chimney"]
    
    gtf.set_train_dataset(opt.data_path,"","",train_dir,classes_list=classes,
                          batch_size=opt.batch_size, num_workers=opt.num_workers)
    
    gtf.set_val_dataset(opt.data_path,"","",val_dir)
    
    # Define the Model Architecture
    modelArchitecture = f"efficientdet-d{opt.compound_coef}.pth" ## Figure out the customModel_path
    
    gtf.set_model(model_name=modelArchitecture, num_gpus=opt.num_gpus,
                  freeze_head=opt.head_only)
    
    # Set Model Hyperparameters    
    gtf.set_hyperparams(optimizer=opt.optim, lr=opt.lr,
                        es_min_delta=opt.es_min_delta, es_patience=opt.es_patience)
    
    # Train    
    gtf.train(num_epochs=opt.num_epochs, val_interval=opt.val_interval,
              save_interval=opt.save_interval)

if __name__ == '__main__':
    opt = get_args()
    train(opt)