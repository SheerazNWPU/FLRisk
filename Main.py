import os
import numpy as np
import time
import sys
import torch, random
from torch.utils.data import Dataset
import pandas as pd
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from Densenet import densenet121
import torch.nn as nn
from Densenet import densenet169
from Densenet import densenet201
from Resnet import resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d, wide_resnet50_2, wide_resnet101_2
from Efficientnet import efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3, efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7
from PIL import Image
from Adaptive_Learning import FederatedServer
from Adaptive_Learning import FederatedClient

#--------------------------------------------------------------------------------   

seed = (int)(sys.argv[1])
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
torch.cuda.manual_seed(seed)


class CustomDataset(Dataset):
    def __init__(self, image_dir, metadata_file, ids_file, transform=None):
        self.metadata = pd.read_excel(metadata_file)
        self.image_ids = np.loadtxt(ids_file, dtype=str)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_name = self.image_ids[idx] + ".jpg"
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert('RGB')
    
        if self.transform:
            image = self.transform(image)
    
        label_er = self.metadata['ER'].map({'Positive': 1, 'Negative': 0}).values[idx]
        label_pr = self.metadata['PR'].map({'Positive': 1, 'Negative': 0}).values[idx]
        label_her2 = self.metadata['HER2'].map({'Positive': 1, 'Negative': 0}).values[idx]
    
        label = [label_er, label_pr, label_her2]
    
        return image, torch.FloatTensor(label), image_path

def Start_Adapt():

    # ---- Neural network parameters: type of the network, is it pre-trained
    # ---- on imagenet, number of classes
    nnIsTrained = True
    nnClassCount = 3  # 14
    num_rounds = 10
    
    
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    transformList = []
    transformList.append(transforms.RandomResizedCrop(224))
    transformList.append(transforms.RandomHorizontalFlip())
    transformList.append(transforms.ToTensor())
    transformList.append(normalize)
    train_transform = transforms.Compose(transformList)
    
    
    
    
    transformList = [
    transforms.Resize(256),
    transforms.FiveCrop(224),  # Generates 5 crops
    transforms.Lambda(lambda crops: transforms.ToTensor()(crops[0])),  # Take only the first crop
    transforms.Lambda(lambda crop: normalize(crop))  # Normalize the single crop
    ]
    transform_test = transforms.Compose(transformList)

    
    train_dataset_1 = CustomDataset(image_dir='/home/15t/Gul/Datasets/Combined_BRCA/WSI/',
                               metadata_file='/home/15t/Gul/Datasets/Combined_BRCA/combined_labels.xlsx',
                               ids_file='/home/15t/Gul/Datasets/Combined_BRCA/dataset-split/train_ids.txt',
                               transform=train_transform)

    val_dataset_1 = CustomDataset(image_dir='/home/15t/Gul/Datasets/Combined_BRCA/WSI/',
                                 metadata_file='/home/15t/Gul/Datasets/Combined_BRCA/combined_labels.xlsx',
                                 ids_file='/home/15t/Gul/Datasets/Combined_BRCA/dataset-split/val_ids.txt',
                                 transform=transform_test)
    
    test_dataset_1 = CustomDataset(image_dir='/home/15t/Gul/Datasets/Combined_BRCA/WSI/',
                                  metadata_file='/home/15t/Gul/Datasets/Combined_BRCA/combined_labels.xlsx',
                                  ids_file='/home/15t/Gul/Datasets/Combined_BRCA/dataset-split/test_ids.txt',
                                  transform=transform_test)
    
    train_dataset_2 = CustomDataset(image_dir='/home/15t/Gul/Datasets/BCNB/WSIs/',
                               metadata_file='/home/15t/Gul/Datasets/BCNB/patient-clinical-data.xlsx',
                               ids_file='/home/15t/Gul/Datasets/BCNB/dataset-splitting/train_id.txt',
                               transform=train_transform)

    val_dataset_2 = CustomDataset(image_dir='/home/15t/Gul/Datasets/BCNB/WSIs/',
                                 metadata_file='/home/15t/Gul/Datasets/BCNB/patient-clinical-data.xlsx',
                                 ids_file='/home/15t/Gul/Datasets/BCNB/dataset-splitting/val_id.txt',
                                 transform=transform_test)
    
    test_dataset_2 = CustomDataset(image_dir='/home/15t/Gul/Datasets/BCNB/WSIs/',
                                  metadata_file='/home/15t/Gul/Datasets/BCNB/patient-clinical-data.xlsx',
                                  ids_file='/home/15t/Gul/Datasets/BCNB/dataset-splitting/test_id.txt',
                                  transform=transform_test)
    train_loader_1 = DataLoader(train_dataset_1, batch_size=32, shuffle=True, num_workers=8)
    val_loader_1 = DataLoader(val_dataset_1, batch_size=32, shuffle=False, num_workers=8)
    test_loader_1 = DataLoader(test_dataset_1, batch_size=32, shuffle=False, num_workers=8)
    
    train_loader_2 = DataLoader(train_dataset_2, batch_size=32, shuffle=True, num_workers=8)
    val_loader_2 = DataLoader(val_dataset_2, batch_size=32, shuffle=False, num_workers=8)
    test_loader_2 = DataLoader(test_dataset_2, batch_size=32, shuffle=False, num_workers=8)

    
    model_zoo = {'r18':resnet18, 'r34':resnet34, 'r50':resnet50, 'r101':resnet101, 'r152':resnet152, 
              ##'d121':densenet121, 'd161':densenet161, 'd169':densenet169, 'd201':densenet201,
              ##'v11':vgg11, 'v13':vgg13, 'v16':vgg16, 'v19':vgg19,
             'eb0':efficientnet_b0, 'eb1':efficientnet_b1, 'eb2':efficientnet_b2, 'eb3':efficientnet_b3,
             'eb4':efficientnet_b4, 'eb5':efficientnet_b5, 'eb6':efficientnet_b6,  'eb7':efficientnet_b7,
             'rx50':resnext50_32x4d, 'wrn50':wide_resnet50_2, 'wrn101':wide_resnet101_2}  
    
    #select the model
    cnn = 'r50'
    nb_class = 3
    global_model = model_zoo[cnn](pretrained=True)
    val_num = 10
    batch_size = 16
    transResize = 224
    transCrop = 224
    if cnn.startswith("r"):
        #global_model.backbone = nn.Sequential(*list(global_model.children())[:-1])  # all layers except final FC
        global_model.fc = nn.Linear(global_model.fc.in_features, nb_class)
        # net.fc = nn.Sequential(nn.Dropout(p=args.drop_out), nn.Linear(net.fc.in_features, nb_class))  # for resnet
    elif cnn.startswith('w'):
        global_model.fc = nn.Linear(global_model.fc.in_features, nb_class)
    elif cnn.startswith('v'):
        global_model.classifier = nn.Linear(global_model.classifier[0].in_features, nb_class) # for VGG
    elif cnn.startswith('d'):
        global_model.classifier = nn.Linear(global_model.classifier.in_features, nb_class)
        # net.classifier = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(net.classifier.in_features, nb_class))  # for densenet
    elif cnn.startswith('e'):
        global_model.classifier._modules['1'] = nn.Linear(global_model.classifier._modules['1'].in_features, nb_class)
    global_model.cuda()
    
    net_file_name = '/home/15t/Gul/SG/sheeraz/result_archive/Multilabel_FEDSpear_R50/max_f1_0.68.pth'
    try:
            try: global_model.load_state_dict(torch.load(net_file_name)['model'])
            except: global_model.load_state_dict(torch.load(net_file_name))
    except:
            global_model = torch.nn.DataParallel(global_model)
            try: global_model.load_state_dict(torch.load(net_file_name)['model'])
            except: global_model.load_state_dict(torch.load(net_file_name))
    
    global_model.cuda()
    global_model.eval()

    store_name = 'Multilabel_FEDSpear_R'
    save_name = os.path.join('/home/15t/Gul/SG/sheeraz/risk_val_pmg_result/', str(val_num), store_name.split('/')[-1], str(seed))
    client_path = '/home/15t/Gul/SG/sheeraz/result_archive/Multilabel_FEDSpear_R50/'
    
    client1 = FederatedClient(train_loader_1, val_loader_1, test_loader_1, save_name, client_path,  device='cuda')
    client2 = FederatedClient(train_loader_2, val_loader_2, test_loader_2, save_name, client_path, device='cuda')
    
    
    server = FederatedServer(global_model, [client1, client2],  nb_class, save_name)
    
    server.train(num_rounds)
    

# --------------------------------------------------------------------------------


# --------------------------------------------------------------------------------

if __name__ == '__main__':
 Start_Adapt() 


