from __future__ import print_function
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,0"
from glob import glob
from os.path import join
import pandas as pd
import numpy as np
import time
import sys
import copy
from sklearn.metrics import multilabel_confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from  Folder import ImageFolder
from torch.utils.data import DataLoader, ConcatDataset
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from tqdm import tqdm
import torch, random
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
from PIL import Image
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, hamming_loss
import torch.distributed as dist
from Densenet import densenet121
from Densenet import densenet169
from Densenet import densenet201
from Resnet import resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d, wide_resnet50_2, wide_resnet101_2
from Efficientnet import efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3, efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7
from PIL import Image
import logging
import random
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn as nn
from utils import *
from torch.utils.data import Dataset
from risk_one_rule import risk_dataset
from risk_one_rule import risk_torch_model
import risk_one_rule.risk_torch_model as risk_model
from common import config as config_risk
from torch.nn.functional import softmax, sigmoid
from scipy.special import softmax

import csv

cfg = config_risk.Configuration(config_risk.global_data_selection, config_risk.global_deep_learning_selection)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

"""Seed and GPU setting"""
seed = (int)(sys.argv[1])
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
torch.cuda.manual_seed(seed)

cudnn.benchmark = True
cudnn.deterministic = True

## Allow Large Images
Image.MAX_IMAGE_PIXELS = None

def hamming_loss(y_true, y_pred):
    # Ensure y_true and y_pred are tensors
    y_true = torch.tensor(y_true)
    y_pred = torch.tensor(y_pred)

    # Convert predictions to binary if they are probabilities
    y_pred = (y_pred > 0.5).float()

    # Calculate the Hamming loss
    loss = torch.mean((y_true != y_pred).float())
    return loss

class DropConnectWrapper(nn.Module):
    def __init__(self, module, drop_prob):
        super(DropConnectWrapper, self).__init__()
        self.module = module
        self.drop_prob = drop_prob

    def forward(self, x):
        # Drop connections with probability drop_prob
        if self.training:
            mask = torch.bernoulli(torch.ones_like(self.module.weight) * (1 - self.drop_prob))
            weight = self.module.weight * mask
        else:
            weight = self.module.weight

        out = F.linear(x, weight, self.module.bias)
        return out
# Define the dataset class
class LabelDependencyGCN(nn.Module):
    def __init__(self, num_labels, hidden_dim=64):
        super(LabelDependencyGCN, self).__init__()
        self.conv1 = GCNConv(num_labels, hidden_dim)  # First GCN layer
        self.conv2 = GCNConv(hidden_dim, num_labels)  # Second GCN layer (back to label space)
    
    def forward(self, logits, edge_index):
        """
        Refines CNN logits using label dependencies.
        
        Args:
            logits (torch.Tensor): CNN output logits of shape [batch_size, num_labels].
            edge_index (torch.Tensor): Edge list tensor defining label dependencies.

        Returns:
            torch.Tensor: Refined logits of shape [batch_size, num_labels].
        """
        x = F.relu(self.conv1(logits, edge_index))  # Apply GCN layer with ReLU activation
        x = self.conv2(x, edge_index)  # Apply second GCN layer
        return x
class LabelTransformer(nn.Module):
    def __init__(self, num_labels, hidden_dim=64, num_heads=4, num_layers=2):
        super(LabelTransformer, self).__init__()
        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers
        )
        self.embedding = nn.Linear(num_labels, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, num_labels)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x, x)
        refined_output = self.output_layer(x)
        return refined_output

class AdaptiveLabelLearningRateScheduler:
    def __init__(self, optimizer, initial_lr, label_count, patience=2, factor=0.5):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.label_count = label_count
        self.patience = patience
        self.factor = factor
        self.performance_history = {label: [] for label in range(label_count)}

    def step(self, label_performance):
        for label in range(self.label_count):
            self.performance_history[label].append(label_performance[label])
            if len(self.performance_history[label]) > self.patience:
                self.performance_history[label].pop(0)
            # Example condition: if performance has not improved, reduce the learning rate
            if len(self.performance_history[label]) == self.patience:
                if self.performance_history[label][-1] < min(self.performance_history[label]):
                    new_lr = self.optimizer.param_groups[label]['lr'] * self.factor
                    #adaptive LR
                    print(new_lr)
                    self.optimizer.param_groups[label]['lr'] = max(new_lr, 1e-6)

    def reset(self):
        for label in range(self.label_count):
            self.optimizer.param_groups[label]['lr'] = self.initial_lr[label]

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
        
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        
        # Create a tensor for true distribution
        true_dist = torch.full_like(pred, self.smoothing / (self.cls - 1))
        
        # Update only the indices of the target
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

# Define the RegularizedLoss class
class RegularizedLoss(nn.Module):
    def __init__(self, base_loss_fn, co_occurrence_matrix, lambda_reg=0.1):
        super(RegularizedLoss, self).__init__()
        self.base_loss_fn = base_loss_fn
        self.co_occurrence_matrix = co_occurrence_matrix
        self.lambda_reg = lambda_reg

    def forward(self, outputs, targets):
        base_loss = self.base_loss_fn(outputs, targets)
        regularization_term = self.lambda_reg * self.compute_regularization(outputs, targets)
        return base_loss + regularization_term

    def compute_regularization(self, outputs, targets):
        # Ensure outputs and targets are 2D tensors with shape (batch_size, num_classes)
        if outputs.dim() == 1:
            outputs = outputs.unsqueeze(0)
        if targets.dim() == 1:
            targets = targets.unsqueeze(0)
    
        batch_size = outputs.size(0)
        reg_loss = 0.0
    
        # Iterate over all pairs of labels
        for i in range(self.co_occurrence_matrix.size(0)):
            for j in range(self.co_occurrence_matrix.size(1)):
                reg_loss += self.co_occurrence_matrix[i, j] * torch.sum(targets[:, i] * outputs[:, j])
    
        # Normalize by batch size
        return reg_loss / batch_size


def compute_and_normalize_co_occurrence_matrix(train_labels):
    """
    Computes and normalizes the co-occurrence matrix from the training labels.

    Args:
        train_labels (torch.Tensor): Tensor of shape (num_samples, num_classes) with binary labels.

    Returns:
        torch.Tensor: Normalized co-occurrence matrix.
    """
    #train_labels_tensor = torch.tensor(train_labels, dtype=torch.float32)
    # Calculate co-occurrence matrix by matrix multiplication
    co_occurrence_matrix = torch.mm(train_labels.T, train_labels)
    
    # Normalize by the maximum value in the matrix to get values between 0 and 1
    normalized_co_occurrence_matrix = co_occurrence_matrix / co_occurrence_matrix.max()
    
    return normalized_co_occurrence_matrix

def output_risk_scores(file_path, id_2_scores, label_index, ground_truth_y, predict_y):
    op_file = open(file_path, 'w+', 1, encoding='utf-8')
    #print(op_file)
    for i in range(len(id_2_scores)):
        #print("CHECK")
        _id = id_2_scores[i][0]
        _risk = id_2_scores[i][1]
        _label_index = label_index.get(_id)
        _str = "{}, {}, {}, {}".format(ground_truth_y[_label_index],
                                       predict_y[_label_index],
                                       _risk,
                                       _id)
        op_file.write(_str + '\n')
    op_file.flush()
    op_file.close()
    return True
def collect_true_labels(dataloader):
    all_labels = []
    for batch_idx, (inputs, targets, paths) in enumerate(dataloader):
        all_labels.append(targets.numpy())  # Collect targets from each batch
    return np.concatenate(all_labels, axis=0)  # Concatenate all labels into one array

def prepare_data_4_risk_data(client_name):
    """
    first, generate , include all_info.csv, train.csv, val.csv, test.csv.
    second, use csvs to generate rules. one rule just judge one class
    :return:
    """
    #print('hello')
    #print(cfg)
    train_data, validation_data, test_data = risk_dataset.load_data(cfg, client_name)
    return train_data, validation_data, test_data

def prepare_data_4_risk_model(train_data, validation_data, test_data):

    rm = risk_torch_model.RiskTorchModel()
    rm.train_data = train_data
    rm.validation_data = validation_data
    rm.test_data = test_data
    return rm

# --------------------------------------------------------------------------------

class FederatedServer():
    def __init__(self, global_model, clients, class_num, save_name):
        self.global_model = global_model
        self.clients = clients
        self.class_num = class_num
        self.max_val_f1 = 0.0
        self.save_name = save_name
    def aggregate(self, client_models):
        global_dict = self.global_model.state_dict()
        for key in global_dict:
            global_dict[key] = torch.stack([client_models[i].state_dict()[key].float() for i in range(len(client_models))], 0).mean(0)
        self.global_model.load_state_dict(global_dict)
    
    def test(self):
        self.global_model.eval()
        y_true, y_pred, y_score = [], [], []
        for client in self.clients:
            client_y_true, client_y_pred, client_y_score, _, _ = client.test(self.global_model)
            y_true.extend(client_y_true)
            y_pred.extend(client_y_pred)
            y_score.extend(client_y_score)
        self.calculate_metrics(y_true, y_pred, y_score)

    def calculate_metrics(self, y_true, y_pred, y_score):
        nb_class = 3
        y_true = np.concatenate(y_true, axis=0)
        y_pred = np.concatenate(y_pred, axis=0)
        y_score = np.concatenate(y_score, axis=0)
        if y_true.ndim == 1:
            print("Warning: y_true is 1D. Reshaping to (num_samples, num_labels)")
            y_true = y_true.reshape(-1, nb_class)  # Assuming nb_class is the number of labels
        
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, nb_class)
        
        if y_score.ndim == 1:
            y_score = y_score.reshape(-1, nb_class)
        predictions_binary = (y_pred > 0.5).astype(int)
        true_labels_binary = y_true.astype(int)
        print(predictions_binary.shape)
        print(true_labels_binary.shape)
        num_labels = 3

        test_accuracies = []
        test_precisions = []
        test_recalls = []
        test_f1_scores = []
        test_aucs = []

        image_wise_accuracies = []
        image_wise_precisions = []
        image_wise_recalls = []
        image_wise_f1_scores = []
        image_wise_aucs = []

        print("\n===== Label-wise Metrics =====")
        print("Label\tAccuracy\tPrecision\tRecall\tF1-Score\tAUC\t\tStd(Acc)\tStd(Prec)\tStd(Rec)\tStd(F1)\tStd(AUC)")

        # Loop through each label
        for label in range(num_labels):
            # Calculate label-wise accuracy, precision, recall, F1 score, and AUC
            label_accuracy = np.mean(predictions_binary[:, label] == true_labels_binary[:, label])
            label_precision = precision_score(true_labels_binary[:, label], predictions_binary[:, label], zero_division=1)
            label_recall = recall_score(true_labels_binary[:, label], predictions_binary[:, label], zero_division=1)
            label_f1_score = f1_score(true_labels_binary[:, label], predictions_binary[:, label], zero_division=1)
        
            # Check if the true labels for the current label contain both 0 and 1 (i.e., not a constant label)
            unique_true_labels = np.unique(true_labels_binary[:, label])
        
            if len(unique_true_labels) > 1:
                # Calculate label-wise AUC (using predicted probabilities)
                label_auc = roc_auc_score(true_labels_binary[:, label], y_score[:, label])
            else:
                # If the label is constant (all 0 or all 1), set AUC to NaN
                label_auc = np.nan
        
            # Append label metrics to the lists
            test_accuracies.append(label_accuracy)
            test_precisions.append(label_precision)
            test_recalls.append(label_recall)
            test_f1_scores.append(label_f1_score)
            test_aucs.append(label_auc)
        
            # Calculate per-image metrics for each image in the batch
            per_image_accuracies = np.mean(predictions_binary[:, label] == true_labels_binary[:, label], axis=0)
            per_image_precisions = [precision_score(true_labels_binary[i, :], predictions_binary[i, :], zero_division=1, average='macro') for i in range(true_labels_binary.shape[0])]
            per_image_recalls = [recall_score(true_labels_binary[i, :], predictions_binary[i, :], zero_division=1, average='macro') for i in range(true_labels_binary.shape[0])]
            per_image_f1_scores = [f1_score(true_labels_binary[i, :], predictions_binary[i, :], zero_division=1, average='macro') for i in range(true_labels_binary.shape[0])]
            
            # Calculate per-image AUCs (using predicted probabilities for each image)
            per_image_aucs = [roc_auc_score(true_labels_binary[i, :], y_score[i, :]) if len(np.unique(true_labels_binary[i, :])) > 1 else np.nan for i in range(true_labels_binary.shape[0])]
        
            # Append per-image metrics (std dev) to the lists
            image_wise_accuracies.append(np.std(per_image_accuracies))
            image_wise_precisions.append(np.std(per_image_precisions))
            image_wise_recalls.append(np.std(per_image_recalls))
            image_wise_f1_scores.append(np.std(per_image_f1_scores))
            image_wise_aucs.append(np.nanstd(per_image_aucs))  # Use np.nanstd to ignore NaN values in std calculation
            # Print label-wise results
            print(f"{label+1}\t{label_accuracy:.4f}\t{label_precision:.4f}\t{label_recall:.4f}\t{label_f1_score:.4f}\t"
              f"{label_auc:.4f}\t{np.std(per_image_accuracies):.4f}\t{np.std(per_image_precisions):.4f}\t{np.std(per_image_recalls):.4f}\t"
              f"{np.std(per_image_f1_scores):.4f}\t{np.nanstd(per_image_aucs):.4f}")


        test_hamming_loss = hamming_loss(true_labels_binary, predictions_binary)
        test_accuracy_mean = np.mean(test_accuracies)
        test_accuracy_std = np.mean(image_wise_accuracies)
        test_precision_mean = np.mean(test_precisions)
        test_precision_std = np.mean(image_wise_precisions)
        test_recall_mean = np.mean(test_recalls)
        test_recall_std = np.mean(image_wise_recalls)
        test_f1_mean = np.mean(test_f1_scores)
        test_f1_std = np.mean(image_wise_f1_scores)
        test_auc_mean = np.nanmean(test_aucs)
        test_auc_std = np.nanmean(image_wise_aucs)

        print("\n===== Overall Results =====")
        print("Metric\t\tMean\t\tStd Dev")
        print(f"Accuracy\t{test_accuracy_mean:.4f}\t{test_accuracy_std:.4f}")
        print(f"Precision\t{test_precision_mean:.4f}\t{test_precision_std:.4f}")
        print(f"Recall\t\t{test_recall_mean:.4f}\t{test_recall_std:.4f}")
        print(f"F1-Score\t{test_f1_mean:.4f}\t{test_f1_std:.4f}")
        print(f"AUC\t\t{test_auc_mean:.4f}\t{test_auc_std:.4f}")
        print(f"Hamming Loss\t{test_hamming_loss:.4f}\tN/A")
        # At the end, load the best model state if needed
    def train(self, num_rounds):
        save_name = self.save_name
        class_num = self.class_num
        clients = self.clients
    
        print(save_name)
        if not os.path.exists(save_name):
            os.makedirs(save_name)

        

        use_cuda = torch.cuda.is_available()
        model = self.global_model
        model.cuda()
        lr_begin = 0.0005
        optimizer = torch.optim.SGD(model.parameters(), lr=lr_begin, momentum=0.9, weight_decay=5e-4)
        adaptive_lr_scheduler = AdaptiveLabelLearningRateScheduler(optimizer=optimizer, initial_lr=[lr_begin]*class_num, label_count=class_num)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        max_test_acc=0
        self.test()
        for epochID in range(0, num_rounds):
            client_models = []
            for idx, client in enumerate(clients, start=1):  # Use existing client instances
                _, train_pre = client.train_t(self.global_model)  # Use train_t on each client
                _, val_pre = client.Valtest(self.global_model)  # Validation
                y_true_test, y_pred_test, y_score_test, test_f1 ,test_pre= client.test(self.global_model)  # Testing
        
                client_name = f"client_{idx}"  # Generate client name dynamically
                train_data, val_data, test_data = prepare_data_4_risk_data(client_name)
                
                risk_data = [train_data, val_data, test_data]
                my_risk_model = prepare_data_4_risk_model(risk_data[0], risk_data[1], risk_data[2])
                # Initialize empty tensors
                train_one_pre = torch.empty((0, class_num), dtype=torch.float64)
                val_one_pre = torch.empty((0, class_num), dtype=torch.float64)
                test_one_pre = torch.empty((0, class_num), dtype=torch.float64)
                
                # Function to process predictions
                def process_predictions(predictions):
                    # Extract the maximum values for each pair of predictions
                    max_values = torch.stack([torch.max(predictions[:, i:i+2], dim=1).values for i in range(0, 6, 2)], dim=1)
                    return max_values
                
                # Function to process labels
                def process_labels(predictions):
                    # Compare pairs and assign 1 or 0 based on which value is greater
                    labels = torch.stack([(predictions[:, i] > predictions[:, i+1]).long() for i in range(0, 6, 2)], dim=1)
                    return labels
                
                # Process train, validation, and test predictions
                train_max_values = process_predictions(train_pre)
                val_max_values = process_predictions(val_pre)
                test_max_values = process_predictions(test_pre)
                
                # Concatenate the processed predictions
                train_one_pre = torch.cat((train_one_pre, train_max_values), dim=0).cpu().numpy()
                val_one_pre = torch.cat((val_one_pre, val_max_values), dim=0).cpu().numpy()
                test_one_pre = torch.cat((test_one_pre, test_max_values), dim=0).cpu().numpy()
                
                # Process train, validation, and test labels
                train_labels = process_labels(train_pre)
                val_labels = process_labels(val_pre)
                test_labels = process_labels(test_pre)
                
                # Move labels to the appropriate device (assuming device is defined)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                train_labels = train_labels.to(device)
                val_labels = val_labels.to(device)
                test_labels = test_labels.to(device)
                
                my_risk_model.train(train_one_pre, val_one_pre, test_one_pre, train_pre.cpu().numpy(),
                                         val_pre.cpu().numpy(),
                                         test_pre.cpu().numpy(), train_labels, val_labels, test_labels, epochID)
                my_risk_model.predict(test_one_pre, test_pre.cpu().numpy(), )
                
                # Save model before fine-tuning
                prev_model = copy.deepcopy(self.global_model)
        
                # Compute average confidence before fine-tuning
                before_confidence = torch.sigmoid(test_f1).mean().item()
                client_model = client.adaptive_fine_tune(my_risk_model, test_labels, test_one_pre, train_pre, val_pre, test_pre, client_name, class_num, epochID, self.global_model)  # Adaptive training
                # Re-test the model after fine-tuning
                _, _, _, test_f1_after, test_pre_after = client.test(client_model)
        
                # Compute average confidence after fine-tuning
                after_confidence = torch.sigmoid(test_pre_after).mean().item()
        
                # Decide whether to keep or rollback
                if after_confidence >= before_confidence:
                    print(f"[{client_name}] Fine-tuning kept (confidence improved: {before_confidence:.4f} ? {after_confidence:.4f})")
                else:
                    print(f"[{client_name}] Rolling back (confidence dropped: {before_confidence:.4f} ? {after_confidence:.4f})")
                    client_model = prev_model  # rollback to previous model

                
                
                client_models.append(client_model)
            
            # Aggregate models after training
            self.aggregate(client_models)
            self.test()




class FederatedClient():
    
    def __init__(self, train_loader, val_loader, test_loader, save_name, client_path, device):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.save_name = save_name
        self.client_path = client_path 
        self.device = device
    
    def train(self, global_model):
        local_model = copy.deepcopy(global_model).to(self.device)
        local_model.train()
        optimizer = optim.SGD(local_model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        criterion = nn.BCEWithLogitsLoss()
    
    def load_name_mapping(self, map_path):
        mapping_df = pd.read_csv(map_path, header=None)
        name_map = {}
        
        for _, row in mapping_df.iterrows():
            anonymized_name = row[0]
            original_path = row[1]
            # Get just the filename from the original path
            original_filename = os.path.basename(original_path)
            name_map[original_filename] = anonymized_name
        
        return name_map
    # --------------------------------------------------------------------------------


    def adaptive_fine_tune(self, my_risk_model, test_labels, test_one_pre, train_pre, val_pre, test_pre, client_name, class_num, epochID, global_model):
        
        
        use_cuda = torch.cuda.is_available()
        model = global_model
        optimizer = optim.SGD(model.parameters(), lr=0.0005, momentum=0.9, weight_decay=5e-4)
        max_test_f1 = 0
        #class_num = train_pre.shape[1]
        LSLLoss = LabelSmoothingLoss(class_num, 0.1)
        LSLLoss1 = LabelSmoothingLoss(class_num, 0.1)
        LSLLoss2 = LabelSmoothingLoss(class_num, 0.1) 
                
        save_name = self.save_name
                
        # setup output
        exp_dir = save_name
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            os.stat(exp_dir)
        except:
            os.makedirs(exp_dir)
                # ---- TRAIN THE NETWORK
        
        test_num = my_risk_model.test_data.data_len
        test_ids = my_risk_model.test_data.data_ids
        test_pred_y = test_labels
        test_true_y = my_risk_model.test_data.true_labels
        risk_scores = my_risk_model.test_data.risk_values
        
        
        id_2_label_index = dict()
        id_2_VaR_risk = []
        for i in range(test_num):
            id_2_VaR_risk.append([test_ids[i], risk_scores[i]])
            id_2_label_index[test_ids[i]] = i
        id_2_VaR_risk = sorted(id_2_VaR_risk, key=lambda item: sum(item[1]), reverse=True)
        print('this is epoch: {}'.format(epochID))
        output_risk_scores('{}/risk_score_epoch_{}_{}.txt'.format(exp_dir, client_name, epochID), id_2_VaR_risk, id_2_label_index, test_true_y, test_pred_y)
        
        all_id_2_risk_desc = []
        num_labels = len(risk_scores[0])
        for label_idx in range(num_labels):
            id_2_risk = []
            for i in range(test_num):
                test_pred = test_one_pre[i][label_idx]  # Prediction for the current label
                m_label = test_pred_y[i][label_idx]     # Predicted label
                t_label = test_true_y[i][label_idx]     # True label
                if m_label == t_label:
                    label_value = 0.0
                else:
                    label_value = 1.0
                id_2_risk.append([test_ids[i], 1 - test_pred])
            id_2_risk_desc = sorted(id_2_risk, key=lambda item: item[1], reverse=True)
            all_id_2_risk_desc.extend(id_2_risk_desc)
        
        output_risk_scores('{}/base_score_epoch_{}_{}.txt'.format(exp_dir, client_name, epochID), all_id_2_risk_desc, id_2_label_index, test_true_y, test_pred_y)
                            
        
        budgets = [10, 20, 50, 100, 200, 300, 400, 500, 1000, 2000, 3000, 4000, 5000]
        risk_correct = [0] * len(budgets)
        base_correct = [0] * len(budgets)
        for i in range(test_num):
            for budget in range(len(budgets)):
                if i < budgets[budget]:
                    pair_id = id_2_VaR_risk[i][0]
                    _index = id_2_label_index.get(pair_id)
                    if test_true_y[_index] != test_pred_y[_index]:
                        risk_correct[budget] += 1
                    pair_id = id_2_risk_desc[i][0]
                    _index = id_2_label_index.get(pair_id)
                    if test_true_y[_index] != test_pred_y[_index]:
                        base_correct[budget] += 1
        
        
        risk_loss_criterion = risk_model.RiskLoss(my_risk_model)
        risk_loss_criterion = risk_loss_criterion.cuda()
        
        rule_mus = torch.tensor(my_risk_model.test_data.get_risk_mean_X_discrete(), dtype=torch.float64).cuda()
        machine_mus = torch.tensor(my_risk_model.test_data.get_risk_mean_X_continue(), dtype=torch.float64).cuda()
        rule_activate = torch.tensor(my_risk_model.test_data.get_rule_activation_matrix(),
                                     dtype=torch.float64).cuda()
        machine_activate = torch.tensor(my_risk_model.test_data.get_prob_activation_matrix(),
                                        dtype=torch.float64).cuda()
        machine_one = torch.tensor(my_risk_model.test_data.machine_label_2_one, dtype=torch.float64).cuda()
        risk_y = torch.tensor(my_risk_model.test_data.risk_labels, dtype=torch.float64).cuda()
        
        
        test_ids = my_risk_model.test_data.data_ids
        print(test_ids)
        test_ids_dict = dict()
        for ids_i in range(len(test_ids)):
            test_ids[ids_i] = os.path.basename(
                test_ids[ids_i])
            test_ids_dict[test_ids[ids_i]] = ids_i
        del my_risk_model
        data_len = len(risk_y)
        
        model.train()
        best_performance = None
        best_model_state = None
        risk_labels = None
        epoch_loss = 0
        all_confidences=[]
        sum_confidence_high = 0.0
        sum_confidence_medium = 0.0
        num_batches_high = 0
        num_batches_medium = 0
        #out_uncertain=None
        TestDataLoader = self.test_loader
        batch_size = 32
        client_path = self.client_path
        map_path = f"{client_path}{client_name}/all_data_map.csv"
        # Load the mapping
        name_map = self.load_name_mapping(map_path)
        for batch_idx, (inputs, targets, paths) in enumerate(TestDataLoader):
            paths_list = list(paths)
            for path_i in range(len(paths_list)):
                original_filename = os.path.basename(paths_list[path_i])
                if original_filename in name_map:
                    paths_list[path_i] = name_map[original_filename]  # Replace with anonymized name
            
            # Convert back to the same type as original paths
            paths = type(paths)(paths_list)
            
            #print(paths)  # Now prints anonymized paths
            optimizer.zero_grad()
        
            idx = batch_idx
            if inputs.shape[0] < batch_size:
                continue
            if use_cuda:
                inputs, targets = inputs.to(device), targets.to(device)
            inputs, targets = Variable(inputs), Variable(targets)
        
        
            index = []
        
            # we just need class_name and image_name
            paths = list(paths)
            for path_i in range(len(paths)):
                paths[path_i] = os.path.basename(
                    paths[path_i])
                # print(paths[path_i])
                #index.append(test_ids_dict.get(paths[path_i], -1))
                index.append(test_ids_dict[paths[path_i]])
            #               print(index)
        
            test_pre_batch = test_pre[index]
            rule_mus_batch = rule_mus[index]
            machine_mus_batch = machine_mus[index]
            rule_activate_batch = rule_activate[index]
            machine_activate_batch = machine_activate[index]
            machine_one_batch = machine_one[index]
        
            # optimizer.zero_grad()
            # _, _, _, output_concat, _, _ = net(inputs)
            chex=1
            if chex == 1:
                if inputs.dim() == 4:
                    bs = inputs.size(0)
                    n_crops = 1
                    c, h, w = inputs.size(1), inputs.size(2), inputs.size(3)
                                
                elif inputs.dim() == 5:
                    bs, n_crops, c, h, w = inputs.size()
                    inputs = inputs.view(-1, c, h, w)  # Reshape to [batch_size * n_crops, c, h, w]
        
            inputs, targets = inputs.cuda(), targets.cuda()
            try:
                x4, xc = model(inputs)
        
            except:
                xc = model(inputs)
        
            if chex == 1:
                xc = xc.cuda().squeeze().view(bs, n_crops, -1).mean(1)
            
            batch_confidence_high = xc.mean().item()
        
            # Update running sum and count
            sum_confidence_high += batch_confidence_high
            num_batches_high += 1
            
            out=xc
            #out_uncertain = out
            y_score = sigmoid(xc.data.cpu()/2)
            out_2=1-out
            out_temp=torch.reshape(out,(-1,1))
            out_2=torch.reshape(out_2,(-1,1))
            out_2D=torch.cat((out_temp,out_2),1)
        
        
            # Compute the risk labels
            risk_labels = risk_loss_criterion(test_pre_batch, rule_mus_batch, machine_mus_batch,
                                      rule_activate_batch, machine_activate_batch,
                                      machine_one_batch, y_score, labels=None)
        
            risk_labels = risk_labels.cuda()
        
            # Step 1: Update for Risky Labels Only (risk_labels == 1)
            mask_risky = (risk_labels == 0).long()
            print('mask_high')
            print(mask_risky)
            
            if mask_risky.sum() > 0:  # Ensure there are risky labels to update
                optimizer.zero_grad()  # Clear previous gradients
                Loss_risky = LSLLoss(out, mask_risky) 
                Loss_risky = Loss_risky.sum() / mask_risky.sum()  # Normalize loss
                Loss_risky.backward(retain_graph=True)  # Backpropagate for risky labels, retain graph for uncertain
                optimizer.step()  # Update model
            
            #epoch_loss += Loss_risky.item()
        
        # Step 2: Evaluate the final selected model on test data
        y_true_test, y_pred_test, y_score_test, test_f1_final, y_score = self.test(model)  # Testing
        #test_acc, test_pre, test_f1_final = FederatedClient.test('/home/15t/Gul/Datasets/{}/test'.format(store_name), model,
        #                                                        'RESNET-101', class_num,
        #                                                        False, 1, 256, 244)
        
        if test_f1_final > max_test_f1:
            max_test_f1 = test_f1_final
            old_models = sorted(glob(join(exp_dir, 'max_*.pth')))
            
            # Remove the oldest saved model if it exists
            if len(old_models) > 0: 
                os.remove(old_models[0])
            
            # Save the new best model based on test accuracy
            torch.save({'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                # 'scheduler': scheduler.state_dict()  # Ensure this is correctly placed if needed
               },  # <- Closing parenthesis for the dictionary
               os.path.join(exp_dir, "max_test_acc_{:.2f}.pth".format(max_test_f1)),
               _use_new_zipfile_serialization=False)
        
            print(f"New best model saved with test f1: {max_test_f1:.2f}")     
        
        return model
     # --------------------------------------------------------------------------------


    def train_t(self, global_model):
    #def train_t(pathImgTest, pathModel, nnArchitecture, class_num, nnIsTrained, batch_size, transResize, transCrop,ckpt=False):
        model = global_model
        model.eval()
        model.cuda()
       
        chex = 1
        testloader = self.train_loader
        batch_hamming_losses = []
        distribution_x4 = []
        distribution_xc = []
        paths = []
        y_pred, y_true, y_score = [], [], []
        with torch.no_grad():

            for _, (inputs, targets, paths_batch) in enumerate(tqdm(testloader, ncols=80)):
                        if chex == 1:
                            if inputs.dim() == 4:
                                bs = inputs.size(0)
                                n_crops = 1
                                c, h, w = inputs.size(1), inputs.size(2), inputs.size(3)
                                
                            elif inputs.dim() == 5:
                                bs, n_crops, c, h, w = inputs.size()
                                inputs = inputs.view(-1, c, h, w)  # Reshape to [batch_size * n_crops, c, h, w]
            
                        
            
                        inputs, targets = inputs.cuda(), targets.cuda()
                        x4, xc = model(inputs)
            
                        if chex == 1: 
                          xc = xc.squeeze().view(bs, n_crops, -1).mean(1)
                          x4 = x4.squeeze().view(bs, n_crops, -1).mean(1)
                          
                        predicted = torch.sigmoid(xc.data) > 0.5
                        predicted = predicted.cpu().numpy().astype(int)
                        
                        y_score.extend(sigmoid(xc.data.cpu()))
                        y_pred.extend(predicted.tolist())
                        y_true.extend(targets.cpu().numpy())
                        # Compute Hamming Loss
                        batch_hamming_loss = hamming_loss(y_true, y_pred)
                        batch_hamming_losses.append(batch_hamming_loss)
                        distribution_x4.extend(x4.cpu().tolist())
                        distribution_xc.extend(xc.cpu().tolist())
                        paths.extend(paths_batch)
                        
                        y_score_t = [_[1] for _ in softmax(xc.data.cpu(), axis=1)]
                        varOutput_f = ([_[1] for _ in softmax(1 - xc.data.cpu(), axis=1)])
                        
                        y_score_t = torch.tensor(y_score_t)
                        varOutput_f = torch.tensor(varOutput_f)
                        varOutput_n = torch.reshape(y_score_t, (-1, 1))
                        varOutput_f = torch.reshape(varOutput_f, (-1, 1))
                        varOutput_n = torch.cat((varOutput_n.cpu(), varOutput_f.cpu()), 1)
                        
            avg_hamming_loss = np.mean(batch_hamming_losses)
            std_hamming_loss = np.std(batch_hamming_losses)
            y_score = [tensor.tolist() for tensor in y_score]
            y_score = np.array(y_score)
            
            y_true = np.concatenate(y_true, axis=0)
            y_pred = np.concatenate(y_pred, axis=0)
            y_score = np.concatenate(y_score, axis=0)
            distribution_xc_per_image = []
            num_labels = 3
            
            # Iterate over each sample and label index
            for i in range(len(distribution_xc)):
                image_predictions = []
                for j in range(num_labels):
                    # Append the raw output value for positive label
                    image_predictions.append(distribution_xc[i][j])
                    # Append the complement of raw output value for negative label
                    image_predictions.append(1 - distribution_xc[i][j])
                distribution_xc_per_image.append(image_predictions)
            # Evaluate predictions
            predictions_binary = (y_pred > 0.5).astype(int)
            true_labels_binary = y_true.astype(int)
            #print(len(distribution_xc_per_image))
            #for pb in distribution_xc_per_image:
            #    print(pb)
            
            predictions_binary = predictions_binary.reshape(-1, num_labels)
            true_labels_binary = true_labels_binary.reshape(-1, num_labels)
            print(f"predictions_binary shape: {predictions_binary.shape}")
            print(f"true_labels_binary shape: {true_labels_binary.shape}")
            #print(predictions_binary.shape[1])
            
        
            train_accuracies = []
            train_precisions = []
            train_recalls = []
            train_f1_scores = []
            
            for label in range(num_labels):
                label_accuracy = np.mean(predictions_binary[:, label] == true_labels_binary[:, label])
                label_precision = precision_score(true_labels_binary[:, label], predictions_binary[:, label], zero_division=0)
                label_recall = recall_score(true_labels_binary[:, label], predictions_binary[:, label])
                label_f1_score = f1_score(true_labels_binary[:, label], predictions_binary[:, label])
        
                train_accuracies.append(label_accuracy)
                train_precisions.append(label_precision)
                train_recalls.append(label_recall)
                train_f1_scores.append(label_f1_score)
        
            train_accuracy = 100.0 * np.mean(train_accuracies)
            train_precision = 100.0 *  np.mean(train_precisions)
            train_recall = 100.0 * np.mean(train_recalls)
            train_f1_score = 100.0 * np.mean(train_f1_scores)
        
            
            print("Dataset \t{:.2f}\t{:.2f}\t{:.2f}\t\t{:.2f}\t{:.2f}\n".format( train_accuracy, train_f1_score,
                                                                                  train_precision, train_recall, avg_hamming_loss
                                                                                  
                                                                                  ))
            
            y_score = torch.Tensor(distribution_xc_per_image)
            
            return train_accuracy, y_score
     # --------------------------------------------------------------------------------


    def test(self, global_model):
   
        model = global_model
        model.eval()
        model.cuda()

        
        chex = 1
        testloader = self.test_loader
        #testloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
        batch_hamming_losses =[]
        distribution_x4 = []
        distribution_xc = []
        paths = []
        y_pred, y_true, y_score, y_score_auc = [], [], [], []
        with torch.no_grad():

            for _, (inputs, targets, paths_batch) in enumerate(tqdm(testloader, ncols=80)):
                        if chex == 1:
                            if inputs.dim() == 4:
                                bs = inputs.size(0)
                                n_crops = 1
                                c, h, w = inputs.size(1), inputs.size(2), inputs.size(3)
                                
                            elif inputs.dim() == 5:
                                bs, n_crops, c, h, w = inputs.size()
                                inputs = inputs.view(-1, c, h, w)  # Reshape to [batch_size * n_crops, c, h, w]
                            #bs, n_crops, c, h, w = inputs.size()
                            #inputs = inputs.view(-1, c, h, w)
            
                        inputs, targets = inputs.cuda(), targets.cuda()
                        #print(model(inputs))
                        x4, xc = model(inputs)
            
                        if chex == 1: 
                          xc = xc.squeeze().view(bs, n_crops, -1).mean(1)
                          x4 = x4.squeeze().view(bs, n_crops, -1).mean(1)
                          
                        predicted = torch.sigmoid(xc.data) > 0.5
                        #print(predicted)
                        predicted = predicted.cpu().numpy().astype(int)
                        y_score.extend(sigmoid(xc.data.cpu()))
                        y_score_auc.extend(torch.sigmoid(xc).cpu().numpy())
                        y_pred.extend(predicted.tolist())
                        y_true.extend(targets.cpu().numpy())
                        batch_hamming_loss = hamming_loss(y_true, y_pred)
                        batch_hamming_losses.append(batch_hamming_loss)
                        distribution_x4.extend(x4.cpu().tolist())
                        distribution_xc.extend(xc.cpu().tolist())
                        paths.extend(paths_batch)
                        #p
                        
                        y_score_t = [_[1] for _ in softmax(xc.data.cpu(), axis=1)]
                        varOutput_f = ([_[1] for _ in softmax(1 - xc.data.cpu(), axis=1)])
                        
                        
                        y_score_t = torch.tensor(y_score_t)
                        varOutput_f = torch.tensor(varOutput_f)
                        varOutput_n = torch.reshape(y_score_t, (-1, 1))
                        varOutput_f = torch.reshape(varOutput_f, (-1, 1))
                        varOutput_n = torch.cat((varOutput_n.cpu(), varOutput_f.cpu()), 1)
                        
            y_true_auc = np.array(y_true)
            y_score_auc = np.array(y_score_auc)
            
            avg_hamming_loss = np.mean(batch_hamming_losses)
            std_hamming_loss = np.std(batch_hamming_losses)
            y_true_test = y_true
            y_pred_test = y_pred
            y_score_test = y_score
            y_true = np.concatenate(y_true, axis=0)
            y_pred = np.concatenate(y_pred, axis=0)
            y_score = np.concatenate(y_score, axis=0)
            distribution_xc_per_image = []
            num_labels = 3
            # Iterate over each sample and label index
            for i in range(len(distribution_xc)):
                image_predictions = []
                for j in range(num_labels):
                    # Append the raw output value for positive label
                    image_predictions.append(distribution_xc[i][j])
                    # Append the complement of raw output value for negative label
                    image_predictions.append(1 - distribution_xc[i][j])
                distribution_xc_per_image.append(image_predictions)
            predictions_binary = (y_pred > 0.5).astype(int)
            true_labels_binary = y_true.astype(int)
            
            num_labels = 3
            predictions_binary = predictions_binary.reshape(-1, num_labels)
            true_labels_binary = true_labels_binary.reshape(-1, num_labels)
            print(f"predictions_binary shape: {predictions_binary.shape}")
            print(f"true_labels_binary shape: {true_labels_binary.shape}")
        
            train_accuracies = []
            train_precisions = []
            train_recalls = []
            train_f1_scores = []
            train_aucs = []
            predictions_binary = predictions_binary.reshape(-1, num_labels)
            true_labels_binary = true_labels_binary.reshape(-1, num_labels)
            for label in range(num_labels):
                label_accuracy = np.mean(predictions_binary[:, label] == true_labels_binary[:, label])
                label_precision = precision_score(true_labels_binary[:, label], predictions_binary[:, label], zero_division=0)
                label_recall = recall_score(true_labels_binary[:, label], predictions_binary[:, label])
                label_f1_score = f1_score(true_labels_binary[:, label], predictions_binary[:, label])
                label_auc = roc_auc_score(y_true_auc[:, label], y_score_auc[:, label])
                
                
                print("Dataset Label \t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\n".format(
                    label, label_accuracy, label_f1_score, label_precision, label_recall, label_auc, avg_hamming_loss, std_hamming_loss))
                
                train_accuracies.append(label_accuracy)
                train_precisions.append(label_precision)
                train_recalls.append(label_recall)
                train_f1_scores.append(label_f1_score)
                train_aucs.append(label_auc)
            train_accuracy = 100.0 * np.mean(train_accuracies)
            train_precision = 100.0 * np.mean(train_precisions)
            train_recall = 100.0 * np.mean(train_recalls)
            train_f1_score = 100.0 * np.mean(train_f1_scores)
            train_auc = 100.0 * np.mean(train_aucs)
            confusion_matrices = multilabel_confusion_matrix(true_labels_binary, predictions_binary)
            
            
            for label in range(num_labels):
                print(f"Confusion Matrix for Label {label}:")
                print(confusion_matrices[label])
                print("\n")
            print("Dataset over all \t{:.2f}\t{:.2f}\t{:.2f}\t\t{:.2f}\t{:.2f}\t{:.2f}\n".format( train_accuracy, train_f1_score,
                                                                                  train_precision, train_recall, train_auc, 
                                                                                  avg_hamming_loss
                                                                                  ))
       
            y_score = torch.Tensor(distribution_xc_per_image)
          
            
            return y_true_test, y_pred_test, y_score_test, train_f1_score, y_score
    # --------------------------------------------------------------------------------

    def Valtest(self, global_model):
    #def Valtest(pathImgTest, pathModel, nnArchitecture, testdataloader, nnIsTrained, batch_size, transResize, transCrop,ckpt=False):
  
        model = global_model
 
        model.eval()
        model.cuda()

        y_score_n = torch.empty([0, 2], dtype=torch.float32)

        chex=1
        

        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transformList = []
        transformList.append(transforms.RandomResizedCrop(224))
        transformList.append(transforms.RandomHorizontalFlip())
        transformList.append(transforms.ToTensor())
        transformList.append(normalize)
        transform_test = transforms.Compose(transformList)
        # transform_test = transforms.Compose(transformList)
        chex=0

        with torch.no_grad():
            
            
            testloader = self.val_loader
            total_hamming_loss = 0
            distribution_x4 = []
            distribution_xc = []
            paths = []
            y_pred, y_true, y_score = [], [], []

            for _, (inputs, targets, paths_batch) in enumerate(tqdm(testloader, ncols=80)):
                        if chex == 1:
                            if inputs.dim() == 4:
                                bs = inputs.size(0)
                                n_crops = 1
                                c, h, w = inputs.size(1), inputs.size(2), inputs.size(3)
                                
                            elif inputs.dim() == 5:
                                bs, n_crops, c, h, w = inputs.size()
                                inputs = inputs.view(-1, c, h, w)  # Reshape to [batch_size * n_crops, c, h, w]
                           
            
                        inputs, targets = inputs.cuda(), targets.cuda()
                        
                        x4, xc = model(inputs)
            
                        if chex == 1: 
                          xc = xc.squeeze().view(bs, n_crops, -1).mean(1)
                          x4 = x4.squeeze().view(bs, n_crops, -1).mean(1)
                          
                        predicted = torch.sigmoid(xc.data) > 0.5
                        #print(predicted)
                        predicted = predicted.cpu().numpy().astype(int)
                        
                        y_score.extend(sigmoid(xc.data.cpu()))
                        y_pred.extend(predicted.tolist())
                        y_true.extend(targets.cpu().numpy())
                        batch_hamming_loss = hamming_loss(y_true, y_pred)
                        total_hamming_loss += batch_hamming_loss
                        distribution_x4.extend(x4.cpu().tolist())
                        distribution_xc.extend(xc.cpu().tolist())
                        
                        paths.extend(paths_batch)

                       
                        y_score_t = [_[1] for _ in softmax(xc.data.cpu(), axis=1)]
                        varOutput_f = ([_[1] for _ in softmax(1 - xc.data.cpu(), axis=1)])
                        y_score_t = torch.tensor(y_score_t)
                        varOutput_f = torch.tensor(varOutput_f)
                        varOutput_n = torch.reshape(y_score_t, (-1, 1))
                        varOutput_f = torch.reshape(varOutput_f, (-1, 1))
                        varOutput_n = torch.cat((varOutput_n.cpu(), varOutput_f.cpu()), 1)
                        y_score_n = torch.cat((varOutput_n.cpu(), y_score_n.cpu()), 0)

               
            avg_hamming_loss = total_hamming_loss / len(testloader)
            y_score = [tensor.tolist() for tensor in y_score]
            y_score = np.array(y_score)
            y_true = np.concatenate(y_true, axis=0)
            y_pred = np.concatenate(y_pred, axis=0)
            y_score = np.concatenate(y_score, axis=0)
            distribution_xc_per_image = []
            num_labels = 3
            # Iterate over each sample and label index
            for i in range(len(distribution_xc)):
                image_predictions = []
                for j in range(num_labels):
                    # Append the raw output value for positive label
                    image_predictions.append(distribution_xc[i][j])
                    # Append the complement of raw output value for negative label
                    image_predictions.append(1 - distribution_xc[i][j])
                distribution_xc_per_image.append(image_predictions)
             # Evaluate predictions
            predictions_binary = (y_pred > 0.5).astype(int)
            true_labels_binary = y_true.astype(int)
            num_labels = 3
            predictions_binary = predictions_binary.reshape(-1, num_labels)
            true_labels_binary = true_labels_binary.reshape(-1, num_labels)
            print(f"predictions_binary shape: {predictions_binary.shape}")
            print(f"true_labels_binary shape: {true_labels_binary.shape}")
        
            train_accuracies = []
            train_precisions = []
            train_recalls = []
            train_f1_scores = []
            predictions_binary = predictions_binary.reshape(-1, num_labels)
            true_labels_binary = true_labels_binary.reshape(-1, num_labels)
            for label in range(num_labels):
                label_accuracy = np.mean(predictions_binary[:, label] == true_labels_binary[:, label])
                label_precision = precision_score(true_labels_binary[:, label], predictions_binary[:, label], zero_division=0)
                label_recall = recall_score(true_labels_binary[:, label], predictions_binary[:, label])
                label_f1_score = f1_score(true_labels_binary[:, label], predictions_binary[:, label])
        
                train_accuracies.append(label_accuracy)
                train_precisions.append(label_precision)
                train_recalls.append(label_recall)
                train_f1_scores.append(label_f1_score)
        
            train_accuracy = 100.0 * np.mean(train_accuracies)
            train_precision = 100.0 * np.mean(train_precisions)
            train_recall = 100.0 * np.mean(train_recalls)
            train_f1_score = 100.0 * np.mean(train_f1_scores)
        
            print("Dataset \t{:.2f}\t{:.2f}\t{:.2f}\t\t{:.2f}\t{:.2f}\n".format( train_accuracy, train_f1_score,
                                                                                  train_precision, train_recall, avg_hamming_loss
                                                                                  
                                                                                  ))
            
            y_score = torch.Tensor(distribution_xc_per_image)
            return train_accuracy,y_score

    




