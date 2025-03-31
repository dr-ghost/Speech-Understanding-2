import soundfile as sf
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset, DataLoader
from torchaudio.transforms import Resample
from models.ecapa_tdnn import ECAPA_TDNN_SMALL
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

from pytorch_metric_learning.losses import ArcFaceLoss

from data import VoxCeleb1, VoxCeleb2

import loratorch as lora

import warnings
warnings.filterwarnings("ignore")

MODEL = 'wavlm_base_plus'
CHECKPOINT = 'Question 1/models/checkpoints/wavlm_base_plus_nofinetune.pth'

def init_model(checkpoint=None) -> torch.nn.Module:
    model = ECAPA_TDNN_SMALL(feat_dim=768, feat_type='wavlm_base_plus', config_path=None)
    
    
    if checkpoint is not None:
        state_dict = torch.load(checkpoint, map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict['model'], strict=False)
    
    return model

def replace_conv1d_with_lora(module, r=4, lora_alpha=1.0, dropout=0.0):
    """
    Recursively replace all nn.Conv1d layers in module with LoRAConv1d.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv1d):
            in_channels = child.in_channels
            out_channels = child.out_channels
            # Support for both int and tuple specifications.
            kernel_size = child.kernel_size if isinstance(child.kernel_size, int) else child.kernel_size[0]
            stride = child.stride if isinstance(child.stride, int) else child.stride[0]
            padding = child.padding if isinstance(child.padding, int) else child.padding[0]
            dilation = child.dilation if isinstance(child.dilation, int) else child.dilation[0]
            groups = child.groups
            bias = child.bias is not None

            new_conv = lora.LoRAConv1d(in_channels, out_channels, kernel_size,
                                  stride=stride, padding=padding, dilation=dilation,
                                  groups=groups, bias=bias, r=r, lora_alpha=lora_alpha, dropout=dropout)
            new_conv.conv.weight.data.copy_(child.weight.data)
            if bias:
                new_conv.conv.bias.data.copy_(child.bias.data)
            setattr(module, name, new_conv)
        else:
            replace_conv1d_with_lora(child, r=r, lora_alpha=lora_alpha, dropout=dropout)

def replace_linear_with_lora(module, r=4, lora_alpha=1.0, dropout=0.0):
    """
    Recursively replace all nn.Linear layers in module with LoRALinear.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            new_linear = lora.LoRALinear(child.in_features, child.out_features, r=r,
                                    lora_alpha=lora_alpha, dropout=dropout, bias=(child.bias is not None))
            new_linear.weight.data.copy_(child.weight.data)
            if child.bias is not None:
                new_linear.bias.data.copy_(child.bias.data)
            setattr(module, name, new_linear)
        else:
            replace_linear_with_lora(child, r=r, lora_alpha=lora_alpha, dropout=dropout)

def train_lora(model: torch.nn.Module, dataset: Dataset, epochs: int = 50, batch_size: int = 32, device: str = 'cuda', model_save: bool = False, save_path: str = 'Question 1/models/checkpoints/wavlm_base_plus_lora.pth') -> tuple:
    """
    Train the model on the dataset.
    """
    def collate_fn(batch):
        # Custom collate function to handle variable-length sequences
        
        wavs, labels = zip(*batch)
        
        padded_wavs = torch.nn.utils.rnn.pad_sequence(wavs, batch_first=True)
        
        labels = torch.tensor(labels)
        
        return padded_wavs, labels
    
    test_dat = dataset.clone()
    test_dat.train = False  
    
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader1 = DataLoader(test_dat, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader2 = DataLoader(test_dat, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    model.to(torch.device(device))
    
    lora.mark_only_lora_as_trainable(model)
    
    toy_emb = model(dataset[0][0].unsqueeze(0).to(torch.device(device)))
    
    
    arc_face_loss = ArcFaceLoss(embedding_size=toy_emb.shape[-1], num_classes=100, margin=0.5, scale=32)
    arc_face_loss.to(torch.device(device))
    
    optim_model = optim.AdamW(model.parameters(), lr=1e-4)
    optim_loss = optim.AdamW(arc_face_loss.parameters(), lr=1e-4)
    
    train_loss = []
    
    test_acc = []
    test_auc = []
    test_f1 = []
    test_precision = []
    test_recall = []

    for epoch in tqdm(range(epochs)):
        for wavs, labels in train_loader:
            model.train()
            
            wavs.to(torch.device(device))
            labels.to(torch.device(device))
            
            optim_model.zero_grad()
            optim_loss.zero_grad()
            
            embs = model(wavs)
            loss = arc_face_loss(embs, labels)
            
            loss.backward()
            optim_model.step()
            optim_loss.step()
            
            
            lora.register_model_param_after_backward(model)
            
            train_loss.append(loss.detach().cpu().item())
        
        # Test the model
        if epoch % 5 == 0:
            
            cos_sim = np.zeros(len(test_loader1.dataset))
            labels = np.zeros(len(test_loader1.dataset))
            
            model.eval()
            for idx, batch in enumerate(zip(test_loader1, test_loader2)):
                i1, i2 = batch
                
                wavs1, labels1 = i1
                wavs2, labels2 = i2
                
                wavs1.to(torch.device(device))
                wavs2.to(torch.device(device))
                labels1.to(torch.device(device))
                labels2.to(torch.device(device))
                
                with torch.no_grad():
                    embs1 = model(wavs1)
                    embs2 = model(wavs2)
                    
                    cos_sim_ = F.cosine_similarity(embs1, embs2).cpu().numpy()
                    
                    cos_sim[idx * batch_size: (idx + 1) * batch_size] = cos_sim_
                    
                    labels[idx * batch_size: (idx + 1) * batch_size] = (labels1 == labels2).cpu().numpy()
                    
            acc_score = accuracy_score(labels, (cos_sim > 0.5).astype(int))
            test_acc.append(acc_score)
            auc_score = roc_auc_score(labels, cos_sim)
            test_auc.append(auc_score)         
            f1 = f1_score(labels, (cos_sim > 0.5).astype(int))
            test_f1.append(f1)
            precision = precision_score(labels, (cos_sim > 0.5).astype(int))
            test_precision.append(precision)
            recall = recall_score(labels, (cos_sim > 0.5).astype(int))
            test_recall.append(recall) 

    if model_save:
        torch.save(lora.lora_state_dict(model), save_path)       
    return train_loss, test_acc, test_auc, test_f1, test_precision, test_recall
            
            


if __name__ == "__main__":
    model = init_model(checkpoint=CHECKPOINT)
    
    # LoRA
    replace_linear_with_lora(model, r=4, lora_alpha=1.0, dropout=0.0)
    
    vox2_d = VoxCeleb2(data_dir='Question 1/data/vox2', sample_freq=16000)
    
    
    