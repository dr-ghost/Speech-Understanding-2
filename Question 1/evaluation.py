import soundfile as sf
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchaudio.transforms import Resample
from models.ecapa_tdnn import ECAPA_TDNN_SMALL
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, roc_curve
from tqdm import tqdm

from data import VoxCeleb1

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

def compute_eer(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    return eer * 100, eer_threshold


def compute_tar(scores, labels, far_level=0.01):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    tar = tpr[np.nanargmin(np.abs(fpr - far_level))]
    return tar

def evaluate(model: torch.nn.Module, vox1: Dataset, eval_file_path: str, batch_size: int = 1, device: torch.DeviceObjType = torch.device('cpu')):
    with open(eval_file_path, 'r') as f:
        lines = f.readlines()
        tmp_lines = [line.split() for line in lines]
        
        batched_lines = [tmp_lines[i:i + batch_size] for i in range(0, len(tmp_lines), batch_size)]
        
        batched_labels = [[int(i[0]) for i in batch] for batch in batched_lines]
        batched_wav1 = [[i[1] for i in batch] for batch in batched_lines]
        batched_wav2 = [[i[2] for i in batch] for batch in batched_lines]
        
        cos_sim = np.zeros(len(lines))
        labels = np.zeros(len(lines))
        
        model.to(device)
        
        model.eval()      
        for idx, batch in tqdm(enumerate(zip(batched_labels, batched_wav1, batched_wav2)), total=len(batched_lines)):
            labels_, wav1, wav2 = batch
            
            wav1 = [vox1[i] for i in wav1]
            min_len1 = min([i.shape[1] for i in wav1])
            wav1 = [i[:, :min_len1] for i in wav1]

            
            wav2 = [vox1[i] for i in wav2]
            min_len2 = min([i.shape[1] for i in wav2])
            wav2 = [i[:, :min_len2] for i in wav2]
            
            
            wav1 = torch.stack(wav1).squeeze_(-2).to(device)
            wav2 = torch.stack(wav2).squeeze_(-2).to(device)
            
            
            with torch.no_grad():
                emb1 = model(wav1)
                emb2 = model(wav2)
                
                cos_sim_ = F.cosine_similarity(emb1, emb2).cpu().numpy()
                
                cos_sim[idx * batch_size: (idx + 1) * batch_size] = cos_sim_
                
                labels[idx * batch_size: (idx + 1) * batch_size] = labels_
                
        auc_score = roc_auc_score(labels, cos_sim)
        acc_score = accuracy_score(labels, (cos_sim > 0.5).astype(int))
        f1 = f1_score(labels, (cos_sim > 0.5).astype(int))
        precision = precision_score(labels, (cos_sim > 0.5).astype(int))
        recall = recall_score(labels, (cos_sim > 0.5).astype(int))
        
        eer, eer_threshold = compute_eer(labels, cos_sim)
        
        tar_1_far = compute_tar(cos_sim, labels, far_level=0.01)
        
        speaker_id_acc = np.mean((cos_sim > eer_threshold) == labels)
        
        print(f"AUC: {auc_score:.4f}, Accuracy: {acc_score:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
        print(f"EER: {eer:.2f}%, TAR@1% FAR: {tar_1_far:.4f}, Speaker Identification Accuracy: {speaker_id_acc:.4f}")
        
        return auc_score, acc_score, f1, precision, recall, eer, tar_1_far, speaker_id_acc        

if __name__ == '__main__':
    model = init_model(CHECKPOINT)
    # print(next(model.parameters()).device)
    vox1_d = VoxCeleb1('Question 1/data/vox1')
    
    evaluate(model, vox1_d, "Question 1/data/vox1_trial_pairs.txt", batch_size=128)
    