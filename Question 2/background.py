import os
import subprocess
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import soundfile as sf
import torchaudio.transforms as T
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib.pyplot as plt

from collections import defaultdict
import random
import librosa
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import warnings
warnings.filterwarnings("ignore")


def download_data(data_dir=os.path.dirname(os.path.abspath(__file__))):
    subprocess.run(['chmod', 'a+x', os.path.join(data_dir, 'download_data.sh')])
    print("script made executable.")
    
    subprocess.run(['bash', 'download_data.sh'], cwd=data_dir)
    print("download_data.sh script executed successfully.")
    
    subprocess.run(
        ["unzip", "-o", os.path.join(data_dir, "audio-dataset-with-10-indian-languages.zip"), "-d", data_dir],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    print("Unzipped audio-dataset-with-10-indian-languages.zip successfully.")
    
    subprocess.run(
        ["rm", os.path.join(data_dir, "audio-dataset-with-10-indian-languages.zip")],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
class IndianLanguagesDataset(Dataset):
    def __init__(self, data_dir, sampling_rate=16000, feat_dim=40, win_len=25, hop_len=10):
        self.data_dir = data_dir
        self.audio_files = []
        self.labels = []
        
        self.langs = []
        
        for lang in os.listdir(data_dir):
            lang_dir = os.path.join(data_dir, lang)
            
            self.langs.append(lang)
            
            if os.path.isdir(lang_dir):
                for file in os.listdir(lang_dir):
                    if file.endswith('.mp3'):
                        self.audio_files.append(os.path.join(lang_dir, file))
                        self.labels.append(lang)
        
        
        self.sr = sampling_rate
        self.feat_dim = feat_dim
        self.win_len = win_len
        self.hop_len = hop_len
        
        melkwargs = {
            'n_fft': 512,
            'win_length': self.win_len,
            'hop_length': self.hop_len,
            'n_mels': self.feat_dim,
            'f_min': 0.0,
            'f_max': self.sr // 2,
            'pad': 0
        }

        self.feature_extract = T.MFCC(sample_rate=self.sr, n_mfcc=feat_dim, log_mels=False,
                                              melkwargs=melkwargs)
        
        self.instance_norm = nn.InstanceNorm1d(self.feat_dim)
        
        self.transform = None
                        
    def mfcc_features(self, waveform):
        with torch.no_grad():
            wav = self.feature_extract(waveform)  + 1e-6
            
            wav = self.instance_norm(wav) #feature normalizatio
            
            return wav
        
    def __len__(self):
        return len(self.audio_files)
    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        label = self.labels[idx]
        
        wavform, sr = librosa.load(audio_path, sr=self.sr)
        
        if self.transform:
            waveform = self.transform(waveform)
            
        feats = self.mfcc_features(torch.tensor(wavform).float())
        feats = feats.squeeze(0).transpose(0, 1)
        
        return feats, label, self.langs.index(label)
        
def visualize_mfcc_spectrogram(mfcc, title):
    plt.figure(figsize=(10, 4))
    plt.imshow(mfcc.T, origin="lower", aspect="auto", cmap="viridis")
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("MFCC Coefficients")
    plt.tight_layout()
    plt.show()
    
def compute_mfcc_statistics(dataset, max_samples_per_lang=50):
    mfcc_stats = defaultdict(list)
    
    
    idxs = []
    while True:
        
        
        idx = random.randint(0, len(dataset) - 1)
        
        if idx in idxs:
            continue
        
        mfcc_feat, label, _ = dataset[idx]
        
        if len(mfcc_stats[label]) >= max_samples_per_lang:
            continue
        
        mfcc_np = mfcc_feat.detach().numpy()
        mfcc_stats[label].append(mfcc_np)
        
        idxs.append(idx)
        
        if min([len(mfcc_stats[label]) for label in dataset.langs]) >= max_samples_per_lang and (
            len(mfcc_stats.keys()) >= len(dataset.langs)
        ):
            break
    
    stats = {}
    for lang, mfcc_list in mfcc_stats.items():
        all_mfcc = np.concatenate(mfcc_list, axis=0)
        mean_coeff = np.mean(all_mfcc, axis=0)
        var_coeff = np.var(all_mfcc, axis=0)
        stats[lang] = (mean_coeff, var_coeff)
    
    return stats

class LanguageClassifier(nn.Module):
    def __init__(self, input_channels, num_classes):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        
        self.b_norm = nn.BatchNorm1d(1)
        
        self.pool = nn.AdaptiveAvgPool1d(128)
        
        self.fc = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        x = self.conv(x)
        x = self.b_norm(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    
def train(model: nn.Module, dataset: Dataset, epochs: int = 25, batch_size: int = 64, device: torch.DeviceObjType = torch.device('cpu'), max_bins: int = 1200) -> tuple:
    
    indices = [random.randint(0, len(dataset) - 1) for _ in range(10_000)]
    
    idxis = []
    
    for i in indices:
        try:
            dataset[i]
            idxis.append(i)
        except:
            continue
        

    dataset_ = Subset(dataset, indices)
        
    train_dataset, val_dataset = train_test_split(dataset_, test_size=0.2, random_state=42)
    
    
    def collate_fn(batch):
        wavs, labels, idxs = zip(*batch)
        
        wavs = [wav[:max_bins, :] for wav in wavs]
        
        return torch.stack(wavs).squeeze_(-2), labels, torch.tensor(idxs)
    
    train_loader =  DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    model.to(device)
    
    opt = optim.AdamW(model.parameters(), lr=1e-3)
    
    criterion = nn.CrossEntropyLoss()
    
    train_loss = []

    val_acc = []
    val_f1 = []
    
    for epoch in tqdm(range(epochs)):
        loss_ = 0
        n_lops = 0
        for wav, label, idx in train_loader:            
            x = wav.to(device)
            y = idx.to(device)
            
            opt.zero_grad()
            
            y_ = model(x)
            
            loss = criterion(y_, y)
            loss.backward()
            opt.step()
            
            loss_ += loss.cpu().item()
            n_lops += 1
        
        train_loss.append(loss_ / n_lops)
        
        gt = np.zeros(len(val_dataset))
        pred = np.zeros(len(val_dataset))
        
        for i, batch in enumerate(val_loader):
            wav, label, idx = batch
            
            
            x = wav.to(device)
            y = idx.to(device)
            
            with torch.no_grad():
                y_ = model(x)
                y_ = torch.argmax(y_, dim=1)
                
                gt[i * batch_size: (i + 1) * batch_size] = y.cpu().numpy()
                pred[i * batch_size: (i + 1) * batch_size] = y_.cpu().numpy()
                
        acc = accuracy_score(gt, pred)
        f1 = f1_score(gt, pred, average='weighted')
                
        val_acc.append(acc)
        val_f1.append(f1)
                
    return train_loss, val_acc, val_f1
        

if __name__ == "__main__":
    download_data()
    pass