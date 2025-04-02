import os
import requests
import subprocess
import soundfile as sf
import torch
import torch.nn.functional as F
import pandas as pd
from torchaudio.transforms import Resample
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as T
import numpy as np

import gdown

import warnings
warnings.filterwarnings("ignore")

def download_data():
    """
    Download the VoxCeleb datasets from Google Drive.
    """
    file_ids = [
        "1AC-Q8dEw1LTPdpEi5ofS04ZmZWdl8BBg",  # Corresponds to vox_1
        "1Onu4jzcyasrxTR1rRT9rl9AfkUrueM6o",  # Corresponds to vox_2
        "1vAoILnZvbYNWbFkVutlqa4qUjoO-t3xL"   # Corresponds to vox_2_text
    ]
    
    # Desired output file names
    file_names = ["vox1.zip", "vox2.zip", "vox2_text.zip"]
    
    # Google Drive direct download URL base format
    base_url = "https://drive.google.com/uc?export=download&id="
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    for file_id, file_name in zip(file_ids, file_names):
        url = base_url + file_id
        output_path = os.path.join(current_dir, file_name)
        print(f"Downloading {file_name} from {url}...")
        gdown.download(url, output_path, quiet=False)
        print(f"Downloaded {file_name} to {output_path}")
        

def download_txt_files():
    url = "https://mm.kaist.ac.kr/datasets/voxceleb/meta/veri_test2.txt"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(current_dir, "vox_celeb_1.txt")
    
    print(f"Downloading text file from {url}...")
    try:
        response = requests.get(url)
        response.raise_for_status()
        print("file getted")
        with open(output_file, "wb") as f:
            f.write(response.content)
        print(f"File downloaded successfully and saved to {output_file}")
    except requests.RequestException as e:
        print(f"Failed to download file: {e}")

def unzipy(directory=os.path.dirname(os.path.abspath(__file__))):
    for filename in os.listdir(directory):
        if filename.lower().endswith('.zip'):
            zip_path = os.path.join(directory, filename)
            base_name = os.path.splitext(filename)[0]
            # Create a temporary directory for extraction
            # temp_dir = os.path.join(directory, f"temp_extract_{base_name}")
            # Final directory name will be the ZIP file name without extension
            final_dir = os.path.join(directory, base_name)
            
            print(f"Extracting {zip_path} ....")
            try:
                # Create temporary directory if it doesn't exist
                # os.makedirs(temp_dir, exist_ok=True)
                # Unzip with -o flag (overwrite) into the temporary directory
                
                t1 = os.listdir(directory)
                
                result = subprocess.run(
                    ["unzip", "-o", zip_path, "-d", directory],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                t2 = os.listdir(directory)
                
                change_d = [i for i in t2 if i not in t1]
                
                extracted_ = os.path.join(directory, change_d[0])
                
                result_mv = subprocess.run(
                    ["mv", extracted_, final_dir],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                subprocess.run(
                    ["rm", zip_path],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                # if result_mv.stdout:
                #     print(result_mv.stdout)
            except subprocess.CalledProcessError as e:
                print(f"Error processing {zip_path}: {e.stderr}")
        else:
            print(f"Skipping {filename}: not a .zip file.")

            
class VoxCeleb1(Dataset):
    def __init__(self, data_dir, transform=None, sample_freq=16000):
        self.data_dir = data_dir
        self.transform = transform
        self.audio_files = [os.path.join(dirpath, f) for dirpath, _, filenames in os.walk(data_dir) for f in filenames if f.endswith('.wav')]
        
        self.sample_freq = sample_freq

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = os.path.join(self.data_dir, self.audio_files[idx])
        
        waveform, sr1 = sf.read(audio_path)

        waveform = torch.from_numpy(waveform).unsqueeze(0).float()
        resample1 = Resample(orig_freq=sr1, new_freq=self.sample_freq)
        waveform = resample1(waveform)

        if self.transform:
            waveform = self.transform(waveform)

        return waveform
    
    def __getitem__(self, path):
        audio_path = os.path.join(self.data_dir, path)
        
        waveform, sr1 = sf.read(audio_path)

        waveform = torch.from_numpy(waveform).unsqueeze(0).float()
        resample1 = Resample(orig_freq=sr1, new_freq=self.sample_freq)
        waveform = resample1(waveform)

        if self.transform:
            waveform = self.transform(waveform)

        return waveform

def m4a_to_wav(data_dir):
    with open(os.path.join(data_dir, 'convert.sh'), 'w') as f:
        content = '''# copy this to root directory of data and 
# chmod a+x convert.sh
# ./convert.sh
# https://unix.stackexchange.com/questions/103920/parallelize-a-bash-for-loop

open_sem(){
    mkfifo pipe-$$
    exec 3<>pipe-$$
    rm pipe-$$
    local i=$1
    for((;i>0;i--)); do
    printf %s 000 >&3
    done
}
run_with_lock(){
    local x
    read -u 3 -n 3 x && ((0==x)) || exit $x
    (
    ( "$@"; )
    printf '%.3d' $? >&3
    )&
}

N=32 # number of vCPU
open_sem $N
for f in $(find . -name "*.m4a"); do
    run_with_lock ffmpeg -loglevel panic -i "$f" -ar 16000 "${f%.*}.wav"
done
'''
        f.write(content)
        print("convert.sh script created successfully.")
        
    subprocess.run(['chmod', 'a+x', os.path.join(data_dir, 'convert.sh')])
    print("convert.sh script made executable.")
        
    subprocess.run(['bash', 'convert.sh'], cwd=data_dir)
    print("convert.sh script executed successfully.")
        
    subprocess.run(['rm', 'convert.sh'], cwd=data_dir)
    print("convert.sh script removed successfully.")


class VoxCeleb2(Dataset):
    def __init__(self, data_dir, transform=None, sample_freq=16000):
        
        to_wav = True
        
        for dirpath, _, filenames in os.walk(data_dir):
            for filename in filenames:
                if filename.endswith('.wav'):
                    to_wav = False
                    break
                
        if to_wav:
            print("wav files not found. Converting to wav format...")
            m4a_to_wav(data_dir)
            print("Converted m4a files to wav format.")
           
         
        self.dirs = os.listdir(data_dir)
        self.dirs.sort()
        
        self.train_dirs = self.dirs[:100]
        self.test_dirs = self.dirs[100:]
        
        self.train_file_paths = [os.path.join(data_dir, i, j, k) for i in self.train_dirs for j in os.listdir(os.path.join(data_dir, i)) for k in os.listdir(os.path.join(data_dir, i, j)) if k.endswith('.wav')]
        self.train_ids = [i.split('/')[-3] for i in self.train_file_paths]
        train_ids = list(set(self.train_ids))
        self.train_dct = {train_ids[i] : i for i in range(len(train_ids))}  
        
        self.test_file_paths = [os.path.join(data_dir, i, j, k) for i in self.test_dirs for j in os.listdir(os.path.join(data_dir, i)) for k in os.listdir(os.path.join(data_dir, i, j)) if k.endswith('.wav')]
        self.test_ids = [i.split('/')[-3] for i in self.test_file_paths]
        test_ids = list(set(self.test_ids))
        self.test_dct = {test_ids[i] : i for i in range(len(test_ids))}  
        
        self.train = True
        
        self.sample_freq = sample_freq
        
        self.transform = None
        
    def __len__(self):
        if self.train:
            return len(self.train_file_paths)
        else:
            return len(self.test_file_paths)

    def __getitem__(self, idx: int):
        audio_path = None
        id_ = None
        if self.train:
            audio_path = self.train_file_paths[idx]
            identity = self.train_ids[idx]
            id_ = self.train_dct[identity]
        else:
            audio_path = self.test_file_paths[idx]
            identity = self.test_ids[idx]
            id_ = self.test_dct[identity]
            
        waveform, sr1 = sf.read(audio_path)

        waveform = torch.from_numpy(waveform).unsqueeze(0).float()
        resample1 = Resample(orig_freq=sr1, new_freq=self.sample_freq)
        waveform = resample1(waveform)

        if self.transform:
            waveform = self.transform(waveform)

        return waveform, id_
    
    def __getitem__(self, path: str):
        audio_path = os.path.join(self.data_dir, path)
        
        waveform, sr1 = sf.read(audio_path)

        waveform = torch.from_numpy(waveform).unsqueeze(0).float()
        resample1 = Resample(orig_freq=sr1, new_freq=self.sample_freq)
        waveform = resample1(waveform)

        if self.transform:
            waveform = self.transform(waveform)

        return waveform
        
class VoxCelebmix(Dataset):
    def __init__(self, data_dir, transform=None, sample_freq=16000):   
        
        self.train_metadata = pd.read_csv(os.path.join(data_dir, 'train','metadata.csv'))
        self.test_metadata = pd.read_csv(os.path.join(data_dir, 'test','metadata.csv'))
        
        self.train = True
        
        self.sample_freq = sample_freq
        
        self.transform = None
        
    def __len__(self):
        if self.train:
            return self.train_metadata.shape[0]
        else:
            return self.test_metadata.shape[0]

    def __getitem__(self, idx: int):
        mix_path = None
        src1_path = None
        src2_path = None
        
        sp_1 = None
        sp_2 = None
        
        mix_len = None
        if self.train:
            mix_path = self.train_metadata['mix_path'][idx]
            src1_path = self.train_metadata['src1_path'][idx]
            src2_path = self.train_metadata['src2_path'][idx]
            
            sp_1 = self.train_metadata['speaker1'][idx]
            sp_2 = self.train_metadata['speaker2'][idx]
            
            mix_len = self.train_metadata['mixture_length'][idx]
            
            
        else:
            mix_path = self.train_metadata['mix_path'][idx]
            src1_path = self.train_metadata['src1_path'][idx]
            src2_path = self.train_metadata['src2_path'][idx]
            
            sp_1 = self.train_metadata['speaker1'][idx]
            sp_2 = self.train_metadata['speaker2'][idx]
            
            mix_len = self.train_metadata['mixture_length'][idx]
            
        waveform_mix, sr1_mix = sf.read(mix_path)

        waveform_mix = torch.from_numpy(waveform_mix).unsqueeze(0).float()
        resample1 = Resample(orig_freq=sr1_mix, new_freq=self.sample_freq)
        waveform_mix = resample1(waveform_mix)

        waveform_s1, sr1_s1 = sf.read(src1_path)

        waveform_s1 = torch.from_numpy(waveform_s1).unsqueeze(0).float()
        resample1 = Resample(orig_freq=sr1_s1, new_freq=self.sample_freq)
        waveform_s1 = resample1(waveform_s1)
        
        waveform_s2, sr1_s2 = sf.read(src2_path)
        
        waveform_s2 = torch.from_numpy(waveform_s2).unsqueeze(0).float()
        resample1 = Resample(orig_freq=sr1_s2, new_freq=self.sample_freq)
        waveform_s2 = resample1(waveform_s2)

        return waveform_mix, waveform_s1, waveform_s2, sp_1, sp_2, mix_len
        
if __name__ == "__main__":
    download_data()
    unzipy()