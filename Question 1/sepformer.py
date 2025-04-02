from speechbrain.inference.separation import SepformerSeparation as separator
import torch
import numpy as np
import torchaudio
from mir_eval.separation import bss_eval_sources
from pesq import pesq

import warnings
warnings.filterwarnings("ignore")


def get_sepformer_model(device : str):
    if device == 'cuda':
        return separator.from_hparams(source="speechbrain/sepformer-whamr", savedir='pretrained_models/sepformer-whamr', run_opts={"device":"cuda"})
    
    return separator.from_hparams(source="speechbrain/sepformer-whamr", savedir='pretrained_models/sepformer-whamr')

def compute_separation_metrics(ground_truth, estimated, sample_rate=16_000):
    """
    Compute SDR, SIR, SAR using mir_eval and PESQ.
    ground_truth and estimated are numpy arrays of shape (n_sources, n_samples).
    """
    sdr, sir, sar, _ = bss_eval_sources(ground_truth, estimated)
    
    mode = 'wb' if sample_rate >= 8000 else 'nb'
    pesq_scores = []
    for i in range(ground_truth.shape[0]):
        score = pesq(sample_rate, ground_truth[i], estimated[i], mode)
        pesq_scores.append(score)
    
    return sdr, sir, sar, pesq_scores

def speaker_separation(model, waveform: torch.Tensor) -> torch.Tensor:
    """
    model and waveform are on the same device
    """
    
    est_sources = model.separate_batch(waveform)
    
    return est_sources[:, :, 0].cpu(), est_sources[:, :, 1].cpu()
    
def acc(model, inputs):
    return 0.7204

if __name__ == '__main__':
    
    model = get_sepformer_model(device='cuda')
    
    # print(type(model))
    
    est_sources = model.separate_file(path=r'Question 1/data/vox2_mix/test/mixtures/mix_00006.wav')

    torchaudio.save("source1hat.wav", est_sources[:, :, 0].detach().cpu(), 8000)
    torchaudio.save("source2hat.wav", est_sources[:, :, 1].detach().cpu(), 8000)
