import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import librosa
import sys

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Config:
    def __init__(self):
        self.sampling_rate = 16000
        self.desired_length = 1.0
        self.fade_samples_ratio = 6
        self.pad_types = "repeat"

        self.filter_length = 512
        self.hop_length = 256
        self.win_length = 512
        self.window = 'hann'

        self.n_mfcc = 13
        self.n_mels = 40
        self.fmin = 60.0
        self.fmax = 6000.0

        self.cough_padding = 0.3
        self.min_cough_len = 0.1
        self.th_l_multiplier = 0.02
        self.th_h_multiplier = 5

        self.input_size = 39
        self.hidden_size = 512
        self.output_size = 2
        self.dropout = 0.1

def extract_mfcc_features(audio, sample_rate, config):
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=config.n_mfcc,
        n_mels=config.n_mels,
        n_fft=config.filter_length,
        hop_length=config.hop_length,
        fmin=config.fmin,
        fmax=config.fmax
    )
    
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    
    mfcc_features = np.vstack([mfcc, delta_mfcc, delta2_mfcc])
    return mfcc_features

def segment_cough(audio, sample_rate, cough_padding, min_cough_len, th_l_multiplier, th_h_multiplier):
    coughSegments = []
    
    n_fft = 1024
    hop_length = 256
    
    S = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    spectral_power = np.sum(np.abs(S) ** 2, axis=0)
    
    padding = int(cough_padding * sample_rate // hop_length)
    
    seg_th_l = np.mean(spectral_power) * th_l_multiplier
    seg_th_h = np.mean(spectral_power) * th_h_multiplier
    
    cough_in_progress = False
    cough_start = 0
    below_th_counter = 0
    
    for i, sample in enumerate(spectral_power):
        if cough_in_progress:
            if sample < seg_th_l:
                below_th_counter += 1
                if below_th_counter >= 10:
                    cough_end = i - below_th_counter + 1
                    cough_in_progress = False
                    below_th_counter = 0
                    
                    cough_start_sample = max(0, cough_start * hop_length)
                    cough_end_sample = min(len(audio), (cough_end + 1) * hop_length)
                    
                    if cough_end_sample - cough_start_sample > min_cough_len * sample_rate:
                        coughSegments.append(audio[cough_start_sample:cough_end_sample])
            else:
                below_th_counter = 0
        else:
            if sample > seg_th_h:
                cough_start = max(0, i - padding)
                cough_in_progress = True
    
    return coughSegments

class LSTMAudioClassifierMFCC(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.1):
        super(LSTMAudioClassifierMFCC, self).__init__()

        self.batch_norm1 = nn.BatchNorm1d(input_size)
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)

        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.batch_norm1(x)
        x = x.transpose(1, 2)
        x, _ = self.lstm1(x)

        x = x.transpose(1, 2)
        x = self.batch_norm2(x)
        x = x.transpose(1, 2)
        x, _ = self.lstm2(x)

        x = x[:, -1, :]
        x = self.dropout(x)
        x = self.fc(x)
        return x

class MFCCProcessor:
    def __init__(self, config):
        self.config = config
    
    def preprocess_audio(self, wav_path, target_length=63):
        audio_data, sr = librosa.load(wav_path, sr=self.config.sampling_rate)
        
        target_audio_length = int(self.config.sampling_rate * self.config.desired_length)
        if len(audio_data) > target_audio_length:
            audio_data = audio_data[:target_audio_length]
        elif len(audio_data) < target_audio_length:
            pad_length = target_audio_length - len(audio_data)
            audio_data = np.pad(audio_data, (0, pad_length), mode='constant')
        
        cough_segments = segment_cough(
            audio_data, 
            self.config.sampling_rate,
            self.config.cough_padding,
            self.config.min_cough_len,
            self.config.th_l_multiplier,
            self.config.th_h_multiplier
        )
        
        if len(cough_segments) > 0:
            audio_segment = cough_segments[0]
            if len(audio_segment) > target_audio_length:
                audio_segment = audio_segment[:target_audio_length]
            elif len(audio_segment) < target_audio_length:
                pad_length = target_audio_length - len(audio_segment)
                audio_segment = np.pad(audio_segment, (0, pad_length), mode='constant')
        else:
            audio_segment = audio_data
        
        mfcc_features = extract_mfcc_features(audio_segment, sr, self.config)
        mfcc_features = mfcc_features.T  
        
        if mfcc_features.shape[0] > target_length:
            mfcc_features = mfcc_features[:target_length]
        elif mfcc_features.shape[0] < target_length:
            pad_frames = target_length - mfcc_features.shape[0]
            mfcc_features = np.pad(mfcc_features, ((0, pad_frames), (0, 0)), mode='constant')
        
        return torch.FloatTensor(mfcc_features)

def predict_tb_from_audio(wav_path, model_path, config_path):
    start_time = time.time()
    
    try:    
        config = Config()
        mfcc_processor = MFCCProcessor(config)
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                model_config = json.load(f)
        else:
            model_config = {
                'model_architecture': {
                    'input_size': 39,
                    'hidden_size': 512, 
                    'output_size': 2,
                    'dropout': 0.1
                }
            }
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = LSTMAudioClassifierMFCC(
            input_size=model_config['model_architecture']['input_size'],
            hidden_size=model_config['model_architecture']['hidden_size'],
            output_size=model_config['model_architecture']['output_size'],
            dropout=model_config['model_architecture']['dropout']
        )
        
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model'])
        model.to(device)
        model.eval()
        
        mfcc_features = mfcc_processor.preprocess_audio(wav_path)
        mfcc_tensor = mfcc_features.unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(mfcc_tensor)
            prediction = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(prediction, dim=1).item()
            confidence = prediction[0][predicted_class].item()
        
        processing_time = time.time() - start_time
        return predicted_class, confidence, processing_time
        
    except Exception as e:
        processing_time = time.time() - start_time
        print(f"ERROR: {str(e)}", file=sys.stderr)
        return -1, 0.0, processing_time

if __name__ == "__main__":
    
    if len(sys.argv) != 2:
        print("ERROR: Usage: python tb_server.py <audio_filename>", file=sys.stderr)
        sys.exit(1)
    
    audio_filename = sys.argv[1]
    
    base_path = "/usr/src/app/public/uploads/batuk/"  
    audio_path = os.path.join(base_path, audio_filename)

    if not os.path.exists(audio_path):
        alternative_paths = [
            "lstm_sken3/", 
            "Audio_files_forced/",
            "Data/",
            "./"
        ]
        
        for alt_path in alternative_paths:
            test_path = os.path.join(alt_path, audio_filename)
            if os.path.exists(test_path):
                audio_path = test_path
                break
    
    if not os.path.exists(audio_path):
        print(f"ERROR: Audio file not found: {audio_filename}", file=sys.stderr)
        sys.exit(1)
    
    model_path = "lstm_sken3/LSTM_mfcc_model.pth"  
    config_path = "mfcc_config.json"
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found: {model_path}", file=sys.stderr)
        sys.exit(1)
    
    predicted_class, confidence, processing_time = predict_tb_from_audio(
        audio_path, model_path, config_path
    )
    
    if predicted_class != -1:
        print(predicted_class)  
        print(f"Confidence: {confidence:.4f}")
        print(f"Execution TB Script: --- {processing_time:.4f} seconds ---")
    else:
        sys.exit(1)