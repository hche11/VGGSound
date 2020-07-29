import os
import cv2
import json
import torch
import csv
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import time
from PIL import Image
import glob
import sys
from scipy import signal
import random
import soundfile as sf

class GetAudioVideoDataset(Dataset):

    def __init__(self, args, mode='train', transforms=None):
        data2path = {}
        classes = []
        classes_ = []
        data = []
        data2class = {}

        with open(args.csv_path + 'stat.csv') as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                classes.append(row[0])

        with open(args.csv_path  + args.test) as f:
            csv_reader = csv.reader(f)
            for item in csv_reader:
                if item[1] in classes and os.path.exists(args.data_path + item[0][:-3] + 'wav')::
                    data.append(item[0])
                    data2class[item[0]] = item[1]

        self.audio_path = args.data_path 
        self.mode = mode
        self.transforms = transforms
        self.classes = sorted(classes)
        self.data2class = data2class

        # initialize audio transform
        self._init_atransform()
        #  Retrieve list of audio and video files
        self.video_files = []
        
        for item in data:
            self.video_files.append(item)
        print('# of audio files = %d ' % len(self.video_files))
        print('# of classes = %d' % len(self.classes))


    def _init_atransform(self):
        self.aid_transform = transforms.Compose([transforms.ToTensor()])


    def __len__(self):
        return len(self.video_files)  

    def __getitem__(self, idx):
        wav_file = self.video_files[idx]
        # Audio
        samples, samplerate = sf.read(self.audio_path + wav_file[:-3]+'wav')

        # repeat in case audio is too short
        resamples = np.tile(samples,10)[:160000]

        resamples[resamples > 1.] = 1.
        resamples[resamples < -1.] = -1.
        frequencies, times, spectrogram = signal.spectrogram(resamples, samplerate, nperseg=512,noverlap=353)
        spectrogram = np.log(spectrogram+ 1e-7)

        mean = np.mean(spectrogram)
        std = np.std(spectrogram)
        spectrogram = np.divide(spectrogram-mean,std+1e-9)

        return spectrogram, resamples,self.classes.index(self.data2class[wav_file]),wav_file


