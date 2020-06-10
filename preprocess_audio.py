import glob
import multiprocessing
import subprocess
import os
import argparse

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--video_input',
        default='',
        type=str,
        help='Input directory path of videos or audios')
    parser.add_argument(
        '--audio_output',
        default='',
        type=str,
        help='Output directory path of videos')
    return parser.parse_args() 

def convert(v):
    subprocess.check_call([
    'ffmpeg',
    '-n',
    '-i', v,
    '-acodec', 'pcm_s16le',
    '-ac','1',
    '-ar','16000',
    args.audio_output + '%s.wav' % v.split('/')[-1][:-4]])

def obtain_list():
    files = []
    txt = glob.glob(args.video_input + '/*.mp4') # '/*.flac'
    for item in txt:
        files.append(item)
    return files

args = get_arguments()
p = multiprocessing.Pool(32)
p.map(convert, obtain_list())

