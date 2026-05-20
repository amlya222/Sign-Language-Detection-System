import datasets.features.video as video_feature
def mock_decode_encode(self, value, **kwargs):
    return value
video_feature.Video.decode_example = mock_decode_encode
video_feature.Video.encode_example = mock_decode_encode

from datasets import load_dataset
import sys

dataset = load_dataset('akasheroor/American-Sign-Language-Dataset', streaming=True)
for item in dataset['train']:
    print("ALL KEYS:", list(item.keys()))
    print("Label:", item.get('label'))
    print("Text:", item.get('text'))
    
    # Just to see the first 5 samples
    continue
