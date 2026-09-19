"""CPU-only paired PSNR/SSIM: qo=0 video vs dense ground truth, and vs the
original per-(qo,ko)-selector video, matching score_first_video.py's method."""
import json
import statistics
import sys
from pathlib import Path

SOURCE = Path('/local2/denghaoran/GridMergeAttention/axis_wh_c8_source_20260918')
sys.path.insert(0, str(SOURCE))
from analysis.evaluate_video_quality import paired_frames
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

reference = Path('/local2/denghaoran/GridMergeAttention/experiments/axis_video_c4_20260918/dense_first20/000-00.mp4')
original_kernel = Path('/local2/denghaoran/GridMergeAttention/experiments/axis_video_c8_20260918/qw_kh_c8_current_r2_v1_20/000-00.mp4')
qo0 = Path('/local2/denghaoran/GridMergeAttention_axis_c8_opt/qo0_check/000-00.mp4')


def score(a, b):
    psnr, ssim = [], []
    for ref, got in paired_frames(a, b, expected_frames=81):
        psnr.append(float(peak_signal_noise_ratio(ref, got, data_range=255)))
        ssim.append(float(structural_similarity(ref, got, channel_axis=2, data_range=255)))
    return {'psnr': statistics.fmean(psnr), 'ssim': statistics.fmean(ssim), 'frames': len(psnr)}


result = {
    'qo0_vs_dense_reference': score(reference, qo0),
    'original_kernel_vs_dense_reference': score(reference, original_kernel),
    'qo0_vs_original_kernel': score(original_kernel, qo0),
}
print(json.dumps(result, indent=2))
