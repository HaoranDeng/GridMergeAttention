"""Per-frame PSNR between two directories of paired videos (same file names).

python analysis/video_pair_psnr.py REFERENCE_DIR TEST_DIR
PSNR is computed per decoded RGB frame with data range 255, averaged over the
frames of a video and then over videos (the paper's definition).
"""
import sys,glob,os
import numpy as np
import imageio.v3 as iio

ref_dir,test_dir=sys.argv[1],sys.argv[2]
scores=[]
for path in sorted(glob.glob(os.path.join(ref_dir,'*.mp4'))):
    other=os.path.join(test_dir,os.path.basename(path))
    if not os.path.exists(other):continue
    a=iio.imread(path).astype(np.float64);b=iio.imread(other).astype(np.float64)
    assert a.shape==b.shape,(path,a.shape,b.shape)
    mse=((a-b)**2).reshape(a.shape[0],-1).mean(1)
    psnr=float(np.mean(10*np.log10(255.0**2/np.maximum(mse,1e-12))));scores.append(psnr)
    print(f'{os.path.basename(path)} frames={a.shape[0]} PSNR={psnr:.2f} dB')
print(f'mean PSNR {np.mean(scores):.2f} dB over {len(scores)} videos')
