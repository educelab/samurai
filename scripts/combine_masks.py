import imageio.v3 as iio
import os
import argparse
import numpy as np
from natsort import natsorted
from pathlib import Path

# returns mask number from file 
def get_mask_num(mask, return_as_int):
    sect = mask.split('/')
    temp = sect[-1].split('_')
    mask_num = temp[1].split('.')[0]
    if return_as_int:
        return int(mask_num)
    return mask_num

def main(args):
    input_dir = args.input_dir
    output_dir = args.output_dir
    frames = args.init_frames

    temp = os.listdir(frames)
    dim0 = len(temp)
    del temp

    mask_groups = [[] for _ in range(dim0)]
    masks = natsorted(os.listdir(input_dir))
    
    # storing masks of same frame together in a new list
    for mask in masks:
        mask_num = get_mask_num(mask, return_as_int=True)
        mask_groups[mask_num].append(os.path.join(input_dir, mask))

    # getting mask size 
    shape = iio.imread(mask_groups[0][0]).shape

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # combining masks and writing the image out
    for group in mask_groups:
        out_mask = np.zeros(shape)
        mask_num = get_mask_num(group[0], return_as_int=False)
        out_name = os.path.join(output_dir, str(mask_num)+'.png')

        for i, fn in enumerate(group, 1):
            img = iio.imread(fn)
            idx = np.where(img == 255)
            out_mask[idx] = i
        
        print(f"Writing out to {out_name}")
        iio.imwrite(out_name, out_mask.astype(np.uint8))
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Input directory of masks.")
    parser.add_argument("--output_dir", required=True, help="Output directory for combined masks.")
    parser.add_argument("--init_frames", required=True, help="Directory of the images the masks were generated from.")
    args = parser.parse_args()
    main(args)

