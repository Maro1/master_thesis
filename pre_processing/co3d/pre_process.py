import argparse
import os
from typing import cast, Optional, Tuple

import numpy as np
import cv2
import torch

from pytorch3d.implicitron.dataset.json_index_dataset import JsonIndexDataset
from pytorch3d.implicitron.tools.config import expand_args_fields
from pytorch3d.transforms import matrix_to_euler_angles, matrix_to_quaternion

from utils import find_angles, find_valid_frames

def parse_args():
    parser = argparse.ArgumentParser(
        prog='CO3D Pre-Process',
        description='Pre-processes CO3D dataset for use with GET3D')
    parser.add_argument('root')
    parser.add_argument('camera_root')
    parser.add_argument('output')
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--num_frames', type=int, default=24)
    parser.add_argument('--category', type=str, default='book')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=-1)

    return parser.parse_args()


def get_frame_data(
    dataset: JsonIndexDataset,
    sequence_name: Optional[str] = None,
    mask_points: bool = True,
    max_frames: int = -1,
    num_workers: int = 0,
    load_dataset_point_cloud: bool = False,
): 
    """
    Make a point cloud by sampling random points from each frame the dataset.
    """

    if len(dataset) == 0:
        raise ValueError("The dataset is empty.")

    if not dataset.load_depths:
        raise ValueError("The dataset has to load depths (dataset.load_depths=True).")

    if mask_points and not dataset.load_masks:
        raise ValueError(
            "For mask_points=True, the dataset has to load masks"
            + " (dataset.load_masks=True)."
        )

    # setup the indices of frames loaded from the dataset db
    sequence_entries = list(range(len(dataset)))
    if sequence_name is not None:
        sequence_entries = [
            ei
            for ei in sequence_entries
            # pyre-ignore[16]
            if dataset.frame_annots[ei]["frame_annotation"].sequence_name
            == sequence_name
        ]
        if len(sequence_entries) == 0:
            raise ValueError(
                f'There are no dataset entries for sequence name "{sequence_name}".'
            )

    # subsample loaded frames if needed
    if (max_frames > 0) and (len(sequence_entries) > max_frames):
        sequence_entries = [
            sequence_entries[i]
            for i in torch.randperm(len(sequence_entries))[:max_frames].sort().values
        ]

    # take only the part of the dataset corresponding to the sequence entries
    sequence_dataset = torch.utils.data.Subset(dataset, sequence_entries)

    # load the required part of the dataset
    loader = torch.utils.data.DataLoader(
        sequence_dataset,
        batch_size=len(sequence_dataset),
        shuffle=False,
        num_workers=num_workers,
        collate_fn=dataset.frame_data_type.collate,
    )

    frame_data = next(iter(loader))  # there's only one batch

    return frame_data

def pre_process(args):
    frame_file = os.path.join(args.root, args.category, "frame_annotations.jgz")
    sequence_file = os.path.join(args.root, args.category, "sequence_annotations.jgz")
    expand_args_fields(JsonIndexDataset)

    dataset = JsonIndexDataset(
                frame_annotations_file=frame_file,
                sequence_annotations_file=sequence_file,
                dataset_root=args.root,
                image_height=args.resolution,
                image_width=args.resolution,
                box_crop=True,
                load_point_clouds=False,
            )

    num_seq = len(dataset.seq_annots.keys())
    end = args.end if args.end != -1 or args.end > num_seq else num_seq

    if args.start >= num_seq:
        return

    sequences = list(dataset.seq_annots.keys())[args.start:end] 

    print('START, END: ' + str(args.start) + ', ' + str(args.end))

    for sequence in sequences:
        if os.path.exists(os.path.join(os.path.abspath(args.output), 'img', args.category, sequence)):
            continue

        sequence_frame_data = get_frame_data(
            dataset,
            sequence_name=sequence,
            mask_points=True,
            num_workers=4,
            load_dataset_point_cloud=True,
        )

        model_path = os.path.join(args.camera_root, sequence, 'output')
        if not os.path.exists(model_path):
            print('WARNING: No model at path: ', model_path)
            continue

        # Get valid frames from COLMAP returned output
        valid_frames = find_valid_frames(model_path)
        while np.max(valid_frames) >= len(sequence_frame_data.image_rgb):
            valid_frames.remove(np.max(valid_frames))
        valid_frames = np.array(valid_frames)

        frame_indices = np.round(np.linspace(0, len(valid_frames) - 1, args.num_frames, dtype='int'))
        frames = valid_frames[frame_indices]

        # Get camera rotation and elevation angles
        angles = find_angles(model_path, frames)
        if angles is None:
            print('WARNING: Cannot find angles for sequence: ', sequence)
            continue

        rotations, elevations = angles 
        camera_folder = os.path.join(os.path.abspath(args.output), 'camera', args.category, sequence)
        os.makedirs(camera_folder, exist_ok=True)

        np.save(os.path.join(camera_folder, 'rotation'), rotations)
        np.save(os.path.join(camera_folder, 'elevation'), elevations)

        i = 1
        for frame_idx in frames:
            image = sequence_frame_data.image_rgb[frame_idx].numpy()
            mask = sequence_frame_data.fg_probability[frame_idx].numpy()

            image = np.swapaxes(image, 0, 1)
            image = np.swapaxes(image, 1, 2)
            image = image * 255
            image = image.astype(np.uint8)

            mask = np.swapaxes(mask, 0, 1)
            mask = np.swapaxes(mask, 1, 2)
            mask = mask * 255
            mask = mask.astype(np.uint8)

            # Apply Otsu's binarization to segmentation mask
            mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            image = cv2.bitwise_and(image, image, mask=mask)

            rgba = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
            rgba[:, :, 3] = mask

            img_folder = os.path.join(os.path.abspath(args.output), 'img', args.category, sequence)
            img_filename = '{0:03}'.format(i) + '.png'
            os.makedirs(img_folder, exist_ok=True)
            i += 1

            cv2.imwrite(os.path.join(img_folder, img_filename), rgba)

if __name__ == '__main__':
    args = parse_args()
    pre_process(args)
