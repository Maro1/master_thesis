import numpy as np

from scipy.spatial.transform import Rotation

from read_write_model import read_model, qvec2rotmat

def find_valid_frames(model_path):
    """
    Read COLMAP model and return valid frames
    """
    model = read_model(model_path, ".txt")
    if model is None:
        print('WARNING: Could not read model at path: ', model_path)
        return model 
    _, images, _ = model
    image_ids = list()
    for image in images:
        image_ids.append(image)
    return sorted(image_ids)


def find_angles(model_path, frames):
    """
    Convert camera rotation to rotation and elevation angles
    """
    model = read_model(model_path, ".txt")
    if model is None:
        print('WARNING: Could not read model at path: ', model_path)
        return model 
    _, images, _ = model

    rotation_angle_list = list()
    elevation_angle_list = list()

    for frame in frames:
        img = images[frame]
        R = qvec2rotmat(img.qvec)

        # translation
        t = img.tvec

        # invert
        t = -R.T @ t
        R = R.T

        euler = Rotation.from_matrix(R).as_euler('YXZ', degrees=True)

        elevation = -euler[1]
        rotation = -euler[0]
        if rotation < 0:
            rotation = 360 + rotation

        rotation_angle_list.append(rotation)
        elevation_angle_list.append(elevation)

    return np.array(rotation_angle_list), np.array(elevation_angle_list)

if __name__ == '__main__':
    print(find_valid_frames('/cluster/home/mathiron/colmap_workspace/573_83801_166148/output'))
