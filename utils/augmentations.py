import random
import numpy as np
from PIL import Image
'''
def invert_affine_matrix(cos_theta, sin_theta, tx, ty, cx, cy):
    det = cos_theta ** 2 + sin_theta ** 2
    inv_cos = cos_theta / det
    inv_sin = sin_theta / det
    inv_tx = (-inv_cos * tx + inv_sin * ty)
    inv_ty = (-inv_sin * tx - inv_cos * ty)

    inv_tx += cx - inv_cos * cx + inv_sin * cy
    inv_ty += cy - inv_sin * cx - inv_cos * cy

    return inv_cos, -inv_sin, inv_tx, inv_sin, inv_cos, inv_ty
'''
def invert_affine_matrix(cos_theta, sin_theta, tx, ty, cx, cy, scale, angle):
    inv_scale = 1.0 / scale
    inv_angle_rad = np.radians(angle)
    inv_cos = inv_scale * np.cos(inv_angle_rad)
    inv_sin = inv_scale * np.sin(inv_angle_rad)

    inv_tx = -inv_cos * tx + inv_sin * ty
    inv_ty = -inv_sin * tx - inv_cos * ty

    inv_tx += cx - inv_cos * cx + inv_sin * cy
    inv_ty += cy - inv_sin * cx - inv_cos * cy

    return inv_cos, -inv_sin, inv_tx, inv_sin, inv_cos, inv_ty

def apply_augmentations(image, annotation, aug_config):
    if random.random() > aug_config.get('probability', 0.0):
        return image, annotation  # No augmentation

    lanes = annotation['lanes']

    # Shared random params
    do_flip = random.random() < 0.5
    angle = random.uniform(-aug_config.get('angle', 10), aug_config.get('angle', 10))
    translate = (
        random.uniform(-aug_config.get('translate', 0.1), aug_config.get('translate', 0.1)) * image.width,
        random.uniform(-aug_config.get('translate', 0.1), aug_config.get('translate', 0.1)) * image.height
    )
    scale = random.uniform(1 - aug_config.get('scale', 0.2), 1 + aug_config.get('scale', 0.2))

    cos_theta = scale * np.cos(np.radians(-angle))
    sin_theta = scale * np.sin(np.radians(-angle))
    tx, ty = translate
    cx, cy = image.width / 2, image.height / 2

    # Image flip
    if do_flip:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)

    # Invert affine for PIL
    #a, b, c, d, e, f = invert_affine_matrix(cos_theta, sin_theta, tx, ty, cx, cy)
    a, b, c, d, e, f = invert_affine_matrix(cos_theta, sin_theta, tx, ty, cx, cy, scale, angle)


    image = image.transform(
        image.size,
        Image.AFFINE,
        (a, b, c, d, e, f),
        resample=Image.BILINEAR
    )

    # Apply forward affine to points
    for lane in lanes:
        if 'points' in lane:
            for point in lane['points']:
                x, y = point

                if do_flip:
                    x = image.width - x

                x -= cx
                y -= cy

                x_new = cos_theta * x - sin_theta * y + tx
                y_new = sin_theta * x + cos_theta * y + ty

                x_new += cx
                y_new += cy

                point[0], point[1] = x_new, y_new

    # Clip points to image bounds
    width, height = image.size
    for lane in lanes:
        if 'points' in lane:
            for point in lane['points']:
                point[0] = max(0, min(point[0], width - 1))
                point[1] = max(0, min(point[1], height - 1))

    return image, annotation
