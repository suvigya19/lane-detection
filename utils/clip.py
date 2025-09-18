def clip_annotation(annotation, image_size):
    width, height = image_size
    for lane in annotation['lanes']:
        if 'points' in lane:
            for point in lane['points']:
                point[0] = max(0, min(point[0], width - 1))
                point[1] = max(0, min(point[1], height - 1))
    return annotation
