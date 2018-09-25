# -*- coding: utf-8 -*-

import numpy as np
import pyopencl as cl
import cv2
import os


def fast_detect(ctx, queue, image, N=12, threshold=10, nonmax=True):
    """Detects FAST keypoints using OpenCL

    :param ctx: OpenCL context
    :param queue: OpenCL command queue
    :param image: image for keypoints detection
    :param N: how many points in circle should be considered, defaults to 12
    :param N: int, optional
    :param threshold: FAST threshold, defaults to 10
    :param threshold: int, optional
    :param nonmax: if True, non-max suppression is applied on keypoints, defaults to True
    :param nonmax: bool, optional
    """

    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    h, w = image.shape

    h_image = cl.image_from_array(ctx, image, 1)
    h_is_keypoint = np.zeros(image.ravel().shape).astype(np.int32)
    h_scores = np.zeros(image.ravel().shape).astype(np.int32)

    d_is_keypoint = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, h_is_keypoint.nbytes)
    d_scores = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, h_scores.nbytes)

    prg = cl.Program(ctx, open(os.path.dirname(__file__) + "\kernel.cl", "r").read()).build()

    prg.fast_detect(queue, (w, h), None,
                    h_image, np.int32(w), np.int32(h), np.int32(N), np.float32(threshold),
                    d_is_keypoint, d_scores)

    cl.enqueue_copy(queue, h_is_keypoint, d_is_keypoint)
    cl.enqueue_copy(queue, h_scores, d_scores)

    h_is_keypoint = h_is_keypoint.reshape(h, w)
    corners = np.asarray(np.where(h_is_keypoint == 1)).T
    scores = np.asarray(np.where(h_scores != 0)).T

    if nonmax:
        sc = np.zeros(image.shape)
        for i in range(len(corners)-1):
            sc[corners[i][0], corners[i][1]] = scores[i]

        nonmax_keypoints = []

        for i in range(len(corners)-1):
            s = scores[i]
            y = corners[i][0]
            x = corners[i][1]

            if s >= sc[y - 1][x + 1] and s >= sc[y - 1][x] and s >= sc[y - 1][x - 1] and s >= sc[y][x + 1] and \
                    s >= sc[y][x - 1] and s >= sc[y + 1][x + 1] and s >= sc[y + 1][x] and s >= sc[y + 1][x - 1]:
                nonmax_keypoints.append(cv2.KeyPoint(x, y, 1.0))

        return nonmax_keypoints

    return [cv2.KeyPoint(y, x, 1.0) for x, y in corners]
