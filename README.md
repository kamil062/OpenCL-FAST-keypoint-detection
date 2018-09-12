# OpenCL FAST keypoint detection

Simple OpenCL implementation of FAST keypoint detection algorithm.

   [Faster and better: A machine learning approach to corner detection, E. Rosten, R. Porter and T. Drummond, PAMI 2010](https://www.edwardrosten.com/work/rosten_2008_faster.pdf)
   
   [Machine learning for high-speed corner detection, E. Rosten and T. Drummond, ECCV 2006](https://www.edwardrosten.com/work/rosten_2006_machine.pdf)

Code example 

```python
import pyopencl as cl
import cv2
import os

from cl_fast import fast_detect

os.environ['PYOPENCL_COMPILER_OUTPUT'] = '0'

def cl_init(platform, device):

    platform = cl.get_platforms()[platform]
    device = platform.get_devices()[device]
    context = cl.Context([device])
    queue = cl.CommandQueue(context)

    return context, queue

if __name__ == "__main__":

    ctx, queue = cl_init(1, 0)

    img = cv2.imread("frame_0.png", -1)

    keypoints = fast_detect(ctx, queue, img)

    img1 = cv2.drawKeypoints(img, keypoints, None, color=(255,0,0), flags=0)

    cv2.imshow("FAST OpenCL", img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
```
