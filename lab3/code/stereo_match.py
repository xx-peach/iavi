
'''
Simple example of stereo image matching and point cloud generation.

Resulting .ply file cam be easily viewed using MeshLab ( http://meshlab.sourceforge.net/ )
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''

def write_ply(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')


def main():
    print('loading images...')
    imgL = cv.imread('./cupL.png')
    imgR = cv.imread('./cupR.png')
    imgL = cv.resize(imgL, (imgL.shape[1] // 2, imgL.shape[0] // 2))
    imgR = cv.resize(imgR, (imgR.shape[1] // 2, imgR.shape[0] // 2))
    imgL = cv.pyrDown(imgL)
    imgR = cv.pyrDown(imgR)
    # imgL = cv.pyrDown(cv.imread(cv.samples.findFile('aloeL.jpg')))  # downscale images for faster processing
    # imgR = cv.pyrDown(cv.imread(cv.samples.findFile('aloeR.jpg')))

    # disparity range is tuned for 'aloe' image pair
    window_size = 5
    min_disp = 4
    num_disp = 76 - min_disp
    # min_disp = 4
    # num_disp = 20-min_disp
    stereo = cv.StereoSGBM_create(minDisparity = min_disp,
        numDisparities = num_disp,
        blockSize = 4,
        P1 = 8*3*window_size**2,
        P2 = 32*3*window_size**2,
        disp12MaxDiff = 1,
        uniquenessRatio = 10,
        speckleWindowSize = 100,
        speckleRange = 100,
        # mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    print('computing disparity...')
    disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

    print('generating 3d point cloud...',)
    h, w = imgL.shape[:2]
    # f = 0.8*w                          # guess for focal length
    f = 394.59508339
    Q = np.float32([[1, 0,  0, -0.5*w],
                    [0, -1, 0, 0.5*h], # turn points 180 deg around x-axis,
                    [0, 0,  0, -f], # so that y-axis looks up
                    [0, 0,  1, 0]])
    # Q = np.float32([[1, 0, 0, 0],
    #             [0, -1, 0, 0], # turn points 180 deg around x-axis,
    #             [0, 0, 0, -f], # so that y-axis looks up
    #             [0, 0, 1, 0]])
    # Q = np.float32([[1, 0, 0, 0],
    #                 [0, 1, 0, 0],
    #                 [0, 0, 1, 0],
    #                 [0, 0, 0, 1]])
    points = cv.reprojectImageTo3D(disp, Q)
    colors = cv.cvtColor(imgL, cv.COLOR_BGR2RGB)
    mask = disp > disp.min()
    out_points = points[mask]
    out_colors = colors[mask]
    out_fn = 'out.ply'
    write_ply(out_fn, out_points, out_colors)
    print('%s saved' % out_fn)

    cv.imshow('left', imgL)
    cv.imshow('disparity', (disp-min_disp)/num_disp)
    cv.waitKey()

    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()