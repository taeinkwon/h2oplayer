import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import cv2
from collections import namedtuple
import os
import struct
import copy


def read_text(text_path, offset=0, half=0):
    with open(text_path, 'r') as txt_file:
        data = txt_file.readline().split(" ")
        data = list(filter(lambda x: x != "", data))

    if half:
        data_list = np.array(data)[offset:half].tolist(
        ) + np.array(data)[half+offset:].tolist()
        return np.array(data_list).reshape((-1, 3)).astype(np.float32)
    else:
        return np.array(data)[offset:].reshape((-1, 3)).astype(np.float32)


def get_xyz_from_depth(depth_img, cam_mtx, points_2d, refl_img=None, ir=False):
    # The valid range is between 50cm and 3.86 m (NFOV unbinned), 5.46 m NFOV 2x2 binned (SW)
    Z_MIN = 0.5  # 0.25
    if ir:
        Z_MAX = 5.00
    else:
        Z_MAX = 3.86  # 2.88

    fx = cam_mtx[0, 0]
    fy = cam_mtx[1, 1]
    cx = cam_mtx[0, 2]
    cy = cam_mtx[1, 2]
    do_colors = refl_img is not None

    x_vec = (points_2d[:, 0]-cx)*(depth_img.T).reshape(1, -1) / fx
    y_vec = (points_2d[:, 1]-cy)*(depth_img.T).reshape(1, -1) / fy
    z_vec = (depth_img.T).reshape(1, -1)
    mask = np.where((Z_MIN < z_vec[0]) & (Z_MAX > z_vec[0]))
    points = np.array([x_vec[0], y_vec[0], z_vec[0]]).T

    if do_colors:
        color = (refl_img/255.).astype(np.float64)
        color = np.swapaxes(color, 0, 1)
        colors = color.reshape((-1, 3))
        colors = np.flip(colors, 1)
        mask_color = np.where((Z_MIN < z_vec[0]) & (Z_MAX > z_vec[0]))

        return points[mask], colors[mask]
    else:
        return points[mask]


def parse_calib_init(path):
    with open(path, 'r') as f:
        lines = f.readlines()

    start_parms_cam0 = -1
    start_parms_cam1 = -1
    for i, l in enumerate(lines):
        if l.strip() == 'depth cam 0':
            start_parms_cam0 = i + 1
        elif l.strip() == 'depth cam 1':
            start_parms_cam1 = i + 1
    assert(start_parms_cam0 > -1)
    assert(start_parms_cam1 > -1)

    def extract_values(start_parms_id):
        w, h = [int(n) for n in lines[start_parms_id].split(' ')]
        fx, fy = lines[start_parms_id+1].split(' ')
        cx, cy = lines[start_parms_id+2].split(' ')
        cam_mtx = np.array([[float(fx), 0, float(cx)], [
                           0, float(fy), float(cy)], [0, 0, 1]])
        dst_k = [float(n) for n in lines[start_parms_id+3].split(' ')]
        dst_p = [float(n) for n in lines[start_parms_id+4].split(' ')]
        dist_coeffs = np.hstack((dst_k[:2], dst_p, dst_k[2:]))
        return w, h, cam_mtx, dist_coeffs

    w0, h0, cam_mtx0, dist_coeffs0 = extract_values(start_parms_cam0)
    new_cam_mtx0, roi0 = cv2.getOptimalNewCameraMatrix(
        cam_mtx0, dist_coeffs0, (w0, h0), alpha=0.)
    cam0 = CameraCalib(cam_mtx=cam_mtx0, dist_coeffs=dist_coeffs0,
                       w=w0, h=h0, new_cam_mtx=new_cam_mtx0, roi=roi0)

    w1, h1, cam_mtx1, dist_coeffs1 = extract_values(start_parms_cam1)
    new_cam_mtx1, roi1 = cv2.getOptimalNewCameraMatrix(
        cam_mtx1, dist_coeffs1, (w1, h1), alpha=0.)
    cam1 = CameraCalib(cam_mtx=cam_mtx1, dist_coeffs=dist_coeffs1,
                       w=w1, h=h1, new_cam_mtx=new_cam_mtx1, roi=roi1)

    return cam0, cam1


def load_refl_img(path, cam):
    img = cv2.imread(path)
    # Slightly increase brightness of AB images,
    # otherwise corners will not be visible most of the times
    img = (img.astype(float) * BRIGHT_CONST).astype(np.uint8)
    img = undistort(img, cam)
    return img


def undistort(img, cam):
    # The two methods should be equivalent to a call to undistort (only the interpolation changes;
    # bilinear interp tends to give noisy depth maps)
    # See https://docs.opencv.org/3.0-beta/modules/imgproc/doc/geometric_transformations.html#undistort
    # (map1, map2) = cv2.initUndistortRectifyMap(cam.cam_mtx, cam.dist_coeffs, np.eye(3), cam.new_cam_mtx, (cam.w, cam.h), cv2.CV_32FC1)
    # return cv2.remap(img, map1, map2, cv2.INTER_NEAREST)
    # return cv2.undistort(img, cam.cam_mtx, cam.dist_coeffs, None, cam.new_cam_mtx)
    (map1, map2) = cv2.initUndistortRectifyMap(cam.cam_mtx, cam.dist_coeffs,
                                               np.eye(3), cam.new_cam_mtx, (cam.w, cam.h), cv2.CV_32FC1)
    return cv2.remap(img, map1, map2, cv2.INTER_NEAREST)


def parse_cam_extrin(path):
    with open(path, 'r') as f:
        lines = f.readlines()

    w, h = [int(n) for n in lines[0].split(' ')]
    fx, fy, cx, cy = lines[1].split(' ')
    cam_mtx = np.array([[float(fx), 0, float(cx)], [
                       0, float(fy), float(cy)], [0, 0, 1]])
    dist_coeffs = np.array([float(lines[n]) for n in range(2, 10)])

    ext_rot = np.array([[float(n) for n in lines[20].split(' ')[0:3]],
                        [float(n) for n in lines[21].split(' ')[0:3]],
                        [float(n) for n in lines[22].split(' ')[0:3]]])
    ext_tran = np.array([float(n)/1000.0 for n in lines[24].split(' ')[0:3]])

    new_cam_mtx, roi = cv2.getOptimalNewCameraMatrix(
        cam_mtx, dist_coeffs, (w, h), alpha=0.)
    cam = CameraCalib(cam_mtx=cam_mtx, dist_coeffs=dist_coeffs, w=w,
                      h=h, R=ext_rot, T=ext_tran, new_cam_mtx=new_cam_mtx, roi=roi)

    return cam


def cal_points2d(depth_img):
    points_2d = []
    for u in np.arange(depth_img.shape[1]):
        for v in np.arange(depth_img.shape[0]):
            points_2d.append(np.array([u, v]).reshape((1, -1)))
    points_2d = np.vstack(points_2d)
    return points_2d


def compute_pcloud(depth_img, refl_img, cam, points_2d, scale_to_m=False, rgb=False, ir=False):

    if scale_to_m:
        depth_img = depth_img.astype(float)/1000.

    # if (rgb==True):
    #    points, colors = get_xyz_from_depth(depth_img, cam.new_cam_mtx, points_2d,refl_img,True)
    # else:
    points, colors = get_xyz_from_depth(
        depth_img, cam, points_2d, refl_img, ir)
    pcloud = o3d.geometry.PointCloud()
    pcloud.points = o3d.utility.Vector3dVector(points)
    pcloud.colors = o3d.utility.Vector3dVector(colors)
    return pcloud


def apply_RT(RT, pcloud):
    #RT = np.load(RT_path)
    R = RT['R']
    T = RT['T']
    transf_mtx = np.vstack(
        (np.hstack((R, T.reshape((-1, 1)))), np.array([0, 0, 0, 1]).reshape((1, -1))))
    inv_transf_mtx = np.linalg.inv(transf_mtx)
    pcloud.transform(inv_transf_mtx)
    return pcloud


def get_pointcloud(depth_path, refl_path, calib, points_2d, rgb=0, ir=False):
    depth_img = cv2.imread(
        depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    refl_img = cv2.imread(refl_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if ir:
        refl_img = cv2.cvtColor(refl_img, cv2.COLOR_GRAY2RGB)
    pcloud = compute_pcloud(depth_img, refl_img, calib,
                            points_2d, scale_to_m=True, ir=ir)
    #pcloud, ind0 = pcloud.remove_radius_outlier(nb_points = NB_POINTS, radius = RADIOUS)
    return pcloud
