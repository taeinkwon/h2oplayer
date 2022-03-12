import argparse
import open3d as o3d
import numpy as np
import copy
import glob
import tqdm
import time
import os
import json
import csv
import torch
import cv2
from utils import *
from scipy.spatial.transform import Rotation as R

from manopth.manolayer import ManoLayer


# H2O Player
class PlayerCallBack:

    def __init__(self, results, frame_list, objects=[], left_hand=[], right_hand=[], object_meshes=[], object_path=[], hand_path=[]):
        self.flag_exit = False
        self.flag_pause = False
        #self.ply = ply
        self.count = 0
        self.frame_list = frame_list
        self.fps = results.fps
        self.rgb = results.rgb

        self.current_rgb = o3d.geometry.Image()
        self.current_ply = o3d.geometry.PointCloud()
        self.current_object = o3d.geometry.TriangleMesh()
        self.current_lh = o3d.geometry.TriangleMesh()
        self.current_rh = o3d.geometry.TriangleMesh()
        self.current_mesh = o3d.geometry.TriangleMesh()
        self.obj_idx = 0
        self.checkbox = results.checkbox
        self.object_path = object_path
        self.objects = objects
        self.left_hand = left_hand
        self.right_hand = right_hand
        self.object_meshes = object_meshes
        self.results = results
        self.hand_path = hand_path
        self.hands = results.hands
        if self.results.ego:
            if self.results.capture_img:
                self.param = o3d.io.read_pinhole_camera_parameters(
                    "viewpoint_1280.json")
                self.fps = 10000
            else:

                self.param = o3d.io.read_pinhole_camera_parameters(
                    "viewpoint.json")
            # with open(self.results.source +'/cam4/calib/undist_calib.json') as f:
            #    self.cam_param  = json.load(f)

    def rerun_callback(self, vis):
        self.count = 0
        return False

    def esc_callback(self, vis):
        self.flag_exit = True
        return False

    def pause_callback(self, vis):
        if self.flag_pause:
            self.flag_pause = False
        else:
            self.flag_pause = True
        return False

    def frame_back_callback(self, vis):
        if self.count > 0:
            self.count -= 1
        else:
            self.count = 0
        print(self.frame_list[self.count])
        return False

    def frame_forward_callback(self, vis):
        self.count += 1

        if self.count >= len(self.frame_list)-1:
            self.count = len(self.frame_list)-1
        print(self.frame_list[self.count])
        return False

    def left_hand_callback(self, vis):
        if str(self.frame_list[self.count]) in self.lh_pose_false.keys():
            del self.lh_pose_false[str(self.frame_list[self.count])]
            print("lh true")
        else:
            self.lh_pose_false[str(self.frame_list[self.count])] = True
            print("lh false")
        with open(self.lhpose_file_path, 'w') as outfile:
            json.dump(self.lh_pose_false, outfile)
        return False

    def right_hand_callback(self, vis):
        if str(self.frame_list[self.count]) in self.rh_pose_false.keys():
            del self.rh_pose_false[str(self.frame_list[self.count])]
            print("rh true")
        else:
            self.rh_pose_false[str(self.frame_list[self.count])] = True
            print("rh false")
        with open(self.rhpose_file_path, 'w') as outfile:
            json.dump(self.rh_pose_false, outfile)
        return False

    def reload_obj_callback(self, vis):
        print("reload obj")
        del self.objects
        del self.left_hand
        del self.right_hand
        self.objects, self.left_hand, self.right_hand = load_meshes(
            object_meshes=self.object_meshes, results=self.results, object_path=self.object_path, hand_path=self.hand_path, objects_flag=self.results.object, hands=self.hands)
        return False

    def run(self, plys, rgbs=[], object_info=[]):
        glfw_key_esc = 256
        glfw_key_space = 32
        glfw_key_r = 82
        glfw_key_comma = 44  # ,
        glfw_key_period = 46  # .
        glfw_key_q = 81
        glfw_key_u = 85
        glfw_key_a = 65
        glfw_key_s = 83
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.register_key_callback(glfw_key_r, self.rerun_callback)
        vis.register_key_callback(glfw_key_esc, self.esc_callback)
        vis.register_key_callback(glfw_key_space, self.pause_callback)
        vis.register_key_callback(glfw_key_comma, self.frame_back_callback)
        vis.register_key_callback(glfw_key_period, self.frame_forward_callback)
        vis.register_key_callback(glfw_key_u, self.reload_obj_callback)
        vis.register_key_callback(glfw_key_a, self.left_hand_callback)
        vis.register_key_callback(glfw_key_s, self.right_hand_callback)
        vis.create_window('pcd', width=960, height=540, left=0, top=0)
        # print("plys",plys)
        if self.rgb:
            empty_frame = np.zeros_like(rgbs[0][0])
            vis2 = o3d.visualization.VisualizerWithKeyCallback()
            vis2.create_window(window_name='rgb', width=960,
                               height=800, left=960, top=0)
            vis2.register_key_callback(glfw_key_r, self.rerun_callback)
            vis2.register_key_callback(glfw_key_esc, self.esc_callback)
            vis2.register_key_callback(glfw_key_space, self.pause_callback)
            vis2.register_key_callback(
                glfw_key_comma, self.frame_back_callback)
            vis2.register_key_callback(
                glfw_key_period, self.frame_forward_callback)
            vis2.register_key_callback(glfw_key_u, self.reload_obj_callback)
            vis2.register_key_callback(glfw_key_a, self.left_hand_callback)
            vis2.register_key_callback(glfw_key_s, self.right_hand_callback)

        if self.results.object or self.results.hands:
            vis3 = o3d.visualization.VisualizerWithKeyCallback()
            vis3.create_window(window_name='object', width=960,
                               height=540, left=0, top=540)
            vis3.register_key_callback(glfw_key_r, self.rerun_callback)
            vis3.register_key_callback(glfw_key_esc, self.esc_callback)
            vis3.register_key_callback(glfw_key_space, self.pause_callback)
            vis3.register_key_callback(
                glfw_key_comma, self.frame_back_callback)
            vis3.register_key_callback(
                glfw_key_period, self.frame_forward_callback)
            vis3.register_key_callback(glfw_key_u, self.reload_obj_callback)
            vis3.register_key_callback(glfw_key_a, self.left_hand_callback)
            vis3.register_key_callback(glfw_key_s, self.right_hand_callback)

            self.current_object = copy.copy(self.objects[0])
            if self.results.hands:

                self.current_object = copy.copy(self.objects[0])
                self.current_lh = copy.copy(self.left_hand[0])
                self.current_rh = copy.copy(self.right_hand[0])
                self.current_mesh = self.current_object
                self.current_mesh_hands = self.current_lh + self.current_rh

                # print(np.asarray(self.current_mesh.triangle_normals))
            else:
                self.current_mesh = self.current_object
            self.obj_idx = object_info[self.count]["idx"]

        if self.results.ego:
            vis4 = o3d.visualization.VisualizerWithKeyCallback()
            if self.results.capture_img:
                vis4.create_window(window_name='ego', width=1280,
                                   height=720, left=960, top=800)
                vis4_opt = vis4.get_render_option()
                #vis4_opt.background_color = np.asarray([0, 0, 0])
            else:
                vis4.create_window(window_name='ego', width=640,
                                   height=360, left=960, top=800)
            vis4.register_key_callback(glfw_key_r, self.rerun_callback)
            vis4.register_key_callback(glfw_key_esc, self.esc_callback)
            vis4.register_key_callback(glfw_key_space, self.pause_callback)
            vis4.register_key_callback(
                glfw_key_comma, self.frame_back_callback)
            vis4.register_key_callback(
                glfw_key_period, self.frame_forward_callback)
            vis4.register_key_callback(glfw_key_u, self.reload_obj_callback)
            vis4.register_key_callback(glfw_key_a, self.left_hand_callback)
            vis4.register_key_callback(glfw_key_s, self.right_hand_callback)
            ctr4 = vis4.get_view_control()

        self.current_ply = copy.copy(plys[0])
        vis_geometry_added = False
        while not self.flag_exit:
            start_time = time.time()

            if self.count >= len(plys):
                vis.poll_events()
                vis.update_renderer()
                continue

            self.current_ply.points = plys[self.count].points
            self.current_ply.colors = plys[self.count].colors

            if (self.results.object or self.results.hands) and vis_geometry_added:
                if self.obj_idx != object_info[self.count]["idx"]:
                    vis.remove_geometry(self.current_mesh)
                    vis3.remove_geometry(self.current_mesh)
                    vis.remove_geometry(self.current_mesh_hands)
                    vis3.remove_geometry(self.current_mesh_hands)
                    if self.results.ego:
                        vis4.remove_geometry(self.current_mesh)
                        vis4.remove_geometry(self.current_mesh_hands)
                    self.current_object = self.objects[self.count]
                    if self.results.hands:
                        self.current_lh = self.left_hand[self.count]
                        self.current_rh = self.right_hand[self.count]
                        self.current_mesh = self.current_object
                        self.current_mesh_hands = self.current_lh + self.current_rh
                    else:
                        self.current_mesh = self.current_object
                    vis.add_geometry(self.current_mesh)
                    vis3.add_geometry(self.current_mesh)
                    vis.add_geometry(self.current_mesh_hands)
                    vis3.add_geometry(self.current_mesh_hands)
                    if self.results.ego:
                        vis4.add_geometry(self.current_mesh)
                        vis4.add_geometry(self.current_mesh_hands)
                    self.obj_idx = object_info[self.count]["idx"]
                self.current_object.vertex_normals = o3d.utility.Vector3dVector(
                    np.asarray(self.objects[self.count].vertex_normals))
                self.current_object.vertex_colors = self.objects[self.count].vertex_colors
                self.current_object.vertices = o3d.utility.Vector3dVector(
                    np.asarray(self.objects[self.count].vertices))
                if self.results.hands:
                    self.current_lh.vertex_normals = o3d.utility.Vector3dVector(
                        np.asarray(self.left_hand[self.count].vertex_normals))
                    self.current_rh.vertex_normals = o3d.utility.Vector3dVector(
                        np.asarray(self.right_hand[self.count].vertex_normals))
                    self.current_lh.vertex_colors = self.left_hand[self.count].vertex_colors
                    self.current_rh.vertex_colors = self.right_hand[self.count].vertex_colors
                    self.current_lh.vertices = o3d.utility.Vector3dVector(
                        np.asarray(self.left_hand[self.count].vertices))
                    self.current_rh.vertices = o3d.utility.Vector3dVector(
                        np.asarray(self.right_hand[self.count].vertices))

                if results.checkbox:
                    if str(self.frame_list[self.count]) in self.obj_pose_false.keys():
                        self.current_object.vertex_colors = o3d.utility.Vector3dVector(np.zeros((np.shape(
                            self.current_object.vertex_colors)[0], np.shape(self.current_object.vertex_colors)[1])))
                    if self.results.hands:
                        if str(self.frame_list[self.count]) in self.lh_pose_false.keys():
                            self.current_lh.vertex_colors = o3d.utility.Vector3dVector(np.zeros((np.shape(
                                self.current_lh.vertex_colors)[0], np.shape(self.current_lh.vertex_colors)[1])))
                        if str(self.frame_list[self.count]) in self.rh_pose_false.keys():
                            self.current_rh.vertex_colors = o3d.utility.Vector3dVector(np.zeros((np.shape(
                                self.current_rh.vertex_colors)[0], np.shape(self.current_rh.vertex_colors)[1])))

                if self.results.hands:
                    #self.current_mesh =  self.current_object + self.current_lh + self.current_rh
                    self.current_mesh.vertex_normals = o3d.utility.Vector3dVector(
                        np.asarray(self.current_object.vertex_normals))
                    self.current_mesh_hands.vertex_normals = o3d.utility.Vector3dVector(np.concatenate(
                        (np.asarray(self.current_lh.vertex_normals), np.asarray(self.current_rh.vertex_normals))))
                    empty_points = o3d.utility.Vector3dVector(
                        np.zeros((778, 3)))
                    if len(np.asarray(self.current_rh.vertices)) == 0:
                        self.current_rh.vertices = o3d.utility.Vector3dVector(
                            empty_points)
                        self.current_rh.vertex_colors = o3d.utility.Vector3dVector(
                            empty_points)
                    if len(np.asarray(self.current_lh.vertices)) == 0:
                        self.current_lh.vertices = o3d.utility.Vector3dVector(
                            empty_points)
                        self.current_lh.vertex_colors = o3d.utility.Vector3dVector(
                            empty_points)
                    #self.current_mesh.vertices = o3d.utility.Vector3dVector(np.concatenate((np.asarray(self.current_object.vertices),np.asarray(self.current_lh.vertices),np.asarray(self.current_rh.vertices))))
                    #self.current_mesh.vertex_colors = o3d.utility.Vector3dVector(np.concatenate((np.asarray(self.current_object.vertex_colors),np.asarray(self.current_lh.vertex_colors),np.asarray(self.current_rh.vertex_colors))))
                    self.current_mesh.vertices = o3d.utility.Vector3dVector(
                        np.asarray(self.current_object.vertices))
                    self.current_mesh.vertex_colors = o3d.utility.Vector3dVector(
                        np.asarray(self.current_object.vertex_colors))
                    self.current_mesh_hands.vertices = o3d.utility.Vector3dVector(np.concatenate(
                        (np.asarray(self.current_lh.vertices), np.asarray(self.current_rh.vertices))))
                    self.current_mesh_hands.vertex_colors = o3d.utility.Vector3dVector(np.concatenate(
                        (np.asarray(self.current_lh.vertex_colors), np.asarray(self.current_rh.vertex_colors))))
                else:
                    self.current_mesh.vertices = o3d.utility.Vector3dVector(
                        np.asarray(self.objects[self.count].vertices))
                    #self.current_mesh =  self.current_object

            if self.rgb and vis_geometry_added:
                self.current_rgb.clear()
                vis2.remove_geometry(self.current_rgb)
                if self.results.cam == 2:
                    self.current_rgb = o3d.geometry.Image(np.concatenate(
                        (rgbs[0][self.count], rgbs[1][self.count]), axis=0))
                elif self.results.cam == 5:

                    self.current_rgb = o3d.geometry.Image(np.concatenate((
                        np.concatenate(
                            (rgbs[0][self.count], rgbs[1][self.count]), axis=1),
                        np.concatenate(
                            (rgbs[2][self.count], rgbs[3][self.count]), axis=1),
                        np.concatenate((empty_frame, rgbs[4][self.count]), axis=1)), axis=0))
                #self.current_rgb= o3d.geometry.Image(rgbs[4])
                vis2.add_geometry(self.current_rgb)

            # In order to put shader
            # if not self.results.capture_img:
            self.current_mesh.compute_vertex_normals()
            if self.results.hands:
                self.current_mesh_hands.compute_vertex_normals()

            if not vis_geometry_added:
                vis.add_geometry(self.current_ply)

                if self.rgb:
                    vis2.add_geometry(self.current_rgb)

                if self.results.object or self.results.hands:
                    vis3.add_geometry(self.current_mesh)
                    vis.add_geometry(self.current_mesh)
                if self.results.hands:
                    vis3.add_geometry(self.current_mesh_hands)
                    vis.add_geometry(self.current_mesh_hands)

                if self.results.ego:
                    # vis4.add_geometry(self.current_ply)
                    if self.results.object or self.results.hands:
                        vis4.add_geometry(self.current_mesh)
                    if self.results.hands:
                        vis4.add_geometry(self.current_mesh_hands)
                    ctr4.convert_from_pinhole_camera_parameters(self.param)
                vis_geometry_added = True

            vis.update_geometry(self.current_ply)
            vis.update_geometry(self.current_mesh)
            if self.results.hands:
                vis.update_geometry(self.current_mesh_hands)
            vis.poll_events()
            vis.update_renderer()
            if self.rgb:
                vis2.update_geometry(self.current_rgb)
                vis2.poll_events()
                vis2.update_renderer()
            if self.results.object or self.results.hands:
                vis3.update_geometry(self.current_mesh)
                if self.results.hands:
                    vis3.update_geometry(self.current_mesh_hands)

                vis3.poll_events()
                vis3.update_renderer()
            if self.results.ego:
                vis4.update_geometry(self.current_mesh)
                if self.results.hands:
                    vis4.update_geometry(self.current_mesh_hands)
                vis4.poll_events()
                vis4.update_renderer()
                cam_pose_path = results.source + \
                    "/cam4/cam_pose/{0:06d}.txt".format(
                        self.frame_list[self.count])
                with open(cam_pose_path, 'r') as txt_file:
                    data = txt_file.readline().split(" ")
                    data = list(filter(lambda x: x != "", data))
                # because it's world to cam
                self.param.extrinsic = np.linalg.inv(
                    np.array(data).astype(np.float).reshape((4, 4)))
                ctr4.convert_from_pinhole_camera_parameters(self.param)

            while True:
                # vis.update_geometry()
                vis.poll_events()
                vis.update_renderer()
                if self.rgb:
                    vis2.update_geometry(self.current_rgb)
                    vis2.poll_events()
                if self.results.object or self.results.hands:
                    vis3.update_geometry(self.current_mesh)
                    if self.results.hands:
                        vis3.update_geometry(self.current_mesh_hands)

                    vis3.poll_events()
                if self.results.ego:
                    vis4.poll_events()
                    vis4.update_renderer()
                if (time.time()-start_time) > (1/self.fps):
                    break
            if self.results.capture_img:
                print("campture image")
                vis4.capture_screen_image(
                    results.source +
                    "/cam4/capture/{0:06d}.png".format(self.frame_list[self.count]))
            if not self.flag_pause:
                self.count += 1
                if self.count >= len(self.frame_list)-1:
                    if self.results.capture_img:
                        break
                    self.count = len(self.frame_list)-1

        vis.destroy_window()


def load_plys(path, object_plys, results, frame_list, object_path=[], hands_path=[], voxel_size=0.01, downsample=True, objects=False, hands=False):
    """
    load pointcloud by path and down samle (if True) based on voxel_size 
    """

    plys = [o3d.geometry.PointCloud() for _ in range(len(path))]
    if results.ply_create:
        color_calib = []
        for cam_num in range(results.cam):
            calib_path = results.source + \
                "/cam{}/cam_intrinsics.txt".format(cam_num)
            cam_fx, cam_fy, cam_cx, cam_cy, _, _ = read_text(
                calib_path).reshape((6))

            color_calib.append(np.array(
                [[cam_fx, 0, cam_cx], [0, cam_fy, cam_cy], [0, 0, 1]]))

        depth_img = cv2.imread(
            results.source+"/cam0/depth/{0:06d}.png".format(0), cv2.IMREAD_ANYDEPTH)

        point2d = cal_points2d(depth_img)
    for idx, frame_num in tqdm.tqdm(enumerate(path)):
        if results.no_ply:
            ply = o3d.geometry.PointCloud()
            plys[idx] = ply
        else:
            if results.ply_create:
                ply = o3d.geometry.PointCloud()
                for cam_num in range(results.cam):
                    undist_depth = results.source + \
                        "/cam{0}//depth/{1:06d}.png".format(
                            cam_num, frame_list[idx])
                    undist_rgb = results.source + \
                        "/cam{0}//rgb/{1:06d}.png".format(
                            cam_num, frame_list[idx])

                    pcloud = get_pointcloud(
                        undist_depth, undist_rgb, color_calib[cam_num], point2d)

                    cam_pose_path = results.source + \
                        "/cam{0}/cam_pose/{1:06d}.txt".format(
                            cam_num, frame_list[idx])
                    with open(cam_pose_path, 'r') as txt_file:
                        data = txt_file.readline().split(" ")
                        data = list(filter(lambda x: x != "", data))

                    cam_pose = np.array(data).astype(np.float).reshape((4, 4))
                    # need to fix this part
                    pcloud.transform(cam_pose)

                    ply += pcloud
            else:
                ply_path = results.source + \
                    "/registered_pcl/{0:06d}.pcd".format(frame_num)
                ply = o3d.io.read_point_cloud(ply_path)
            if downsample == True:
                ply_down = ply.voxel_down_sample(voxel_size=voxel_size)
                objects = False
                hands = False
                if hands:
                    hands_pcl = o3d.io.read_point_cloud(hands_path[idx])
                    hands_pcl.colors = o3d.utility.Vector3dVector(np.reshape(
                        [209/255., 163/255., 164/255.]*np.shape(hands_pcl.points)[0], (-1, 3)))

                    #(209, 163, 164)
                    # print(hands_pcl)
                    ply_down += hands_pcl
                plys[idx] = ply_down
            else:
                plys[idx] = ply
    return plys


def load_meshes(object_meshes, results, object_path=[], hand_path=[], objects_flag=False, hands=False):
    """
    load pointcloud by path and down samle (if True) based on voxel_size 
    """
    objects = [o3d.geometry.TriangleMesh()for _ in range(len(object_path))]
    left_hand = [o3d.geometry.TriangleMesh()for _ in range(len(hand_path))]
    right_hand = [o3d.geometry.TriangleMesh()for _ in range(len(hand_path))]
    ncomps = 45
    torch.set_default_tensor_type('torch.DoubleTensor')
    mano_right = ManoLayer(
        mano_root='mano/models', use_pca=False, ncomps=ncomps, flat_hand_mean=True, side='right')
    mano_left = ManoLayer(
        mano_root='mano/models', use_pca=False, ncomps=ncomps, flat_hand_mean=True, side='left')
    for idx in tqdm.trange(len(object_path)):
        mesh = o3d.geometry.TriangleMesh()
        left_hand_mesh = o3d.geometry.TriangleMesh()
        right_hand_mesh = o3d.geometry.TriangleMesh()
        if objects_flag:
            obj_pose = {}
            load_obj = np.loadtxt(object_path[idx])
            obj_pose["idx"] = load_obj[0]
            obj_pose['pose'] = load_obj[1:].reshape((4, 4))
            extrinsic_matrix = np.loadtxt("{0}/cam4/cam_pose/{1:06d}.txt".
                                          format(results.source, int(object_path[idx][-10:-4]))).reshape((4, 4))
            # print("obj_pose",obj_pose)
            temp_mesh = copy.copy(object_meshes[int(obj_pose["idx"])])
            transform_mtx = np.dot(extrinsic_matrix, obj_pose["pose"])
            # nump.linalg.inv()
            temp_mesh.transform(transform_mtx)
            triangles_temp = np.asarray(temp_mesh.triangles)
            temp_mesh.triangles = o3d.utility.Vector3iVector(triangles_temp)
            mesh = temp_mesh
        if hands:
            # try:

            extrinsic_matrix = np.loadtxt("{0}/cam2/cam_pose/{1:06d}.txt".
                                          format(results.source, int(object_path[idx][-10:-4]))).reshape((4, 4))
            hand_pose = {}
            load_hand = np.loadtxt(hand_path[idx])

            hand_pose["left_pose"] = [load_hand[4:52]]
            hand_pose["left_tran"] = [load_hand[1:4]]
            hand_pose["left_shape"] = [load_hand[52:62]]

            left_rot = R.from_rotvec(load_hand[4:7])
            left_mat = np.concatenate((np.concatenate((left_rot.as_matrix(), np.array(
                hand_pose['left_tran']).T), axis=1), [[0, 0, 0, 1]]), axis=0)
            left_mat_proj = np.dot(extrinsic_matrix, left_mat)
            left_rotvec = R.from_matrix(left_mat_proj[:3, :3]).as_rotvec()
            hand_pose["left_pose"][0][:3] = left_rotvec

            _, mano_keypoints_3d_left = mano_left(torch.tensor(hand_pose["left_pose"]),
                                                  torch.tensor(hand_pose["left_shape"]))
            left_hand_origin = mano_keypoints_3d_left[0][0] / 1000.0
            origin_left = torch.unsqueeze(
                left_hand_origin, 1) + torch.tensor(hand_pose["left_tran"]).T
            left_mat_proj = np.dot(
                extrinsic_matrix, np.concatenate((origin_left, [[1]])))
            new_left_trans = torch.tensor(
                left_mat_proj.T[0, :3]) - left_hand_origin
            # print("new_left_trans",new_left_trans)
            hand_pose["left_tran"] = new_left_trans

            hand_pose["right_pose"] = [load_hand[66:114]]
            hand_pose["right_tran"] = [load_hand[63:66]]
            hand_pose["right_shape"] = [load_hand[114:124]]

            right_rot = R.from_rotvec(load_hand[66:69])
            right_mat = np.concatenate((np.concatenate((right_rot.as_matrix(), np.array(
                hand_pose['right_tran']).T), axis=1), [[0, 0, 0, 1]]), axis=0)
            right_mat_proj = np.dot(extrinsic_matrix, right_mat)
            right_rotvec = R.from_matrix(right_mat_proj[:3, :3]).as_rotvec()

            hand_pose["right_pose"][0][:3] = right_rotvec
            _, mano_keypoints_3d_right = mano_right(torch.tensor(hand_pose["right_pose"]),
                                                    torch.tensor(hand_pose["right_shape"]))
            right_hand_origin = mano_keypoints_3d_right[0][0] / 1000.0
            origin_right = torch.unsqueeze(
                right_hand_origin, 1) + torch.tensor(hand_pose["right_tran"]).T
            right_mat_proj = np.dot(
                extrinsic_matrix, np.concatenate((origin_right, [[1]])))
            new_right_trans = torch.tensor(
                right_mat_proj.T[0, :3]) - right_hand_origin
            hand_pose["right_tran"] = new_right_trans

            random_pose = torch.tensor(hand_pose["left_pose"])
            random_tran = torch.tensor(hand_pose["left_tran"])
            random_shape = torch.tensor(hand_pose["left_shape"])

            if load_hand[0] == 0:
                mesh_lh = o3d.geometry.TriangleMesh()
            else:

                #mano_layer = ManoLayer(mano_root='mano/models', use_pca=True, ncomps=ncomps, flat_hand_mean=False, side='left')

                hand_verts, hand_joints = mano_left(random_pose, random_shape)
                hand_verts_scaled = hand_verts/1000.0 + random_tran
                triangles = mano_left.th_faces
                mesh_lh = o3d.geometry.TriangleMesh()
                mesh_lh.vertices = o3d.utility.Vector3dVector(
                    hand_verts_scaled.detach().numpy()[0])
                mesh_lh.triangles = o3d.utility.Vector3iVector(
                    np.asarray(triangles)[:len(triangles):])
                mesh_lh.vertex_colors = o3d.utility.Vector3dVector(np.reshape(
                    [209/255., 163/255., 164/255.]*np.shape(mesh_lh.vertices)[0], (-1, 3)))

            random_pose = torch.tensor(hand_pose["right_pose"])
            random_tran = torch.tensor(hand_pose["right_tran"])
            random_shape = torch.tensor(hand_pose["right_shape"])
            # print("random_shape",random_shape)
            if load_hand[62] == 0:
                mesh_rh = o3d.geometry.TriangleMesh()
            else:
                #mano_layer = ManoLayer(mano_root='mano/models', use_pca=True, ncomps=ncomps, flat_hand_mean=False, side='right')

                hand_verts, hand_joints = mano_right(random_pose, random_shape)
                hand_verts_scaled = hand_verts/1000.0 + random_tran
                hand_joints_scaled = hand_joints/1000.0 + random_tran
                triangles = mano_right.th_faces
                mesh_rh = o3d.geometry.TriangleMesh()
                mesh_rh.vertices = o3d.utility.Vector3dVector(
                    hand_verts_scaled.detach().numpy()[0])
                mesh_rh.triangles = o3d.utility.Vector3iVector(
                    np.asarray(triangles)[:len(triangles):])

                # Skin color
                mesh_rh.vertex_colors = o3d.utility.Vector3dVector(np.reshape(
                    [209/255., 163/255., 164/255.]*np.shape(mesh_rh.vertices)[0], (-1, 3)))

            left_hand[idx] = mesh_lh
            right_hand[idx] = mesh_rh
        objects[idx] = mesh

    print("objects", objects)
    return objects, left_hand, right_hand


def load_rgbs(path, results, action_lists):
    """
    load pointcloud by path and down samle (if True) based on voxel_size 
    """
    action_txt = ['background',
                  'grab book',
                  'grab espresso',
                  'grab lotion',
                  'grab spray',
                  'grab milk',
                  'grab cocoa',
                  'grab chips',
                  'grab cappuccino',
                  'place book',
                  'place espresso',
                  'place lotion',
                  'place spray',
                  'place milk',
                  'place cocoa',
                  'place chips',
                  'place cappuccino',
                  'open lotion',
                  'open milk',
                  'open chips',
                  'close lotion',
                  'close milk',
                  'close chips',
                  'pour milk',
                  'take out espresso',
                  'take out cocoa',
                  'take out chips',
                  'take out cappuccino',
                  'put in espresso',
                  'put in cocoa',
                  'put in cappuccino',
                  'apply lotion',
                  'apply spray',
                  'read book',
                  'read espresso',
                  'spray spray',
                  'squeeze lotion']

    rgbs_allcam = []
    for cam_num, sub_path in enumerate(path):
        rgbs = [np.array([[]]) for _ in range(len(path[0]))]
        # for idx in trange(len(path[0])):
        for idx, file_path in tqdm.tqdm(enumerate(sub_path)):
            (base, file_name) = os.path.split(file_path)
            #pil_img = Image.open(path[cam_num][idx])
            cv_img = cv2.imread(path[cam_num][idx])
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

            if results.action_info and cam_num == 0:
                font = cv2.FONT_HERSHEY_SIMPLEX
                # action_lists
                actions = int(action_lists[int(file_name[:6])])
                offset = 1
                # for action in actions:

                cv2.putText(
                    cv_img, action_txt[actions], (10, 80*offset), font, 3, (255, 0, 0), 5, cv2.LINE_AA)
                offset += 1

            if results.bbox:
                points = read_text(
                    "{0}/cam{1}//obj_pose/{2:06d}.txt".format(results.source, cam_num, int(file_name[:6])), 1)
                cam_fx, cam_fy, cam_cx, cam_cy, _, _ = read_text(
                    "{0}/cam{1}//cam_intrinsics.txt".format(results.source, cam_num)).reshape((6))
                camera_matrix = np.array(
                    [[cam_fx, 0, cam_cx], [0, cam_fy, cam_cy], [0, 0, 1]])
                # print(points)
                img_points, _ = cv2.projectPoints(points, np.array([[0, 0, 0]]).astype(
                    np.float32), np.array([[0, 0, 0]]).astype(np.float32), camera_matrix, None)
                cv2.line(cv_img, tuple(img_points[1][0]), tuple(
                    img_points[2][0]), (0, 255, 0), 5)
                cv2.line(cv_img, tuple(img_points[2][0]), tuple(
                    img_points[3][0]), (0, 255, 0), 5)
                cv2.line(cv_img, tuple(img_points[3][0]), tuple(
                    img_points[4][0]), (0, 255, 0), 5)
                cv2.line(cv_img, tuple(img_points[4][0]), tuple(
                    img_points[1][0]), (0, 255, 0), 5)

                cv2.line(cv_img, tuple(img_points[1][0]), tuple(
                    img_points[5][0]), (0, 255, 0), 5)
                cv2.line(cv_img, tuple(img_points[2][0]), tuple(
                    img_points[6][0]), (0, 255, 0), 5)
                cv2.line(cv_img, tuple(img_points[3][0]), tuple(
                    img_points[7][0]), (0, 255, 0), 5)
                cv2.line(cv_img, tuple(img_points[4][0]), tuple(
                    img_points[8][0]), (0, 255, 0), 5)

                cv2.line(cv_img, tuple(img_points[5][0]), tuple(
                    img_points[6][0]), (0, 255, 0), 5)
                cv2.line(cv_img, tuple(img_points[6][0]), tuple(
                    img_points[7][0]), (0, 255, 0), 5)
                cv2.line(cv_img, tuple(img_points[7][0]), tuple(
                    img_points[8][0]), (0, 255, 0), 5)
                cv2.line(cv_img, tuple(img_points[8][0]), tuple(
                    img_points[5][0]), (0, 255, 0), 5)

            if results.hand_proj:
                points = read_text("{0}/cam{1}/hand_pose/{2:06d}.txt".format(
                    results.source, cam_num, int(file_name[:6])), 1, 64)
                cam_fx, cam_fy, cam_cx, cam_cy, _, _ = read_text(
                    "{0}/cam{1}//cam_intrinsics.txt".format(results.source, cam_num)).reshape((6))
                camera_matrix = np.array(
                    [[cam_fx, 0, cam_cx], [0, cam_fy, cam_cy], [0, 0, 1]])

                rot_vec = np.array([[0, 0, 0]]).astype(np.float32)
                tran_vec = np.array([[0, 0, 0]]).astype(np.float32)

                img_points, _ = cv2.projectPoints(
                    points, rot_vec, tran_vec, camera_matrix, None)
                img_points = np.round(img_points).astype(int)

                # left hand
                line_thickness = 2
                cv2.line(cv_img, tuple(img_points[1][0]), tuple(
                    img_points[2][0]), (0, 0, 255), line_thickness)
                cv2.line(cv_img, tuple(img_points[2][0]), tuple(
                    img_points[3][0]), (0, 0, 255), line_thickness)
                cv2.line(cv_img, tuple(img_points[3][0]), tuple(
                    img_points[4][0]), (0, 0, 255), line_thickness)

                cv2.line(cv_img, tuple(img_points[5][0]), tuple(
                    img_points[6][0]), (0, 0, 255), line_thickness)
                cv2.line(cv_img, tuple(img_points[6][0]), tuple(
                    img_points[7][0]), (0, 0, 255), line_thickness)
                cv2.line(cv_img, tuple(img_points[7][0]), tuple(
                    img_points[8][0]), (0, 0, 255), line_thickness)

                cv2.line(cv_img, tuple(img_points[9][0]), tuple(
                    img_points[10][0]), (0, 0, 255), line_thickness)
                cv2.line(cv_img, tuple(img_points[10][0]), tuple(
                    img_points[11][0]), (0, 0, 255), line_thickness)
                cv2.line(cv_img, tuple(img_points[11][0]), tuple(
                    img_points[12][0]), (0, 0, 255), line_thickness)

                cv2.line(cv_img, tuple(img_points[13][0]), tuple(
                    img_points[14][0]), (0, 0, 255), line_thickness)
                cv2.line(cv_img, tuple(img_points[14][0]), tuple(
                    img_points[15][0]), (0, 0, 255), line_thickness)
                cv2.line(cv_img, tuple(img_points[15][0]), tuple(
                    img_points[16][0]), (0, 0, 255), line_thickness)

                cv2.line(cv_img, tuple(img_points[17][0]), tuple(
                    img_points[18][0]), (0, 0, 255), line_thickness)
                cv2.line(cv_img, tuple(img_points[18][0]), tuple(
                    img_points[19][0]), (0, 0, 255), line_thickness)
                cv2.line(cv_img, tuple(img_points[19][0]), tuple(
                    img_points[20][0]), (0, 0, 255), line_thickness)

                cv2.line(cv_img, tuple(img_points[0][0]), tuple(
                    img_points[1][0]), (0, 0, 255), line_thickness)
                cv2.line(cv_img, tuple(img_points[0][0]), tuple(
                    img_points[5][0]), (0, 0, 255), line_thickness)
                cv2.line(cv_img, tuple(img_points[0][0]), tuple(
                    img_points[9][0]), (0, 0, 255), line_thickness)
                cv2.line(cv_img, tuple(img_points[0][0]), tuple(
                    img_points[13][0]), (0, 0, 255), line_thickness)
                cv2.line(cv_img, tuple(img_points[0][0]), tuple(
                    img_points[17][0]), (0, 0, 255), line_thickness)

                # right hand
                offset = 21
                cv2.line(cv_img, tuple(
                    img_points[offset+1][0]), tuple(img_points[offset+2][0]), (255, 0, 0), line_thickness)
                cv2.line(cv_img, tuple(
                    img_points[offset+2][0]), tuple(img_points[offset+3][0]), (255, 0, 0), line_thickness)
                cv2.line(cv_img, tuple(
                    img_points[offset+3][0]), tuple(img_points[offset+4][0]), (255, 0, 0), line_thickness)

                cv2.line(cv_img, tuple(
                    img_points[offset+5][0]), tuple(img_points[offset+6][0]), (255, 0, 0), line_thickness)
                cv2.line(cv_img, tuple(
                    img_points[offset+6][0]), tuple(img_points[offset+7][0]), (255, 0, 0), line_thickness)
                cv2.line(cv_img, tuple(
                    img_points[offset+7][0]), tuple(img_points[offset+8][0]), (255, 0, 0), line_thickness)

                cv2.line(cv_img, tuple(
                    img_points[offset+9][0]), tuple(img_points[offset+10][0]), (255, 0, 0), line_thickness)
                cv2.line(cv_img, tuple(img_points[offset+10][0]), tuple(
                    img_points[offset+11][0]), (255, 0, 0), line_thickness)
                cv2.line(cv_img, tuple(img_points[offset+11][0]), tuple(
                    img_points[offset+12][0]), (255, 0, 0), line_thickness)

                cv2.line(cv_img, tuple(img_points[offset+13][0]), tuple(
                    img_points[offset+14][0]), (255, 0, 0), line_thickness)
                cv2.line(cv_img, tuple(img_points[offset+14][0]), tuple(
                    img_points[offset+15][0]), (255, 0, 0), line_thickness)
                cv2.line(cv_img, tuple(img_points[offset+15][0]), tuple(
                    img_points[offset+16][0]), (255, 0, 0), line_thickness)

                cv2.line(cv_img, tuple(img_points[offset+17][0]), tuple(
                    img_points[offset+18][0]), (255, 0, 0), line_thickness)
                cv2.line(cv_img, tuple(img_points[offset+18][0]), tuple(
                    img_points[offset+19][0]), (255, 0, 0), line_thickness)
                cv2.line(cv_img, tuple(img_points[offset+19][0]), tuple(
                    img_points[offset+20][0]), (255, 0, 0), line_thickness)

                cv2.line(cv_img, tuple(
                    img_points[offset+0][0]), tuple(img_points[offset+1][0]), (255, 0, 0), line_thickness)
                cv2.line(cv_img, tuple(
                    img_points[offset+0][0]), tuple(img_points[offset+5][0]), (255, 0, 0), line_thickness)
                cv2.line(cv_img, tuple(
                    img_points[offset+0][0]), tuple(img_points[offset+9][0]), (255, 0, 0), line_thickness)
                cv2.line(cv_img, tuple(
                    img_points[offset+0][0]), tuple(img_points[offset+13][0]), (255, 0, 0), line_thickness)
                cv2.line(cv_img, tuple(
                    img_points[offset+0][0]), tuple(img_points[offset+17][0]), (255, 0, 0), line_thickness)

            if results.capture_skeleton:
                if cam_num == 4:
                    cv2.imwrite(
                        'capture_skeleton/{0:06d}.png'.format(int(file_name[:6])), cv_img)
            rgb = np.asarray(cv_img)
            rgbs[idx] = rgb

        rgbs_allcam.append(rgbs)

    return rgbs_allcam


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="h2oplayer, visualize the data including rgb images, meshes and pointcloudes for the H2O dataset.")

    parser.add_argument("--source", action="store",
                        help="source directory", required=True)
    parser.add_argument("--obj_source", action="store",
                        help="object directory", default="/media/taein/dataset/objects/")

    parser.add_argument("--obj_file", action="store_true",
                        default=False, help="use .obj file for 3D meshes of objects")
    parser.add_argument("--fps", action="store", type=int, default=30,
                        help="framerate of the player, default is 30")
    parser.add_argument("--downsample", action="store_true",
                        default=False, help="down sampling")
    parser.add_argument("--voxelsize", action="store",
                        default=0.01, help="voxel size")

    parser.add_argument("--object", action="store_true",
                        default=False, help="show object mesh")
    parser.add_argument("--rgb", action="store_true",
                        default=False, help="show rgb images")
    parser.add_argument("--hands", action="store_true",
                        default=False, help="show hands mesh")
    parser.add_argument("--start", action="store", type=int,
                        default=0, help="start frame")
    parser.add_argument("--end", action="store", type=int,
                        default=-1, help="end frame")

    parser.add_argument("--checkbox", action="store_true",
                        default=False, help="check failed frames for object pose")
    parser.add_argument("--action_info", action="store_true",
                        default=False, help="give info about action labels")
    parser.add_argument("--ego", action="store_true",
                        help="Rendering for egocentric view")
    parser.add_argument("--bbox", action="store_true",
                        help="add bbox for rgb images")
    parser.add_argument("--hand_proj", action="store_true",
                        help="add hand projection for rgb images")

    parser.add_argument("--no_ply", action="store_true",
                        help="not loading plys")
    parser.add_argument("--capture_img", action="store_true",
                        help="capture image from egocentric cam")
    parser.add_argument("--capture_skeleton", action="store_true",
                        help="capture image from egocentric cam")
    parser.add_argument("--cam", action="store", type=int,
                        default=5, help="number of cameras")
    parser.add_argument("--ply_create", action="store_true", default=False,
                        help="Create point clouds from rgbd images")

    results = parser.parse_args()

    # parser.add_argument("--ego", help="for egocentric view",
    #                action="store_true")

    if results.capture_img:
        capture_dir = results.source + "/cam4/capture"
        if not os.path.exists(capture_dir):
            os.mkdir(capture_dir)

    frame_num = 65532
    for cam_num in range(results.cam):
        frame_num = min(len(glob.glob(results.source +
                        "/cam{}//rgb/".format(cam_num)+"*.png")), frame_num)

    action_lists = {}

    startframe = results.start
    if results.end == -1:
        endframe = frame_num-1
    else:
        endframe = results.end
    #hands_path = sorted(glob.glob(results.source + "/registered_hands/*.ply"))[startframe:endframe]
    ply_path = range(frame_num)[startframe:endframe]

    if results.capture_img:
        ego_render_dir = "{}/cam{}//render/".format(
            results.source, results.cam-1)
        if not os.path.exists(ego_render_dir):
            os.makedirs(ego_render_dir)

    for frame in range(startframe, endframe+1):
        action_path = results.source + \
            '/cam4/action_label/{0:06d}.txt'.format(frame)
        action_idx = np.loadtxt(action_path)
        # print("action_idx",action_idx)
        action_lists[frame] = action_idx
    hand_path = []
    # print(obj_pose)
    # print(ply_path)

    object_path = []
    object_info = []

    rgb_path = []
    frame_list = []
    for cam_num in range(results.cam):
        rgb_path.append([])
    for ply_indi_path in ply_path:
        frame_list.append(ply_indi_path)
        hand_path.append(
            results.source + "/cam2/hand_pose_mano/{0:06d}.txt".format(ply_indi_path))
        object_path.append(
            results.source + "/cam4/obj_pose_rt/{0:06d}.txt".format(ply_indi_path))

        for cam_num in range(results.cam):
            rgb_path[cam_num].append(
                results.source + "/cam{0}//rgb/{1:06d}.png".format(cam_num, ply_indi_path))

    for idx in range(len(object_path)):
        obj_pose = {}
        obj_pose_init = {}
        with open(object_path[0]) as json_data:
            load_obj = np.loadtxt(json_data)
            obj_pose_init["idx"] = load_obj[0]
            obj_pose_init['pose'] = load_obj[1:].reshape((4, 4))
            # print("obj_pose_init",obj_pose_init)
        if os.path.exists(object_path[idx]):
            with open(object_path[idx]) as json_data:

                load_obj = np.loadtxt(json_data)
                obj_pose["idx"] = load_obj[0]
                obj_pose['pose'] = load_obj[1:].reshape((4, 4))

        else:
            obj_pose = obj_pose_init
        obj_pose['frame'] = idx
        object_info.append(obj_pose)

    class_names = ["background", "book", "espresso", "lotion", "lotion_spray", "milk", "ovomaltine",
                   "pringles", "starbucks"]

    obj_path = results.obj_source
    object_meshes = []
    for class_name in class_names:
        if results.obj_file:
            object_meshes.append(o3d.io.read_triangle_mesh(
                "{0}/{1}/{1}.obj".format(obj_path, class_name)))
        else:
            object_meshes.append(o3d.io.read_triangle_mesh(
                "{0}/{1}/{1}.ply".format(obj_path, class_name)))
    object_plys = []

    rgbs = []
    if results.rgb:
        rgbs = load_rgbs(rgb_path, results, action_lists)
    objects = []
    left_hand = []
    right_hand = []
    if results.object or results.hands:
        objects, left_hand, right_hand = load_meshes(object_meshes=object_meshes, results=results,
                                                     object_path=object_path, hand_path=hand_path, objects_flag=results.object, hands=results.hands)

    plys = load_plys(ply_path, object_plys, results, frame_list,  object_path, hand_path, float(
        results.voxelsize), downsample=results.downsample, objects=False, hands=False)

    vis = PlayerCallBack(results, frame_list, objects, left_hand,
                         right_hand, object_meshes, object_path, hand_path)

    vis.run(plys, rgbs, object_info)
