# import openmesh as om
import torch
import cv2
import numpy as np
import sys
import os
from os.path import join
from utils.ObjIO import *
from tqdm import tqdm
from Render.mesh import  compute_normal
from utils.cam_util import *
sys.path.append("GenEvalData")
import RenderUtils


class RenderColorMesh(object):
    def __init__(self,w,h,fx,fy,cam_t=np.array([0,0,10]),camera_type='o') -> None:
        super().__init__()
        self.B_MIN = np.array([-1.0, -0.5, -1.0])
        self.B_MAX = np.array([1.0, 2.28, 1.0])
        self.light_dire = [0.0, 0.0, -1.0]
        self.ambient_strength = 0.4
        self.light_strength = 0.6
        self.camera_dict = {}
        self.camera_dict['InterMat'] = np.array([[fx, 0, w/2], [0, fy, h/2], [0, 0, 1]])
        self.camera_dict['ExterMat'] = np.array([[1., 0., 0., cam_t[0]],
                                                 [0., -1., 0., cam_t[1]],
                                                 [0., 0., -1., cam_t[2]],
                                                 [0., 0., 0., 1.]])
        self.camera_dict['CameraReso'] = np.array([w, h])
        self.camera_type=camera_type

    def set_camera(self, camera):
        self.camera_dict['InterMat']=camera.get_intrinsic_matrix()
        self.camera_dict['CameraReso'] = np.array([camera.width, camera.height])
        self.camera_dict['ExterMat'] = camera.get_extrinsic_matrix()
    def render_tex_mesh_func(self, fv_indices, tri_uvs, tri_normals, tex_img, vps, camera_dict):
        if self.camera_type=='o':
            proj_pixels, z_vals, v_status = self.orthogonal_mesh_vps(vps, camera_dict)
        if self.camera_type=='p':
            proj_pixels, z_vals, v_status = self.project_mesh_vps(vps, camera_dict)
        tri_proj_pixels = (proj_pixels[fv_indices]).reshape(-1, 6)
        tri_z_vals = z_vals[fv_indices]  # [n_f, 3]
        tri_status = (v_status[fv_indices]).all(axis=1)  # [n_f]

        cam_w = camera_dict["CameraReso"][0]
        cam_h = camera_dict["CameraReso"][1]
        ex_mat = camera_dict["ExterMat"]

        depth_img = np.ones((cam_h, cam_w), np.float32) * 100.0
        rgb_img = np.zeros((cam_h, cam_w, 3), np.float32)
        mask_img = np.zeros((cam_h, cam_w), np.int32)

        w_light_dx = self.light_dire[0]
        w_light_dy = self.light_dire[1]
        w_light_dz = self.light_dire[2]

        c_light_dx = ex_mat[0, 0] * w_light_dx + ex_mat[0, 1] * w_light_dy + ex_mat[0, 2] * w_light_dz
        c_light_dy = ex_mat[1, 0] * w_light_dx + ex_mat[1, 1] * w_light_dy + ex_mat[1, 2] * w_light_dz
        c_light_dz = ex_mat[2, 0] * w_light_dx + ex_mat[2, 1] * w_light_dy + ex_mat[2, 2] * w_light_dz

        ambient_strength = self.ambient_strength
        light_strength = self.light_strength

        RenderUtils.render_tex_mesh(
            tri_normals, tri_uvs, tri_proj_pixels, tri_z_vals, tri_status, tex_img, depth_img, rgb_img, mask_img,
            c_light_dx, c_light_dy, c_light_dz, ambient_strength, light_strength
            # if(study time > 10h):
            #     print("stop neijuan!")
        )
        # np.concatenate
        depth_img[mask_img < 0.5] = 0
        return rgb_img, depth_img, mask_img
    def set_tex_mesh(self,mesh_path,tex_path):
        mesh=load_obj_data(mesh_path)
        if not self.check_mesh_bbox(mesh['v']):
            print("Error, the bounding box of the mesh is out of the pre-defined range.")
            exit(0)
        if mesh['vn'] is None or mesh['vn'].shape != mesh['v'].shape:
            mesh['vn'] = compute_normal(mesh['v'], mesh['f'])
        self.vertex=mesh['v']
        self.face = mesh['f']
        self.normals= mesh['vn']
        self.vt= mesh['vt']
        self.ft= mesh['ft']
        self.tex_img = np.ascontiguousarray((cv2.imread(tex_path).astype(np.float32) / 255.0)[:, :, ::-1])  # BGR to RGB
        self.tri_uvs = (mesh['vt'][mesh['ft']]).reshape(-1, 6)
    def render_tex_mesh(self, R):
        vertex=self.vertex.dot(R)
        normals=self.normals.dot(R)
        tri_normals = (normals[self.face]).reshape(-1, 9)
        rgb_img, depth_img, mask_img = self.render_tex_mesh_func(
            self.face, self.tri_uvs, tri_normals, self.tex_img, vertex, self.camera_dict
        )
        return rgb_img, depth_img, mask_img

    def render_color_mesh_func(self, fv_indices, tri_colors, tri_normals, vps, camera_dict):
        if self.camera_type=='o':
            proj_pixels, z_vals, v_status = self.orthogonal_mesh_vps(vps, camera_dict)
        if self.camera_type=='p':
            proj_pixels, z_vals, v_status = self.project_mesh_vps(vps, camera_dict)
        # proj_pixels, z_vals, v_status = self.orthogonal_mesh_vps(vps, camera_dict)
        tri_proj_pixels = (proj_pixels[fv_indices]).reshape(-1, 6)  # [n_f, 6]
        tri_z_vals = z_vals[fv_indices]  # [n_f, 3]
        tri_status = (v_status[fv_indices]).all(axis=1)  # [n_f]

        cam_w = camera_dict["CameraReso"][0]
        cam_h = camera_dict["CameraReso"][1]
        ex_mat = camera_dict["ExterMat"]

        depth_img = np.ones((cam_h, cam_w), np.float32) * 100.0
        rgb_img = np.ones((cam_h, cam_w, 3), np.float32)
        mask_img = np.zeros((cam_h, cam_w), np.int32)

        w_light_dx = self.light_dire[0]
        w_light_dy = self.light_dire[1]
        w_light_dz = self.light_dire[2]

        c_light_dx = ex_mat[0, 0] * w_light_dx + ex_mat[0, 1] * w_light_dy + ex_mat[0, 2] * w_light_dz
        c_light_dy = ex_mat[1, 0] * w_light_dx + ex_mat[1, 1] * w_light_dy + ex_mat[1, 2] * w_light_dz
        c_light_dz = ex_mat[2, 0] * w_light_dx + ex_mat[2, 1] * w_light_dy + ex_mat[2, 2] * w_light_dz

        ambient_strength = self.ambient_strength
        light_strength = self.light_strength

        RenderUtils.render_color_mesh(
            tri_normals, tri_colors, tri_proj_pixels, tri_z_vals, tri_status, depth_img, rgb_img, mask_img,
            c_light_dx, c_light_dy, c_light_dz, ambient_strength, light_strength
        )
        depth_img[mask_img < 0.5] = 0.0
        return rgb_img, depth_img, mask_img
    def set_mesh(self,mesh_path):
        mesh=load_obj_data(mesh_path)
        if not self.check_mesh_bbox(mesh['v']):
            print("Error, the bounding box of the mesh is out of the pre-defined range.")
            exit(0)
        if mesh['vn'] is None or mesh['vn'].shape != mesh['v'].shape:
            mesh['vn'] = compute_normal(mesh['v'], mesh['f'])
        self.tri_normals = (mesh['vn'][mesh['f']]).reshape(-1, 9)
        self.tri_colors = (mesh['vc'][mesh['f']]).reshape(-1, 9)
        self.vertex=mesh['v']
        self.face = mesh['f']
        self.normals= mesh['vn']
    def render_color_mesh(self, R):
        vertex=self.vertex.dot(R.T)
        normals=self.normals.dot(R.T)
        tri_normals = (normals[self.face]).reshape(-1, 9)
        rgb_img, depth_img, mask_img = self.render_color_mesh_func(
            self.face, self.tri_colors, tri_normals, vertex, self.camera_dict
        )
        return rgb_img, depth_img, mask_img

    def check_mesh_bbox(self, vps):
        min_vp = vps.min(axis=0)
        max_vp = vps.max(axis=0)

        res = (min_vp > self.B_MIN) * (max_vp < self.B_MAX)
        res = res.all()

        return res

    def project_mesh_vps(self, world_vps, camera_dict):
        ex_mat = camera_dict["ExterMat"]
        in_mat = camera_dict["InterMat"]
        cam_reso = camera_dict["CameraReso"]

        cam_w = cam_reso[0]
        cam_h = cam_reso[1]
        ex_Rmat = ex_mat[:3, :3]
        ex_Tvec = ex_mat[:3, 3:]

        fx = in_mat[0, 0]
        fy = in_mat[1, 1]
        cx = in_mat[0, 2]
        cy = in_mat[1, 2]

        cam_vps = ex_Rmat.dot(world_vps.T) + ex_Tvec
        pixel_x = fx * (cam_vps[0, :] / cam_vps[2, :]) + cx
        pixel_y = fy * (cam_vps[1, :] / cam_vps[2, :]) + cy

        vps_status = (pixel_x > 0) * (pixel_x < cam_w) * (pixel_y > 0) * (pixel_y < cam_h)
        proj_pixel = np.stack([pixel_x, pixel_y], axis=1)

        return proj_pixel, cam_vps[2, :], vps_status
    def orthogonal_mesh_vps(self, world_vps, camera_dict):
        ex_mat = camera_dict["ExterMat"]
        in_mat = camera_dict["InterMat"]
        cam_reso = camera_dict["CameraReso"]

        cam_w = cam_reso[0]
        cam_h = cam_reso[1]
        ex_Rmat = ex_mat[:3, :3]
        ex_Tvec = ex_mat[:3, 3:]

        fx = in_mat[0, 0]
        fy = in_mat[1, 1]
        cx = in_mat[0, 2]
        cy = in_mat[1, 2]

        cam_vps = ex_Rmat.dot(world_vps.T)+ex_Tvec
        pixel_x = fx * cam_vps[0, :] + cx
        pixel_y = fy * cam_vps[1, :] + cy

        vps_status = (pixel_x > 0) * (pixel_x < cam_w) * (pixel_y > 0) * (pixel_y < cam_h)
        proj_pixel = np.stack([pixel_x, pixel_y], axis=1)

        return proj_pixel, cam_vps[2, :], vps_status
if __name__ == "__main__":
    color_mesh_path = "dataset_example/FRONT_mesh_normalized.obj"
    tt = RenderColorMesh(w=512,h=512,fx=512,fy=512,cam_t=np.array([0,0,10]),camera_type='o')
    # tt.set_mesh(color_mesh_path)
    tt.set_tex_mesh('./dataset_example/mesh_data/rp_dennis_posed_004/rp_dennis_posed_004_100k.obj','./dataset_example/mesh_data/rp_dennis_posed_004/rp_dennis_posed_004_dif_2k.jpg')
    view_num=360
    for view_id in tqdm(range(0,view_num)):
        R=make_rotate(0,view_id*np.pi/180,0)
        # rgb_img, depth_img, mask_img=tt.render_color_mesh(R)
        rgb_img, depth_img, mask_img = tt.render_tex_mesh(R)
        cv2.imwrite(join("./TempData/TexMesh", "O_RGB_%04d.png" % view_id), (rgb_img * 255)[:, :, ::-1])
        # cv2.imwrite(join("./TempData/ColorMesh", "O_depth_%04d.png" % view_id), (depth_img * 10000).astype(np.uint16))
        cv2.imwrite(join("./TempData/TexMesh", "O_mask_%04d.png" % view_id), (mask_img).astype(np.uint8))
    # tex_mesh_path = "TempData/SampleData/rp_dennis_posed_004_100k.obj"
    # tex_img_path = "TempData/SampleData/rp_dennis_posed_004_dif_2k.jpg"
    # tt = RenderColorMesh()
    # tt.render_tex_mesh(tex_mesh_path, tex_img_path, "./TempData/TexMesh", "tex")