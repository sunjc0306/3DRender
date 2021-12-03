import numpy as np
import os
import cv2 as cv
from tqdm import tqdm
from Render.camera import Camera
from Render.gl.color_render import ColorRender
from Render.gl.normal_render import NormalRender
from utils.ObjIO import *
from utils.cam_util import *
from Render.mesh import  compute_normal
view_num = 1
cam_f = 5000
cam_dist = 10
pre_cam=np.array([0.8845,-0.0299,0.1250])
cam_t=np.array([0,0,10])#pred_cam[:, 0:1] * cam_c * cam_tz / cam_f
img_resw = 512
img_resh = 512
cam_type='p'
class SmplVtx(object):
    """
    Local class used to load and store SMPL's vertices coordinate at rest pose
    with mean shape
    """
    def __init__(self):
        self.smpl_vtx_std = np.loadtxt('vertices.txt')
        min_x = np.min(self.smpl_vtx_std[:, 0])
        max_x = np.max(self.smpl_vtx_std[:, 0])
        min_y = np.min(self.smpl_vtx_std[:, 1])
        max_y = np.max(self.smpl_vtx_std[:, 1])
        min_z = np.min(self.smpl_vtx_std[:, 2])
        max_z = np.max(self.smpl_vtx_std[:, 2])

        self.smpl_vtx_std[:, 0] = (self.smpl_vtx_std[:, 0]-min_x)/(max_x-min_x)
        self.smpl_vtx_std[:, 1] = (self.smpl_vtx_std[:, 1]-min_y)/(max_y-min_y)
        self.smpl_vtx_std[:, 2] = (self.smpl_vtx_std[:, 2]-min_z)/(max_z-min_z)


_smpl_vtx = SmplVtx()


def get_smpl_semantic_code():
    """gets semantic code definition on SMPL model"""
    return _smpl_vtx.smpl_vtx_std
def main():
    mesh=load_obj_data('dataset_example/FRONT_smpl_normalized.obj')
    # mesh=load_obj_data('test/pre_smpl2.obj')
    # mesh['v'][:,0]=(mesh['v'][:,0]+pre_cam[1])*pre_cam[0]/(5000/(256))*10
    # mesh['v'][:,1]=(mesh['v'][:,1]+pre_cam[2])*pre_cam[0]/(5000/(256))*10
    # mesh['v'][:,2]=(mesh['v'][:,2])*pre_cam[0]/(5000/(256))*10
    # mesh=save_obj_data(mesh,'test/pre_smpl1.obj')
    if mesh['vn'] is None or mesh['vn'].shape!=mesh['v'].shape:
        mesh['vn']=compute_normal(mesh['v'],mesh['f'])
    # color=0.5*(norm+1)
    mesh['vc']=get_smpl_semantic_code()
    rndr = ColorRender(width=img_resw, height=img_resh)
    rndr_noraml=NormalRender(width=img_resw, height=img_resh)
    # mesh['vc']= np.array([[0.65098039, 0.74117647, 0.85882353]]).repeat(mesh['v'].shape[0],0)
    # rndr.set_mesh(vertices=vertices,faces=faces,color=color,faces_clr=faces)
    rndr.set_mesh(vertices=mesh['v'],faces=mesh['f'],color=mesh['vc'],faces_clr=mesh['f'],norms=mesh['vn'],faces_nml=mesh['f'])
    rndr_noraml.set_mesh(vertices=mesh['v'],faces=mesh['f'],norms=mesh['vn'],face_normals=mesh['f'])
    rndr.set_norm_mat(axis=np.array([1,-1,-1]))
    rndr_noraml.set_norm_mat(axis=np.array([1,1,1]))
    cam = Camera(width=img_resw, height=img_resh,focal=cam_f,camera_type=cam_type)
    cam.center=cam_t
    # cam_params = generate_cameras(dist=cam_dist, view_num=view_num)
    # sh_list = []
    for view_id in tqdm(range(0,view_num)):
        R=make_rotate(0,view_id*np.pi/180,0)
        rndr.rot_matrix=R
        rndr_noraml.rot_matrix=R
        cam.sanity_check()
        rndr.set_camera(cam)
        rndr.display()
        rndr_noraml.set_camera(cam)
        rndr_noraml.display()
        out_all_f = rndr.get_color(0)
        out_mask = out_all_f[:, :, 3]
        out_all_f = cv.cvtColor(out_all_f, cv.COLOR_RGBA2BGR)
        cv.imwrite(os.path.join('./test/%04d_rgb.png' % view_id), np.uint8(out_all_f * 255))
        # out_all_f = rndr_noraml.get_color(0)
        # out_mask = out_all_f[:, :, 3]
        # out_all_f = cv.cvtColor(out_all_f, cv.COLOR_RGBA2BGR)
        # cv.imwrite(os.path.join('./test/pn%04d.jpg' % view_id), np.uint8(out_all_f * 255))
        cv.imwrite(os.path.join('./test/%04d_mask.png' % view_id), np.uint8(out_mask * 255))
    # index=4


if __name__ == '__main__':
    main()