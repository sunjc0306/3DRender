import numpy as np
import os
import cv2 as cv
import glob
import math
import random
import pyexr
from tqdm import tqdm
import scipy.io as sio
import prt.sh_util as sh_util
import copy
from Render.camera import Camera
from Render.mesh import load_obj_mesh, compute_tangent, compute_normal, load_obj_mesh_mtl

from utils.cam_util import *
view_num = 360
cam_f = 5000
cam_dist = 10
cam_t=np.array([0,0,cam_dist])
img_resw = 512
img_resh = 512

def read_data(item):
    """reads data """
    mesh_filename = glob.glob(os.path.join(item, '*.obj'))[0]  # assumes one .obj file
    text_filename = glob.glob(os.path.join(item, '*.jpg'))[0]  # assumes one .jpg file
    assert os.path.exists(mesh_filename) and os.path.exists(text_filename)
    vertices, faces, normals, faces_normals, textures, face_textures \
        = load_obj_mesh(mesh_filename, with_normal=True, with_texture=True)
    texture_image = cv.imread(text_filename)
    texture_image = cv.cvtColor(texture_image, cv.COLOR_BGR2RGB)
    prt_filename=os.path.join(item, 'bounce/prt_data.mat')
    assert os.path.exists(prt_filename)
    prt_data = sio.loadmat(prt_filename)
    prt, face_prt = prt_data['bounce0'], prt_data['face']
    return vertices, faces, normals, faces_normals, textures, face_textures, texture_image, prt, face_prt
    # import pdb
    # pdb.set_trace()
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--index', type=int)
args = parser.parse_args()
index = args.index
def main():
    shs = np.load('./env_sh.npy')
    egl = False
    data_item='./dataset_example/mesh_data/rp_dennis_posed_004/'
    from Render.gl.prt_render import PRTRender
    rndr = PRTRender(width=img_resw, height=img_resh, ms_rate=1.0)
    rndr_uv = PRTRender(width=img_resw, height=img_resh, uv_mode=True)
    vertices, faces, normals, faces_normals, textures, face_textures, \
        texture_image, prt, face_prt = read_data(data_item)

    cam = Camera(width=img_resw, height=img_resh, focal=cam_f, near=0.1, far=40,camera_type='p')
    cam.sanity_check()
    vertsMean=0
    scaleMin=1
    rndr.set_norm_mat(scaleMin, vertsMean)
    tan, bitan = compute_tangent(vertices, faces, normals, textures, face_textures)
    rndr.set_mesh(vertices, faces, normals, faces_normals, textures, face_textures, prt, face_prt, tan, bitan)
    rndr.set_albedo(texture_image)
    rndr_uv.set_mesh(vertices, faces, normals, faces_normals, textures, face_textures, prt, face_prt, tan, bitan)
    rndr_uv.set_albedo(texture_image)
    cam.center=cam_t
    for view_id in tqdm(range(0,view_num)):
        R=make_rotate(0,view_id*np.pi/180,0)
        rndr.rot_matrix=R.T
        cam.sanity_check()
        rndr.set_camera(cam)
        rndr.display()
        out_all_f = rndr.get_color(2)
        out_mask = out_all_f[:, :, 3]
        out_all_f = cv.cvtColor(out_all_f, cv.COLOR_RGBA2BGR)

        cv.imwrite(os.path.join('./test/%04d.jpg' % view_id), np.uint8(out_all_f * 255))
        # cv.imwrite(os.path.join('./test/%04d.png' % idx), np.uint8(out_mask * 255))

if __name__ == '__main__':
    main()