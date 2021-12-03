import numpy as np
import os
import cv2 as cv
from tqdm import tqdm
from opendr.camera import ProjectPoints
from opendr.renderer import ColoredRenderer
from utils.ObjIO import *
from utils.cam_util import *
from Render.mesh import  compute_normal
view_num = 360
cam_f = 5000
cam_dist = 10
cam_t=np.array([0,0,cam_dist])
img_resw = 512
img_resh = 512
def main():
    mesh=load_obj_data('dataset_example/FRONT_mesh_normalized.obj')
    if mesh['vn'] is None or mesh['vn'].shape!=mesh['v'].shape:
        mesh['vn']=compute_normal(mesh['v'],mesh['f'])
    # color=0.5*(norm+1)
    # mesh['vc']=
    rndr = ColoredRenderer()
    cam = ProjectPoints(rt=np.array([0,0,0]), t=cam_t, f=np.array([cam_f,cam_f]), c= np.array([img_resw, img_resh]) / 2. , k=np.zeros(5))
    # rndr.background_image=np.zeros([img_resw, img_resh])
    # cam_params = generate_cameras(dist=cam_dist, view_num=view_num)
    rndr.camera=cam
    rndr.frustum = {'near': 0.4,
                       'far': 40,
                       'height': img_resh,
                       'width': img_resw}
    for view_id in tqdm(range(0,view_num)):
        R=make_rotate(0,view_id*np.pi/180,0)
        rndr.set(v=mesh['v'].dot(R.T), f=mesh['f'], vc=mesh['vc'], bgcolor=np.ones(3))
        # rndr.camera=cam
        out_all_f = rndr.r
        # out_mask = out_all_f[:, :, 3]
        # out_all_f = cv.cvtColor(out_all_f, cv.COLOR_RGB2BGR)
        cv.imwrite(os.path.join('./test/pc%04d.jpg' % view_id), np.uint8(out_all_f[:,:,::-1] * 255))
        visMap = rndr.visibility_image

        # (h,w,3), barycentric weights for each tri-face

        # (h,w): {0,1} maskImage, 1-mask, 0-bg
        maskImage = np.asarray(visMap != 4294967295, np.uint32).reshape(visMap.shape)
        cv.imwrite(os.path.join('./test/pcm%04d.jpg' % view_id), np.uint8(maskImage * 255))
        # out_all_f = rndr_noraml.get_color(0)
        # out_mask = out_all_f[:, :, 3]
        # out_all_f = cv.cvtColor(out_all_f, cv.COLOR_RGBA2BGR)
        # cv.imwrite(os.path.join('./test/pn%04d.jpg' % view_id), np.uint8(out_all_f * 255))
        # cv.imwrite(os.path.join('./test/%04d.png' % idx), np.uint8(out_mask * 255))
    # index=4


if __name__ == '__main__':
    main()