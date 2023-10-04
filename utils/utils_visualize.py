import torch
import cv2
import os
import numpy as np
from tqdm import tqdm
import trimesh
from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from body_visualizer.tools.vis_tools import colors
from body_visualizer.mesh.mesh_viewer import MeshViewer
from body_visualizer.mesh.sphere import points_to_spheres
import trimesh.util as util
from psbody.mesh import Mesh

from datetime import datetime

os.environ['PYOPENGL_PLATFORM'] = 'egl'

"""
# --------------------------------
# CheckerBoard, from Xianghui Xie
# --------------------------------
"""

class CheckerBoard:
    def __init__(self, white=(247, 246, 244), black=(146, 163, 171)):
        self.white = np.array(white)/255.
        self.black = np.array(black)/255.
        self.verts, self.faces, self.texts = None, None, None
        self.offset = None

    def init_checker(self, offset, plane='xz', xlength=500, ylength=200, square_size=0.5):
        "generate checkerboard and prepare v, f, t"
        checker = self.gen_checker_xy(self.black, self.white, square_size, xlength, ylength)
        rot = np.eye(3)
        if plane == 'xz':
            # rotate around x-axis by 90
            rot[1, 1] = rot[2, 2] = 0
            rot[1, 2] = -1
            rot[2, 1] = 1
        elif plane == 'yz':
            raise NotImplemented
        checker.v = np.matmul(checker.v, rot.T)

        # apply offsets
        checker.v += offset
        self.offset = offset

        self.verts, self.faces, self.texts = self.prep_checker_rend(checker)

    def get_rends(self):
        return self.verts, self.faces, self.texts

    def append_checker(self, checker):
        "append another checker"
        v, f, t = checker.get_rends()
        nv = self.verts.shape[1]
        self.verts = torch.cat([self.verts, v], 1)
        self.faces = torch.cat([self.faces, f+nv], 1)
        self.texts = torch.cat([self.texts, t], 1)

    @staticmethod
    def gen_checkerboard(square_size=0.5, total_size=50.0, plane='xz'):
        "plane: the checkboard is in parallal to which plane"
        checker = CheckerBoard.gen_checker_xy(square_size, total_size)
        rot = np.eye(3)
        if plane == 'xz':
            # rotate around x-axis by 90, so that the checker plane is perpendicular to y-axis
            rot[1, 1] = rot[2, 2] = 0
            rot[1, 2] = -1
            rot[2, 1] = 1
        elif plane == 'yz':
            raise NotImplemented
        checker.v = np.matmul(checker.v, rot.R)
        return checker

    def prep_checker_rend(self, checker:Mesh):
        verts = torch.from_numpy(checker.v.astype(np.float32)).cuda().unsqueeze(0)
        faces = torch.from_numpy(checker.f.astype(int)).cuda().unsqueeze(0)
        nf = checker.f.shape[0]
        texts = torch.zeros(1, nf, 4, 4, 4, 3).cuda()
        for i in range(nf):
            texts[0, i, :, :, :, :] = torch.tensor(checker.fc[i], dtype=torch.float32).cuda()
        return verts, faces, texts

    @staticmethod
    def gen_checker_xy(black, white, square_size=0.5, xlength=50.0, ylength=50.0):
        """
        generate a checker board in parallel to x-y plane
        starting from (0, 0) to (xlength, ylength), in meters
        return: psbody.Mesh
        """
        xsquares = int(xlength / square_size)
        ysquares = int(ylength / square_size)
        verts, faces, texts = [], [], []
        fcount = 0
        # black = torch.tensor([0, 0, 0.], dtype=torch.float32).cuda()
        # white = torch.tensor([1., 1., 1.], dtype=torch.float32).cuda()
        # white = np.array([247, 246, 244]) / 255.
        # black = np.array([146, 163, 171]) / 255.
        for i in range(xsquares):
            for j in range(ysquares):
                p1 = np.array([i * square_size, j * square_size, 0])
                p2 = np.array([(i + 1) * square_size, j * square_size, 0])
                p3 = np.array([(i + 1) * square_size, (j + 1) * square_size, 0])

                verts.extend([p1, p2, p3])
                faces.append([fcount * 3, fcount * 3 + 1, fcount * 3 + 2])
                fcount += 1

                p1 = np.array([i * square_size, j * square_size, 0])
                p2 = np.array([(i + 1) * square_size, (j + 1) * square_size, 0])
                p3 = np.array([i * square_size, (j + 1) * square_size, 0])

                verts.extend([p1, p2, p3])
                faces.append([fcount * 3, fcount * 3 + 1, fcount * 3 + 2])
                fcount += 1

                if (i + j) % 2 == 0:
                    texts.append(black)
                    texts.append(black)
                else:
                    texts.append(white)
                    texts.append(white)
                    

                    
                    
        # now compose as mesh
        mesh = Mesh(v=np.array(verts), f=np.array(faces), fc=np.array(texts))
        # mesh.write_ply("/BS/xxie2020/work/hoi3d/utils/checkerboards/mychecker.ply")
        mesh.v += np.array([-5, -5, 0])
        return mesh

    @staticmethod
    def from_meshes(meshes, yaxis_up=True, xlength=50, ylength=20):
        """
        initialize checkerboard ground from meshes
        """
        vertices = [x.v for x in meshes]
        if yaxis_up:
            # take ymin
            y_off = np.min(np.concatenate(vertices, 0), 0)
        else:
            # take ymax
            y_off = np.min(np.concatenate(vertices, 0), 0)
        offset = np.array([xlength/2, y_off[1], ylength/2]) # center to origin
        checker = CheckerBoard()
        checker.init_checker(offset, xlength=xlength, ylength=ylength)
        return checker

    @staticmethod
    def from_verts(verts, yaxis_up=True, xlength=5, ylength=5, square_size=0.2):
        """
        verts: (1, N, 3)
        """
        if yaxis_up:
            y_off = torch.min(verts[0], 0)[0].cpu().numpy()
        else:
            y_off = torch.max(verts[0], 0)[0].cpu().numpy()
        # print(verts.shape, y_off.shape)
        offset = np.array([-xlength/2, y_off[1], -ylength/2])
        print(offset, torch.min(verts[0], 0)[0].cpu().numpy(), torch.max(verts[0], 0)[0].cpu().numpy())
        checker = CheckerBoard()
        checker.init_checker(offset, xlength=xlength, ylength=ylength, square_size=square_size)
        return checker


"""
# --------------------------------
# Visualize avatar using body pose information and body model
# --------------------------------
"""




def save_animation(body_pose, savepath, bm, fps = 60, resolution = (800,800), identifier=''):
    imw, imh = resolution
    mv = MeshViewer(width=imw, height=imh, use_offscreen=True)
    faces = c2c(bm.f)
    img_array = []
    for fId in tqdm(range(body_pose.v.shape[0])):
        body_mesh = trimesh.Trimesh(vertices=c2c(body_pose.v[fId]),
                                    faces=faces,
                                    vertex_colors=np.tile(colors['purple'], (6890, 1)))


        generator = CheckerBoard()
        checker = generator.gen_checker_xy(generator.black, generator.white)
        checker_mesh = trimesh.Trimesh(checker.v,checker.f,process=False,face_colors=checker.fc)

        body_mesh.apply_transform(trimesh.transformations.rotation_matrix(-90, (0, 0, 10)))
        body_mesh.apply_transform(trimesh.transformations.rotation_matrix(30, (10, 0, 0)))
        body_mesh.apply_transform(trimesh.transformations.scale_matrix(0.5))

        checker_mesh.apply_transform(trimesh.transformations.rotation_matrix(-90, (0, 0, 10)))
        checker_mesh.apply_transform(trimesh.transformations.rotation_matrix(30, (10, 0, 0)))
        checker_mesh.apply_transform(trimesh.transformations.scale_matrix(0.5))

        mv.set_static_meshes([body_mesh])
        # mv.set_static_meshes([checker_mesh,body_mesh])
        body_image = mv.render(render_wireframe=False)
        body_image = body_image.astype(np.uint8)
        body_image = cv2.cvtColor(body_image, cv2.COLOR_BGR2RGB)
        
        img_array.append(body_image)
    savepath = savepath.replace('.avi', '')
    savepath = savepath + '_' + identifier + '.avi'
    # img_array = img_array[-100:]
    out = cv2.VideoWriter(savepath,cv2.VideoWriter_fourcc(*'DIVX'), fps, resolution)
     
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


def save_single(body_pose, savepath, bm, fps = 60, resolution = (800,800), identifier=''):
    imw, imh = resolution
    mv = MeshViewer(width=imw, height=imh, use_offscreen=True)
    faces = c2c(bm.f)
    img_array = []
    for fId in range(body_pose.v.shape[0]):
        body_mesh = trimesh.Trimesh(vertices=c2c(body_pose.v[fId]),
                                    faces=faces,
                                    vertex_colors=np.tile(colors['purple'], (6890, 1)))


        generator = CheckerBoard()
        checker = generator.gen_checker_xy(generator.black, generator.white)
        checker_mesh = trimesh.Trimesh(checker.v,checker.f,process=False,face_colors=checker.fc)

        body_mesh.apply_transform(trimesh.transformations.rotation_matrix(-90, (0, 0, 10)))
        body_mesh.apply_transform(trimesh.transformations.rotation_matrix(30, (10, 0, 0)))
        body_mesh.apply_transform(trimesh.transformations.scale_matrix(0.5))

        checker_mesh.apply_transform(trimesh.transformations.rotation_matrix(-90, (0, 0, 10)))
        checker_mesh.apply_transform(trimesh.transformations.rotation_matrix(30, (10, 0, 0)))
        checker_mesh.apply_transform(trimesh.transformations.scale_matrix(0.5))

        mv.set_static_meshes([body_mesh])
        # mv.set_static_meshes([checker_mesh,body_mesh])
        body_image = mv.render(render_wireframe=False)
        body_image = body_image.astype(np.uint8)
        body_image = cv2.cvtColor(body_image, cv2.COLOR_BGR2RGB)

        img_array.append(body_image)
        # breakpoint()
        savepath = savepath.replace('.avi', '')
        # savepath = savepath + '_' + str(datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p")) + '.jpg'
        savepath = savepath + '_' + identifier + '.jpg'
        
        cv2.imwrite(savepath, body_image)
    # out = cv2.VideoWriter(savepath,cv2.VideoWriter_fourcc(*'DIVX'), fps, resolution)
     
    # for i in range(len(img_array)):
    #     out.write(img_array[i])
    # out.release()


def save_animation_gt(body_pose, pred_transformer, noisy_body, predbody_pose, savepath, bm, text="", fps = 60, resolution = (800,800)):
    imw, imh = resolution
    mv = MeshViewer(width=imw, height=imh, use_offscreen=True)
    faces = c2c(bm.f)
    img_array = []
    for fId in range(body_pose.v.shape[0]):
        body_mesh = trimesh.Trimesh(vertices=c2c(body_pose.v[fId]), faces=faces, vertex_colors=np.tile(colors['purple'], (6890, 1)))
        pred_transformer_mesh = trimesh.Trimesh(vertices=c2c(pred_transformer.v[fId]), faces=faces, vertex_colors=np.tile(colors['purple'], (6890, 1)))
        noisy_body_mesh = trimesh.Trimesh(vertices=c2c(noisy_body.v[fId]), faces=faces, vertex_colors=np.tile(colors['purple'], (6890, 1)))
        predbody_mesh = trimesh.Trimesh(vertices=c2c(predbody_pose.v[fId]), faces=faces, vertex_colors=np.tile(colors['purple'], (6890, 1)))
        
        idx = str(fId).zfill(4)
        newpath = savepath.replace('.avi', '_frame_'+idx+'.png')
        # generator = CheckerBoard()
        # checker = generator.gen_checker_xy(generator.black, generator.white)
        # checker_mesh = trimesh.Trimesh(checker.v,checker.f,process=False,face_colors=checker.fc)

        body_mesh.apply_transform(trimesh.transformations.rotation_matrix(-90, (0, 0, 10)))
        body_mesh.apply_transform(trimesh.transformations.rotation_matrix(30, (10, 0, 0)))
        body_mesh.apply_transform(trimesh.transformations.scale_matrix(0.5))
        
        pred_transformer_mesh.apply_transform(trimesh.transformations.rotation_matrix(-90, (0, 0, 10)))
        pred_transformer_mesh.apply_transform(trimesh.transformations.rotation_matrix(30, (10, 0, 0)))
        pred_transformer_mesh.apply_transform(trimesh.transformations.scale_matrix(0.5))
        
        noisy_body_mesh.apply_transform(trimesh.transformations.rotation_matrix(-90, (0, 0, 10)))
        noisy_body_mesh.apply_transform(trimesh.transformations.rotation_matrix(30, (10, 0, 0)))
        noisy_body_mesh.apply_transform(trimesh.transformations.scale_matrix(0.5))

        predbody_mesh.apply_transform(trimesh.transformations.rotation_matrix(-90, (0, 0, 10)))
        predbody_mesh.apply_transform(trimesh.transformations.rotation_matrix(30, (10, 0, 0)))
        predbody_mesh.apply_transform(trimesh.transformations.scale_matrix(0.5))


        mv.set_static_meshes([body_mesh])
        body_image = mv.render(render_wireframe=False)
        body_image = body_image.astype(np.uint8)
        body_image = cv2.cvtColor(body_image, cv2.COLOR_BGR2RGB)
        cv2.putText(img=body_image, org=(10, 20),fontScale=1, text="GT", fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255,0,0), thickness=1)

        mv.set_static_meshes([pred_transformer_mesh])
        pred_transformer_image = mv.render(render_wireframe=False)
        pred_transformer_image = pred_transformer_image.astype(np.uint8)
        pred_transformer_image = cv2.cvtColor(pred_transformer_image, cv2.COLOR_BGR2RGB)
        cv2.putText(img=pred_transformer_image, org=(10, 20),fontScale=1, text="Transformer", fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255,0,0), thickness=1)
        
        mv.set_static_meshes([predbody_mesh])
        predbody_image = mv.render(render_wireframe=False)
        predbody_image = predbody_image.astype(np.uint8)
        predbody_image = cv2.cvtColor(predbody_image, cv2.COLOR_BGR2RGB)
        cv2.putText(img=predbody_image, org=(10, 20),fontScale=1, text="afterDDPM", fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255,0,0), thickness=1)

        mv.set_static_meshes([noisy_body_mesh])
        noisy_body_image = mv.render(render_wireframe=False)
        noisy_body_image = noisy_body_image.astype(np.uint8)
        noisy_body_image = cv2.cvtColor(noisy_body_image, cv2.COLOR_BGR2RGB)
        cv2.putText(img=noisy_body_image, org=(10, 20),fontScale=1, text="Noisy", fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255,0,0), thickness=1)

        cv2.putText(img=predbody_image,  org=(10, 50), fontScale=0.56, text=text, fontFace=cv2.FONT_HERSHEY_PLAIN, color=(0,0,255), thickness=1, lineType=cv2.LINE_AA)
        joint_frames = np.concatenate([body_image, pred_transformer_image, predbody_image, noisy_body_image], 1)
        # img_array.append(joint_frames)
        cv2.imwrite(newpath, joint_frames)


def save_animation_notr(body_pose, noisy_body, predbody_pose, savepath, bm, text="", fps = 60, resolution = (800,800)):
    imw, imh = resolution
    mv = MeshViewer(width=imw, height=imh, use_offscreen=True)
    faces = c2c(bm.f)
    img_array = []
    for fId in range(body_pose.v.shape[0]):
        body_mesh = trimesh.Trimesh(vertices=c2c(body_pose.v[fId]), faces=faces, vertex_colors=np.tile(colors['purple'], (6890, 1)))
        # pred_transformer_mesh = trimesh.Trimesh(vertices=c2c(pred_transformer.v[fId]), faces=faces, vertex_colors=np.tile(colors['purple'], (6890, 1)))
        noisy_body_mesh = trimesh.Trimesh(vertices=c2c(noisy_body.v[fId]), faces=faces, vertex_colors=np.tile(colors['purple'], (6890, 1)))
        predbody_mesh = trimesh.Trimesh(vertices=c2c(predbody_pose.v[fId]), faces=faces, vertex_colors=np.tile(colors['purple'], (6890, 1)))
        
        idx = str(fId).zfill(4)
        newpath = savepath.replace('.avi', '_frame_'+idx+'.png')
        # generator = CheckerBoard()
        # checker = generator.gen_checker_xy(generator.black, generator.white)
        # checker_mesh = trimesh.Trimesh(checker.v,checker.f,process=False,face_colors=checker.fc)

        body_mesh.apply_transform(trimesh.transformations.rotation_matrix(-90, (0, 0, 10)))
        body_mesh.apply_transform(trimesh.transformations.rotation_matrix(30, (10, 0, 0)))
        body_mesh.apply_transform(trimesh.transformations.scale_matrix(0.5))
        
        # pred_transformer_mesh.apply_transform(trimesh.transformations.rotation_matrix(-90, (0, 0, 10)))
        # pred_transformer_mesh.apply_transform(trimesh.transformations.rotation_matrix(30, (10, 0, 0)))
        # pred_transformer_mesh.apply_transform(trimesh.transformations.scale_matrix(0.5))
        
        noisy_body_mesh.apply_transform(trimesh.transformations.rotation_matrix(-90, (0, 0, 10)))
        noisy_body_mesh.apply_transform(trimesh.transformations.rotation_matrix(30, (10, 0, 0)))
        noisy_body_mesh.apply_transform(trimesh.transformations.scale_matrix(0.5))

        predbody_mesh.apply_transform(trimesh.transformations.rotation_matrix(-90, (0, 0, 10)))
        predbody_mesh.apply_transform(trimesh.transformations.rotation_matrix(30, (10, 0, 0)))
        predbody_mesh.apply_transform(trimesh.transformations.scale_matrix(0.5))


        mv.set_static_meshes([body_mesh])
        body_image = mv.render(render_wireframe=False)
        body_image = body_image.astype(np.uint8)
        body_image = cv2.cvtColor(body_image, cv2.COLOR_BGR2RGB)
        cv2.putText(img=body_image, org=(10, 20),fontScale=1, text="GT", fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255,0,0), thickness=1)

        # mv.set_static_meshes([pred_transformer_mesh])
        # pred_transformer_image = mv.render(render_wireframe=False)
        # pred_transformer_image = pred_transformer_image.astype(np.uint8)
        # pred_transformer_image = cv2.cvtColor(pred_transformer_image, cv2.COLOR_BGR2RGB)
        # cv2.putText(img=pred_transformer_image, org=(10, 20),fontScale=1, text="Transformer", fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255,0,0), thickness=1)
        
        mv.set_static_meshes([predbody_mesh])
        predbody_image = mv.render(render_wireframe=False)
        predbody_image = predbody_image.astype(np.uint8)
        predbody_image = cv2.cvtColor(predbody_image, cv2.COLOR_BGR2RGB)
        cv2.putText(img=predbody_image, org=(10, 20),fontScale=1, text="afterDDPM", fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255,0,0), thickness=1)

        mv.set_static_meshes([noisy_body_mesh])
        noisy_body_image = mv.render(render_wireframe=False)
        noisy_body_image = noisy_body_image.astype(np.uint8)
        noisy_body_image = cv2.cvtColor(noisy_body_image, cv2.COLOR_BGR2RGB)
        cv2.putText(img=noisy_body_image, org=(10, 20),fontScale=1, text="Noisy", fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255,0,0), thickness=1)

        cv2.putText(img=predbody_image,  org=(10, 50), fontScale=0.56, text=text, fontFace=cv2.FONT_HERSHEY_PLAIN, color=(0,0,255), thickness=1, lineType=cv2.LINE_AA)
        joint_frames = np.concatenate([body_image, predbody_image, noisy_body_image], 1)
        # img_array.append(joint_frames)
        cv2.imwrite(newpath, joint_frames)


def save_animation_singles(body_pose, pred_transformer, noisy_body, predbody_pose, savepath, bm, text="", fps = 60, resolution = (800,800)):
    imw, imh = resolution
    mv = MeshViewer(width=imw, height=imh, use_offscreen=True)
    faces = c2c(bm.f)
    img_array = []
    for fId in range(body_pose.v.shape[0]):
        body_mesh = trimesh.Trimesh(vertices=c2c(body_pose.v[fId]), faces=faces, vertex_colors=np.tile(colors['purple'], (6890, 1)))
        pred_transformer_mesh = trimesh.Trimesh(vertices=c2c(pred_transformer.v[fId]), faces=faces, vertex_colors=np.tile(colors['purple'], (6890, 1)))
        noisy_body_mesh = trimesh.Trimesh(vertices=c2c(noisy_body.v[fId]), faces=faces, vertex_colors=np.tile(colors['purple'], (6890, 1)))
        predbody_mesh = trimesh.Trimesh(vertices=c2c(predbody_pose.v[fId]), faces=faces, vertex_colors=np.tile(colors['purple'], (6890, 1)))
        
        idx = str(fId).zfill(4)
        newpath = savepath.replace('.avi', '_frame_'+idx+'.png')
        # generator = CheckerBoard()
        # checker = generator.gen_checker_xy(generator.black, generator.white)
        # checker_mesh = trimesh.Trimesh(checker.v,checker.f,process=False,face_colors=checker.fc)

        body_mesh.apply_transform(trimesh.transformations.rotation_matrix(-90, (0, 0, 10)))
        body_mesh.apply_transform(trimesh.transformations.rotation_matrix(30, (10, 0, 0)))
        body_mesh.apply_transform(trimesh.transformations.scale_matrix(0.5))
        
        pred_transformer_mesh.apply_transform(trimesh.transformations.rotation_matrix(-90, (0, 0, 10)))
        pred_transformer_mesh.apply_transform(trimesh.transformations.rotation_matrix(30, (10, 0, 0)))
        pred_transformer_mesh.apply_transform(trimesh.transformations.scale_matrix(0.5))
        
        noisy_body_mesh.apply_transform(trimesh.transformations.rotation_matrix(-90, (0, 0, 10)))
        noisy_body_mesh.apply_transform(trimesh.transformations.rotation_matrix(30, (10, 0, 0)))
        noisy_body_mesh.apply_transform(trimesh.transformations.scale_matrix(0.5))

        predbody_mesh.apply_transform(trimesh.transformations.rotation_matrix(-90, (0, 0, 10)))
        predbody_mesh.apply_transform(trimesh.transformations.rotation_matrix(30, (10, 0, 0)))
        predbody_mesh.apply_transform(trimesh.transformations.scale_matrix(0.5))


        mv.set_static_meshes([body_mesh])
        body_image = mv.render(render_wireframe=False)
        body_image = body_image.astype(np.uint8)
        body_image = cv2.cvtColor(body_image, cv2.COLOR_BGR2RGB)
        cv2.putText(img=body_image, org=(10, 20),fontScale=1, text="GT", fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255,0,0), thickness=1)

        mv.set_static_meshes([pred_transformer_mesh])
        pred_transformer_image = mv.render(render_wireframe=False)
        pred_transformer_image = pred_transformer_image.astype(np.uint8)
        pred_transformer_image = cv2.cvtColor(pred_transformer_image, cv2.COLOR_BGR2RGB)
        cv2.putText(img=pred_transformer_image, org=(10, 20),fontScale=1, text="Transformer", fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255,0,0), thickness=1)
        
        mv.set_static_meshes([predbody_mesh])
        predbody_image = mv.render(render_wireframe=False)
        predbody_image = predbody_image.astype(np.uint8)
        predbody_image = cv2.cvtColor(predbody_image, cv2.COLOR_BGR2RGB)
        cv2.putText(img=predbody_image, org=(10, 20),fontScale=1, text="afterDDPM", fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255,0,0), thickness=1)

        mv.set_static_meshes([noisy_body_mesh])
        noisy_body_image = mv.render(render_wireframe=False)
        noisy_body_image = noisy_body_image.astype(np.uint8)
        noisy_body_image = cv2.cvtColor(noisy_body_image, cv2.COLOR_BGR2RGB)
        cv2.putText(img=noisy_body_image, org=(10, 20),fontScale=1, text="Noisy", fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255,0,0), thickness=1)

        cv2.putText(img=predbody_image,  org=(10, 50), fontScale=0.56, text=text, fontFace=cv2.FONT_HERSHEY_PLAIN, color=(0,0,255), thickness=1, lineType=cv2.LINE_AA)
        # joint_frames = np.concatenate([body_image, pred_transformer_image, predbody_image, noisy_body_image], 1)
        # img_array.append(joint_frames)
        finalpath = newpath.replace('.png', '_gt.png')
        cv2.imwrite(finalpath, body_image)
        finalpath = newpath.replace('.png', '_Avatar.png')
        cv2.imwrite(finalpath, pred_transformer_image)
        finalpath = newpath.replace('.png', '_DDPM.png')
        cv2.imwrite(finalpath, predbody_image)
        finalpath = newpath.replace('.png', '_noise.png')
        cv2.imwrite(finalpath, noisy_body_image)