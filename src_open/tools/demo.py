import os
import torch
from pathlib import Path
import numpy as np
from omegaconf import DictConfig, OmegaConf
import cv2
import copy
import warnings
import pickle
import glob
from tqdm import tqdm

from ..utils.geometry.wrappers import Pose, Camera
from ..models import get_model
from ..utils.lightening_utils import MyLightningLogger, convert_old_model, load_model_weight
from ..dataset.utils import read_image, resize, numpy_image_to_torch, crop, zero_pad, get_imgaug_seq
from ..utils.utils import project_correspondences_line, get_closest_template_view_index,\
    get_closest_k_template_view_index, get_bbox_from_p2d
from ..models.deep_ac import calculate_basic_line_data

from scipy.spatial.transform import Rotation as R

@torch.no_grad()
def main(cfg):

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu_id

    logger = MyLightningLogger('DeepAC', cfg.save_dir)
    logger.dump_cfg(cfg, 'demo_cfg.yml')
    assert ('load_cfg' in cfg)
    # assert ('load_model' in cfg)
    assert (Path(cfg.load_cfg).exists())
    # assert (Path(cfg.load_model).exists())
    train_cfg = OmegaConf.load(cfg.load_cfg)
    data_conf = train_cfg.data
    logger.dump_cfg(train_cfg, 'train_cfg.yml')

    model = get_model(train_cfg.models.name)(train_cfg.models)
    ckpt = torch.load(cfg.load_model, map_location='cpu')
    if "pytorch-lightning_version" not in ckpt:
        warnings.warn(
            "Warning! Old .pth checkpoint is deprecated. "
            "Convert the checkpoint with tools/convert_old_checkpoint.py "
        )
        ckpt = convert_old_model(ckpt)
    load_model_weight(model, ckpt, logger)
    logger.info("Loaded model weight from {}".format(cfg.load_model))
    model.cuda()
    model.eval()

    fore_learn_rate = cfg.fore_learn_rate
    back_learn_rate = cfg.back_learn_rate

    obj_name = cfg.obj_name
    data_dir = cfg.data_dir
    img_dir = os.path.join(data_dir, 'img')
    pose_path = os.path.join(img_dir, 'pose.txt')
    K_path = os.path.join(data_dir, 'K.txt')
    template_path = os.path.join(data_dir, obj_name, 'pre_render', f'{obj_name}.pkl')
    camera_pose_path = os.path.join(data_dir, 'camera_pose.txt')

    # Load Templates
    with open(template_path, "rb") as pkl_handle:
        pre_render_dict = pickle.load(pkl_handle)
    head = pre_render_dict['head']
    num_sample_contour_points = head['num_sample_contour_point']
    template_views = torch.from_numpy(pre_render_dict['template_view']).type(torch.float32)
    orientations = torch.from_numpy(pre_render_dict['orientation_in_body']).type(torch.float32)

    K = torch.from_numpy(np.loadtxt(K_path)).type(torch.float32)

    # Load initial pose
    poses = torch.from_numpy(np.loadtxt(pose_path)).type(torch.float32)
    init_pose = poses[0]
    init_R = init_pose[:9].reshape(3, 3)
    init_t = init_pose[9:] * cfg.geometry_unit_in_meter

    # in-plane rotation
    #wall_open: x -90
    #four_way: x 90
    #one_way: x 90
    #whisen_view: x -90
    if "wall_open" in data_dir or "whisen_view" in data_dir:
        x_axis = -90.0
    else:
        x_axis = 90.0
    init_rot = torch.from_numpy(R.from_euler('x', x_axis, degrees=True).as_matrix()).type(torch.float32)
    init_R = init_R @ init_rot

    # y-axis negation
    init_R[:, 1] = -init_R[:, 1]
    init_R[1, :] = -init_R[1, :]
    init_t[1] = -init_t[1]

    init_pose = Pose.from_Rt(init_R, init_t)

    # Read input image sequence
    img_lists = glob.glob(img_dir + '/*.png', recursive=True) + glob.glob(img_dir + '/*.jpg', recursive=True)
    img_lists.sort()


    def read_camera_pose_as_pose(file_path):
        # Read c2AcCore Pose
        poses_data = []  # 순수 텐서를 저장할 리스트
        with open(file_path, 'r') as f:
            for line in f:
                # 데이터 파싱
                data = list(map(float, line.strip().split()))
                translation = data[:3]  # x, y, z
                quaternion = np.array(data[3:])  # w, x, y, z

                # 쿼터니언 -> 회전 행렬 변환
                rotation_matrix = R.from_quat(quaternion).as_matrix()  # (3, 3)

                if "whisen_view" in data_dir:
                    z_angle = 0
                else:
                    z_angle = -90 # 휘센 뷰 제외 rotation offset 적용
                z_rotation = R.from_euler('z', z_angle, degrees=True).as_matrix()
                rotation_matrix = rotation_matrix @ z_rotation

                # y-axis flip
                rotation_matrix[:, 1] = -rotation_matrix[:, 1]
                rotation_matrix[1, :] = -rotation_matrix[1, :]
                translation[1] = -translation[1]

                # Pose 데이터 생성
                R_tensor = torch.tensor(rotation_matrix, dtype=torch.float32)  # 회전 행렬
                t_tensor = torch.tensor(translation, dtype=torch.float32)  # 번역 벡터
                pose_data = torch.cat([R_tensor.flatten(), t_tensor])  # 1D 텐서로 결합
                poses_data.append(pose_data)

        # 텐서를 스택하고 Pose 객체로 변환
        poses_tensor = torch.stack(poses_data)  # (N, 12)
        print("Stacked pose tensor shape:", poses_tensor.shape)
        return Pose(poses_tensor)

    # 카메라 포즈 읽기 및 변환
    camera_pose = read_camera_pose_as_pose(camera_pose_path)

    # # 첫 번째 프레임의 camera pose 읽기
    # first_pose_data = camera_pose._data[0]  # 첫 번째 행 데이터
    # rot = first_pose_data[:9].reshape(3, 3)  # 회전 행렬 (3x3)
    # trans = first_pose_data[9:]  # 번역 벡터 (3,)
    #
    # # 결과 출력
    # print("First frame rotation matrix (R):")
    # print(rot)
    # print("First frame translation vector (t):")
    # print(trans)
    # print(camera_pose[0])
    def preprocess_image(img, bbox2d, camera):
        bbox2d[2:] += data_conf.crop_border * 2
        img, camera, bbox = crop(img, bbox2d, camera=camera, return_bbox=True)

        scales = (1, 1)
        if isinstance(data_conf.resize, int):
            if data_conf.resize_by == 'max':
                # print('img shape', img.shape)
                # print('img path', image_path)
                img, scales = resize(img, data_conf.resize, fn=max)
            elif (data_conf.resize_by == 'min' or (data_conf.resize_by == 'min_if' and min(*img.shape[:2]) < data_conf.resize)):
                img, scales = resize(img, data_conf.resize, fn=min)
        elif len(data_conf.resize) == 2:
            img, scales = resize(img, list(data_conf.resize))
        if scales != (1, 1):
            camera = camera.scale(scales)

        img, h_crop, w_crop = zero_pad(data_conf.pad, img)
        img = img.astype(np.float32)
        return numpy_image_to_torch(img), camera, h_crop, w_crop

    def update_validity_with_crop(centers_in_image, centers_valid, w_crop, h_crop):
        """
        centers_in_image의 (x, y) 값이 w_crop 또는 h_crop보다 크면 valid를 False로 설정
        """
        # centers_in_image의 x, y 좌표 가져오기
        x_coords = centers_in_image[..., 0]
        y_coords = centers_in_image[..., 1]

        # 유효성 검사: (x < w_crop) & (y < h_crop) 인 경우만 valid
        valid_crop = torch.logical_and(x_coords < w_crop, y_coords < h_crop)

        # 기존 valid와 논리곱 (both conditions must be True)
        centers_valid = torch.logical_and(centers_valid, valid_crop)

        return centers_valid

    if cfg.output_video:
        video = cv2.VideoWriter(os.path.join(logger.log_dir, obj_name + ".avi"),  # 
                                cv2.VideoWriter_fourcc('M', 'P', '4', '2'), 30, (640, 480))

    if cfg.output_image:
        frame_output_dir = os.path.join(logger.log_dir, "frames")
        os.makedirs(frame_output_dir, exist_ok=True)

    for i, img_path in enumerate(tqdm(img_lists)):
        ori_image = read_image(img_path)
        height, width = ori_image.shape[:2]
        intrinsic_param = torch.tensor([width, height, K[0], K[4], K[2], K[5]], dtype=torch.float32)
        ori_camera = Camera(intrinsic_param)

        indices = get_closest_k_template_view_index(init_pose, orientations,
                                                    data_conf.get_top_k_template_views * data_conf.skip_template_view)
        closest_template_views = torch.stack([template_views[ind * num_sample_contour_points:(ind + 1) * num_sample_contour_points, :]
                                                for ind in indices[::data_conf.skip_template_view]])
        closest_orientations_in_body = orientations[indices[::data_conf.skip_template_view]]
        data_lines = project_correspondences_line(closest_template_views[0], init_pose, ori_camera)
        bbox2d = get_bbox_from_p2d(data_lines['centers_in_image'])
        # print(bbox2d)
        img, camera, h_crop, w_crop = preprocess_image(ori_image, bbox2d.numpy().copy(), ori_camera)

        # # Crop image 확인
        # img_numpy = img
        # img_numpy = img_numpy.detach().cpu().numpy()
        # img_numpy = np.transpose(img_numpy, (1, 2, 0))
        # cv2.imshow('cropped',img_numpy)
        # cv2.waitKey(0)

        if i == 0:
            _, _, centers_in_image, centers_valid, normals_in_image, foreground_distance, background_distance, _ =\
                calculate_basic_line_data(closest_template_views[None][:, 0], init_pose[None]._data, camera[None]._data, 1, 0)
            total_fore_hist, total_back_hist = \
                model.histogram.calculate_histogram(img[None], centers_in_image, centers_valid, normals_in_image, 
                                                    foreground_distance, background_distance, True)

        centers_valid = update_validity_with_crop(centers_in_image, centers_valid, w_crop, h_crop)

        original_closest_template_views = closest_template_views.detach().clone()
        original_closest_orientations_in_body = closest_orientations_in_body.detach().clone()

        lost = False
        data = {
            'image': img[None].cuda(),
            'camera': camera[None].cuda(),
            'body2view_pose': init_pose[None].cuda(),
            'closest_template_views': closest_template_views[None].cuda(),
            'closest_orientations_in_body': closest_orientations_in_body[None].cuda(),
            'fore_hist': total_fore_hist.cuda(),
            'back_hist': total_back_hist.cuda()
        }
        pred = model._forward(data, visualize=False, tracking=True)

        #for camera pose debugging
        # lost = True

        # print(f'Frame {i}: centers_valid_ratio: {torch.mean(centers_valid.float())}')
        # 중요한 Contour가 밖으로 나갔을 때 반영하는 방법?
        if torch.mean(centers_valid.float()) <= 0.6:
            lost = True
        if lost:
            pred['opt_body2view_pose'][-1][0] = init_pose[None].cuda()
            pred['closest_template_views'] = original_closest_template_views[None].cuda()
            pred['closest_orientations_in_body'] = original_closest_orientations_in_body[None].cuda()

        if cfg.output_video and cfg.output_image:
            pred['optimizing_result_imgs'] = []
            model.visualize_optimization(pred['opt_body2view_pose'][-1], pred)
            ori_image = read_image(img_path)
            crop_border = data_conf.crop_border
            x_center, y_center, w, h = bbox2d
            w += 2 * crop_border
            h += 2 * crop_border
            x1 = int (x_center - w / 2 )
            y1 = int (y_center - h / 2 )
            x2 = int (x_center + w / 2 )
            y2 = int (y_center + h / 2 )
            x1 = max(x1, 0)
            y1 = max(y1, 0)
            x2 = min(x2, ori_image.shape[1])
            y2 = min(y2, ori_image.shape[0])
            # cv2.imshow('check', ori_image)
            resized_pred_image = cv2.resize(pred['optimizing_result_imgs'][0][0][:h_crop, :w_crop], (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
            ori_image[y1:y2, x1:x2] = resized_pred_image

            if lost:
                ori_image = cv2.UMat(ori_image)
                cv2.putText(ori_image, "LOST", (520, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 5, cv2.LINE_AA)
            video.write(ori_image)

            # 프레임 이미지 저장 경로 설정
            frame_path = os.path.join(frame_output_dir, f"frame_{i:04d}.png")
            # 병합된 이미지 저장
            cv2.imwrite(frame_path, ori_image)

        init_pose = (Pose.from_Rt(camera_pose[i + 1].R, camera_pose[i + 1].t)).inv() @ Pose.from_Rt(camera_pose[i].R, camera_pose[i].t) @ pred['opt_body2view_pose'][-1][0].cpu()
        # init_pose = pred['opt_body2view_pose'][-1][0].cpu()
        index = get_closest_template_view_index(init_pose, orientations)
        # print(f"Frame {i+1} index: {index}")
        # init_pose = pred['opt_body2view_pose'][-1][0].cpu()
        # index = get_closest_template_view_index(init_pose, orientations)
        # print(f"for Frame {i}, template index : {index}")
        closest_template_view = template_views[index*num_sample_contour_points:(index+1)*num_sample_contour_points, :]
        _, _, centers_in_image, centers_valid, normals_in_image, foreground_distance, background_distance, _ =\
            calculate_basic_line_data(closest_template_view[None], init_pose[None]._data, camera[None]._data, 1, 0)
        fore_hist, back_hist = \
            model.histogram.calculate_histogram(img[None], centers_in_image, centers_valid, normals_in_image, 
                                                foreground_distance, background_distance, True)
        total_fore_hist = (1 - fore_learn_rate) * total_fore_hist + fore_learn_rate * fore_hist
        total_back_hist = (1 - back_learn_rate) * total_back_hist + back_learn_rate * back_hist

    video.release()