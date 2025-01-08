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
    init_rot = torch.from_numpy(R.from_euler('x', -90, degrees=True).as_matrix()).type(torch.float32) #blender상에서 보니 wall_open.obj에 local하게 orientation이 x 90이 먹여져있었음.
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
        # Read Camera to ACCore Pose
        poses_data = []  # 순수 텐서를 저장할 리스트
        with open(file_path, 'r') as f:
            for line in f:
                # 데이터 파싱
                data = list(map(float, line.strip().split()))
                translation = data[:3]  # x, y, z
                quaternion = np.array(data[3:])  # w, x, y, z

                # 쿼터니언 -> 회전 행렬 변환
                rotation_matrix = R.from_quat(quaternion).as_matrix()  # (3, 3)

                z_angle = -90 # 휘센 뷰 제외 rotation offset 적용
                z_rotation = R.from_euler('z', z_angle, degrees=True).as_matrix()
                rotation_matrix = rotation_matrix @ z_rotation

                # Y축 플립
                rotation_matrix[:, 1] = -rotation_matrix[:, 1]  # Y축 플립
                rotation_matrix[1, :] = -rotation_matrix[1, :]  # Y축 플립
                translation[1] = -translation[1]  # Y축 플립

                # Pose 데이터 생성
                R_tensor = torch.tensor(rotation_matrix, dtype=torch.float32)  # 회전 행렬
                t_tensor = torch.tensor(translation, dtype=torch.float32)  # 번역 벡터
                pose_data = torch.cat([R_tensor.flatten(), t_tensor])  # 1D 텐서로 결합
                poses_data.append(pose_data)

        # 텐서를 스택하고 Pose 객체로 변환
        poses_tensor = torch.stack(poses_data)  # (N, 12)
        print("Stacked pose tensor shape:", poses_tensor.shape)
        return Pose(poses_tensor)

    # 파일 경로
    camera_pose_path = "/home/ohj/DeepAC/data/wall_open/camera_pose.txt"
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

        img, = zero_pad(data_conf.pad, img)
        img = img.astype(np.float32)
        return numpy_image_to_torch(img), camera

    if cfg.output_video:
        video = cv2.VideoWriter(os.path.join(logger.log_dir, obj_name + ".avi"),  # 
                                cv2.VideoWriter_fourcc('M', 'P', '4', '2'), 30, cfg.output_size)
        # 추가: 원본 이미지 기반으로 저장할 비디오 생성
        full_video = cv2.VideoWriter(os.path.join(logger.log_dir, obj_name + "_full.avi"),
                                     cv2.VideoWriter_fourcc('M', 'P', '4', '2'), 30, (640, 480))
    if cfg.output_image:
        frame_output_dir = os.path.join(logger.log_dir, "frames")
        os.makedirs(frame_output_dir, exist_ok=True)

    prediction_thresholds = {
        'min_valid_ratio': 0.95,     # 최소 유효 중심점 비율
        'min_fore_hist': 0.00,       # 최소 전경 히스토그램 평균값
        'min_back_hist': 0.00,       # 최소 배경 히스토그램 평균값
        'max_pose_diff': 0.5,        # 최대 포즈 변화량
        'min_pose_diff': 0.05       # 최소 포즈 변화량
    }
    world_object_pose = None
    cnt = 0
    # 추적 실패 감지 함수
    def detect_tracking_failure(centers_valid, fore_hist, back_hist, init_pose, opt_pose, prediction_thresholds):
        """
        추적 실패를 감지하고 로그를 출력.
        """
        failure_detected = False
        log_messages = []

        # 조건 1: 유효한 중심점 비율
        valid_ratio = torch.sum(centers_valid) / centers_valid.numel()
        if valid_ratio < prediction_thresholds['min_valid_ratio']:
            failure_detected = True
            log_messages.append(
                f"Low valid centers ratio: {valid_ratio:.2f} (threshold: {prediction_thresholds['min_valid_ratio']})")

        # 조건 2: 히스토그램 신뢰도
        if fore_hist.mean() < prediction_thresholds['min_fore_hist']:
            failure_detected = True
            log_messages.append(
                f"Low foreground histogram mean: {fore_hist.mean():.2f} (threshold: {prediction_thresholds['min_fore_hist']})")
        if back_hist.mean() < prediction_thresholds['min_back_hist']:
            failure_detected = True
            log_messages.append(
                f"Low background histogram mean: {back_hist.mean():.2f} (threshold: {prediction_thresholds['min_back_hist']})")

        # 두 Pose 간의 차이 계산
        rotation_diff = torch.norm(opt_pose.R - init_pose.R)  # 회전 행렬 차이
        translation_diff = torch.norm(opt_pose.t - init_pose.t)  # 번역 벡터 차이
        pose_diff = rotation_diff + translation_diff

        # 추적 실패 감지
        if pose_diff > prediction_thresholds['max_pose_diff']:
            failure_detected = True
            log_messages.append(
                f"Large pose difference detected: {pose_diff:.2f} (threshold: {prediction_thresholds['max_pose_diff']})")

        return failure_detected, log_messages


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
        img, camera = preprocess_image(ori_image, bbox2d.numpy().copy(), ori_camera)

        if i == 0:
            _, _, centers_in_image, centers_valid, normals_in_image, foreground_distance, background_distance, _ =\
                calculate_basic_line_data(closest_template_views[None][:, 0], init_pose[None]._data, camera[None]._data, 1, 0)
            total_fore_hist, total_back_hist = \
                model.histogram.calculate_histogram(img[None], centers_in_image, centers_valid, normals_in_image, 
                                                    foreground_distance, background_distance, True)

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

        # 추적 실패 감지
        failure_detected, log_messages = detect_tracking_failure(
            centers_valid, total_fore_hist, total_back_hist, init_pose, pred['opt_body2view_pose'][-1][0].cpu(),
            prediction_thresholds
        )
        #
        # if failure_detected:
        #     print(f"Frame {i}: Tracking failure detected.")
        #     for message in log_messages:
        #         print(message)
        #
        #     cnt = cnt + 1
        #     if cnt >= 2:
        #         print(f"Recover Pose.")
        #         # 월드 좌표계에서 카메라의 포즈 생성
        #         camera_pose_matrix = Pose.from_Rt(camera_pose[i].R, camera_pose[i].t)
        #         # print("Recovered Camera Pose Matrix (World to Camera): ", camera_pose_matrix.R, camera_pose_matrix.t)
        #
        #         # 추적 실패 시, world_object_pose와 camera_pose[i-1]을 이용하여 객체 포즈 복구
        #         recovered_pose = camera_pose_matrix.inv() @ world_object_pose
        #         pred['opt_body2view_pose'][-1][0] = recovered_pose
        #         # print("Recovered Object Pose (Camera Space): ", recovered_pose.R, recovered_pose.t)
        # else:
        #     cnt = 0
        #     # 실패하지 않았고 포즈 차이가 너무 작지 않을 때 업데이트
        #     pose_diff = torch.norm(pred['opt_body2view_pose'][-1][0].cpu()._data - init_pose._data)
        #     valid_ratio = torch.sum(centers_valid) / centers_valid.numel()
        #     if pose_diff < prediction_thresholds['min_pose_diff'] and valid_ratio > prediction_thresholds['min_valid_ratio']:
        #         print(f"Update Object World Pose.")
        #         # 월드 좌표계에서 카메라의 포즈 생성
        #         camera_pose_matrix = Pose.from_Rt(camera_pose[i].R, camera_pose[i].t)
        #         # print("Camera Pose Matrix (World to Camera): ", camera_pose_matrix.R, camera_pose_matrix.t)
        #
        #         # 월드 좌표계에서 객체 포즈 업데이트
        #         world_object_pose = camera_pose_matrix @ pred['opt_body2view_pose'][-1][0].cpu()
        #         # print("Updated World Object Pose: ", world_object_pose.R, world_object_pose.t)

        # if i == 0:
        #     world_object_pose = Pose.from_Rt(camera_pose[i].R, camera_pose[i].t) @ init_pose
        #     pred['opt_body2view_pose'][-1][0] = init_pose
        # else:
        #     camera_pose_matrix = Pose.from_Rt(camera_pose[i].R, camera_pose[i].t)
        #     recovered_pose = camera_pose_matrix.inv() @ world_object_pose
        #     pred['opt_body2view_pose'][-1][0] = recovered_pose

        if cfg.output_video:
            pred['optimizing_result_imgs'] = []
            model.visualize_optimization(pred['opt_body2view_pose'][-1], pred)
            video.write(cv2.resize(pred['optimizing_result_imgs'][0][0], cfg.output_size))

        # 원본 이미지 위에 예측값 오버레이
        overlay_image = ori_image.copy()
        if 'lines' not in data_lines:
            print("Generating 'lines' key from available data...")
            centers = data_lines.get('centers_in_image', None)
            normals = data_lines.get('normals_in_image', None)

            if centers is not None and normals is not None:
                # centers와 normals로 lines 생성
                lines = []
                for center, normal in zip(centers, normals):
                    start = center
                    end = center + normal * 10  # normal 방향으로 10 픽셀 이동
                    lines.append((start.tolist(), end.tolist()))
                data_lines['lines'] = lines
            else:
                print("Error: Unable to generate 'lines'. Missing centers_in_image or normals_in_image.")
                data_lines['lines'] = []  # 빈 리스트로 초기화
        for line in data_lines['lines']:
            pt1 = tuple(map(int, line[0]))
            pt2 = tuple(map(int, line[1]))
            # cv2.line(overlay_image, pt1, pt2, (0, 255, 0), 2)  # 초록색 선
            cv2.circle(overlay_image, pt1, radius=5, color=(0, 255, 0), thickness=-1)

        # 원본 이미지 비디오에 저장
        full_video.write(overlay_image)

        init_pose = (Pose.from_Rt(camera_pose[i + 1].R, camera_pose[i + 1].t)).inv() @ Pose.from_Rt(camera_pose[i].R, camera_pose[i].t) @ pred['opt_body2view_pose'][-1][0].cpu()
        index = get_closest_template_view_index(init_pose, orientations)

        if cfg.output_image:
            # 원본 이미지 위에 예측값 오버레이
            overlay_image = ori_image.copy()
            if 'lines' not in data_lines:
                print("Generating 'lines' key from available data...")
                centers = data_lines.get('centers_in_image', None)
                normals = data_lines.get('normals_in_image', None)

                if centers is not None and normals is not None:
                    # centers와 normals로 lines 생성
                    lines = []
                    for center, normal in zip(centers, normals):
                        start = center
                        end = center + normal * 10  # normal 방향으로 10 픽셀 이동
                        lines.append((start.tolist(), end.tolist()))
                    data_lines['lines'] = lines
                else:
                    print("Error: Unable to generate 'lines'. Missing centers_in_image or normals_in_image.")
                    data_lines['lines'] = []  # 빈 리스트로 초기화
            for line in data_lines['lines']:
                pt1 = tuple(map(int, line[0]))
                pt2 = tuple(map(int, line[1]))
                #cv2.line(overlay_image, pt1, pt2, (0, 255, 0), 2)  # 초록색 선
                cv2.circle(overlay_image, pt1, radius=5, color=(0, 255, 0), thickness=-1)

            # # 프레임 이미지 저장 경로 설정
            # frame_path = os.path.join(frame_output_dir, f"frame_{i:04d}.png")
            #
            # # 오버레이된 이미지 저장
            # cv2.imwrite(frame_path, overlay_image)

            if isinstance(index, (torch.Tensor, np.ndarray)):
                index_value = index.item()  # tensor(1027) -> 1027로 변환
            elif isinstance(index, int):
                index_value = index  # 이미 정수라면 그대로 사용
            else:
                raise ValueError(f"Unsupported index type: {type(index)}")
            print(index_value)
                # Index 기반으로 오른쪽 이미지 로드
            index_str = str(index_value).zfill(6)  # 6자리로 패딩
            mask_image_path = f"/home/ohj/DeepAC/data/wall_open/wall_open/pre_render/mask/{index_str}.jpg"

            if os.path.exists(mask_image_path):
                mask_image = cv2.imread(mask_image_path)
                if mask_image is None:
                    print(f"Error: Unable to load image from {mask_image_path}")
                    mask_image = np.zeros_like(overlay_image)  # 빈 이미지 대체
            else:
                print(f"Error: File does not exist at {mask_image_path}")
                mask_image = np.zeros_like(overlay_image)  # 빈 이미지 대체

            if overlay_image.shape[:2] != mask_image.shape[:2]:
                print("Resizing mask_image to match overlay_image dimensions.")
                mask_image = cv2.resize(mask_image, (overlay_image.shape[1], overlay_image.shape[0]))

            # 데이터 타입 맞추기
            if overlay_image.dtype != mask_image.dtype:
                print("Converting mask_image to match overlay_image dtype.")
                mask_image = mask_image.astype(overlay_image.dtype)

            # 두 이미지를 나란히 붙이기
            combined_image = cv2.hconcat([overlay_image, mask_image])

            # 프레임 이미지 저장 경로 설정
            frame_path = os.path.join(frame_output_dir, f"frame_{i:04d}.png")

            # 병합된 이미지 저장
            cv2.imwrite(frame_path, combined_image)

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
    full_video.release()