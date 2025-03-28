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
import csv

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

    # Setting
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu_id

    logger = MyLightningLogger('DeepAC', cfg.save_dir)
    logger.dump_cfg(cfg, 'demo_cfg.yml')
    assert ('load_cfg' in cfg)
    assert (Path(cfg.load_cfg).exists())
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

    # Parameter Setting (Path)
    obj_name = cfg.obj_name
    data_dir = cfg.data_dir
    img_dir = os.path.join(data_dir, 'img')
    pose_path = os.path.join(img_dir, 'pose.txt')
    k_path = os.path.join(data_dir, 'k.txt')
    template_path = os.path.join(data_dir, obj_name, 'pre_render', f'{obj_name}.pkl')
    camera_pose_path = os.path.join(data_dir, 'camera_pose.txt')

    # Load Templates
    with open(template_path, "rb") as pkl_handle:
        pre_render_dict = pickle.load(pkl_handle)
    head = pre_render_dict['head']
    num_sample_contour_points = head['num_sample_contour_point']
    template_views = torch.from_numpy(pre_render_dict['template_view']).type(torch.float32)
    orientations = torch.from_numpy(pre_render_dict['orientation_in_body']).type(torch.float32)

    # Load Camera Intrinsics
    # ready by numpy, convert to torch
    K = torch.from_numpy(np.loadtxt(k_path)).type(torch.float32)

    # Load initial pose
    poses = torch.from_numpy(np.loadtxt(pose_path)).type(torch.float32)
    init_pose = poses[0]
    init_R = init_pose[:9].reshape(3, 3)
    init_t = init_pose[9:] * cfg.geometry_unit_in_meter

    # unity to opencv in-plane rotation
    #wall_open: x -90
    #four_way: x 90
    #one_way: x 90
    #whisen_view: x -90
    if "wall_open" in data_dir or "whisen_view" in data_dir:
        x_axis = -90.0 #Unity
        #x_axis = 0.0 #Not Unity
    else:
        x_axis = 90.0
        #x_axis = 0.0 #Not Unity
    init_rot = torch.from_numpy(R.from_euler('x', x_axis, degrees=True).as_matrix()).type(torch.float32)
    init_R = init_R @ init_rot

    # y-axis negation #Unity
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

    def skip(i):
        if i >= 210 and i <= 220:
            skip_result = True
        else:
            skip_result = False
        return skip_result
    # def calculate_pose_difference(init_pose, optimized_pose):
    #     # 1. Rotation difference
    #     init_R = init_pose[:3, :3]  # 초기 회전 행렬
    #     opt_R = optimized_pose[:3, :3]  # 최적화된 회전 행렬
    #     relative_R = opt_R @ init_R.T  # 상대 회전 행렬
    #     euler_angles = R.from_matrix(relative_R.cpu().numpy()).as_euler('xyz', degrees=True)
    #     x_diff_rot, y_diff_rot, z_diff_rot = euler_angles  # x, y, z축 회전 차이 (도 단위)
    #
    #     # 2. Translation difference
    #     init_t = init_pose[:3, 3]  # 초기 변환 벡터
    #     opt_t = optimized_pose[:3, 3]  # 최적화된 변환 벡터
    #     translation_diff = opt_t - init_t  # [dx, dy, dz]
    #     dx_trans, dy_trans, dz_trans = translation_diff.tolist()
    #
    #     # 3. 결과 반환
    #     return {
    #         'rotation_diff': {'x': x_diff_rot, 'y': y_diff_rot, 'z': z_diff_rot},
    #         'translation_diff': {'x': dx_trans, 'y': dy_trans, 'z': dz_trans}
    #     }

    if cfg.output_video:
        video = cv2.VideoWriter(os.path.join(logger.log_dir, obj_name + ".avi"),  # 
                                cv2.VideoWriter_fourcc('M', 'P', '4', '2'), 30, (640, 480))

    if cfg.output_image:
        frame_output_dir = os.path.join(logger.log_dir, "frames")
        os.makedirs(frame_output_dir, exist_ok=True)

    if cfg.output_closest_template:
        merged_frame_output_dir = os.path.join(logger.log_dir, "frames_with_template")
        os.makedirs(merged_frame_output_dir, exist_ok=True)

    with open('output.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = ['img_idx',
                  '[0][0]', '[0][1]', '[0][2]','[0][3]',
                  '[1][0]', '[1][1]', '[1][2]','[1][3]',
                  '[2][0]', '[2][1]', '[2][2]','[2][3]',
                  '[3][0]', '[3][1]', '[3][2]','[3][3]',]
        writer.writerow(header)

        for i, img_path in enumerate(tqdm(img_lists)):
            ori_image = read_image(img_path)
            height, width = ori_image.shape[:2]
            intrinsic_param = torch.tensor([width, height, K[0], K[4], K[2], K[5]], dtype=torch.float32)
            ori_camera = Camera(intrinsic_param)

            indices = get_closest_k_template_view_index(init_pose, orientations,
                                                        data_conf.get_top_k_template_views * data_conf.skip_template_view)
            index = indices[0] # If lost, use this
            closest_template_views = torch.stack([template_views[ind * num_sample_contour_points:(ind + 1) * num_sample_contour_points, :]
                                                    for ind in indices[::data_conf.skip_template_view]])
            closest_orientations_in_body = orientations[indices[::data_conf.skip_template_view]]
            data_lines = project_correspondences_line(closest_template_views[0], init_pose, ori_camera)
            bbox2d = get_bbox_from_p2d(data_lines['centers_in_image'])
            # print(bbox2d.numpy())
            center_x, center_y, wid, hei = bbox2d
            x1 = int(center_x - wid / 2)
            y1 = int(center_y - hei / 2)
            x2 = int(center_x + wid / 2)
            y2 = int(center_y + hei / 2)
            # print(x1, y1, x2, y2)
            # 이미지 범위 내에 있는지 확인
            is_not_within_bounds = (x1 > width and x2 > width) or (y1 > height and y2 > height)
            # print(is_within_bounds)
            if is_not_within_bounds == False:
                img, camera, h_crop, w_crop = preprocess_image(ori_image, bbox2d.numpy().copy(), ori_camera)
            else:
                img = numpy_image_to_torch(ori_image.astype(np.float32))
                camera = ori_camera
                h_crop = height
                w_crop = width

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
                lost = False
            # if i == 81 or i == 200:
            #     print(f'Frame {i}: {centers_in_image.numpy()}{centers_valid.numpy()}')
            # if torch.mean(centers_valid.float()) < 1.0:
            #     print(f'Frame {i}: {centers_in_image.numpy()}{centers_valid.numpy()}')
            # set invalid if point goes out of crop
            centers_valid = update_validity_with_crop(centers_in_image, centers_valid, w_crop, h_crop)
            # if i == 200:
            #     print(f'Frame {i}: {centers_in_image.numpy()}{centers_valid.numpy()}')
            #for camera pose debugging
            # lost = True

            # print(f'Frame {i}: centers_valid_ratio: {torch.mean(centers_valid.float())}')
            # 중요한 Contour가 밖으로 나갔을 때 반영하는 방법?
            valid_rate = torch.mean(centers_valid.float())
            if lost:
                if valid_rate > 0.9 and skip(i) == False:
                    lost = False
                    _, _, centers_in_image, centers_valid, normals_in_image, foreground_distance, background_distance, _ = \
                        calculate_basic_line_data(closest_template_views[None][:, 0], init_pose[None]._data,
                                                  camera[None]._data, 1, 0)
                    total_fore_hist, total_back_hist = \
                        model.histogram.calculate_histogram(img[None], centers_in_image, centers_valid, normals_in_image,
                                                            foreground_distance, background_distance, True)
            else:
                if valid_rate < 0.7 or skip(i):
                    lost = True
                    original_closest_template_views = closest_template_views.detach().clone()
                    original_closest_orientations_in_body = closest_orientations_in_body.detach().clone()

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

            if not lost and i % 10 == 0:
                refined_pose_obj = pred['opt_body2view_pose'][-1][0]
                refined_t = refined_pose_obj.t.detach().cpu().clone()
                refined_R = refined_pose_obj.R.detach().cpu().clone()
                F = torch.diag(torch.tensor([1.0, -1.0, 1.0]))
                local_rot_inv = torch.from_numpy(R.from_euler('x', 90.0, degrees=True).as_matrix()).type(torch.float32)
                refined_R = (F @ refined_R @ F) @ local_rot_inv
                refined_t = F @ refined_t
                t_np = refined_t.numpy().flatten()
                R_np = refined_R.numpy()

                T = np.eye(4)
                T[:3, :3] = R_np
                T[:3, 3] = t_np
                T_flat = T.flatten().tolist()
                row = [i] + T_flat
                writer.writerow(row)

            if lost:
                pred['opt_body2view_pose'][-1][0] = init_pose[None].cuda()
                pred['closest_template_views'] = original_closest_template_views[None].cuda()
                pred['closest_orientations_in_body'] = original_closest_orientations_in_body[None].cuda()
            # else:
                # print(f"Frame {i} : init_pose {init_pose.numpy()}")
                #print(calculate_pose_difference(init_pose, pred['opt_body2view_pose'][-1][0]))

            # print(f"Frame {i} : optimized_pose {pred['opt_body2view_pose'][-1][0].cpu().numpy()}")
            if cfg.output_video and cfg.output_image and cfg.output_closest_template:
                pred['optimizing_result_imgs'] = []
                model.visualize_optimization(pred['opt_body2view_pose'][-1], pred)
                ori_image = read_image(img_path)
                ori_image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
                if is_not_within_bounds == False:
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

                ori_image = cv2.UMat(ori_image)
                #if lost:
                    #cv2.putText(ori_image, "LOST", (520, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 5, cv2.LINE_AA)
                video.write(ori_image)
                ori_image = ori_image.get() # UMat to numpy

                # 프레임 이미지 저장 경로 설정
                frame_path = os.path.join(frame_output_dir, f"frame_{i:04d}.png")
                # 병합된 이미지 저장
                cv2.imwrite(frame_path, ori_image)

                merged_frame_path = os.path.join(merged_frame_output_dir, f"frame_{i:04d}.png")

                closest_template_path = os.path.join(data_dir, obj_name, 'pre_render', 'mask', str(index.item()).zfill(6) + '.jpg')
                closest_template_img = read_image(closest_template_path)

                if ori_image.shape[0] != closest_template_img.shape[0]:
                    # 원본 이미지 높이
                    target_height = ori_image.shape[0]

                    # 원본 비율 유지한 새로운 너비 계산
                    orig_height, orig_width = closest_template_img.shape[:2]
                    new_width = int((target_height / orig_height) * orig_width)

                    # 비율 유지하면서 새로운 크기로 변환
                    closest_template_img = cv2.resize(closest_template_img, (new_width, target_height))

                # 좌우로 병합 (hstack 사용)
                merged_image = np.hstack((ori_image, closest_template_img))

                # 병합된 이미지 저장
                cv2.imwrite(merged_frame_path, merged_image)


            t_distance_threshold = 0.3
            t_distance = torch.norm(camera_pose[i + 1].t - camera_pose[i].t).item()
            # print(t_distance)
            if t_distance < t_distance_threshold:
                init_pose = Pose.from_Rt(camera_pose[i + 1].R, camera_pose[i + 1].t).inv() @ Pose.from_Rt(camera_pose[i].R, camera_pose[i].t) @ pred['opt_body2view_pose'][-1][0].cpu()
            else:
                print(f"Frame {i} :Loop Closure! (t_distance: {t_distance})")
                init_pose = pred['opt_body2view_pose'][-1][0].cpu()
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