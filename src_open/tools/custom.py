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

from ..network.server import NetworkServer

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
    template_path = os.path.join(data_dir, obj_name, 'pre_render', f'{obj_name}.pkl')

    # Load Templates
    with open(template_path, "rb") as pkl_handle:
        pre_render_dict = pickle.load(pkl_handle)
    head = pre_render_dict['head']
    num_sample_contour_points = head['num_sample_contour_point']
    template_views = torch.from_numpy(pre_render_dict['template_view']).type(torch.float32)
    orientations = torch.from_numpy(pre_render_dict['orientation_in_body']).type(torch.float32)

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
                                cv2.VideoWriter_fourcc('M', 'P', '4', '2'), 30, (800,600))

    if cfg.output_image:
        frame_output_dir = os.path.join(logger.log_dir, "frames")
        os.makedirs(frame_output_dir, exist_ok=True)

    if cfg.output_closest_template:
        merged_frame_output_dir = os.path.join(logger.log_dir, "frames_with_template")
        os.makedirs(merged_frame_output_dir, exist_ok=True)

    server = NetworkServer(host='192.168.0.6', port=7777, save_img=True, save_data=True)
    server.start()
    print("[MainThread] Server started, waiting for frames...")


    i = 0;
    try:
        while True:
            # 최신 프레임을 가져온다.
            frame_id, header, original_img = server.get_latest_frame()
            if frame_id is None:
                continue
            # intrinsic matrix K
            # initial pose
            # camera pose
            # ori_image
            fx = header[8]
            fy = header[9]
            cx = header[10]
            cy = header[11]
            width = header[12]
            height = header[13]
            #hmd_translation = header[16:19]
            #hmd_rotation = header[19:23]
            camera_pose = header[16:23]
            object_translation = header[23:26]
            object_rotation = header[26:30]

            # header에서 추출한 값들을 출력하여 확인합니다.
            # print("=== Header에서 추출한 값들 ===")
            # print(f"fx: {fx}")
            # print(f"fy: {fy}")
            # print(f"cx: {cx}")
            # print(f"cy: {cy}")
            # print(f"Header에 기록된 이미지 너비: {width}")
            # print(f"Header에 기록된 이미지 높이: {height}")
            # print(f"Camera pose (header[16:23]): {camera_pose}")
            # print(f"Object translation: {object_translation}")
            # print(f"Object rotation (quaternion): {object_rotation}")

            K_np = np.array([
                [fx], [0], [cx],
                [0], [fy], [cy],
                [0], [0], [1]
            ], dtype=np.float32)
            K = torch.from_numpy(K_np)
            # print("\n=== Intrinsic Matrix K ===")
            # print(K)

            init_t = torch.tensor(object_translation, dtype=torch.float32) * cfg.geometry_unit_in_meter
            init_R_np = R.from_quat(object_rotation).as_matrix()
            init_R = torch.from_numpy(init_R_np).type(torch.float32)

            # print("\n[초기 변환 전]")
            # print("Translation (init_t):", init_t)
            # print("Rotation matrix (init_R):")
            # print(init_R)
            F = torch.diag(torch.tensor([1.0, -1.0, 1.0]))
            local_rot = torch.from_numpy(R.from_euler('x', -90.0, degrees=True).as_matrix()).type(torch.float32)
            init_R = F @ init_R @ local_rot @ F
            init_t = F @ init_t
            init_pose = Pose.from_Rt(init_R, init_t)
            # print("\n=== 최종 Initial Pose ===")
            # print(init_pose)

            ori_image = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            # height, width = ori_image.shape[:2]
            # print(f"img로 계산된 이미지 너비: {width}")
            # print(f"img로 계산된 이미지 높이: {height}")


            #########################################################3
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

            # if i == 0:
            if True:
                _, _, centers_in_image, centers_valid, normals_in_image, foreground_distance, background_distance, _ =\
                    calculate_basic_line_data(closest_template_views[None][:, 0], init_pose[None]._data, camera[None]._data, 1, 0)
                total_fore_hist, total_back_hist = \
                    model.histogram.calculate_histogram(img[None], centers_in_image, centers_valid, normals_in_image,
                                                        foreground_distance, background_distance, True)
                lost = False
            centers_valid = update_validity_with_crop(centers_in_image, centers_valid, w_crop, h_crop)
            valid_rate = torch.mean(centers_valid.float())

            # lost = True

            if lost:
                if valid_rate > 0.9:
                    lost = False
                    _, _, centers_in_image, centers_valid, normals_in_image, foreground_distance, background_distance, _ = \
                        calculate_basic_line_data(closest_template_views[None][:, 0], init_pose[None]._data,
                                                  camera[None]._data, 1, 0)
                    total_fore_hist, total_back_hist = \
                        model.histogram.calculate_histogram(img[None], centers_in_image, centers_valid, normals_in_image,
                                                            foreground_distance, background_distance, True)
            else:
                if valid_rate < 0.7:
                # if True:
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

            # Send process
            # 예시로 refined_pose와 camera_pose를 임의 값으로 설정:
            #refined_pose = (0, 0, 0, 0, 0, 0, 1)  # pos(0,0,0), rot(0,0,0,1)
            #camera_pose = (1, 1, 1, 0, 0, 0, 1)   # pos(1,1,1), rot(0,0,0,1)

            # pred['opt_body2view_pose'][-1][0]가 Pose 객체라고 가정합니다.
            refined_pose_obj = pred['opt_body2view_pose'][-1][0]

            # Pose 객체 내부의 translation과 rotation 추출 (torch.Tensor)
            refined_t = refined_pose_obj.t.detach().cpu().clone()  # shape: (3,)
            refined_R = refined_pose_obj.R.detach().cpu().clone()  # shape: (3,3)

            # print("=== 원래 refined_pose_obj (Python 좌표계) ===")
            # print("Translation:", refined_t)
            # print("Rotation matrix:", refined_R)


            # refined_t[2] = -refined_t[2]
            # refined_R[:, 2] = -refined_R[:, 2]
            # refined_R[2, :] = -refined_R[2, :]

            # [1] y-축 부호 반전: (초기에는 Python에서 Unity로 변환할 때 반전했던 것을 역으로 적용)
            F = torch.diag(torch.tensor([1.0, -1.0, 1.0]))
            local_rot_inv = torch.from_numpy(R.from_euler('x', 90.0, degrees=True).as_matrix()).type(torch.float32) # wall, view
            # local_rot_inv = torch.from_numpy(R.from_euler('x', -90.0, degrees=True).as_matrix()).type(torch.float32) # four way, one way

            # [2] x축 +90도 회전: 초기에는 -90도를 적용했으므로, 역으로 +90도를 적용. <- wall / view
            refined_R = (F @ refined_R @ F) @ local_rot_inv
            refined_t = F @ refined_t
            # print("\n=== Unity 좌표계로 변환 후 refined_pose ===")
            # print("Transformed Translation:", refined_t)
            # print("Transformed Rotation matrix:", refined_R)

            # 최종 refined_pose: (tx, ty, tz, qx, qy, qz, qw) 7개의 float 값으로 구성
            t_np = refined_t.numpy().flatten()  # (3,)
            R_np = refined_R.numpy()  # (3,3)
            quat_np = R.from_matrix(R_np).as_quat()  # 결과: [qx, qy, qz, qw]

            refined_pose = t_np.tolist() + quat_np.tolist()
            # print("\n최종 refined_pose (Unity 전송용):", refined_pose)
            if (lost):
                lost_state = 1.0
            else:
                lost_state = 0.0
            # lost 여부도 같이 보내야함
            server.send_response_pose(refined_pose, camera_pose, lost_state)

            if lost:
                pred['opt_body2view_pose'][-1][0] = init_pose[None].cuda()
                pred['closest_template_views'] = original_closest_template_views[None].cuda()
                pred['closest_orientations_in_body'] = original_closest_orientations_in_body[None].cuda()

            if cfg.output_video and cfg.output_image and cfg.output_closest_template:
                pred['optimizing_result_imgs'] = []
                model.visualize_optimization(pred['opt_body2view_pose'][-1], pred)
                ori_image = original_img;
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
                if lost:
                    cv2.putText(ori_image, "LOST", (520, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 5, cv2.LINE_AA)
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

            init_pose = pred['opt_body2view_pose'][-1][0].cpu()
            index = get_closest_template_view_index(init_pose, orientations)
            closest_template_view = template_views[index*num_sample_contour_points:(index+1)*num_sample_contour_points, :]
            _, _, centers_in_image, centers_valid, normals_in_image, foreground_distance, background_distance, _ =\
                calculate_basic_line_data(closest_template_view[None], init_pose[None]._data, camera[None]._data, 1, 0)
            fore_hist, back_hist = \
                model.histogram.calculate_histogram(img[None], centers_in_image, centers_valid, normals_in_image,
                                                    foreground_distance, background_distance, True)
            total_fore_hist = (1 - fore_learn_rate) * total_fore_hist + fore_learn_rate * fore_hist
            total_back_hist = (1 - back_learn_rate) * total_back_hist + back_learn_rate * back_hist
            i = i + 1
    except KeyboardInterrupt:
        pass
    finally:
        video.release()
        server.stop()