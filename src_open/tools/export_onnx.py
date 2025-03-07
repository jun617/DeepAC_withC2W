import os
import torch
from pathlib import Path
import numpy as np
from omegaconf import DictConfig, OmegaConf
import copy
import warnings
from torch import nn

# 프로젝트 내부 모듈 (실제 경로에 맞게 수정 필요)
#from ..utils.geometry.wrappers import Pose
from ..models import get_model
from ..utils.lightening_utils import MyLightningLogger, convert_old_model, load_model_weight

def reparameterize_model(model: torch.nn.Module) -> nn.Module:
    """학습 시 사용된 다중 분기 구조를 추론용 단일 분기로 재구성합니다."""
    model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'reparameterize'):
            module.reparameterize()
    return model

class CropAndResizeImage(nn.Module):
    def __init__(self, resize, pad_size, crop_border):
        super(CropAndResizeImage, self).__init__()
        self.resize = resize
        self.pad_size = pad_size
        self.crop_border = crop_border

    def forward(self, image, bbox, camera_data_input):
        _, height, width, _ = image.shape
        x1, x2, y1, y2 = bbox
        x1 = (x1 - self.crop_border).clamp(min=0, max=width - 1)
        x2 = (x2 + self.crop_border).clamp(min=0, max=width - 1)
        y1 = (y1 - self.crop_border).clamp(min=0, max=height - 1)
        y2 = (y2 + self.crop_border).clamp(min=0, max=height - 1)
        img = image[:, int(y1):int(y2+1), int(x1):int(x2+1), :3]
        img = img.permute(0, 3, 1, 2)
        # 카메라 데이터 처리는 필요에 따라 추가 구현
        _, _, h, w = img.shape
        scale = self.resize / max(h, w)
        h_new, w_new = int(round(h * scale)), int(round(w * scale))
        img = torch.nn.functional.interpolate(img, size=(h_new, w_new), mode='bilinear')
        img_padded = torch.zeros((1, 3, self.pad_size, self.pad_size), dtype=torch.float)
        img_padded[:, :, :h_new, :w_new] = img
        img_padded = img_padded / 255
        return img_padded

def main(cfg):
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu_id

    logger = MyLightningLogger('DeepAC', cfg.save_dir)
    logger.dump_cfg(cfg, 'deploy_cfg.yml')
    assert 'load_cfg' in cfg
    assert Path(cfg.load_cfg).exists()
    train_cfg = OmegaConf.load(cfg.load_cfg)
    logger.dump_cfg(train_cfg, 'train_cfg.yml')

    # 모델 생성 및 가중치 로드
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
    model.eval().cpu()

    # 모델 구성 요소 분리
    histogram_model = model.histogram
    extractor_model = model.extractor
    contour_feature_model = model.contour_feature_map_extractor
    boundary_predictor_model = model.boundary_predictor
    derivative_calculator_model = model.derivative_calculator

    # 배포 입력 데이터 로드
    deploy_input = np.load('data/deploy_input.npz')
    image_input = torch.from_numpy(deploy_input['image'])
    pose_data_input = torch.from_numpy(deploy_input['pose'])
    camera_data_input = torch.from_numpy(deploy_input['camera'])
    template_views = torch.from_numpy(deploy_input['template_views'])
    template_view = template_views[:, 0]

    # contour_feature_model에 사용할 다중 스케일 입력 준비 (3 스케일)
    image_inputs = []
    camera_data_inputs = []
    for i in range(3):
        h, w = image_input.shape[2:]
        image_scale = 2 ** (2 - i)
        image_scaled = torch.nn.functional.interpolate(image_input, size=(h // int(image_scale), w // int(image_scale)))
        image_inputs.append(image_scaled)
        camera_data_inputs.append(camera_data_input / image_scale)
    pose_data_inputs = [pose_data_input, pose_data_input, pose_data_input]

    # 1. histogram_model ONNX 내보내기
    histogram_dummy_input = (image_input, pose_data_input, camera_data_input, template_view)
    jit_histogram_model = torch.jit.trace(histogram_model, histogram_dummy_input).eval()
    onnx_histogram_path = os.path.join(logger.log_dir, "histogram.onnx")
    torch.onnx.export(
        jit_histogram_model,
        histogram_dummy_input,
        onnx_histogram_path,
        input_names=["image", "pose_data", "camera_data", "template_view"],
        output_names=["fore_hist", "back_hist"],
        opset_version=11
    )
    logger.info("Exported histogram ONNX model to {}".format(onnx_histogram_path))

    # histogram 모델의 출력을 후속 모델의 입력으로 사용
    fore_hist, back_hist = jit_histogram_model(image_input, pose_data_input, camera_data_input, template_view)

    # 2. extractor_model ONNX 내보내기
    extractor_model = reparameterize_model(extractor_model)
    jit_extractor_model = torch.jit.trace(extractor_model, image_input).eval()
    onnx_extractor_path = os.path.join(logger.log_dir, "extractor.onnx")
    torch.onnx.export(
        jit_extractor_model,
        image_input,
        onnx_extractor_path,
        input_names=["image"],
        output_names=["feature0", "feature1", "feature2"],
        opset_version=11
    )
    logger.info("Exported extractor ONNX model to {}".format(onnx_extractor_path))
    feature_inputs = jit_extractor_model(image_input)
    if not isinstance(feature_inputs, (list, tuple)):
        feature_inputs = [feature_inputs]

    # 3. contour_feature_model ONNX 내보내기 (3 스케일 모두)
    jit_contour_feature_models = []
    for i in range(3):
        contour_dummy_input = (
            image_inputs[i],
            feature_inputs[i],
            pose_data_inputs[i],
            camera_data_inputs[i],
            template_view,
            fore_hist,
            back_hist
        )
        jit_model = torch.jit.trace(contour_feature_model, contour_dummy_input).eval()
        jit_contour_feature_models.append(jit_model)
        onnx_contour_path = os.path.join(logger.log_dir, f"contour_feature_extractor_{i}.onnx")
        torch.onnx.export(
            jit_model,
            contour_dummy_input,
            onnx_contour_path,
            input_names=["image", "feature", "pose_data", "camera_data", "template_view", "fore_hist", "back_hist"],
            output_names=[
                "normals_in_image", "centers_in_image", "centers_in_body",
                "lines_image_pf_segments", "lines_image_pb_segments", "valid_data_line",
                "lines_amplitude", "lines_slop", "lines_feature"
            ],
            opset_version=11
        )
        logger.info("Exported contour feature ONNX model (scale {}) to {}".format(i, onnx_contour_path))

    # 첫 번째 스케일의 contour feature 모델 출력을 후속 모델의 입력으로 사용
    outputs = jit_contour_feature_models[0](image_inputs[0], feature_inputs[0], pose_data_inputs[0],
                                             camera_data_inputs[0], template_view, fore_hist, back_hist)
    (normals_in_image, centers_in_image, centers_in_body,
     lines_image_pf_segments, lines_image_pb_segments, valid_data_line,
     lines_amplitude, lines_slop, lines_feature) = outputs

    # 4. boundary_predictor_model ONNX 내보내기
    boundary_dummy_input = (lines_feature, lines_image_pf_segments, lines_image_pb_segments, lines_slop, lines_amplitude)
    jit_boundary_predictor_model = torch.jit.trace(boundary_predictor_model, boundary_dummy_input).eval()
    onnx_boundary_path = os.path.join(logger.log_dir, "boundary_predictor.onnx")
    torch.onnx.export(
        jit_boundary_predictor_model,
        boundary_dummy_input,
        onnx_boundary_path,
        input_names=["lines_feature", "lines_image_pf_segments", "lines_image_pb_segments", "lines_slop", "lines_amplitude"],
        output_names=["distributions", "distribution_mean", "distribution_uncertainties"],
        opset_version=11
    )
    logger.info("Exported boundary predictor ONNX model to {}".format(onnx_boundary_path))
    distributions, distribution_mean, distribution_uncertainties = jit_boundary_predictor_model(*boundary_dummy_input)

    # 5. derivative_calculator_model ONNX 내보내기
    derivative_dummy_input = (
        normals_in_image, centers_in_image, centers_in_body,
        pose_data_input, camera_data_input, valid_data_line,
        distributions, distribution_mean, distribution_uncertainties
    )
    jit_derivative_calculator_model = torch.jit.trace(derivative_calculator_model, derivative_dummy_input).eval()
    onnx_derivative_path = os.path.join(logger.log_dir, "derivative_calculator.onnx")
    torch.onnx.export(
        jit_derivative_calculator_model,
        derivative_dummy_input,
        onnx_derivative_path,
        input_names=[
            "normals_in_image", "centers_in_image", "centers_in_body", "pose_data",
            "camera_data", "valid_data_line", "distributions", "distribution_mean", "distribution_uncertainties"
        ],
        output_names=["gradient", "hessian"],
        opset_version=11
    )
    logger.info("Exported derivative calculator ONNX model to {}".format(onnx_derivative_path))

if __name__ == "__main__":
    # 예시 설정 (실제 환경에 맞게 경로 등을 수정하세요)
    from omegaconf import OmegaConf
    dummy_cfg = OmegaConf.create({
        "gpu_id": "0",
        "save_dir": "./logs",
        "load_cfg": "./path/to/load_cfg.yaml",
        "load_model": "./path/to/model_checkpoint.pth"
    })
    main(dummy_cfg)
