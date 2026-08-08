#!/usr/bin/env python3
"""Run official GEM-X inference and export a small KineTrace interchange file.

This script is intentionally executed by GEM-X's isolated Python environment.
It must not import the KineTrace package or share its dependency set.
"""

from __future__ import annotations

import argparse
import base64
import functools
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-soma-npz", type=Path, required=True)
    parser.add_argument("--static-camera", action="store_true")
    parser.add_argument(
        "--sam3d-batch-size",
        type=int,
        default=int(os.getenv("KINETRACE_GEMX_SAM3D_BATCH_SIZE", "1")),
    )
    return parser.parse_args()


def report_progress(value: float, stage: str) -> None:
    print(f"KINETRACE_PROGRESS {value:.2f} {stage}", flush=True)


def encode_mesh(vertices, faces) -> dict[str, object]:
    """Quantize animated SOMA geometry into a compact browser-ready payload."""
    coordinates = vertices.numpy().astype(np.float32, copy=True)
    coordinates[..., 1:] *= -1.0
    minimum = coordinates.min(axis=(0, 1))
    maximum = coordinates.max(axis=(0, 1))
    offset = (minimum + maximum) * 0.5
    scale = np.maximum((maximum - minimum) / 65534.0, 1e-8)
    quantized = np.clip(
        np.rint((coordinates - offset) / scale),
        -32767,
        32767,
    ).astype("<i2")
    face_indices = faces.numpy()
    if face_indices.size == 0 or int(face_indices.max()) > 65535:
        raise RuntimeError("SOMA mesh cannot be represented with 16-bit face indices")
    triangles = face_indices.astype("<u2", copy=False)
    return {
        "name": "GEM-X SOMA",
        "encoding": "int16-le-base64",
        "frameCount": int(coordinates.shape[0]),
        "vertexCount": int(coordinates.shape[1]),
        "faceCount": int(triangles.shape[0]),
        "offset": offset.tolist(),
        "scale": scale.tolist(),
        "vertices": base64.b64encode(quantized.tobytes()).decode("ascii"),
        "faces": base64.b64encode(triangles.tobytes()).decode("ascii"),
    }


def export_soma_npz(
    torch,
    body_params,
    soma,
    fps: float,
    destination: Path,
    root_translation_offset=None,
) -> None:
    """Preserve GEM-X's native SOMA rotations for MetaHuman retargeting."""
    from soma import save_soma_npz

    global_orient = body_params["global_orient"]
    has_sequence_batch = global_orient.ndim == 3

    def sequence(name):
        value = body_params[name]
        if has_sequence_batch:
            if value.shape[0] != 1:
                raise RuntimeError(f"Cannot export batched SOMA parameter '{name}'")
            value = value[0]
        return value.detach().float()

    global_orient = sequence("global_orient")
    body_pose = sequence("body_pose")
    transl = sequence("transl")
    identity_coeffs = sequence("identity_coeffs")
    scale_params = sequence("scale_params")

    frame_count = global_orient.shape[0]
    if global_orient.shape != (frame_count, 3):
        raise RuntimeError(f"Unexpected SOMA global orientation shape {global_orient.shape}")
    if body_pose.numel() != frame_count * 76 * 3:
        raise RuntimeError(f"Unexpected SOMA body pose shape {body_pose.shape}")
    if transl.shape != (frame_count, 3):
        raise RuntimeError(f"Unexpected SOMA translation shape {transl.shape}")
    if root_translation_offset is not None:
        root_translation_offset = root_translation_offset.detach().float()
        while root_translation_offset.ndim > 2 and root_translation_offset.shape[0] == 1:
            root_translation_offset = root_translation_offset[0]
        if root_translation_offset.shape != transl.shape:
            raise RuntimeError(
                "SOMA ground offset does not match translation shape "
                f"{root_translation_offset.shape} != {transl.shape}"
            )
        transl = transl + root_translation_offset.to(transl.device)

    # SOMA's interchange format starts with a virtual Root. GEM-X predicts the
    # 77 deform joints beginning at Hips, so prepend an identity Root and let
    # save_soma_npz omit it in the standard no-Root representation.
    body_pose = body_pose.reshape(frame_count, 76, 3)
    poses = torch.cat([global_orient[:, None], body_pose], dim=1)
    poses = torch.cat([torch.zeros_like(poses[:, :1]), poses], dim=1)

    underlying_soma = soma.soma
    joint_names = list(underlying_soma.rig_data["joint_names"])
    if len(joint_names) != 78:
        raise RuntimeError(f"Unexpected SOMA rig joint count {len(joint_names)}")

    # GEM-X stores a uniform global scale in column zero followed by MHR's 68
    # body-part scales. The canonical SOMA identity field expects only the 68
    # MHR values, so retain global scale as explicit metadata.
    averaged_scale = scale_params.mean(dim=0, keepdim=True)
    save_soma_npz(
        destination,
        poses,
        transl,
        joint_names=joint_names,
        identity_model_type=underlying_soma.identity_model_type,
        identity_coeffs=identity_coeffs.mean(dim=0, keepdim=True),
        scale_params=averaged_scale[:, 1:],
        joint_orient=underlying_soma._t_pose_orient,
        unit="meters",
        keep_root=False,
        extra_arrays={
            "fps": np.array(fps, dtype=np.float32),
            "global_scale": averaged_scale[:, :1].detach().cpu().numpy(),
            "source": np.array("GEM-X"),
        },
    )


def _rotation_between(torch, source, target):
    source = source / torch.linalg.norm(source, dim=-1, keepdim=True).clamp_min(1e-8)
    target = target / torch.linalg.norm(target, dim=-1, keepdim=True).clamp_min(1e-8)
    cross = torch.cross(source, target, dim=-1)
    cosine = (source * target).sum(dim=-1, keepdim=True)
    sine_squared = (cross * cross).sum(dim=-1, keepdim=True)
    skew = torch.zeros((*source.shape[:-1], 3, 3), device=source.device)
    skew[..., 0, 1] = -cross[..., 2]
    skew[..., 0, 2] = cross[..., 1]
    skew[..., 1, 0] = cross[..., 2]
    skew[..., 1, 2] = -cross[..., 0]
    skew[..., 2, 0] = -cross[..., 1]
    skew[..., 2, 1] = cross[..., 0]
    identity = torch.eye(3, device=source.device).expand_as(skew)
    return identity + skew + (skew @ skew) * (
        (1.0 - cosine) / sine_squared.clamp_min(1e-8)
    )[..., None]


def _straighten_contact_fingers(
    torch,
    vertices,
    joints,
    skinning_weights,
    wrist_id,
    base_ids,
    chain_specs,
    strength,
):
    """Place contacted fingers in a stable, anatomically spaced palm-local fan."""
    source_vertices = vertices
    vertex_displacement = torch.zeros_like(vertices)
    palm_forward = joints[:, list(base_ids)].mean(dim=1) - joints[:, wrist_id]
    palm_forward[:, 1] = 0.0
    palm_forward = palm_forward / torch.linalg.norm(
        palm_forward, dim=-1, keepdim=True
    ).clamp_min(1e-8)
    palm_lateral = joints[:, base_ids[0]] - joints[:, base_ids[-1]]
    palm_lateral[:, 1] = 0.0
    palm_lateral -= palm_forward * (palm_lateral * palm_forward).sum(dim=-1, keepdim=True)
    palm_lateral = palm_lateral / torch.linalg.norm(
        palm_lateral, dim=-1, keepdim=True
    ).clamp_min(1e-8)

    for chain, angles_degrees in chain_specs:
        joint_ids = list(chain)
        chain_joints = joints[:, joint_ids]
        segment_vectors = chain_joints[:, 1:] - chain_joints[:, :-1]
        segment_lengths = torch.linalg.norm(segment_vectors, dim=-1)
        angles = torch.deg2rad(torch.tensor(angles_degrees, device=vertices.device))
        directions = (
            palm_forward[:, None] * torch.cos(angles)[None, :, None]
            + palm_lateral[:, None] * torch.sin(angles)[None, :, None]
        )
        offsets = directions * segment_lengths[..., None]
        natural_joints = chain_joints[:, :1] + torch.cat(
            [
                torch.zeros((vertices.shape[0], 1, 3), device=vertices.device),
                torch.cumsum(offsets, dim=1),
            ],
            dim=1,
        )
        target_joints = torch.lerp(chain_joints, natural_joints, strength[:, None, None])

        for bone_index, joint_id in enumerate(joint_ids[:-1]):
            rotation = _rotation_between(
                torch,
                segment_vectors[:, bone_index],
                target_joints[:, bone_index + 1] - target_joints[:, bone_index],
            )
            transformed = target_joints[:, bone_index, None] + torch.einsum(
                "tij,tvj->tvi",
                rotation,
                source_vertices - chain_joints[:, bone_index, None],
            )
            bone_weight = skinning_weights[:, joint_id + 1]
            vertex_displacement += bone_weight[None, :, None] * (transformed - source_vertices)
        joints[:, joint_ids] = target_joints

    return source_vertices + vertex_displacement, joints


def correct_contact_hands(torch, soma, vertices, joints, prediction):
    """Plant high-confidence contact palms without touching airborne hands."""
    net_outputs = prediction.get("net_outputs")
    logits = net_outputs.get("static_conf_logits") if isinstance(net_outputs, dict) else None
    if logits is None:
        ground = vertices[..., 1].amin()
        vertices = vertices.clone()
        joints = joints.clone()
        vertices[..., 1] -= ground
        joints[..., 1] -= ground
        return vertices, joints

    if vertices.ndim == 4 and vertices.shape[0] == 1:
        vertices = vertices[0]
        joints = joints[0]
    if logits.ndim == 3 and logits.shape[0] == 1:
        logits = logits[0]
    logits = logits.to(device=vertices.device)
    if logits.shape[0] != vertices.shape[0] or logits.shape[-1] < 6:
        return vertices, joints

    # GEM-X grounds its global skeleton. Normalize with the actual mesh surface
    # as the official global renderer does, then plant contact hands on y=0.
    ground = vertices[..., 1].amin()
    vertices = vertices.clone()
    joints = joints.clone()
    vertices[..., 1] -= ground
    joints[..., 1] -= ground

    from gem.utils.rotation_conversions import (
        axis_angle_to_matrix,
        matrix_to_axis_angle,
    )

    skinning_weights = soma.soma.skinning_weights.to(vertices.device)
    hand_specs = (
        # SOMA indices omit its dummy Root; skinning-weight indices include it.
        (
            14,
            (19, 24, 29, 34),
            (tuple(range(19, 24)), tuple(range(24, 29)), tuple(range(29, 34)), tuple(range(34, 39))),
            (
                (tuple(range(15, 19)), (42.0, 35.0, 28.0)),
                (tuple(range(19, 24)), (7.0, 6.0, 5.0, 4.0)),
                (tuple(range(24, 29)), (1.0, 0.5, 0.0, 0.0)),
                (tuple(range(29, 34)), (-3.0, -2.0, -1.0, 0.0)),
                (tuple(range(34, 39)), (-9.0, -7.0, -5.0, -3.0)),
            ),
            tuple(range(14, 39)),
            tuple(range(15, 40)),
            4,
        ),
        (
            42,
            (47, 52, 57, 62),
            (tuple(range(47, 52)), tuple(range(52, 57)), tuple(range(57, 62)), tuple(range(62, 67))),
            (
                (tuple(range(43, 47)), (42.0, 35.0, 28.0)),
                (tuple(range(47, 52)), (7.0, 6.0, 5.0, 4.0)),
                (tuple(range(52, 57)), (1.0, 0.5, 0.0, 0.0)),
                (tuple(range(57, 62)), (-3.0, -2.0, -1.0, 0.0)),
                (tuple(range(62, 67)), (-9.0, -7.0, -5.0, -3.0)),
            ),
            tuple(range(42, 67)),
            tuple(range(43, 68)),
            5,
        ),
    )
    for (
        wrist_id,
        base_ids,
        finger_chains,
        flatten_chains,
        joint_ids,
        weight_ids,
        contact_channel,
    ) in hand_specs:
        contact_strength = ((logits[:, contact_channel].sigmoid() - 0.45) / 0.1).clamp(
            0.0, 1.0
        )
        wrist_height_strength = ((0.28 - joints[:, wrist_id, 1]) / 0.06).clamp(0.0, 1.0)
        extension_ratios = []
        for chain in finger_chains:
            chain_joints = joints[:, list(chain)]
            direct = torch.linalg.norm(chain_joints[:, -1] - chain_joints[:, 0], dim=-1)
            length = torch.linalg.norm(chain_joints[:, 1:] - chain_joints[:, :-1], dim=-1).sum(
                dim=-1
            )
            extension_ratios.append(direct / length.clamp_min(1e-8))
        finger_extension = torch.stack(extension_ratios, dim=-1).mean(dim=-1)
        open_hand_strength = ((finger_extension - 0.8) / 0.04).clamp(0.0, 1.0)
        # Static wrists can also mean gripping equipment. Only ground low,
        # extended hands; leave raised or curled hands entirely untouched.
        strength = contact_strength * wrist_height_strength * open_hand_strength
        if float(strength.max()) <= 0.0:
            continue

        wrist = joints[:, wrist_id]
        forward = joints[:, list(base_ids)].mean(dim=1) - wrist
        lateral = joints[:, base_ids[0]] - joints[:, base_ids[-1]]
        normal = torch.cross(forward, lateral, dim=-1)
        target_normal = torch.zeros_like(normal)
        target_normal[:, 1] = torch.where(normal[:, 1] >= 0.0, 1.0, -1.0)
        full_rotation = _rotation_between(torch, normal, target_normal)
        rotation_vector = matrix_to_axis_angle(full_rotation) * strength[:, None]
        rotation = axis_angle_to_matrix(rotation_vector)

        centered_vertices = vertices - wrist[:, None]
        transformed_vertices = wrist[:, None] + torch.einsum(
            "tij,tvj->tvi", rotation, centered_vertices
        )
        hand_weight = skinning_weights[:, list(weight_ids)].sum(dim=-1).clamp(0.0, 1.0)
        core_vertices = hand_weight > 0.5
        if not bool(core_vertices.any()):
            continue
        vertical_shift = -transformed_vertices[:, core_vertices, 1].amin(dim=1)
        transformed_vertices[..., 1] += vertical_shift[:, None]
        vertex_blend = strength[:, None, None] * hand_weight[None, :, None]
        vertices = torch.lerp(vertices, transformed_vertices, vertex_blend)

        selected_joints = joints[:, list(joint_ids)]
        transformed_joints = wrist[:, None] + torch.einsum(
            "tij,tkj->tki", rotation, selected_joints - wrist[:, None]
        )
        transformed_joints[..., 1] += vertical_shift[:, None]
        joints[:, list(joint_ids)] = torch.lerp(
            selected_joints,
            transformed_joints,
            strength[:, None, None],
        )
        vertices, joints = _straighten_contact_fingers(
            torch,
            vertices,
            joints,
            skinning_weights,
            wrist_id,
            base_ids,
            flatten_chains,
            strength,
        )

    final_ground = vertices[..., 1].amin()
    vertices[..., 1] -= final_ground
    joints[..., 1] -= final_ground
    return vertices, joints


def main() -> None:
    args = parse_args()
    gemx_root = Path.cwd().resolve()
    sys.path.insert(0, str(gemx_root))

    import cv2
    import hydra
    import torch

    # GEM-X checkpoints contain numpy objects; torch 2.6+ otherwise rejects them.
    torch.load = functools.partial(torch.load, weights_only=False)

    from gem.utils.net_utils import detach_to_cpu, to_cuda
    from gem.utils.sam3db_extractor import SAM3DBExtractor
    from gem.utils.soma_utils.soma_layer import SomaLayer
    from gem.utils.vitpose_extractor import VitPoseExtractor
    from scripts.demo.demo_soma import (
        _build_cfg,
        load_data_dict,
        resolve_ckpt_path,
        run_preprocess,
    )

    # Upstream batches 16 DINO frames at once, which exceeds the 8 GB GPUs
    # KineTrace targets. Override only the default used by demo_soma; users can
    # raise it through KINETRACE_GEMX_SAM3D_BATCH_SIZE on larger cards.
    original_extract_video_features = SAM3DBExtractor.extract_video_features
    original_extract_keypoints = VitPoseExtractor.extract

    def extract_keypoints_with_progress(self, *extract_args, **extract_kwargs):
        report_progress(0.22, "Detecting 77 body and hand keypoints")
        result = original_extract_keypoints(self, *extract_args, **extract_kwargs)
        report_progress(0.38, "2D body and hand keypoints ready")
        return result

    def extract_video_features_memory_safe(self, *extract_args, **extract_kwargs):
        report_progress(0.42, "Extracting SAM-3D image features")
        extract_kwargs.setdefault("batch_size", max(args.sam3d_batch_size, 1))
        result = original_extract_video_features(self, *extract_args, **extract_kwargs)
        report_progress(0.68, "SAM-3D image features ready")
        return result

    VitPoseExtractor.extract = extract_keypoints_with_progress
    SAM3DBExtractor.extract_video_features = extract_video_features_memory_safe

    demo_args = SimpleNamespace(
        video=str(args.video.resolve()),
        output_root=str(args.output_root.resolve()),
        static_cam=args.static_camera,
        verbose=False,
        render_mhr=False,
        sam3d_ckpt_path=None,
        sam3d_mhr_path=None,
        ckpt=None,
        exp="gem_soma_regression",
        retarget=False,
    )
    cfg = _build_cfg(demo_args)
    report_progress(0.08, "Detecting and tracking the actor")
    run_preprocess(cfg)
    report_progress(0.70, "Preparing global motion reconstruction")
    data = load_data_dict(cfg)

    result_path = Path(cfg.paths.hpe_results)
    if result_path.exists():
        report_progress(0.86, "Loading cached GEM-X motion")
        prediction = torch.load(result_path)
    else:
        report_progress(0.74, "Running GEM-X global motion model")
        model = hydra.utils.instantiate(cfg.model, _recursive_=False)
        model.load_pretrained_model(resolve_ckpt_path(cfg))
        model = model.eval().cuda()
        prediction = model.predict(data, static_cam=cfg.static_cam, postproc=True)
        torch.save(detach_to_cpu(prediction), result_path)
        del model
        torch.cuda.empty_cache()
    report_progress(0.88, "Building the reconstructed SOMA body")

    body_params = prediction.get("body_params_global") or prediction.get(
        "pred_body_params_global"
    )
    if body_params is None:
        raise RuntimeError("GEM-X prediction has no global SOMA body parameters")

    soma = SomaLayer(
        data_root="inputs/soma_assets",
        low_lod=False,
        device="cuda:0",
        identity_model_type="mhr",
        mode="warp",
    )
    with torch.no_grad():
        soma_output = soma(**to_cuda(body_params))
        report_progress(0.90, "Stabilizing detected palms and fingers")
        vertices, joints = correct_contact_hands(
            torch,
            soma,
            soma_output["vertices"],
            soma_output["joints"],
            prediction,
        )
        root_translation_offset = (
            joints[..., 0, :] - soma_output["joints"][..., 0, :]
        )
        joints = joints.detach().float().cpu()
        vertices = vertices.detach().float().cpu()
        faces = soma.faces.detach().long().cpu()
    if joints.ndim == 4 and joints.shape[0] == 1:
        joints = joints[0]
    if vertices.ndim == 4 and vertices.shape[0] == 1:
        vertices = vertices[0]

    keypoints = torch.load(cfg.paths.vitpose)
    if isinstance(keypoints, tuple):
        keypoints = keypoints[0]
    keypoints = torch.as_tensor(keypoints).detach().float().cpu()

    capture = cv2.VideoCapture(str(cfg.video_path))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if fps <= 1.0 or fps > 240.0:
        fps = 30.0

    report_progress(0.91, "Exporting native SOMA motion for MetaHuman")
    export_soma_npz(
        torch,
        body_params,
        soma,
        fps,
        args.output_soma_npz,
        root_translation_offset=root_translation_offset,
    )

    output = {
        "fps": fps,
        "width": width,
        "height": height,
        "joints": joints.tolist(),
        "keypoints2d": keypoints.tolist(),
        "mesh": encode_mesh(vertices, faces),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    report_progress(0.92, "Serializing reconstructed motion")
    args.output_json.write_text(json.dumps(output, separators=(",", ":")), encoding="utf-8")
    report_progress(0.93, "Reconstruction ready for KineTrace")


if __name__ == "__main__":
    main()
