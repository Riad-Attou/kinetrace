import base64
import struct

from kinetrace.gemx import _parse_progress_line, adapt_gemx_result


def test_gemx_soma_motion_maps_to_canonical_body_and_hands() -> None:
    joints = [[float(index), float(index + 1), float(index + 2)] for index in range(77)]
    keypoints = [[float(index * 2), float(index * 3), 0.8] for index in range(77)]
    mesh = {
        "name": "GEM-X SOMA",
        "encoding": "int16-le-base64",
        "frameCount": 2,
        "vertexCount": 3,
        "faceCount": 1,
        "offset": [0.0, 0.0, 0.0],
        "scale": [0.001, 0.001, 0.001],
        "vertices": base64.b64encode(struct.pack("<18h", *range(18))).decode(),
        "faces": base64.b64encode(struct.pack("<3H", 0, 1, 2)).decode(),
    }

    result = adapt_gemx_result(
        {
            "fps": 25.0,
            "width": 200,
            "height": 300,
            "joints": [joints, joints],
            "keypoints2d": [keypoints, keypoints],
            "mesh": mesh,
        },
        "capture.mp4",
    )

    assert result["metadata"]["engine"] == "gemx"
    assert result["metadata"]["frameCount"] == 2
    assert result["metadata"]["durationMs"] == 40
    frame = result["frames"][0]
    assert len(frame["body"]) == 33
    assert len(frame["hands"]["left"]) == 21
    assert len(frame["hands"]["right"]) == 21

    left_wrist = frame["body"][15]
    assert left_wrist["name"] == "left_wrist"
    assert left_wrist["world"] == {"x": 14.0, "y": -15.0, "z": -16.0}
    assert left_wrist["x"] == 0.14
    assert left_wrist["y"] == 0.14
    assert frame["hands"]["left"][0]["world"] == left_wrist["world"]
    assert result["mesh"] == mesh
    assert "meshVertices" not in frame
    assert result["quality"]["poseCoverage"] == 1.0
    assert result["quality"]["leftHandCoverage"] == 1.0


def test_gemx_adapter_rejects_wrong_joint_count() -> None:
    payload = {
        "joints": [[[0.0, 0.0, 0.0]]],
        "keypoints2d": [[[0.0, 0.0, 1.0]]],
    }

    try:
        adapt_gemx_result(payload, "bad.mp4")
    except ValueError as error:
        assert "unexpected SOMA joint count" in str(error)
    else:
        raise AssertionError("Expected malformed SOMA data to be rejected")


def test_gemx_progress_marker_is_parsed_and_bounded() -> None:
    assert _parse_progress_line("KINETRACE_PROGRESS 0.42 Extracting SAM-3D image features") == (
        0.42,
        "Extracting SAM-3D image features",
    )
    assert _parse_progress_line("noise KINETRACE_PROGRESS 1.5 Finished") == (0.93, "Finished")
    assert _parse_progress_line("ordinary model output") is None
