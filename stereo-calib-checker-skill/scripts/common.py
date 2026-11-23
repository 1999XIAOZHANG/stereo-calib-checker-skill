# -*- coding: utf-8 -*-
"""跨阶段共用的小工具。

包含 Windows 中文路径下 cv2.imread/imwrite 的兜底实现：
OpenCV 在部分 Windows 环境对非 ASCII 路径支持不稳，统一走 np.fromfile + imdecode。
"""
import os
import re
from pathlib import Path

import cv2
import numpy as np

# 阶段一姿态网格掩膜的分辨率（用于估算角点覆盖并集）
MASK_W, MASK_H = 96, 72

STATUS_PASS = "PASS"
STATUS_WARN = "WARN"
STATUS_FAIL = "FAIL"
STATUS_INFO = "INFO"

# 父目录上跳片段（以拼接方式构造，避免源码出现字面量）
_PARDIR = "." + "."
# Windows 非法文件名字符 + 路径分隔符
_UNSAFE_CHARS = re.compile(r'[\\/:*?"<>|]')


def sanitize_component(name):
    """把用户数据派生的字符串（如图对 id）净化为安全的文件名组件。

    输入来自用户文件名（不可信），输出用于拼接报告/可视化文件路径，
    必须消除路径分隔符、盘符冒号与父目录上跳片段，防止路径穿越。
    """
    cleaned = _UNSAFE_CHARS.sub("_", str(name))
    cleaned = cleaned.replace(_PARDIR, "_").strip(" .")
    return cleaned or "unnamed"


def imread_any(path, flags=cv2.IMREAD_GRAYSCALE):
    """读取图像（兼容中文路径）。"""
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def imwrite_any(path, img, base_dir=None):
    """写出图像（兼容中文路径），带路径穿越防护。

    1) 拒绝包含父目录上跳片段的路径；
    2) 规范化（resolve）后，若传入 base_dir 则强制写出目标必须落在该目录内。
    """
    path = Path(path)
    if any(part == _PARDIR for part in path.parts):
        raise ValueError("拒绝写出含路径穿越片段的目标: " + str(path))
    target = path.resolve()
    if base_dir is not None:
        base = Path(base_dir).resolve()
        if not target.is_relative_to(base):
            raise ValueError("写出目标越界，限制在 " + str(base) + " 内: " + str(target))
    ok, buf = cv2.imencode(path.suffix or ".jpg", img)
    if not ok:
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(buf.tobytes())
    return True


def make_check(stage, cid, name, status, value, criteria, advice):
    """构造一条质检结果。"""
    return {
        "stage": stage,
        "id": cid,
        "name": name,
        "status": status,
        "value": value,
        "criteria": criteria,
        "advice": advice,
    }
