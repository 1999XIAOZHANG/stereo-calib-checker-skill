# -*- coding: utf-8 -*-
"""阶段一：采集充分性质检（标定前）。

把张正友标定法原文的可执行化检查：
- 每张图提供 2 个内参约束，理论下限 3 张，实践建议 >=10 张
- 姿态禁区：板面始终平行于像平面（纯平移）/ 仅绕光轴旋转 —— 均导致退化
- 角点应覆盖画面大部分区域（畸变估计依赖边缘观测）
"""
import math

import cv2
import numpy as np

from common import (MASK_H, MASK_W, STATUS_FAIL, STATUS_INFO, STATUS_PASS,
                    STATUS_WARN, imread_any, make_check)

MIN_PAIRS_THEORY = 3    # 张正友标定理论下限
MIN_PAIRS_PRACTICE = 10 # 实践建议下限


def detect_corners(gray, bw, bh):
    """角点检测：优先 SB（亚像素级），失败后回退经典算法 + cornerSubPix。"""
    flags = (cv2.CALIB_CB_NORMALIZE_IMAGE
             | cv2.CALIB_CB_EXHAUSTIVE
             | cv2.CALIB_CB_ACCURACY)
    ok, pts = cv2.findChessboardCornersSB(gray, (bw, bh), flags)
    if ok:
        return True, pts, "SB"
    ok, pts = cv2.findChessboardCorners(
        gray, (bw, bh),
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
    if ok:
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
        pts = cv2.cornerSubPix(gray, pts, (5, 5), (-1, -1), crit)
        return True, pts, "classic"
    return False, None, None


def hull_mask(pts, size):
    """角点凸包栅格化为小掩膜，返回 (掩膜, 单图覆盖率)。"""
    w, h = size
    p = pts.reshape(-1, 2).astype(np.float32)
    scale = np.array([MASK_W / w, MASK_H / h], dtype=np.float32)
    hull = cv2.convexHull(p * scale).astype(np.int32)
    mask = np.zeros((MASK_H, MASK_W), np.uint8)
    cv2.fillPoly(mask, [hull], 1)
    return mask, float(mask.mean())


def build_objp(bw, bh, square):
    """世界坐标：棋盘内角点，单位为方格边长。

    注意与 findChessboardCorners 的角点顺序对齐：行主序、列方向（bw）为快轴，
    因此生成 (x, y) 需用 mgrid[0:bw, 0:bh].T —— 顺序写反会导致网格"转置"，
    标定直接崩溃。
    """
    objp = np.zeros((bw * bh, 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:bw, 0:bh].T.reshape(-1, 2) * square
    return objp


def pose_metrics(objpoints, imgpoints, size):
    """粗标定后从 rvecs 分解每张图的姿态：倾斜角、面内旋转角、板面法向量。"""
    _, _, _, rvecs, _ = cv2.calibrateCamera(objpoints, imgpoints, size, None, None)
    tilts, inplanes, normals = [], [], []
    for rv in rvecs:
        r, _ = cv2.Rodrigues(rv)
        # 倾斜角：板面法向量与光轴夹角（0° = 板面正对相机）
        tilts.append(math.degrees(math.acos(np.clip(r[2, 2], -1.0, 1.0))))
        # 面内旋转：板面 x 轴在像平面内的滚转角
        inplanes.append(math.degrees(math.atan2(r[1, 0], r[0, 0])))
        normals.append(r[:, 2])
    # 两两板面法向量夹角：衡量姿态多样性
    pair_angles = []
    for i in range(len(normals)):
        for j in range(i + 1, len(normals)):
            cosang = np.clip(float(np.dot(normals[i], normals[j])), -1.0, 1.0)
            pair_angles.append(math.degrees(math.acos(cosang)))
    return tilts, inplanes, pair_angles


def run_stage1(ctx, checks):
    pairs = ctx["pairs"]
    bw, bh = ctx["board"]["w"], ctx["board"]["h"]

    valid, failed = [], []
    union = np.zeros((MASK_H, MASK_W), np.uint8)
    size = None

    for p in pairs:
        gl = imread_any(p["left"])
        gr = imread_any(p["right"])
        if gl is None or gr is None:
            failed.append({"id": p["id"], "reason": "图像读取失败"})
            continue
        if size is None:
            size = (gl.shape[1], gl.shape[0])
        okl, pl, ml = detect_corners(gl, bw, bh)
        okr, pr, _ = detect_corners(gr, bw, bh)
        if not okl or not okr:
            why = "左目角点检测失败" if not okl else "右目角点检测失败"
            failed.append({"id": p["id"], "reason": why})
            continue
        sharp = float(cv2.Laplacian(gl, cv2.CV_64F).var())
        bright = float(gl.mean())
        clipped = float(((gl < 5) | (gl > 250)).mean())
        mask, cov = hull_mask(pl, size)
        union |= mask
        valid.append({
            "id": p["id"], "pts_left": pl, "pts_right": pr, "method": ml,
            "left_path": str(p["left"]), "right_path": str(p["right"]),
            "sharpness": sharp, "brightness": bright,
            "clipped": clipped, "coverage": cov,
        })

    ctx["valid"], ctx["failed"], ctx["size"] = valid, failed, size
    n, total = len(valid), len(pairs)

    # ---- 检查 1：有效图像对数量 ----
    if n < MIN_PAIRS_THEORY:
        st, adv = STATUS_FAIL, (
            f"张正友标定理论下限为 {MIN_PAIRS_THEORY} 张（每张提供 2 个内参约束），"
            "当前数量无法求解，请补拍")
    elif n < MIN_PAIRS_PRACTICE:
        st, adv = STATUS_WARN, "建议积累 10~20 张姿态多样的图像，参数更稳定"
    else:
        st, adv = STATUS_PASS, "数量充分"
    checks.append(make_check(1, "num_pairs", "有效图像对数量", st,
                             f"{n} / {total} 对", "<3 FAIL；3~9 WARN；>=10 PASS", adv))

    # ---- 检查 2：角点检测失败清单 ----
    if failed:
        frac = len(failed) / max(total, 1)
        st = STATUS_WARN if frac > 0.4 else STATUS_INFO
        detail = "；".join(f"{f['id']}({f['reason']})" for f in failed)
        checks.append(make_check(1, "detect_fail", "角点检测失败图像", st,
                                 detail, "失败比例 >40% WARN",
                                 "失败图已被自动剔除；若比例偏高请检查棋盘规格/对焦/曝光"))
    else:
        checks.append(make_check(1, "detect_fail", "角点检测失败图像", STATUS_PASS,
                                 "无", ">40% WARN", "全部图像检测成功"))

    # ---- 姿态多样性（张正友原文退化判据） ----
    tilts, inplanes, pair_angles = [], [], []
    if n >= MIN_PAIRS_THEORY:
        objpoints = [build_objp(bw, bh, ctx["square"])] * n
        imgpoints = [v["pts_left"] for v in valid]
        tilts, inplanes, pair_angles = pose_metrics(objpoints, imgpoints, size)
        ctx["pose"] = {"tilts": tilts, "inplanes": inplanes,
                       "pair_angles": pair_angles}
        max_tilt = max(tilts)
        max_pair = max(pair_angles) if pair_angles else 0.0
        std_inplane = float(np.std(inplanes)) if n > 1 else 0.0

        # 检查 3：最大倾斜角（是否全部正对）
        if max_tilt < 10:
            st, adv = STATUS_FAIL, (
                "全部板面近乎正对像平面 —— 张正友标定的典型退化配置，"
                "请补拍多角度倾斜照片（推荐倾斜 15°~45°，朝不同方向）")
        elif max_tilt < 15:
            st, adv = STATUS_WARN, "倾斜姿态偏少，建议补充更大倾角的图像"
        else:
            st, adv = STATUS_PASS, "存在足够的倾斜观测"
        checks.append(make_check(1, "max_tilt", "板面最大倾斜角", st,
                                 f"{max_tilt:.1f}°", "<10° FAIL；10~15° WARN", adv))

        # 检查 4：板面朝向多样性
        if max_pair < 10:
            st, adv = STATUS_FAIL, "所有板面朝向几乎相同（疑似仅平移拍摄），退化风险高"
        elif max_pair < 20:
            st, adv = STATUS_WARN, "板面朝向变化偏小，建议多方向倾斜拍摄"
        else:
            st, adv = STATUS_PASS, "姿态多样性良好"
        checks.append(make_check(1, "pose_spread", "板面朝向最大夹角", st,
                                 f"{max_pair:.1f}°", "<10° FAIL；10~20° WARN", adv))

        # 检查 5：面内旋转多样性（软性建议，圆周标准差避免 ±180° 绕回虚高）
        if n > 1:
            ang = np.radians(inplanes)
            r_bar = abs(np.mean(np.exp(1j * ang)))
            std_inplane = float(np.degrees(np.sqrt(-2 * np.log(max(r_bar, 1e-9)))))
        else:
            std_inplane = 0.0
        if std_inplane < 5:
            st, adv = STATUS_INFO, "面内旋转几乎不变，建议适当旋转标定板（绕板法线）"
        else:
            st, adv = STATUS_PASS, "面内旋转分布合理"
        checks.append(make_check(1, "inplane", "面内旋转角标准差", st,
                                 f"{std_inplane:.1f}°", "<5° INFO（软性建议）", adv))
    else:
        checks.append(make_check(1, "max_tilt", "板面最大倾斜角", STATUS_INFO,
                                 "未评估", "有效图像 <3 张", "有效图像不足，跳过姿态分析"))

    # ---- 检查 6：角点覆盖并集 ----
    cov_union = float(union.mean())
    if cov_union < 0.5:
        st, adv = STATUS_WARN, "角点集中在画面局部，边缘畸变观测不足；建议让棋盘出现在画面各区域（含四角）"
    else:
        st, adv = STATUS_PASS, "覆盖良好"
    checks.append(make_check(1, "coverage", "角点画面覆盖率（并集）", st,
                             f"{cov_union * 100:.0f}%", ">=50% PASS（经验值）", adv))

    # ---- 检查 7：清晰度 ----
    min_sharp = min(v["sharpness"] for v in valid) if valid else 0.0
    if min_sharp < 100:
        st, adv = STATUS_WARN, "存在疑似模糊图像（Laplacian 方差偏低），角点定位精度可能受影响"
    else:
        st, adv = STATUS_PASS, "清晰度正常"
    checks.append(make_check(1, "sharpness", "最小清晰度（Laplacian 方差）", st,
                             f"{min_sharp:.0f}", ">=100 PASS（经验值，与分辨率相关）", adv))

    # ---- 检查 8：曝光 ----
    bad_exp = [v["id"] for v in valid if v["brightness"] < 40 or v["brightness"] > 220]
    bad_clip = [v["id"] for v in valid if v["clipped"] > 0.01]
    if bad_exp or bad_clip:
        st, adv = STATUS_WARN, "存在过暗/过曝或大面积截断的图像，请调整曝光后重拍"
    else:
        st, adv = STATUS_PASS, "曝光正常"
    bmin = min((v["brightness"] for v in valid), default=0)
    bmax = max((v["brightness"] for v in valid), default=0)
    checks.append(make_check(1, "exposure", "曝光（灰度均值范围）", st,
                             f"{bmin:.0f}~{bmax:.0f}", "40~220 且截断像素 <1%", adv))

    if n < MIN_PAIRS_THEORY:
        ctx["abort"] = True
