# -*- coding: utf-8 -*-
"""阶段二：标定参数合理性体检。

单目标定（左右各一次）+ 双目外参标定，检查：
- 重投影误差（总体 + 逐图，定位应剔除的坏图）
- 内参合理性（fx≈fy、主点居中、畸变系数量级）
- 双目外参合理性（基线长度、R 接近单位阵——并排刚性假设）
"""
import math

import cv2
import numpy as np

from common import STATUS_FAIL, STATUS_INFO, STATUS_PASS, STATUS_WARN, make_check
from stage1_acquire import build_objp

# 重投影误差阈值（px，棋盘格标定的经验值）
RMS_GOOD, RMS_OK = 0.5, 1.0


def per_view_errors(objpoints, rvecs, tvecs, k, dist, imgpoints):
    """逐图平均重投影误差（px）。"""
    errs = []
    for obj, rv, tv, obs in zip(objpoints, rvecs, tvecs, imgpoints):
        proj, _ = cv2.projectPoints(obj, rv, tv, k, dist)
        proj = proj.reshape(-1, 2)
        obs = obs.reshape(-1, 2)
        errs.append(float(np.mean(np.linalg.norm(proj - obs, axis=1))))
    return errs


def _mirror_variants(g):
    """角点网格的 4 种镜像组合（左右目检测起始角可能不同）。"""
    return [g, g[:, ::-1], g[::-1], g[::-1, ::-1]]


def align_right_corners(pts_l, pts_r, bw, bh):
    """对齐左右角点网格排序。

    双目两视角的棋盘起角可能互为镜像，而 stereoCalibrate 要求 objp[i]
    在左右目中对应同一物理角点。注意棋盘点集自身镜像对称，单应误差
    区分不了镜像组合，因此：
    - 单应重投影误差仅做有效性校验（排除非共面/错检的坏图对）
    - 组合选择用未校正垂直视差 |dy| 打分：并排双目 rig 的同名点
      未校正 dy 只有几个像素，而 180° 翻转组合会差约半个棋盘高度；
      反射组合则与单目标定收敛矛盾（不会通过阶段二的 RMS 检查）
    返回 (对齐后的右目角点, 单应误差 px)。
    """
    gl = pts_l.reshape(bh, bw, 2).astype(np.float32)
    gr = pts_r.reshape(bh, bw, 2).astype(np.float32)
    src = gl.reshape(-1, 1, 2)
    hom_err = float("inf")
    best_score, best_pts = None, None
    for v in _mirror_variants(gr):
        dst = v.reshape(-1, 1, 2)
        h, _ = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        if h is None:
            continue
        proj = cv2.perspectiveTransform(src, h)
        err = float(np.mean(np.linalg.norm(proj.reshape(-1, 2) - v.reshape(-1, 2), axis=1)))
        hom_err = min(hom_err, err)
        score = float(np.mean(np.abs(gl[..., 1] - v[..., 1])))
        if best_score is None or score < best_score:
            best_score, best_pts = score, v
    if best_pts is None:
        return pts_r, float("inf")
    return best_pts.reshape(-1, 1, 2).astype(np.float32), hom_err


def _rms_check(checks, name, rms):
    if rms < RMS_GOOD:
        st, adv = STATUS_PASS, "优秀"
    elif rms < RMS_OK:
        st, adv = STATUS_WARN, "可用但偏高：检查角点质量、棋盘平整度，或剔除坏图后重标"
    else:
        st, adv = STATUS_FAIL, "误差过大，标定结果不可信，建议重新采集"
    checks.append(make_check(2, name, name.replace("_", " "), st,
                             f"{rms:.3f} px",
                             f"<{RMS_GOOD} PASS；{RMS_GOOD}~{RMS_OK} WARN；>{RMS_OK} FAIL", adv))
    return st


def run_stage2(ctx, checks, expected_baseline=None):
    valid = ctx["valid"]
    size = ctx["size"]
    bw, bh = ctx["board"]["w"], ctx["board"]["h"]
    square = ctx["square"]

    objpoints = [build_objp(bw, bh, square)] * len(valid)
    pts_l = [v["pts_left"] for v in valid]
    pts_r = [v["pts_right"] for v in valid]

    rms_l, k1, d1, rv1, tv1 = cv2.calibrateCamera(objpoints, pts_l, size, None, None)
    rms_r, k2, d2, rv2, tv2 = cv2.calibrateCamera(objpoints, pts_r, size, None, None)
    e_l = per_view_errors(objpoints, rv1, tv1, k1, d1, pts_l)
    e_r = per_view_errors(objpoints, rv2, tv2, k2, d2, pts_r)

    # 左右角点网格排序对齐（镜像 4 选 1），再送入双目标定
    pts_r_aligned, align_errs = [], []
    for v, pl, pr in zip(valid, pts_l, pts_r):
        ar, e = align_right_corners(pl, pr, bw, bh)
        pts_r_aligned.append(ar)
        align_errs.append(e)
        v["align_err"] = e

    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
    rms_s, _, _, _, _, rot, tvec, _, _ = cv2.stereoCalibrate(
        objpoints, pts_l, pts_r_aligned, k1, d1, k2, d2, size,
        flags=cv2.CALIB_FIX_INTRINSIC, criteria=crit)

    # ---- 检查 0：左右角点配准质量 ----
    max_align = max(align_errs) if align_errs else float("inf")
    if max_align > 5.0:
        bad = [v["id"] for v, e in zip(valid, align_errs) if e > 5.0]
        st, adv = STATUS_WARN, (f"以下图对左右角点无法用单应对齐：{'、'.join(bad)}，"
                                "疑似左右图非同一场景或棋盘被遮挡，建议剔除后重跑")
    else:
        st, adv = STATUS_PASS, "全部图对左右角点配准一致"
    checks.append(make_check(2, "pair_align", "左右角点配准（单应误差）", st,
                             f"最大 {max_align:.2f} px", "<=5 px PASS", adv))

    ctx["calib"] = {
        "K1": k1, "d1": d1, "K2": k2, "d2": d2,
        "R": rot, "T": tvec,
        "rms_left": float(rms_l), "rms_right": float(rms_r), "rms_stereo": float(rms_s),
        "per_view_left": e_l, "per_view_right": e_r,
    }

    # ---- 检查 1~3：重投影误差 ----
    _rms_check(checks, "rms_left", rms_l)
    _rms_check(checks, "rms_right", rms_r)
    _rms_check(checks, "rms_stereo", rms_s)

    # ---- 检查 4：逐图离群（建议剔除重标） ----
    mean_l = float(np.mean(e_l))
    outlier_ids = [valid[i]["id"] for i in range(len(valid))
                   if max(e_l[i], e_r[i]) > max(2.0 * mean_l, mean_l + 1.0)]
    if outlier_ids:
        st, adv = STATUS_WARN, (
            "以下图像误差显著偏高，剔除后重新标定可能改善："
            + "、".join(outlier_ids) + "（用 --exclude-ids 排除后重跑验证）")
    else:
        st, adv = STATUS_PASS, "无离群图像"
    worst = max(max(e_l), max(e_r)) if e_l else 0.0
    checks.append(make_check(2, "outlier_views", "逐图重投影误差离群", st,
                             f"最差 {worst:.3f} px", "单图 > max(2×均值, 均值+1.0)", adv))

    # ---- 检查 5：fx ≈ fy ----
    ratios = [abs(k1[0, 0] / k1[1, 1] - 1), abs(k2[0, 0] / k2[1, 1] - 1)]
    if max(ratios) > 0.05:
        st, adv = STATUS_WARN, "fx 与 fy 偏差 >5%：方形像素假设不成立或标定不佳，请复核"
    else:
        st, adv = STATUS_PASS, "方形像素假设成立"
    checks.append(make_check(2, "fxfy", "fx/fy 一致性", st,
                             f"偏差 {max(ratios) * 100:.1f}%", "<=5% PASS", adv))

    # ---- 检查 6：主点位置 ----
    w, h = size
    off_x = max(abs(k1[0, 2] - w / 2) / w, abs(k2[0, 2] - w / 2) / w)
    off_y = max(abs(k1[1, 2] - h / 2) / h, abs(k2[1, 2] - h / 2) / h)
    if off_x > 0.1 or off_y > 0.15:
        st, adv = STATUS_WARN, "主点偏离图像中心较多：可能是姿态覆盖不足，建议补拍画面边缘的图像"
    else:
        st, adv = STATUS_PASS, "主点位置合理"
    checks.append(make_check(2, "principle_point", "主点居中性", st,
                             f"偏移 x:{off_x * 100:.1f}% y:{off_y * 100:.1f}%",
                             "x<=10%、y<=15% PASS", adv))

    # ---- 检查 7：畸变系数量级 ----
    k1_coef = float(d1.flat[0])
    if abs(k1_coef) > 0.6:
        st, adv = STATUS_WARN, "k1 量级异常偏大，通常是角点错检或姿态单一导致，请复核"
    elif k1_coef > 0:
        st, adv = STATUS_INFO, "k1>0（枕形畸变）较少见，请结合图像确认"
    else:
        st, adv = STATUS_PASS, "正常（k1<0，桶形畸变）"
    checks.append(make_check(2, "distortion", "畸变系数 k1", st,
                             f"{k1_coef:.3f}", "|k1|<=0.6 且 k1<0 常见", adv))

    # ---- 检查 8：基线 ----
    baseline = float(np.linalg.norm(tvec))
    unit = "×方格边长"
    if expected_baseline is not None:
        rel = abs(baseline - expected_baseline) / expected_baseline
        if rel > 0.5:
            st, adv = STATUS_WARN, (f"与预期基线 {expected_baseline}（与 --square 同单位）"
                                    f"偏差 {rel * 100:.0f}%，请核对尺子或排查外参")
        else:
            st, adv = STATUS_PASS, "与预期基线一致"
        crit_txt = f"偏差<=50%（预期 {expected_baseline}）"
    elif baseline < 0.2 or baseline > 200:
        st, adv = STATUS_WARN, "基线长度超出常见范围，请确认 --square 与实际尺子"
        crit_txt = "0.2~200 方格边长"
    else:
        st, adv = STATUS_PASS, "基线量级合理（如需精确核验请传 --expected-baseline）"
        crit_txt = "0.2~200 方格边长"
    checks.append(make_check(2, "baseline", "双目基线长度", st,
                             f"{baseline:.2f} {unit}", crit_txt, adv))

    # ---- 检查 9：双目刚性（R 接近单位阵） ----
    ang = math.degrees(math.acos(np.clip((np.trace(rot) - 1) / 2, -1.0, 1.0)))
    if ang < 1:
        st, adv = STATUS_PASS, "两相机光轴近乎平行（并排刚性安装）"
    elif ang < 3:
        st, adv = STATUS_WARN, "两相机存在小角度相对旋转，请确认安装是否松动"
    else:
        st, adv = STATUS_WARN, "相对旋转较大：若非并排安装可忽略本项"
    checks.append(make_check(2, "rig_rotation", "双目相对旋转角", st,
                             f"{ang:.2f}°", "<1° PASS；1~3° WARN（并排刚性假设）", adv))
