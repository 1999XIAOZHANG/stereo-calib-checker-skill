# -*- coding: utf-8 -*-
"""阶段三：极线对齐验证（标定后）。

立体校正后，左右图像的同名点应位于同一行（垂直视差 dy≈0）。
双路验证互为印证：
- 棋盘角点路：角点网格顺序已知，配对精确，直接算 dy
- 通用特征点路：SIFT 匹配，可验证非标定板区域的极线质量
"""
import cv2
import numpy as np
from pathlib import Path

from common import (STATUS_FAIL, STATUS_INFO, STATUS_PASS, STATUS_WARN,
                    imread_any, imwrite_any, make_check, sanitize_component)
from stage1_acquire import detect_corners

# 极线垂直视差阈值（px）
EP_GOOD, EP_OK = 0.5, 1.0


def _variants(g):
    """角点网格的 4 种镜像组合（左右目网格排序可能存在镜像）。"""
    return [g, g[:, ::-1], g[::-1], g[::-1, ::-1]]


def grid_dy(pl, pr, bw, bh):
    """对齐左右角点网格排序后返回 dy 数组（y_left - y_right）。"""
    gl = pl.reshape(bh, bw, 2).astype(np.float64)
    gr = pr.reshape(bh, bw, 2).astype(np.float64)
    best_err, best_dy = None, None
    for a in _variants(gl):
        for b in _variants(gr):
            dy = a[..., 1] - b[..., 1]
            err = float(np.abs(dy).mean())
            if best_err is None or err < best_err:
                best_err, best_dy = err, dy.ravel()
    return best_dy


def feature_dy(rect_l, rect_r, ratio=0.75):
    """SIFT + 比率测试匹配，返回 (dy, 内点掩膜)。"""
    sift = cv2.SIFT_create(nfeatures=4000)
    k1, d1 = sift.detectAndCompute(rect_l, None)
    k2, d2 = sift.detectAndCompute(rect_r, None)
    if d1 is None or d2 is None or len(k1) < 10 or len(k2) < 10:
        return None, None
    bf = cv2.BFMatcher()
    raw = bf.knnMatch(d1, d2, k=2)
    good = [m for m, n in raw if m.distance < ratio * n.distance]
    if len(good) < 20:
        return None, None
    yl = np.array([k1[m.queryIdx].pt[1] for m in good])
    yr = np.array([k2[m.trainIdx].pt[1] for m in good])
    dy = yl - yr
    med = np.median(dy)
    mad = np.median(np.abs(dy - med))
    keep = np.abs(dy - med) < max(3 * 1.4826 * mad, 1.0)
    return dy, keep


def save_anaglyph(rect_l, rect_r, out_dir, pair_id, line_step=40):
    """红蓝立体图 + 水平极线：极线对齐好则线条在同一高度穿过左右内容。"""
    h, w = rect_l.shape
    ana = np.zeros((h, w, 3), np.uint8)
    ana[..., 0] = rect_l          # 左目 -> 红
    ana[..., 1] = rect_r          # 右目 -> 青
    ana[..., 2] = rect_r
    for y in range(0, h, line_step):
        cv2.line(ana, (0, y), (w, y), (0, 255, 255), 1)
    path = Path(out_dir) / ("anaglyph_" + pair_id + ".jpg")
    imwrite_any(path, ana, base_dir=out_dir)


def save_side_by_side(rect_l, rect_r, out_dir, pair_id, line_step=40):
    h, w = rect_l.shape
    combo = np.hstack([rect_l, rect_r])
    combo = cv2.cvtColor(combo, cv2.COLOR_GRAY2BGR)
    for y in range(0, h, line_step):
        cv2.line(combo, (0, y), (2 * w, y), (0, 0, 255), 1)
    path = Path(out_dir) / ("sbs_" + pair_id + ".jpg")
    imwrite_any(path, combo, base_dir=out_dir)


def run_stage3(ctx, checks, out_dir, max_vis=6):
    calib = ctx["calib"]
    size = ctx["size"]
    bw, bh = ctx["board"]["w"], ctx["board"]["h"]

    r1, r2, p1, p2, _, _, _ = cv2.stereoRectify(
        calib["K1"], calib["d1"], calib["K2"], calib["d2"], size,
        calib["R"], calib["T"], flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    m1x, m1y = cv2.initUndistortRectifyMap(calib["K1"], calib["d1"], r1, p1, size, cv2.CV_32FC1)
    m2x, m2y = cv2.initUndistortRectifyMap(calib["K2"], calib["d2"], r2, p2, size, cv2.CV_32FC1)

    chess_dy_all, feat_dy_all, feat_total = [], [], 0
    feat_outlier_frac = []
    per_pair = []
    artifacts = []

    for idx, v in enumerate(ctx["valid"]):
        gl = imread_any(v["left_path"])
        gr = imread_any(v["right_path"])
        if gl is None or gr is None:
            continue
        rect_l = cv2.remap(gl, m1x, m1y, cv2.INTER_LINEAR)
        rect_r = cv2.remap(gr, m2x, m2y, cv2.INTER_LINEAR)

        # 棋盘角点路
        row = {"id": v["id"], "chess_mean": None, "feat_n": 0}
        okl, pl, _ = detect_corners(rect_l, bw, bh)
        okr, pr, _ = detect_corners(rect_r, bw, bh)
        if okl and okr:
            dy = grid_dy(pl, pr, bw, bh)
            chess_dy_all.append(dy)
            row["chess_mean"] = float(np.abs(dy).mean())

        # 通用特征点路
        dy, keep = feature_dy(rect_l, rect_r)
        if dy is not None and keep is not None and keep.sum() >= 10:
            feat_dy_all.append(dy[keep])
            feat_total += int(keep.sum())
            feat_outlier_frac.append(1.0 - keep.mean())
            row["feat_n"] = int(keep.sum())
        per_pair.append(row)

        # 可视化（仅前 max_vis 对；pair_id 已在入口净化，这里再防御一次）
        if idx < max_vis:
            sid = sanitize_component(v["id"])
            save_anaglyph(rect_l, rect_r, out_dir, sid)
            save_side_by_side(rect_l, rect_r, out_dir, sid)
            artifacts += [f"anaglyph_{sid}.jpg", f"sbs_{sid}.jpg"]

    ctx["rect_stats"] = {"per_pair": per_pair}
    ctx["artifacts"] = artifacts

    # ---- 检查 1：棋盘角点极线误差（均值） ----
    if not chess_dy_all:
        checks.append(make_check(3, "ep_chess_mean", "极线误差·棋盘(均值)", STATUS_INFO,
                                 "校正图上未检出棋盘", "-", "无法评估"))
    else:
        allc = np.concatenate(chess_dy_all)
        mean_abs = float(np.abs(allc).mean())
        if mean_abs < EP_GOOD:
            st, adv = STATUS_PASS, "极线对齐良好"
        elif mean_abs < EP_OK:
            st, adv = STATUS_WARN, "存在轻微失配：剔除离群图重标、或补拍姿态多样图像"
        else:
            st, adv = STATUS_FAIL, "极线未对齐，立体校正不可用，请重新标定"
        worst = max(per_pair, key=lambda r: r["chess_mean"] or 0)
        checks.append(make_check(3, "ep_chess_mean", "极线误差·棋盘(均值)", st,
                                 f"{mean_abs:.3f} px（最差图对 {worst['id']}）",
                                 f"<{EP_GOOD} PASS；{EP_GOOD}~{EP_OK} WARN；>{EP_OK} FAIL", adv))

        # ---- 检查 2：棋盘角点极线误差（P95） ----
        p95 = float(np.quantile(np.abs(allc), 0.95))
        if p95 < 1.0:
            st, adv = STATUS_PASS, "尾部分布健康"
        elif p95 < 2.0:
            st, adv = STATUS_WARN, "存在局部超差点，检查角点排序/棋盘平整度"
        else:
            st, adv = STATUS_FAIL, "大量角点偏离极线，标定或数据存在问题"
        checks.append(make_check(3, "ep_chess_p95", "极线误差·棋盘(P95)", st,
                                 f"{p95:.3f} px", "<1 PASS；1~2 WARN；>2 FAIL", adv))

    # ---- 检查 3：通用特征点极线误差 ----
    if not feat_dy_all or feat_total < 50:
        checks.append(make_check(3, "ep_feat", "极线误差·特征点(P95)", STATUS_INFO,
                                 "纹理特征不足", "-", "场景纹理过少，仅供参考棋盘路结果"))
    else:
        allf = np.concatenate(feat_dy_all)
        p95 = float(np.quantile(np.abs(allf), 0.95))
        if p95 < 1.0:
            st, adv = STATUS_PASS, "全图极线质量良好"
        elif p95 < 2.0:
            st, adv = STATUS_WARN, "非标定板区域存在轻微失配（可能是镜头畸变模型不足）"
        else:
            st, adv = STATUS_FAIL, "全图极线失配严重，立体校正不可信"
        checks.append(make_check(3, "ep_feat", "极线误差·特征点(P95)", st,
                                 f"{p95:.3f} px（{feat_total} 个内点）",
                                 "<1 PASS；1~2 WARN；>2 FAIL", adv))

        # ---- 检查 4：特征匹配离群比例 ----
        frac = float(np.mean(feat_outlier_frac))
        if frac > 0.3:
            st, adv = STATUS_WARN, "匹配离群比例偏高：纹理弱或存在重复纹理区域"
        else:
            st, adv = STATUS_PASS, "匹配质量正常"
        checks.append(make_check(3, "feat_outlier", "特征匹配离群比例", st,
                                 f"{frac * 100:.0f}%", "<=30% PASS", adv))
