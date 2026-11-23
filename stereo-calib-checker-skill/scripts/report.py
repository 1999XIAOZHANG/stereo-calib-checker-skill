# -*- coding: utf-8 -*-
"""质检报告渲染：JSON（给 Agent 读）+ Markdown（给人看）。"""
import numpy as np

STATUS_ICON = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌", "INFO": "ℹ️"}
STAGE_NAME = {1: "阶段一 · 采集充分性（张正友原文判据）",
              2: "阶段二 · 标定参数合理性",
              3: "阶段三 · 极线对齐验证"}


def overall_verdict(checks):
    statuses = {c["status"] for c in checks}
    if "FAIL" in statuses:
        return "FAIL"
    if "WARN" in statuses:
        return "WARN"
    return "PASS"


def to_jsonable(obj):
    """递归把 numpy 类型转成原生 Python，便于 json.dumps。"""
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def summarize(ctx):
    """提取报告摘要数字（含标定结果矩阵，供 Agent 复用）。"""
    s = {
        "pairs_total": len(ctx["pairs"]),
        "pairs_valid": len(ctx["valid"]),
        "pairs_failed": ctx["failed"],
        "image_size": list(ctx["size"]) if ctx.get("size") else None,
        "board": dict(ctx["board"]),
    }
    calib = ctx.get("calib")
    if calib:
        s["rms"] = {"left": calib["rms_left"], "right": calib["rms_right"],
                    "stereo": calib["rms_stereo"]}
        s["K1"] = np.round(calib["K1"], 2).tolist()
        s["K2"] = np.round(calib["K2"], 2).tolist()
        s["dist1"] = np.round(np.ravel(calib["d1"]), 5).tolist()
        s["dist2"] = np.round(np.ravel(calib["d2"]), 5).tolist()
        s["baseline"] = float(np.linalg.norm(calib["T"]))
        s["R"] = np.round(calib["R"], 5).tolist()
        s["T"] = np.round(np.ravel(calib["T"]), 5).tolist()
    if ctx.get("pose"):
        s["pose_max_tilt_deg"] = round(max(ctx["pose"]["tilts"]), 1)
    if ctx.get("rect_stats"):
        s["rect_per_pair"] = ctx["rect_stats"]["per_pair"]
    return to_jsonable(s)


def _table(rows, header):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def build_markdown(ctx, checks, verdict, elapsed):
    s = summarize(ctx)
    lines = []
    lines.append("# 双目标定质检报告")
    lines.append("")
    lines.append(f"**总体结论：{STATUS_ICON[verdict]} {verdict}**（耗时 {elapsed:.1f}s）")
    lines.append("")
    lines.append(f"- 图像对：{s['pairs_valid']} 有效 / {s['pairs_total']} 总计"
                 + (f"，失败：{s['pairs_failed']}" if s["pairs_failed"] else ""))
    if s.get("image_size"):
        lines.append(f"- 图像尺寸：{s['image_size'][0]}×{s['image_size'][1]}，"
                     f"棋盘内角点：{s['board']['w']}×{s['board']['h']}，"
                     f"方格边长：{s['board']['square']}")
    if s.get("rms"):
        lines.append(f"- 重投影误差：左 {s['rms']['left']:.3f}px / "
                     f"右 {s['rms']['right']:.3f}px / 双目 {s['rms']['stereo']:.3f}px；"
                     f"基线 {s['baseline']:.2f} 方格边长")
    lines.append("")

    for stage in (1, 2, 3):
        rows = [(STATUS_ICON[c["status"]], c["name"], c["value"],
                 c["criteria"], c["advice"])
                for c in checks if c["stage"] == stage]
        if not rows:
            continue
        lines.append(f"## {STAGE_NAME[stage]}")
        lines.append("")
        lines.append(_table(rows, ("状态", "检查项", "实测", "判据", "建议")))
        lines.append("")

    # 附录：逐图姿态与误差明细（用于挑选剔除对象 / 复现退化实验）
    if ctx.get("pose") and ctx.get("calib"):
        tilts = ctx["pose"]["tilts"]
        inpl = ctx["pose"]["inplanes"]
        e_l = ctx["calib"]["per_view_left"]
        e_r = ctx["calib"]["per_view_right"]
        rect = {r["id"]: r for r in ctx.get("rect_stats", {}).get("per_pair", [])}
        rows = []
        for i, v in enumerate(ctx["valid"]):
            chess = rect.get(v["id"], {}).get("chess_mean")
            rows.append((v["id"], f"{tilts[i]:.1f}°", f"{inpl[i]:.1f}°",
                         f"{e_l[i]:.3f}px", f"{e_r[i]:.3f}px",
                         f"{chess:.3f}px" if chess else "-"))
        lines.append("## 附录 · 逐图明细")
        lines.append("")
        lines.append(_table(rows, ("图对", "倾斜角", "面内旋转",
                                   "左目误差", "右目误差", "校正后棋盘|dy|")))
        lines.append("")

    if ctx.get("artifacts"):
        lines.append("## 可视化产物")
        lines.append("")
        lines.extend(f"- {a}" for a in ctx["artifacts"])
        lines.append("")

    lines.append("---")
    lines.append("*阈值为针对 640×480~4K 棋盘格标定的经验值，请结合相机分辨率与精度要求调整。*")
    lines.append("*本报告为辅助预检，不替代正式的标定验证流程。*")
    return "\n".join(lines)
