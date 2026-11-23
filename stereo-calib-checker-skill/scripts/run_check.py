#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""双目标定质检 Skill —— 主入口。

三阶段流水线：
  1) 采集充分性（角点/数量/姿态多样性/覆盖/清晰度/曝光 —— 张正友原文判据）
  2) 标定参数合理性（重投影误差/内参/基线/双目刚性）
  3) 极线对齐验证（校正后棋盘角点 + 通用特征点双路垂直视差）

用法：
  python scripts/run_check.py --input data --board-w 9 --board-h 6
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True  # 不生成 __pycache__，保持 skill 目录纯净

import cv2  # noqa: F401,E402  (提前触发依赖缺失报错，提示更友好)

from common import sanitize_component
from report import build_markdown, overall_verdict, summarize, to_jsonable
from stage1_acquire import run_stage1
from stage2_calibrate import run_stage2
from stage3_rectify import run_stage3

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def find_pairs(input_dir, left_token="left", right_token="right"):
    """按文件名中的 left/right 记号配对，配对键为去掉记号后的余部（如 '01'）。

    配对键来自用户文件名（不可信），经 sanitize_component 净化后再作为 id，
    防止后续拼报告/可视化文件路径时发生路径穿越。
    """
    files = [p for p in sorted(input_dir.iterdir()) if p.suffix.lower() in IMG_EXTS]
    lefts, rights = {}, {}
    for f in files:
        stem = f.stem.lower()
        if left_token in stem:
            lefts[sanitize_component(stem.replace(left_token, "", 1))] = f
        elif right_token in stem:
            rights[sanitize_component(stem.replace(right_token, "", 1))] = f
    keys = sorted(set(lefts) & set(rights))
    return [{"id": k, "left": lefts[k], "right": rights[k]} for k in keys]


def parse_ids(txt):
    return [t.strip() for t in txt.split(",") if t.strip()]


def main():
    try:  # Windows GBK 控制台兜底
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser(
        description="双目标定数据质检：采集充分性 / 参数合理性 / 极线对齐")
    ap.add_argument("--input", required=True, help="图像对所在目录")
    ap.add_argument("--left-token", default="left", help="左图文件名记号（默认 left）")
    ap.add_argument("--right-token", default="right", help="右图文件名记号（默认 right）")
    ap.add_argument("--board-w", type=int, default=9, help="棋盘内角点列数（默认 9）")
    ap.add_argument("--board-h", type=int, default=6, help="棋盘内角点行数（默认 6）")
    ap.add_argument("--square", type=float, default=1.0,
                    help="方格边长（决定基线单位，默认 1.0）")
    ap.add_argument("--expected-baseline", type=float, default=None,
                    help="预期基线长度（与 --square 同单位），用于核验外参")
    ap.add_argument("--out", default="out", help="报告输出目录（默认 out）")
    ap.add_argument("--max-vis", type=int, default=6, help="最多保存前 N 对可视化图")
    ap.add_argument("--only-ids", default=None, help="仅使用指定图对，如 01,02,03")
    ap.add_argument("--exclude-ids", default=None, help="排除指定图对，如 05,07")
    args = ap.parse_args()

    input_dir = Path(args.input)
    if not input_dir.is_dir():
        sys.exit(f"[错误] 输入目录不存在: {input_dir}")

    pairs = find_pairs(input_dir, args.left_token.lower(), args.right_token.lower())
    if args.only_ids:
        keep = set(parse_ids(args.only_ids))
        pairs = [p for p in pairs if p["id"] in keep]
    if args.exclude_ids:
        drop = set(parse_ids(args.exclude_ids))
        pairs = [p for p in pairs if p["id"] not in drop]
    if not pairs:
        sys.exit("[错误] 未找到任何左右图像对（检查文件名是否含 left/right 记号）")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ctx = {
        "pairs": pairs,
        "board": {"w": args.board_w, "h": args.board_h, "square": args.square},
        "square": args.square,
    }
    checks = []
    t0 = time.time()

    run_stage1(ctx, checks)
    if not ctx.get("abort"):
        run_stage2(ctx, checks, expected_baseline=args.expected_baseline)
        run_stage3(ctx, checks, out_dir, max_vis=args.max_vis)

    verdict = overall_verdict(checks)
    elapsed = time.time() - t0

    md = build_markdown(ctx, checks, verdict, elapsed)
    (out_dir / "report.md").write_text(md, encoding="utf-8")

    payload = {"verdict": verdict, "summary": summarize(ctx), "checks": to_jsonable(checks)}
    (out_dir / "report.json").write_text(
        json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")

    icon = {"PASS": "[PASS]", "WARN": "[WARN]", "FAIL": "[FAIL]"}[verdict]
    print(f"\n===== 双目标定质检：{icon} {verdict} =====")
    for c in checks:
        mark = {"PASS": "+", "WARN": "!", "FAIL": "X", "INFO": "i"}[c["status"]]
        print(f"  [{mark}] {c['name']}: {c['value']}")
    print(f"\n报告: {out_dir / 'report.md'}")
    print(f"JSON : {out_dir / 'report.json'}")
    if ctx.get("artifacts"):
        print(f"可视化: {out_dir}（{len(ctx['artifacts'])} 张校验图）")
    return 0 if verdict != "FAIL" else 1


if __name__ == "__main__":
    sys.exit(main())
