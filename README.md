# stereo-calib-checker

> 双目（立体）相机标定质检 Agent Skill —— 标定完不知道参数能不能用？一测便知。

仓库地址：<https://github.com/1999XIAOZHANG/stereo-calib-checker-skill>

给一组双目棋盘格标定图像做"体检"，回答三个问题：

1. **这组数据能不能标？** 采集充分性检查，判据来自张正友标定法原文（≥3 张理论下限、纯平移/正对姿态退化检测、角点覆盖充分性）
2. **标出来的参数可信吗？** 逐图重投影误差、fx≈fy、主点、畸变系数量级、基线长度、双目刚性
3. **立体校正到底齐没齐？** 校正后左右同名点垂直视差 |dy| 双路验证（棋盘角点 + SIFT 通用特征点），并生成红蓝立体图

输出 `report.md`（人读，含逐图明细）+ `report.json`（Agent 读），每项检查给出 **PASS / WARN / FAIL + 实测值 + 判据 + 修复建议**。

## 仓库结构

```
stereo-calib-checker/
├── stereo-calib-checker-skill/   ← Skill 本体（上传 SkillHub 的就是这个目录，保持纯净）
│   ├── SKILL.md                  # Agent 技能说明
│   ├── scripts/                  # 三阶段流水线脚本
│   ├── download_samples.py       # OpenCV 官方样例数据下载（多源回退）
│   └── requirements.txt
├── demo/                         # 本地演示数据与报告（gitignore，不入库）
├── README.md
└── LICENSE
```

## 快速开始

在仓库根目录执行：

```bash
pip install -r stereo-calib-checker-skill/requirements.txt

# 下载 OpenCV 官方样例数据到 demo/data（13 对：01~09 与 11~14，官方没有 10 号图）
python stereo-calib-checker-skill/download_samples.py --out demo/data

# 全量质检
python stereo-calib-checker-skill/scripts/run_check.py --input demo/data --out demo/out_full
```

> 注意：演示数据与输出报告统一放在 `demo/`（已被 gitignore），避免污染 skill 目录。

更多用法（退化对照实验、剔除坏图、接入 Agent）见 [skill 内 README](stereo-calib-checker-skill/README.md) 与 [SKILL.md](stereo-calib-checker-skill/SKILL.md)。

## 在 OpenCV 官方数据上的验证结果

| 指标 | 实测 |
|---|---|
| 立体重投影误差 | 0.267 px |
| 双目基线 | 3.32 方格边长（与已知值吻合） |
| 极线垂直视差（棋盘） | 均值 0.109 px / P95 0.302 px |
| 总体结论 | WARN（覆盖率 45%、非棋盘区特征点 P95=1.3px 两条诚实提示） |

## 相关文章

- CSDN：《写了个双目标定质检 Agent Skill》（链接待补充）

## License

MIT
