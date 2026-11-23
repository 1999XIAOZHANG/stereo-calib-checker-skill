# stereo-calib-checker：双目标定质检 Agent Skill

标定完不知道参数能不能用？喂给 Agent 这个 Skill，一测便知。

**三阶段体检流水线：**

| 阶段 | 检查内容 | 依据 |
|---|---|---|
| ① 采集充分性 | 角点检测、图像对数量、**姿态多样性**（退化检测）、角点覆盖、清晰度、曝光 | 张正友标定法原文判据 |
| ② 参数合理性 | 逐图重投影误差、fx≈fy、主点居中、畸变系数量级、基线长度、双目刚性 | 标定常识 + 工程经验 |
| ③ 极线对齐验证 | 立体校正后左右同名点垂直视差 dy：棋盘角点路 + SIFT 特征点路双路互证 | 立体视觉基本约束 |

输出 `report.md`（人读）+ `report.json`（Agent 读），每项检查给出 **PASS / WARN / FAIL + 实测值 + 判据 + 修复建议**，并生成红蓝立体极线叠加图等可视化。

## 快速开始

> 提示：示例数据与输出报告请放在 skill 目录**之外**（如下面的 `../demo/`），
> 保持 skill 目录纯净，方便整体上传 SkillHub。

```bash
pip install -r requirements.txt

# 下载 OpenCV 官方样例数据（共 13 对：01~09 与 11~14，官方没有 10 号图）
python download_samples.py --out ../demo/data

# ① 全量质检（健康数据 → 立体 RMS 0.27px、极线均值 0.11px）
python scripts/run_check.py --input ../demo/data --out ../demo/out_full

# ② 张正友理论下限实验：只用 2 对 → 直接 FAIL 并中止后续阶段
python scripts/run_check.py --input ../demo/data --only-ids 01,03 --out ../demo/out_two

# ③ 数量不足实验：3 对 → WARN（理论够用，工程不建议）
python scripts/run_check.py --input ../demo/data --only-ids 01,02,03 --out ../demo/out_three

# ④ 姿态偏正对实验：挑倾斜角最小的 3 对（01/03/04）→ 姿态指标显著变差
python scripts/run_check.py --input ../demo/data --only-ids 01,03,04 --out ../demo/out_frontal
```

自己的数据：把左右图对放进一个目录，文件名含 `left` / `right` 记号即可（如 `left01.jpg` ↔ `right01.jpg`）。

## 常用参数

| 参数 | 说明 | 默认 |
|---|---|---|
| `--board-w / --board-h` | 棋盘内角点数（格子数−1） | 9 / 6 |
| `--square` | 方格边长，决定基线单位 | 1.0 |
| `--expected-baseline` | 预期基线（与 square 同单位），核验外参 | 无 |
| `--only-ids / --exclude-ids` | 子集实验 / 剔除坏图 | 无 |
| `--max-vis` | 保存前 N 对可视化 | 6 |

## 关键指标怎么读

- **板面最大倾斜角 / 朝向最大夹角**：全部正对、只平移拍摄是张正友标定的典型退化配置 → 工具直接 FAIL
- **重投影误差**：<0.5px 优秀，>1px 不可信；逐图明细帮你定位该剔哪张
- **极线垂直视差 |dy|**：校正好后应接近 0；均值 <0.5px 优秀，>1px 说明校正不可用
- 报告附录给出每对图的倾斜角、面内旋转、逐图误差、校正后 |dy|，方便挑选剔除对象

## 接入 Agent

参考 `SKILL.md`（Anthropic Agent Skill 格式）：Agent 读取 SKILL.md 后即可调用本脚本，并把 `report.json` 转述成自然人话反馈。

## 局限性

- 仅支持平面棋盘格（暂不支持 ChArUco / 圆形网格）
- 阈值为经验值，未按分辨率自适应
- 双目刚性检查假设并排同向安装的常规双目架
- 辅助预检用途，不替代正式验证流程

## License

MIT
