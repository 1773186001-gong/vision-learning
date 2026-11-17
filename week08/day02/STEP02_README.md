# Day02 Step 2：同步随机水平翻转

详细学习交接见 [DAY02_STEP02_GPT_HANDOFF.md](DAY02_STEP02_GPT_HANDOFF.md)。

新增实现：`paired_horizontal_flip.py`；真实实验脚本：`experiment_horizontal_flip.py`。
复用原有 `PairedLetterbox`，顺序为 Letterbox → 同步翻转。

在项目根目录的 PowerShell 中运行：

```powershell
Set-Location D:\python\test\vision-learning
$env:PYTHONIOENCODING='utf-8'
& D:\python\.venv\Scripts\python.exe -B .\week08\day02\experiment_horizontal_flip.py `
  --data-root .\data\LLVIP_1000 `
  --output-dir .\week08\day02\outputs\step02_horizontal_flip `
  --seed 42
```

该目录已保存本次真实结果。重跑请使用新的输出目录名，例如 `step02_horizontal_flip_run02`，脚本不会覆盖非空目录。

结果文件：`terminal_output.txt`（完整日志）、`results.json`（坐标/面积/决策/检查结果及哈希）、4 张 comparison 图与 `overview.jpg`。
未修改 Day01、Step1 文件或产物；不含模型训练。

接入原 Dataset 的方式（脚本已实际验证）：

```python
pipeline = PairedLetterboxAndFlip(
    split="train",  # val/test 自动禁用翻转
    train_flip_p=0.5,
    generator=torch.Generator().manual_seed(42),
)
dataset = LLVIPDataset(data_root, split="train", modality="paired", transforms=pipeline)
loader = DataLoader(dataset, batch_size=2, num_workers=0,
                    collate_fn=detection_collate_fn)
```

复现实验限定 CPU、num_workers=0、相同种子和读取顺序。此处的显式 Generator 会随调用推进；不要在每次取样时重新初始化。以后启用多 worker 时需要独立设计 worker 的种子，不在本步骤扩展。
