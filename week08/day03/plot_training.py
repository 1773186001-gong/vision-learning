"""在本地已有matplotlib的环境，根据真实训练CSV生成曲线；不运行模型。"""
import argparse
import csv
from pathlib import Path
from ir_common import plot_history, LOSS_KEYS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if (output / 'loss_curve.png').exists():
        raise FileExistsError('loss_curve.png已存在，不覆盖')
    with args.csv.open(encoding='utf-8-sig', newline='') as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError('真实训练CSV没有数据')
    history = [dict(epoch=int(row['epoch']), **{
        key: float(row[key]) for key in ('total_loss',) + LOSS_KEYS + ('AP50','mAP50_95')
    }) for row in rows]
    output.mkdir(parents=True, exist_ok=True)
    if not plot_history(history, output):
        raise RuntimeError('当前绘图环境缺少matplotlib；未生成曲线')
    print(f'已从真实CSV生成：{output / "loss_curve.png"}')


if __name__ == '__main__':
    main()
