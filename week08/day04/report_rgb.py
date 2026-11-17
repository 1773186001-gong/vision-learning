"""只在RGB训练、选模、最终测试及案例生成成功后生成真实报告。"""
import json
from pathlib import Path
from rgb_common import comparison, ir, read_json, read_reference, write_new_json


def write_report(output, config, history, best_epoch, scores, selection, timing):
    reference=read_reference()
    compared=comparison(config,history,best_epoch,scores,reference)
    write_new_json(output/'rgb_vs_ir.json',compared)
    report=['# Week08 Day04：RGB单模态完整训练真实结果',
            '10轮训练、验证集选模、一次最终测试和缓存出图已完成。历史IR指标仅用于对照。',
            '## 模型、环境与输入',
            '模型、预训练文件、两类预测头、优化器及评估均复用Day03；只向模型输入visible RGB Tensor。'+
            '未从IR微调权重初始化。Dataset仍读取IR做既有配对处理，IR Tensor不传入模型。',
            '输入640×640；原图1280×1024内容缩到640×512，上下各64。内部min/max=640。'+
            'train/val/test=700/150/150，场景隔离，train翻转p=0.5，val/test p=0。',
            '```json\n'+json.dumps(config,ensure_ascii=False,indent=2)+'\n```',
            '## 十轮Loss与验证指标',
            '|Epoch|total|classifier|box_reg|objectness|rpn_box|lr|val P|val R|val AP50|val mAP50:95|seconds|peak allocated bytes|peak reserved bytes|',
            '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    fields=('total_loss',)+ir.LOSS_KEYS+('lr','precision','recall','AP50','mAP50_95','epoch_seconds')
    for row in history:
        values=[str(row['epoch'])]+[f'{row[k]:.6f}' for k in fields]
        values += [str(row[k]) for k in ('peak_allocated_bytes','peak_reserved_bytes')]
        report.append('|'+ '|'.join(values)+'|')
    report += ['## 最佳模型与正式test',f'最佳Epoch={best_epoch}，按val mAP50:95选出；best.pth为最终模型，last.pth为第10轮状态。',
               '```json\n'+json.dumps(scores,indent=2)+'\n```',
               '评估与IR共用同一函数：单person类101点插值，IoU=.50:.05:.95；'+
               'P/R为score≥.5、IoU≥.5，每图最多100预测，模型score下限.001。'+
               '没有COCO crowd/ignore或面积分层，不能声称是官方COCO API完整指标。',
               '## RGB与IR实际对照',
               '|指标|IR历史实测|RGB本次实测|RGB减IR|','|---|---:|---:|---:|']
    for key in ('precision','recall','AP50','mAP50_95'):
        report.append(f'|{key}|{reference["test"][key]:.6f}|{scores[key]:.6f}|{scores[key]-reference["test"][key]:+.6f}|')
    report += ['```json\n'+json.dumps(compared,ensure_ascii=False,indent=2)+'\n```',
               '耗时、显存为单次运行记录；两次子集和协议一致但不代表统计显著结论，未做多seed实验，不能推断融合效果。',
               '## 显存与计时', '```json\n'+json.dumps(timing,ensure_ascii=False,indent=2)+'\n```',
               'epoch_seconds与IR定义一致，为该轮训练和验证，不含随后保存/绘图或最终test。'+
               'GPU峰值为PyTorch allocator统计，不等于整卡占用。run计时从本脚本入口起至产物检查前，'+
               '不含外部启动器，真实进程结束时间和退出码另见exit_status.json。',
               '## 预测案例','```json\n'+json.dumps(selection,ensure_ascii=False,indent=2)+'\n```',
               '绿色GT、红色预测、person及置信分数；按真实TP/FP/FN分类，优先IR已展示的编号；'+
               '不足3张时记录缺额，类别可以重叠。选择依据见prediction_examples/case_details.json。',
               '## 运行命令', '```cmd\n'+config['command']+'\n```',
               '输出位置：`'+str(output)+'`。本地对应路径：'+
               '`D:\\python\\test\\vision-learning\\week08\\day04\\outputs\\'+output.name+'`。',
               '复现须选择新的空run目录，不能覆盖本次实验；test仅正式评估一次，图像由缓存生成。'+
               '独立val评估及缓存出图命令见Day04 README。',
               '## 实际问题与文件',
               '本报告仅在全流程成功后生成；成功并不表示没有任何warning。完整console日志保留启动、检查及warning，'+
               '异常记录为failure_*.json，无OOM/非有限数值时才可能运行到此。初始本地GPU/训练状态不得冒充服务器实测。',
               '主要产物：config.json、preflight.json、smoke_test.json、initialization.json、training_log.csv、'+
               'val_epoch_01.json至val_epoch_10.json、逐轮epoch_NN.pth、best.pth、last.pth、loss_curve.png、'+
               'test_started.json、test_metrics.json、test_predictions.pth、prediction_examples/、rgb_vs_ir.json、'+
               'result_checks.json、protected_after.json、completed.json、exit_status.json及完整console日志。',
               '## 交给ChatGPT学习',
               '请全程中文，每次一个明确步骤，不主动测验；代码完成不代表已掌握。'+
               '请解释公平模态对照、预训练与IR微调初始化的区别、四项Loss和空标注、val选模与一次test、'+
               'AP阈值与定位误差、实际漏检/误检案例、峰值allocated/reserved及续训状态。'+
               '请区分真实观察和原因推测；下一步先分析本次结果，不自动开展RGB+IR融合。']
    path=output/'DAY04_RGB_TRAINING_GPT_HANDOFF.md'
    with path.open('x',encoding='utf-8') as stream:
        stream.write('\n\n'.join(report).replace('|\n\n|','|\n|')+'\n')
    return compared
