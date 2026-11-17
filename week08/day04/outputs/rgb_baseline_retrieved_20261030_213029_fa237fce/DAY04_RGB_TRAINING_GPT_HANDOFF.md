# Week08 Day04：RGB单模态完整训练真实结果

10轮训练、验证集选模、一次最终测试和缓存出图已完成。历史IR指标仅用于对照。

## 模型、环境与输入

模型、预训练文件、两类预测头、优化器及评估均复用Day03；只向模型输入visible RGB Tensor。未从IR微调权重初始化。Dataset仍读取IR做既有配对处理，IR Tensor不传入模型。

输入640×640；原图1280×1024内容缩到640×512，上下各64。内部min/max=640。train/val/test=700/150/150，场景隔离，train翻转p=0.5，val/test p=0。

```json
{
  "model": "fasterrcnn_mobilenet_v3_large_320_fpn",
  "epochs": 10,
  "batch_size": 1,
  "num_workers": 0,
  "seed": 42,
  "lr": 0.001,
  "momentum": 0.9,
  "weight_decay": 0.0005,
  "scheduler": null,
  "amp": false,
  "letterbox": 640,
  "min_size": 640,
  "max_size": 640,
  "selection": "val mAP50_95",
  "modality": "visible",
  "model_input": "visible only; no IR features",
  "data_root": "D:\\vision-learning-server\\data\\LLVIP_1000",
  "manifest_sha256": "8bfefe942c968a7bba05d694e5c6d25548f29d4be35e5f91874ffaa274ecd4eb",
  "weights": {
    "source": "https://download.pytorch.org/models/fasterrcnn_mobilenet_v3_large_320_fpn-907ea3f9.pth",
    "path": "C:\\Users\\gxbb\\.cache\\torch\\hub\\checkpoints\\fasterrcnn_mobilenet_v3_large_320_fpn-907ea3f9.pth",
    "exists": true,
    "sha256": "907ea3f91ff92242bc1baea8049276a3e76bca48ce7560bd268cc029f37977b5"
  },
  "torch": "2.14.1+cu126",
  "torchvision": "0.29.1+cu126",
  "python": "3.14.6",
  "python_executable": "D:\\cv-server-env\\Scripts\\python.exe",
  "cuda_build": "12.6",
  "device": "cuda:0",
  "gpu": "NVIDIA GeForce RTX 3060 Laptop GPU",
  "cpu_threads": 2,
  "train_flip_probability": 0.5,
  "val_test_flip_probability": 0.0,
  "score_floor": 0.001,
  "max_detections": 100,
  "pr_score_threshold": 0.5,
  "pr_iou": 0.5,
  "cudnn_benchmark": false,
  "cudnn_deterministic": true,
  "deterministic_algorithms_required": false,
  "tf32_matmul": false,
  "tf32_cudnn": true,
  "keep_awake_requested": true,
  "protected_hashes": {
    "week08/day01/llvip_dataset.py": "03c2b8fef00f7bc12c68f6284a5b25a5d48003b5f52287211d1b631aa7d31c1a",
    "week08/day02/paired_letterbox.py": "43c797a374ccd5189775db18d8663957f5a0f1b79b5f2796585b9fe89698713d",
    "week08/day02/paired_horizontal_flip.py": "c68ffec6618c87b732be85d4ad60b5225c699d917b1961534d4add9030b2aa6c",
    "week08/day03/ir_common.py": "6cf2ce268feb314d937a9476f88e1db53722eb9aa791571efe44af0f07abe73e",
    "week08/day03/train_ir.py": "f2f85eeb356fefb8f9a3646d6dd134a255051918e87a404d1645f54cd2df108e",
    "week08/day03/evaluate_ir.py": "e1fa78925596a5fff40e696f6d52ad814b489cb5451acaa86ec5c7d0bc738b67",
    "week08/day03/predict_ir.py": "f27c87ce2830aa5bdfea5cd5a73120259f077f418851d30071946b3727d83917",
    "week08/day03/outputs/ir_baseline_run01/best.pth": "9cafdba85453a29ea4d0b25b491617515832faf602690de916eadfcfc9c7cfa2",
    "week08/day03/outputs/ir_baseline_run01/completed.json": "fbd3e9dc2df455e1d261e82e09992c5f5e8613394a4c315740514ce17da74b12",
    "week08/day03/outputs/ir_baseline_run01/config.json": "532e6db1114d067557a479cdf7f131dd4ae5d81f167a626161b12fb9153f63fe",
    "week08/day03/outputs/ir_baseline_run01/console_1793326681737098000.txt": "a6587232158ae87425f10c22288a4bf302744cd928fc055db196ec146867e14a",
    "week08/day03/outputs/ir_baseline_run01/DAY03_IR_TRAINING_GPT_HANDOFF.md": "ff053c0e177b1bfa348bb1d308025f123d011df01bdf171689a80303485c7a2c",
    "week08/day03/outputs/ir_baseline_run01/last.pth": "d1055247066cdd8e5c4e565e58419440b5a028a0042804feaf583838f02263f2",
    "week08/day03/outputs/ir_baseline_run01/loss_curve.png": "edc3236ff052fd00b252b5d1c96261c9d7a2b84be4ea083a04620a5caf44b281",
    "week08/day03/outputs/ir_baseline_run01/prediction_examples/false_positive_190040.png": "5ce524be55f31e705d2b2eb4fc4e35020f53cd0d58d40f504df7d61f30a3077a",
    "week08/day03/outputs/ir_baseline_run01/prediction_examples/false_positive_190474.png": "03092876c5c88a1511c213881f5d5bc98361bbe6894a83ae83d8958d21427c87",
    "week08/day03/outputs/ir_baseline_run01/prediction_examples/false_positive_190513.png": "af70761fad2133f3534eb29e3b38e5ec6f6e5f7fd60b19b5cd1eee1e47e04057",
    "week08/day03/outputs/ir_baseline_run01/prediction_examples/miss_190040.png": "52768483d4dcec18e38dd841cf6420d5558a65e0bd2bbe6ed090baf03722d021",
    "week08/day03/outputs/ir_baseline_run01/prediction_examples/miss_190080.png": "7bd2f717b6d14f945158d0d83807171d8e0cf6425de7a48e7e47ded9d64c062c",
    "week08/day03/outputs/ir_baseline_run01/prediction_examples/miss_190119.png": "08675e2159d9d8d622b317f8cb33e769791ed880d801a0f821026221582af813",
    "week08/day03/outputs/ir_baseline_run01/prediction_examples/selection.json": "1f3e431500f87744ddcc62aa67e2cd14d893a9a20d89a374c3c8ac8023f6f3f9",
    "week08/day03/outputs/ir_baseline_run01/prediction_examples/success_190001.png": "2da7f4a3497db38253cb3a05b970327f73295a8c111a586b89d3e119034609b0",
    "week08/day03/outputs/ir_baseline_run01/prediction_examples/success_190159.png": "6aa6f97ac8c9281a9d633c06d7845f1cf73d484f696131a94056ea741f4ee10b",
    "week08/day03/outputs/ir_baseline_run01/prediction_examples/success_190199.png": "1ee6fbcf8890d4b7119f7f0190967d53c00c72131ef51ba3e5c40c063008d85f",
    "week08/day03/outputs/ir_baseline_run01/smoke_test.json": "7850d1ab1f974f7e58d06b4bcb1e060597ddd86e4c3f4cb7e3ca9c8a4fa9e428",
    "week08/day03/outputs/ir_baseline_run01/test_metrics.json": "a70bbd68db0fe1ec41e3a8f32151d6fd6701d26f1b84d56953c37e4a15bfb5d2",
    "week08/day03/outputs/ir_baseline_run01/test_predictions.pth": "79ae89f650520a80ada1996b0cb9c8f2079846ff62b8d07cd29b7d5e4fae3b78",
    "week08/day03/outputs/ir_baseline_run01/test_started.json": "ffd28ec3a01095e815f52af752a30f24ca582983d1b8aacc65a65ab5dfce05b5",
    "week08/day03/outputs/ir_baseline_run01/training_log.csv": "91ad4edd6e6d344d4bd45115b2b068c0e18c372e16b9cf94032d21a60838c1f9",
    "week08/day03/outputs/ir_baseline_run01/val_epoch_01.json": "9bf5cda066402191a365ca252ebb241ce4be5e63d0828557fa540b8471205c48",
    "week08/day03/outputs/ir_baseline_run01/val_epoch_02.json": "cb71c298058275d0674bd35a3375690d9f2114e2ff2dc8886e1dc08a9517cf11",
    "week08/day03/outputs/ir_baseline_run01/val_epoch_03.json": "92622b5125af9303c916e45f428021d5f463ac7098494f09c244d079cbe9c562",
    "week08/day03/outputs/ir_baseline_run01/val_epoch_04.json": "6771cc3838ca23fd698d1f290cd3433c054c28619b2452ef838862ec55972b57",
    "week08/day03/outputs/ir_baseline_run01/val_epoch_05.json": "d17588f026e3fc4af8c680bd2ea2418daaa7d8400fc4bd98d734c05e7fb19e71",
    "week08/day03/outputs/ir_baseline_run01/val_epoch_06.json": "108c47086fb968d4c2b44ef9cd5d6d4303a4744f57fb1747acce2010626df8a9",
    "week08/day03/outputs/ir_baseline_run01/val_epoch_07.json": "0fe643401132a9ca3afc58b2b6ea4908643be3c524f51dee8d658aed5418a4a4",
    "week08/day03/outputs/ir_baseline_run01/val_epoch_08.json": "ac3b4d98a1f556d71d9035b3586a90f72aec2df940ba21f8cfff132c03df5e13",
    "week08/day03/outputs/ir_baseline_run01/val_epoch_09.json": "65d0d4be712d0f5a020dd14ce4a453655232e769a5dd44bc9e032b59d5a16665",
    "week08/day03/outputs/ir_baseline_run01/val_epoch_10.json": "f5e65d1b14b5e4f1cd05b45f6a31553024b972bc702ab540813ef37e37555a83",
    "week08/day03/outputs/ir_baseline_run01_launch_1793326678503699800.log": "6b72e566afb9ef6ea24abe7acd9b411eef6b4ed595d3b5de460c864710f68250"
  },
  "day04_runtime_hashes": {
    "check_day04.py": "6b7cb0dca2646d8195ada5abcadd5c626bb56388506bc489deefeb7b8591953d",
    "evaluate_rgb.py": "da291eb3236fdff68dd8ce90d0c261d414b6412b675d3f25e748d526b15baed5",
    "monitor_rgb.ps1": "165bdd79de7caafd4198e3400ebf73b5523485bee6f2b66a55895dd309ceeea9",
    "predict_rgb.py": "2e9e31ca29b8fe4c9596070ad4302f00239dabb68897eb1106847b4ba58e4c1d",
    "preflight_rgb.py": "6fe014c763ff4ace0912e715b5a9e929ea78e2a2310c6490f672cd17948da40e",
    "protected_inputs.json": "c05abd0d5f6879a09569f011273a84fd7031a62a44da86a4f9fe5877dac9507f",
    "REMOTE_READONLY_CHECK.ps1": "8c03b7d9a00482c93cffd407d88cbbfe3a63cd75bee6f809715ed572a2be0a57",
    "report_rgb.py": "73a981a3021ebcedd0102caa5de56646aae633e35733bbb55bc33fa89239c601",
    "rgb_common.py": "eb252028e35b1cb109a0ac0885a9ac409fb9fe969f0436a97600dbe4efaf972b",
    "start_rgb_background.ps1": "af5c278884d3d34493b5d21f4e3fc8ac4235be9eed5d43a012c63ef1bf0a477d",
    "train_rgb.py": "23c019da273658b42a42b45cdb14aa60e386954e1dac84983daa096273863d88"
  },
  "command": "D:\\cv-server-env\\Scripts\\python.exe -B D:\\vision-learning-server\\week08\\day04\\train_rgb.py --data-root D:\\vision-learning-server\\data\\LLVIP_1000 --output D:\\vision-learning-server\\week08\\day04\\outputs\\rgb_baseline_run01 --device cuda:0 --keep-awake"
}
```

## 十轮Loss与验证指标

|Epoch|total|classifier|box_reg|objectness|rpn_box|lr|val P|val R|val AP50|val mAP50:95|seconds|peak allocated bytes|peak reserved bytes|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|1|0.506035|0.182500|0.292804|0.022808|0.007924|0.001000|0.715942|0.593750|0.642439|0.244894|128.296562|395375104|484442112|
|2|0.561106|0.195942|0.344784|0.013746|0.006635|0.001000|0.696237|0.622596|0.637320|0.238748|128.024973|394217984|486539264|
|3|0.609644|0.203031|0.388923|0.011541|0.006149|0.001000|0.779762|0.629808|0.665828|0.246970|129.746651|392636928|486539264|
|4|0.607672|0.199229|0.392790|0.009913|0.005740|0.001000|0.677184|0.670673|0.676349|0.264635|132.265144|394009088|486539264|
|5|0.634413|0.206509|0.412777|0.009502|0.005624|0.001000|0.713137|0.639423|0.670006|0.263865|134.888796|393999872|486539264|
|6|0.629659|0.198611|0.417487|0.008340|0.005221|0.001000|0.759104|0.651442|0.661033|0.251245|136.449270|381565952|486539264|
|7|0.620926|0.197094|0.410201|0.008464|0.005165|0.001000|0.777448|0.629808|0.666329|0.264234|136.906467|394217984|486539264|
|8|0.602126|0.190048|0.399625|0.007469|0.004984|0.001000|0.756233|0.656250|0.670617|0.260349|137.308055|379776000|486539264|
|9|0.623210|0.187932|0.423306|0.007139|0.004833|0.001000|0.661058|0.661058|0.643049|0.242347|138.724401|384073728|486539264|
|10|0.619015|0.190770|0.416792|0.006698|0.004755|0.001000|0.770349|0.637019|0.651752|0.255090|139.098995|381775872|486539264|

## 最佳模型与正式test

最佳Epoch=4，按val mAP50:95选出；best.pth为最终模型，last.pth为第10轮状态。

```json
{
  "precision": 0.4917763157894737,
  "recall": 0.8591954022988506,
  "AP50": 0.8001258873790847,
  "mAP50_95": 0.36830385695197043,
  "TP": 299,
  "FP": 309,
  "FN": 49,
  "score_threshold": 0.5,
  "protocol": "single_person_101pt_all_area_max100_no_ignore; not official COCO API"
}
```

评估与IR共用同一函数：单person类101点插值，IoU=.50:.05:.95；P/R为score≥.5、IoU≥.5，每图最多100预测，模型score下限.001。没有COCO crowd/ignore或面积分层，不能声称是官方COCO API完整指标。

## RGB与IR实际对照

|指标|IR历史实测|RGB本次实测|RGB减IR|
|---|---:|---:|---:|
|precision|0.850416|0.491776|-0.358639|
|recall|0.882184|0.859195|-0.022989|
|AP50|0.898907|0.800126|-0.098781|
|mAP50_95|0.471626|0.368304|-0.103322|

```json
{
  "fair_settings_match": true,
  "changed_model_input": "infrared -> visible",
  "protocol": "single_person_101pt_all_area_max100_no_ignore; not official COCO API",
  "IR": {
    "precision": 0.850415512465374,
    "recall": 0.882183908045977,
    "AP50": 0.8989071207511944,
    "mAP50_95": 0.4716263012372224,
    "TP": 307,
    "FP": 54,
    "FN": 41,
    "score_threshold": 0.5,
    "protocol": "single_person_101pt_all_area_max100_no_ignore; not official COCO API"
  },
  "RGB": {
    "precision": 0.4917763157894737,
    "recall": 0.8591954022988506,
    "AP50": 0.8001258873790847,
    "mAP50_95": 0.36830385695197043,
    "TP": 299,
    "FP": 309,
    "FN": 49,
    "score_threshold": 0.5,
    "protocol": "single_person_101pt_all_area_max100_no_ignore; not official COCO API"
  },
  "RGB_minus_IR": {
    "precision": -0.3586391966759003,
    "recall": -0.02298850574712641,
    "AP50": -0.09878123337210964,
    "mAP50_95": -0.10332244428525195,
    "TP": -8,
    "FP": 255,
    "FN": 8
  },
  "IR_best_epoch": 9,
  "RGB_best_epoch": 4,
  "IR_epoch_seconds_sum": 1324.373884600005,
  "RGB_epoch_seconds_sum": 1341.7093137999618,
  "IR_peak_allocated_bytes": 393544192.0,
  "RGB_peak_allocated_bytes": 395375104,
  "IR_peak_reserved_bytes": 499122176.0,
  "RGB_peak_reserved_bytes": 486539264,
  "limitation": "单次、单seed、场景隔离子集；耗时非稳定基准；不推断融合成绩"
}
```

耗时、显存为单次运行记录；两次子集和协议一致但不代表统计显著结论，未做多seed实验，不能推断融合效果。

## 显存与计时

```json
{
  "epoch_seconds_sum": 1341.7093137999618,
  "run_seconds_before_report": 1403.191015400007,
  "max_epoch_peak_allocated_bytes": 395375104,
  "max_epoch_peak_reserved_bytes": 486539264
}
```

epoch_seconds与IR定义一致，为该轮训练和验证，不含随后保存/绘图或最终test。GPU峰值为PyTorch allocator统计，不等于整卡占用。run计时从本脚本入口起至产物检查前，不含外部启动器，真实进程结束时间和退出码另见exit_status.json。

## 预测案例

```json
{
  "success": [
    "200183",
    "200228",
    "200258"
  ],
  "miss": [
    "190040",
    "190080",
    "190119"
  ],
  "false_positive": [
    "190001",
    "190159",
    "190199"
  ]
}
```

绿色GT、红色预测、person及置信分数；按真实TP/FP/FN分类，优先IR已展示的编号；不足3张时记录缺额，类别可以重叠。选择依据见prediction_examples/case_details.json。

## 运行命令

```cmd
D:\cv-server-env\Scripts\python.exe -B D:\vision-learning-server\week08\day04\train_rgb.py --data-root D:\vision-learning-server\data\LLVIP_1000 --output D:\vision-learning-server\week08\day04\outputs\rgb_baseline_run01 --device cuda:0 --keep-awake
```

输出位置：`D:\vision-learning-server\week08\day04\outputs\rgb_baseline_run01`。本地对应路径：`D:\python\test\vision-learning\week08\day04\outputs\rgb_baseline_run01`。

复现须选择新的空run目录，不能覆盖本次实验；test仅正式评估一次，图像由缓存生成。独立val评估及缓存出图命令见Day04 README。

## 实际问题与文件

本报告仅在全流程成功后生成；成功并不表示没有任何warning。完整console日志保留启动、检查及warning，异常记录为failure_*.json，无OOM/非有限数值时才可能运行到此。初始本地GPU/训练状态不得冒充服务器实测。

主要产物：config.json、preflight.json、smoke_test.json、initialization.json、training_log.csv、val_epoch_01.json至val_epoch_10.json、逐轮epoch_NN.pth、best.pth、last.pth、loss_curve.png、test_started.json、test_metrics.json、test_predictions.pth、prediction_examples/、rgb_vs_ir.json、result_checks.json、protected_after.json、completed.json、exit_status.json及完整console日志。

## 交给ChatGPT学习

请全程中文，每次一个明确步骤，不主动测验；代码完成不代表已掌握。请解释公平模态对照、预训练与IR微调初始化的区别、四项Loss和空标注、val选模与一次test、AP阈值与定位误差、实际漏检/误检案例、峰值allocated/reserved及续训状态。请区分真实观察和原因推测；下一步先分析本次结果，不自动开展RGB+IR融合。
