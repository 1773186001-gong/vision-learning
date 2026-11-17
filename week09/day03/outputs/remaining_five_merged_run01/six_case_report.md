# Day03 六例固定验证GT：RPN筛选追踪

030269/G1复用已验收捕获，新增五张完整val图片，每张IR/Fusion各一次原eval前向，共10次新增前向。未训练、未运行test、未修改预算或阈值。

本表主口径为Day02一致的CPU float32 IoU；float64仅作独立诊断，另存于CSV和核验JSON。候选数按真实索引追踪；每阶段的最佳框可能不同。

## 十二项逐阶段观测

|案例/模型|阶段|候选数|最大IoU f32 CPU|≥0.75数|最佳ID（原flat索引）|
|---|---|---:|---:|---:|---:|
|030269/G1 IR|decoded|13500|0.793667018414|2|2201|
|030269/G1 IR|pre_nms_topn|450|0.793667018414|2|2201|
|030269/G1 IR|clipped|450|0.793667018414|2|2201|
|030269/G1 IR|small_box_filtered|450|0.793667018414|2|2201|
|030269/G1 IR|score_filtered|109|0.793667018414|2|2201|
|030269/G1 IR|layer_nms|57|0.793667018414|2|2201|
|030269/G1 IR|post_nms_topn|57|0.793667018414|2|2201|
|030269/G1 Fusion|decoded|13500|0.750537335873|1|2201|
|030269/G1 Fusion|pre_nms_topn|450|0.750537335873|1|2201|
|030269/G1 Fusion|clipped|450|0.750537335873|1|2201|
|030269/G1 Fusion|small_box_filtered|450|0.750537335873|1|2201|
|030269/G1 Fusion|score_filtered|158|0.750537335873|1|2201|
|030269/G1 Fusion|layer_nms|105|0.605291903019|0|2501|
|030269/G1 Fusion|post_nms_topn|105|0.605291903019|0|2501|
|030308/G1 IR|decoded|13500|0.878793776035|5|6636|
|030308/G1 IR|pre_nms_topn|450|0.878793776035|5|6636|
|030308/G1 IR|clipped|450|0.878793776035|5|6636|
|030308/G1 IR|small_box_filtered|450|0.878793776035|5|6636|
|030308/G1 IR|score_filtered|94|0.878793776035|4|6636|
|030308/G1 IR|layer_nms|58|0.878793776035|2|6636|
|030308/G1 IR|post_nms_topn|58|0.878793776035|2|6636|
|030308/G1 Fusion|decoded|13500|0.712219297886|0|636|
|030308/G1 Fusion|pre_nms_topn|450|0.712219297886|0|636|
|030308/G1 Fusion|clipped|450|0.712219297886|0|636|
|030308/G1 Fusion|small_box_filtered|450|0.712219297886|0|636|
|030308/G1 Fusion|score_filtered|149|0.712219297886|0|636|
|030308/G1 Fusion|layer_nms|105|0.712219297886|0|636|
|030308/G1 Fusion|post_nms_topn|105|0.712219297886|0|636|
|030205/G0 IR|decoded|13500|0.851542413235|8|3806|
|030205/G0 IR|pre_nms_topn|450|0.851542413235|8|3806|
|030205/G0 IR|clipped|450|0.851542413235|8|3806|
|030205/G0 IR|small_box_filtered|450|0.851542413235|8|3806|
|030205/G0 IR|score_filtered|55|0.851542413235|8|3806|
|030205/G0 IR|layer_nms|31|0.851542413235|2|3806|
|030205/G0 IR|post_nms_topn|31|0.851542413235|2|3806|
|030205/G0 Fusion|decoded|13500|0.729380309582|0|9506|
|030205/G0 Fusion|pre_nms_topn|450|0.729380309582|0|9506|
|030205/G0 Fusion|clipped|450|0.729380309582|0|9506|
|030205/G0 Fusion|small_box_filtered|450|0.729380309582|0|9506|
|030205/G0 Fusion|score_filtered|64|0.729380309582|0|9506|
|030205/G0 Fusion|layer_nms|43|0.696796298027|0|9492|
|030205/G0 Fusion|post_nms_topn|43|0.696796298027|0|9492|
|170100/G1 IR|decoded|13500|0.769244015217|1|13256|
|170100/G1 IR|pre_nms_topn|450|0.769244015217|1|13256|
|170100/G1 IR|clipped|450|0.769244015217|1|13256|
|170100/G1 IR|small_box_filtered|450|0.769244015217|1|13256|
|170100/G1 IR|score_filtered|121|0.578785657883|0|4601|
|170100/G1 IR|layer_nms|67|0.578785657883|0|4601|
|170100/G1 IR|post_nms_topn|67|0.578785657883|0|4601|
|170100/G1 Fusion|decoded|13500|0.845996797085|3|4901|
|170100/G1 Fusion|pre_nms_topn|450|0.845996797085|3|4901|
|170100/G1 Fusion|clipped|450|0.845996797085|3|4901|
|170100/G1 Fusion|small_box_filtered|450|0.845996797085|3|4901|
|170100/G1 Fusion|score_filtered|98|0.845996797085|3|4901|
|170100/G1 Fusion|layer_nms|57|0.845996797085|3|4901|
|170100/G1 Fusion|post_nms_topn|57|0.845996797085|3|4901|
|130126/G3 IR|decoded|13500|0.875696659088|5|3516|
|130126/G3 IR|pre_nms_topn|450|0.875696659088|5|3516|
|130126/G3 IR|clipped|450|0.875696659088|5|3516|
|130126/G3 IR|small_box_filtered|450|0.875696659088|5|3516|
|130126/G3 IR|score_filtered|239|0.875696659088|5|3516|
|130126/G3 IR|layer_nms|171|0.664702534676|0|3221|
|130126/G3 IR|post_nms_topn|150|0.664702534676|0|3221|
|130126/G3 Fusion|decoded|13500|0.846598207951|2|3521|
|130126/G3 Fusion|pre_nms_topn|450|0.846598207951|2|3521|
|130126/G3 Fusion|clipped|450|0.846598207951|2|3521|
|130126/G3 Fusion|small_box_filtered|450|0.846598207951|2|3521|
|130126/G3 Fusion|score_filtered|274|0.846598207951|2|3521|
|130126/G3 Fusion|layer_nms|201|0.846598207951|1|3521|
|130126/G3 Fusion|post_nms_topn|150|0.846598207951|1|3521|
|170455/G1 IR|decoded|13500|0.750012934208|1|10646|
|170455/G1 IR|pre_nms_topn|450|0.750012934208|1|10646|
|170455/G1 IR|clipped|450|0.750012934208|1|10646|
|170455/G1 IR|small_box_filtered|450|0.750012934208|1|10646|
|170455/G1 IR|score_filtered|133|0.750012934208|1|10646|
|170455/G1 IR|layer_nms|92|0.750012934208|1|10646|
|170455/G1 IR|post_nms_topn|92|0.750012934208|1|10646|
|170455/G1 Fusion|decoded|13500|0.674085140228|0|10642|
|170455/G1 Fusion|pre_nms_topn|450|0.674085140228|0|10642|
|170455/G1 Fusion|clipped|450|0.674085140228|0|10642|
|170455/G1 Fusion|small_box_filtered|450|0.674085140228|0|10642|
|170455/G1 Fusion|score_filtered|149|0.674085140228|0|10642|
|170455/G1 Fusion|layer_nms|104|0.674085140228|0|10642|
|170455/G1 Fusion|post_nms_topn|104|0.674085140228|0|10642|

## 逐项解释与固定候选轨迹

### 030269/G1 IR

- 高IoU候选首次出现：decoded；有达标框的阶段：['decoded', 'pre_nms_topn', 'clipped', 'small_box_filtered', 'score_filtered', 'layer_nms', 'post_nms_topn']。
- 达标框由有到全部消失：[]；解码后由无到有：[]。
- 阶段最佳flat索引依次为：['2201', '2201', '2201', '2201', '2201', '2201', '2201']。
- 解码到最终最大IoU变化（最终减解码）：0.000000000000；含真实裁剪带来的几何变化，不把差值全部解释为删框损失。
- Day02=0.7936670184135437；当前=0.7936670184135437；差值=0。

曾达标的固定flat索引为[2201, 8201]。这些ID的完整去留、分数、排名及同ID裁剪前后值见候选轨迹与clip_changes；阶段最佳框另列，不将二者混同。
- ID 2201：实际经过['decoded', 'pre_nms_topn', 'clipped', 'small_box_filtered', 'score_filtered', 'layer_nms', 'post_nms_topn']；首次删除=最终保留。
- ID 8201：实际经过['decoded', 'pre_nms_topn', 'clipped', 'small_box_filtered', 'score_filtered', 'layer_nms', 'post_nms_topn']；首次删除=最终保留。

### 030269/G1 Fusion

- 高IoU候选首次出现：decoded；有达标框的阶段：['decoded', 'pre_nms_topn', 'clipped', 'small_box_filtered', 'score_filtered']。
- 达标框由有到全部消失：['layer_nms']；解码后由无到有：[]。
- 阶段最佳flat索引依次为：['2201', '2201', '2201', '2201', '2201', '2501', '2501']。
- 解码到最终最大IoU变化（最终减解码）：-0.145245432854；含真实裁剪带来的几何变化，不把差值全部解释为删框损失。
- Day02=0.6052919030189514；当前=0.6052919030189514；差值=0。

曾达标的固定flat索引为[2201]。这些ID的完整去留、分数、排名及同ID裁剪前后值见候选轨迹与clip_changes；阶段最佳框另列，不将二者混同。
- ID 2201：实际经过['decoded', 'pre_nms_topn', 'clipped', 'small_box_filtered', 'score_filtered']；首次删除=layer_nms。

### 030308/G1 IR

- 高IoU候选首次出现：decoded；有达标框的阶段：['decoded', 'pre_nms_topn', 'clipped', 'small_box_filtered', 'score_filtered', 'layer_nms', 'post_nms_topn']。
- 达标框由有到全部消失：[]；解码后由无到有：[]。
- 阶段最佳flat索引依次为：['6636', '6636', '6636', '6636', '6636', '6636', '6636']。
- 解码到最终最大IoU变化（最终减解码）：0.000000000000；含真实裁剪带来的几何变化，不把差值全部解释为删框损失。
- Day02=0.8787937760353088；当前=0.8787937760353088；差值=0。

曾达标的固定flat索引为[636, 936, 6636, 6936, 12171]。这些ID的完整去留、分数、排名及同ID裁剪前后值见候选轨迹与clip_changes；阶段最佳框另列，不将二者混同。
- ID 636：实际经过['decoded', 'pre_nms_topn', 'clipped', 'small_box_filtered', 'score_filtered', 'layer_nms', 'post_nms_topn']；首次删除=最终保留。
- ID 936：实际经过['decoded', 'pre_nms_topn', 'clipped', 'small_box_filtered', 'score_filtered']；首次删除=layer_nms。
- ID 6636：实际经过['decoded', 'pre_nms_topn', 'clipped', 'small_box_filtered', 'score_filtered', 'layer_nms', 'post_nms_topn']；首次删除=最终保留。
- ID 6936：实际经过['decoded', 'pre_nms_topn', 'clipped', 'small_box_filtered', 'score_filtered']；首次删除=layer_nms。
- ID 12171：实际经过['decoded', 'pre_nms_topn', 'clipped', 'small_box_filtered']；首次删除=score_filtered。

### 030308/G1 Fusion

- 高IoU候选首次出现：七阶段均无；有达标框的阶段：[]。
- 达标框由有到全部消失：[]；解码后由无到有：[]。
- 阶段最佳flat索引依次为：['636', '636', '636', '636', '636', '636', '636']。
- 解码到最终最大IoU变化（最终减解码）：0.000000000000；含真实裁剪带来的几何变化，不把差值全部解释为删框损失。
- Day02=0.7122192978858948；当前=0.7122192978858948；差值=0。

曾达标的固定flat索引为[]。这些ID的完整去留、分数、排名及同ID裁剪前后值见候选轨迹与clip_changes；阶段最佳框另列，不将二者混同。

### 030205/G0 IR

- 高IoU候选首次出现：decoded；有达标框的阶段：['decoded', 'pre_nms_topn', 'clipped', 'small_box_filtered', 'score_filtered', 'layer_nms', 'post_nms_topn']。
- 达标框由有到全部消失：[]；解码后由无到有：[]。
- 阶段最佳flat索引依次为：['3806', '3806', '3806', '3806', '3806', '3806', '3806']。
- 解码到最终最大IoU变化（最终减解码）：0.000000000000；含真实裁剪带来的几何变化，不把差值全部解释为删框损失。
- Day02=0.8515424132347107；当前=0.8515424132347107；差值=0。

曾达标的固定flat索引为[3491, 3492, 3506, 3792, 3806, 9492, 9506, 9806]。这些ID的完整去留、分数、排名及同ID裁剪前后值见候选轨迹与clip_changes；阶段最佳框另列，不将二者混同。
- ID 3491：实际经过['decoded', 'pre_nms_topn', 'clipped', 'small_box_filtered', 'score_filtered']；首次删除=layer_nms。
- ID 3492：实际经过['decoded', 'pre_nms_topn', 'clipped', 'small_box_filtered', 'score_filtered']；首次删除=layer_nms。
- ID 3506：实际经过['decoded', 'pre_nms_topn', 'clipped', 'small_box_filtered', 'score_filtered']；首次删除=layer_nms。
- ID 3792：实际经过['decoded', 'pre_nms_topn', 'clipped', 'small_box_filtered', 'score_filtered']；首次删除=layer_nms。
- ID 3806：实际经过['decoded', 'pre_nms_topn', 'clipped', 'small_box_filtered', 'score_filtered', 'layer_nms', 'post_nms_topn']；首次删除=最终保留。
- ID 9492：实际经过['decoded', 'pre_nms_topn', 'clipped', 'small_box_filtered', 'score_filtered']；首次删除=layer_nms。
- ID 9506：实际经过['decoded', 'pre_nms_topn', 'clipped', 'small_box_filtered', 'score_filtered']；首次删除=layer_nms。
- ID 9806：实际经过['decoded', 'pre_nms_topn', 'clipped', 'small_box_filtered', 'score_filtered', 'layer_nms', 'post_nms_topn']；首次删除=最终保留。

### 030205/G0 Fusion

- 高IoU候选首次出现：七阶段均无；有达标框的阶段：[]。
- 达标框由有到全部消失：[]；解码后由无到有：[]。
- 阶段最佳flat索引依次为：['9506', '9506', '9506', '9506', '9506', '9492', '9492']。
- 解码到最终最大IoU变化（最终减解码）：-0.032584011555；含真实裁剪带来的几何变化，不把差值全部解释为删框损失。
- Day02=0.6967962980270386；当前=0.6967962980270386；差值=0。

曾达标的固定flat索引为[]。这些ID的完整去留、分数、排名及同ID裁剪前后值见候选轨迹与clip_changes；阶段最佳框另列，不将二者混同。

### 170100/G1 IR

- 高IoU候选首次出现：decoded；有达标框的阶段：['decoded', 'pre_nms_topn', 'clipped', 'small_box_filtered']。
- 达标框由有到全部消失：['score_filtered']；解码后由无到有：[]。
- 阶段最佳flat索引依次为：['13256', '13256', '13256', '13256', '4601', '4601', '4601']。
- 解码到最终最大IoU变化（最终减解码）：-0.190458357334；含真实裁剪带来的几何变化，不把差值全部解释为删框损失。
- Day02=0.5787856578826904；当前=0.5787856578826904；差值=0。

曾达标的固定flat索引为[13256]。这些ID的完整去留、分数、排名及同ID裁剪前后值见候选轨迹与clip_changes；阶段最佳框另列，不将二者混同。
- ID 13256：实际经过['decoded', 'pre_nms_topn', 'clipped', 'small_box_filtered']；首次删除=score_filtered。

### 170100/G1 Fusion

- 高IoU候选首次出现：decoded；有达标框的阶段：['decoded', 'pre_nms_topn', 'clipped', 'small_box_filtered', 'score_filtered', 'layer_nms', 'post_nms_topn']。
- 达标框由有到全部消失：[]；解码后由无到有：[]。
- 阶段最佳flat索引依次为：['4901', '4901', '4901', '4901', '4901', '4901', '4901']。
- 解码到最终最大IoU变化（最终减解码）：0.000000000000；含真实裁剪带来的几何变化，不把差值全部解释为删框损失。
- Day02=0.8459967970848083；当前=0.8459967970848083；差值=0。

曾达标的固定flat索引为[4901, 10901, 13256]。这些ID的完整去留、分数、排名及同ID裁剪前后值见候选轨迹与clip_changes；阶段最佳框另列，不将二者混同。
- ID 4901：实际经过['decoded', 'pre_nms_topn', 'clipped', 'small_box_filtered', 'score_filtered', 'layer_nms', 'post_nms_topn']；首次删除=最终保留。
- ID 10901：实际经过['decoded', 'pre_nms_topn', 'clipped', 'small_box_filtered', 'score_filtered', 'layer_nms', 'post_nms_topn']；首次删除=最终保留。
- ID 13256：实际经过['decoded', 'pre_nms_topn', 'clipped', 'small_box_filtered', 'score_filtered', 'layer_nms', 'post_nms_topn']；首次删除=最终保留。

### 130126/G3 IR

- 高IoU候选首次出现：decoded；有达标框的阶段：['decoded', 'pre_nms_topn', 'clipped', 'small_box_filtered', 'score_filtered']。
- 达标框由有到全部消失：['layer_nms']；解码后由无到有：[]。
- 阶段最佳flat索引依次为：['3516', '3516', '3516', '3516', '3516', '3221', '3221']。
- 解码到最终最大IoU变化（最终减解码）：-0.210994124413；含真实裁剪带来的几何变化，不把差值全部解释为删框损失。
- Day02=0.6647025346755981；当前=0.6647025346755981；差值=0。

曾达标的固定flat索引为[3216, 3516, 3521, 9216, 9516]。这些ID的完整去留、分数、排名及同ID裁剪前后值见候选轨迹与clip_changes；阶段最佳框另列，不将二者混同。
- ID 3216：实际经过['decoded', 'pre_nms_topn', 'clipped', 'small_box_filtered', 'score_filtered']；首次删除=layer_nms。
- ID 3516：实际经过['decoded', 'pre_nms_topn', 'clipped', 'small_box_filtered', 'score_filtered']；首次删除=layer_nms。
- ID 3521：实际经过['decoded', 'pre_nms_topn', 'clipped', 'small_box_filtered', 'score_filtered']；首次删除=layer_nms。
- ID 9216：实际经过['decoded', 'pre_nms_topn', 'clipped', 'small_box_filtered', 'score_filtered']；首次删除=layer_nms。
- ID 9516：实际经过['decoded', 'pre_nms_topn', 'clipped', 'small_box_filtered', 'score_filtered']；首次删除=layer_nms。

### 130126/G3 Fusion

- 高IoU候选首次出现：decoded；有达标框的阶段：['decoded', 'pre_nms_topn', 'clipped', 'small_box_filtered', 'score_filtered', 'layer_nms', 'post_nms_topn']。
- 达标框由有到全部消失：[]；解码后由无到有：[]。
- 阶段最佳flat索引依次为：['3521', '3521', '3521', '3521', '3521', '3521', '3521']。
- 解码到最终最大IoU变化（最终减解码）：0.000000000000；含真实裁剪带来的几何变化，不把差值全部解释为删框损失。
- Day02=0.8465982079505920；当前=0.8465982079505920；差值=0。

曾达标的固定flat索引为[3516, 3521]。这些ID的完整去留、分数、排名及同ID裁剪前后值见候选轨迹与clip_changes；阶段最佳框另列，不将二者混同。
- ID 3516：实际经过['decoded', 'pre_nms_topn', 'clipped', 'small_box_filtered', 'score_filtered']；首次删除=layer_nms。
- ID 3521：实际经过['decoded', 'pre_nms_topn', 'clipped', 'small_box_filtered', 'score_filtered', 'layer_nms', 'post_nms_topn']；首次删除=最终保留。

### 170455/G1 IR

- 高IoU候选首次出现：decoded；有达标框的阶段：['decoded', 'pre_nms_topn', 'clipped', 'small_box_filtered', 'score_filtered', 'layer_nms', 'post_nms_topn']。
- 达标框由有到全部消失：[]；解码后由无到有：[]。
- 阶段最佳flat索引依次为：['10646', '10646', '10646', '10646', '10646', '10646', '10646']。
- 解码到最终最大IoU变化（最终减解码）：0.000000000000；含真实裁剪带来的几何变化，不把差值全部解释为删框损失。
- Day02=0.7500129342079163；当前=0.7500129342079163；差值=0。

曾达标的固定flat索引为[10646]。这些ID的完整去留、分数、排名及同ID裁剪前后值见候选轨迹与clip_changes；阶段最佳框另列，不将二者混同。
- ID 10646：实际经过['decoded', 'pre_nms_topn', 'clipped', 'small_box_filtered', 'score_filtered', 'layer_nms', 'post_nms_topn']；首次删除=最终保留。

### 170455/G1 Fusion

- 高IoU候选首次出现：七阶段均无；有达标框的阶段：[]。
- 达标框由有到全部消失：[]；解码后由无到有：[]。
- 阶段最佳flat索引依次为：['10642', '10642', '10642', '10642', '10642', '10642', '10642']。
- 解码到最终最大IoU变化（最终减解码）：0.000000000000；含真实裁剪带来的几何变化，不把差值全部解释为删框损失。
- Day02=0.6740851402282715；当前=0.6740851402282715；差值=0。

曾达标的固定flat索引为[]。这些ID的完整去留、分数、排名及同ID裁剪前后值见候选轨迹与clip_changes；阶段最佳框另列，不将二者混同。

## 解码质量与筛选变化的对照

|案例|IR解码最大IoU|Fusion解码最大IoU|IR最终|Fusion最终|IR筛选变化|Fusion筛选变化|
|---|---:|---:|---:|---:|---:|---:|
|030269/G1|0.793667018|0.750537336|0.793667018|0.605291903|0.000000000|-0.145245433|
|030308/G1|0.878793776|0.712219298|0.878793776|0.712219298|0.000000000|0.000000000|
|030205/G0|0.851542413|0.729380310|0.851542413|0.696796298|0.000000000|-0.032584012|
|170100/G1|0.769244015|0.845996797|0.578785658|0.845996797|-0.190458357|0.000000000|
|130126/G3|0.875696659|0.846598208|0.664702535|0.846598208|-0.210994124|0.000000000|
|170455/G1|0.750012934|0.674085140|0.750012934|0.674085140|0.000000000|0.000000000|

## 重复模式与证据边界

达标候选全部消失的实际阶段分组：{"layer_nms": ["030269/G1 Fusion", "130126/G3 IR"], "score_filtered": ["170100/G1 IR"]}。
七阶段均未达标的项：['030308/G1 Fusion', '030205/G0 Fusion', '170455/G1 Fusion']。这些是按规则选择的六例，不能估计全val发生比例。
操作索引与同ID坐标能证实在哪一步删除、裁剪后是否达标；这些观察不能证实融合特征、anchor回归或某个NMS抑制者就是根因。没有记录抑制者配对，也不从两个框重叠猜测实际抑制关系。
解码阶段已有差异时，上游候选生成机制仍需另行验证；候选存在却筛选丢失时，后续可优先只读检查对应分数排名与筛选去留。当前证据不支持直接修改融合模块。
裁剪范围为含Letterbox padding的完整640×640；GT完整位于范围内。对有限有效框，同ID裁剪IoU在精确几何下不降，float64核验atol=1e-6。未过Top-N的框不做反事实裁剪。

## 边界样本170455/G1 IR

历史IoU=0.7500129342079163，当前=0.7500129342079163，实际差=0；历史覆盖=True，当前覆盖=True。0.75阈值没有附加容差。

所有核验见merged_verification.json，各模型原始捕获与原state/config/恢复证据保存在子目录；全程来源、真实命令、退出码和受保护文件SHA前后值见run_metadata.json、process_status.json及run.log。完成本步骤后停止，不启动新实验。
