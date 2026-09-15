# 硕士论文大纲（整合简版）

**题目**：Posture and Activity Detection in Manufacturing Environment Using 3D Vision and AI

**作者**：Fanbo Meng
**机构**：Chalmers University of Technology，合作方 Viscando AB

本版以 2026 年 6 月 2 日的大纲为基础，保留原版“评估框架—实验结果”的清晰结构，并结合后续实验增加独立的 GPU 与实时实现章节。

---

## 第一章：引言

- 介绍制造环境中的人体工程学问题，以及 RULA、REBA 等人工评估方法的作用和局限。
- 说明使用相机自动分析人体姿态、关节角度和关键危险姿势的意义。
- 提出本文的实际定位：不追求医疗或实验室级精度，而是研究系统能否提供稳定、可解释并足以支持人体工程学判断的结果。
- 说明遮挡、拍摄距离、相机视角和计算速度等实际困难。
- 明确项目边界：本文完成并评估的是更大工业人体工程学系统中的视觉处理核心，包括姿态估计、双目重建、统一角度评价和计算性能验证；实时采集、传输、用户界面、风险提示及完整的人体工程学决策链路不属于本文已经实现的范围。

### Research Questions

- **RQ1 — Accuracy:** *How closely do stereo-based SKT and monocular FastSAM3D agree with an Xsens-derived reference in estimating ergonomically relevant human posture?*
- **RQ2 — Robustness:** *How does the performance of the two approaches vary across different distances, viewpoints, and occlusion conditions?*
- **RQ3 — Deployment feasibility:** *How feasible is the real-time deployment of the selected SKT pipeline for industrial ergonomic assessment?*

其中，RQ1 中的 Accuracy 表示与 Xsens-derived reference 的一致程度，并不表示由绝对 Ground Truth 验证的物理准确性。

- 最后说明研究范围和主要贡献；活动识别暂不作为单独的实验任务。

## 第二章：背景与相关工作

- 介绍人体工程学评估为什么依赖关节角度，并说明本文不实现完整的 RULA/REBA 自动评分。
- 说明为什么核心精度指标是关节角度而非 3D 关节位置误差（MPJPE）：工人的工效学风险由姿态决定，RULA/REBA 等方法直接按关节角度是否落入危险区间分档，角度误差与风险等级一致性有直接对应关系；而某个关节在三维空间中偏离了多少厘米，本身并不直接对应风险高低。
- 补充技术层面的原因：MPJPE 还需要先把 SKT（立体几何度量）、FastSAM3D（单目 + 人体先验尺度）、Xsens（IMU + 运动学模型）三套原理不同的系统对齐到同一坐标系和尺度，这是一个比角度计算更难、某种程度上不适定的附加问题；角度由局部三个关节点计算得到，天然不受平移、旋转和尺度差异影响，不需要先解决这一问题。
- 介绍 YOLO 在人体关键点检测中的作用，重点放在本文实际使用的 YOLOv8m 和 YOLO11l。
- 以足以支撑 SKT 实现的数学深度介绍双目视觉：针孔投影 $s\tilde{x}=K[R\mid t]\tilde{X}$、内外参数与镜头畸变、极线几何、rectification、DLT triangulation、reprojection error 和 positive-depth check。推导 $Z=fB/d$ 以及一阶误差关系 $|\delta Z|\approx Z^2|\delta d|/(fB)$，为后文的距离限制提供理论基础，并配置一张精简的双目几何示意图。
- 介绍单目三维人体姿态估计和人体先验，并引出 FastSAM3D。
- 介绍 IMU 人体运动捕捉和运动学模型，并引出 Xsens。
- 从原理上区分三类数据来源：FastSAM3D 是“视觉观测 + 强人体先验”，SKT 是“视觉观测 + 几何测量”，Xsens 是“身体传感器 + 运动学模型”。
- 介绍 Xsens 的工作方式及局限，明确它是外部参考系统，不是绝对 Ground Truth。

## 第三章：实验数据与系统方法

这一章回答一个连续的问题：**同一动作场景如何被三种系统记录，并经过各自的处理路径，最终形成能够在统一框架下同步和比较的关节角度序列？** 本章把“数据从哪里来”和“系统如何处理数据”连成一条完整链路，但不在这里定义评价指标或报告实验结果。

### 3.1 研究设计、整体流程与系统角色

- 以 recording 为基本单位说明实验设计：在同一动作或对应场景中获取双目视频、单目视频和 Xsens 数据；逐项注明哪些 recording 确实是同步采集，哪些只是条件相近或后续对齐，不能笼统称为完全同步。
- 用一张“总—分—总”流程图表示：顶部仅放三类系统所观察的动作或场景；随后按 Viscando 双目-SKT、单目 FastSAM3D 和穿戴式 Xsens 分开采集和处理，因为设备、标定与源时间信息并不共享；最后再经过共同关节映射、有效性处理、时间对齐和配置追溯形成 evaluation-ready angle sequences。下一章才从这些序列分出绝对角度和 motion 两条评价路径。
- 明确三种系统的性质：SKT 是“视觉观测 + 几何测量”，FastSAM3D 是“视觉观测 + 人体先验”，Xsens 是“身体传感器 + 运动学模型”。SKT 和 FastSAM3D 是被比较的视觉方法；Xsens 是同步外部参考，不作为第三个候选视觉方法参与排名，也不被视为绝对 Ground Truth。
- 区分原始或接收到的采集文件与项目生成的处理结果。相机视频及 Xsens 导出的 MVNX 是本项目接收到的输入；MVNX 已包含厂商的 sensor fusion，并非未经处理的原始 IMU 信号。SKT 的 NPZ、FastSAM3D/EasyErgo 的 TRC 以及后续角度文件属于系统输出，而不是新的原始数据集。

### 3.2 数据集、采集环境与用途

#### 3.2.1 数据集与录制场景

- 介绍 2025 年数据和 2026 年 Assar 数据，包括被试、任务、动作、录制时长、拍摄距离、视角和遮挡情况。
- 说明双目相机、Webcam 和 Xsens 在每个 recording 中的实际采集关系，以及各数据集在论文中的用途。
- 第一次出现 A255/A257 时，明确它们是 2026 年现场两台完整 Viscando 双目传感器的设备编号；每台设备各自包含左右相机并独立形成一套 SKT 输入，并非把四个视角合并成一套重建系统。
- 区分用于主要方法比较、消融实验、距离/视角/遮挡分析和部署测试的 recording，避免把不同实验条件的数据直接混在一起。

#### 3.2.2 采集设备与原始文件

- 双目系统：每条 SKT 输入使用一台 Viscando 双目传感器，说明其左右相机视频、分辨率、名义帧率、硬件 frame ID 或时间信息；单目系统：输入 FastSAM3D 的 Webcam 视频及其分辨率、帧率和与其他输入的录制关系；Xsens：MVNX 文件、名义采样率和 calibration 记录。增加一张精简的双栏硬件图，分别展示 Viscando 传感器和被试穿戴 Xsens 的状态。
- 用一张“采集信息表”汇总 recording、设备、输入文件、分辨率、帧率、时长、同步关系和缺失模态。表中明确标识 MVNX 为接收到的 Xsens 导出文件，并将项目生成的 NPZ、TRC 和角度结果列为后续系统输出。

#### 3.2.3 双目相机标定与验证

- 说明相机位置、41~cm 的物理双目基线、内外参数、分辨率和录制设置，以及标定数据来自何时、用于哪些 recording。指出标定恢复出的约 41.00--41.27~cm baseline 与物理结构一致，可作为尺度合理性的内部检查，但不是人体关节位置精度的独立证据。
- 固定三套已存档标定：2025 相机对使用 `camera_params_2025.npz`；2026 A255 使用 SensorCalibration 生成的 `camera_params_A255.npz`；A257 由于 SensorCalibration 有效帧不足，使用 Sensor + SiteCalibration 生成的 `camera_params_A257.npz`。所有实验配置引用对应的固定文件，不在姿态结果上反向调整标定。
- 用一张精简表报告图像尺寸、有效标定帧数、基线长度、rectified vertical disparity 和 rigid-alignment RMSE。这些是标定靶在已接受标定帧上的内部一致性指标，不是人体场景中的独立位置 Ground Truth。完整的 $K,D,R,T,E,F$ 矩阵和搜索配置移入附录。
- 本节只交代实验配置和标定结果；标定参数如何用于矫正、投影矩阵和三角测量放在 3.3.1。

#### 3.2.4 2025 录制的活动分段标注

- 定义第 5.1 节使用的四类人工场景标注：Baseline（17--32 s 与 220--240 s 的正常行走）；Dynamic Action（66--69 s 的短时深蹲）；Occlusion（32--62 s、87--97 s、130--140 s 与 164--170 s）；Environmental Interference（140--160 s 的椅子互动与 214--218 s 的箱子搬运）。时间边界以整秒近似报告。
- 历史记录曾在 156--160 s 另加 `Squatting (Check)` 标签，但它与 140--160 s 的 chair-interaction 区间重叠；当前结果继承实际分析顺序，将该重叠段唯一归入 Environmental Interference，不重复计入 Dynamic Action。
- 说明这些标签是为场景分组分析对录像进行的人工视觉标注，而不是设备自动产生的 Ground Truth。

#### 3.2.5 数据用途与证据边界

- 用一张简表说明每个 recording 用于角度比较、运动比较、消融实验、鲁棒性分析还是部署测试，并对应到相关 Research Question。
- 说明哪些条件属于受控变化，哪些因素可能相互混杂。标出用于单自由度肘关节屈伸分析的 recording；为什么选择该动作由第 4.2 节解释。
- 说明被试数量、动作覆盖、缺失帧、录制长度和可用 recording 等限制。
- 说明距离、视角、动作和数据年份之间可能存在的混杂关系，因此哪些结果属于受控比较，哪些只能作为观察性证据。
- 说明实验没有采集独立的光学 Ground Truth；Xsens 是同步外部参考，而不是绝对真值。

### 3.3 三种系统的处理路径与输出

三小节采用统一叙述顺序：**输入是什么 → 如何处理 → 为什么需要这一步 → 输出是什么 → 有哪些可观察的质量信息或方法限制。** SKT 部分严格对应 Figure 3.3 的六个编号步骤：(1) 用 hardware ID 配对左右帧；(2) 检测 COCO-17 关键点并分别跟踪同一目标；(3) 对 2D 点做 rectification 并检查 stereo pair；(4) 进行 soft epipolar adjustment、confidence-weighted DLT 和 disparity/positive-depth/reprojection 检查；(5) 用邻近 2D 观测恢复短时失败并重新三角化；(6) 保存 3D joints 以及质量和追溯信息。增加一张 2025 真实帧示例，展示左右骨架叠加与 epipolar error、reprojection error 和深度，但明确它是处理示例而非结果图。

#### 3.3.1 SKT：双目视觉观测与几何测量

- 按实际执行顺序介绍主结果使用的最终管线：左右帧同步配对、YOLO 关键点检测与目标跟踪、二维关键点坐标校正、左右人物与关节点对应检查、加权三角测量、时间邻域恢复，以及三维关节点和质量信号输出。
- 说明 confidence-weighted triangulation、soft epipolar constraint 和重建产生的置信度、极线误差、重投影误差、视差等几何质量信号。严格区分推理/三角测量阶段的接受规则与 3.5 的评估阶段掩码。
- 定义正视差要求和 45 px 最小视差这一物理有效性边界，并结合 $Z=fB/d$ 说明在本研究标定参数下对应约 10 m 的最大深度。最终方法不展开早期 1.5 px 设置的开发过程。
- 说明小幅且左右不一致的二维关键点误差如何被双目深度放大，并进一步传播到关节角度。
- 将这套配置统一称为 **canonical SKT pipeline（最终主管线）**，不再把历史名称 V2 当作它的同义词。

#### 3.3.2 FastSAM3D：单目视觉观测与人体先验

- 将本节明确拆成三层：（1）论文公开的 Fast SAM 3D Body 单目重建；（2）Aitor Iriondo Pascual 提供的 EasyErgo/OpenSim 生物力学转换；（3）本论文对 TRC 的读取和共同关节映射。明确上游模型和转换器都不是本论文的贡献。
- 用好理解的系统流程解释上游方法：单路 RGB Webcam、人体与粗略二维姿态定位、相机/FOV 信息、图像特征编码、MHR 人体解码，以及三维网格和关节输出。说明它把图像证据与学习到的人体表示结合起来，不能写成像双目系统一样直接测得深度。
- 将用户提供的流程重新绘制成“方法边界图”：人工输入的被试身高进入人体/模型尺度设置，随后是 skeleton/marker translation、practical corrections 和 OpenSim/IK/export 分支。本论文实际使用的 TRC 用实线路径表示，没有参与评价的下游产品单独表示。
- 明确把生物力学转换归于 Aitor，并引用 Fast SAM 3D Body 论文和 Aitor 的 FastSAM3DToOpenSim 公开仓库。说明身高用于被试尺度，不是额外视觉观测；smoothing/grounding 只作为外部上游流程记录，具体参数不可得，不能改写成本论文统一施加的滤波。
- 说明本文实际使用的 TRC：2025 数据为 12.5 Hz、42 markers、毫米；2026 数据为 30 Hz、46 markers、毫米。TRC header 没有声明具有语义的坐标轴约定，因此保留原始坐标轴，并在 marker 映射后使用不受共同旋转、平移和统一尺度影响的局部角度。
- 说明版本边界：Aitor 的实现后来仍在继续调整，当前公开仓库记录的是 24-marker、米制、Y-up 导出。凡是它与本项目归档的 42/46-marker EasyErgo 文件不一致，以实际归档文件和分析代码为准；公开仓库只用于确认贡献归属和总体转换用途，不能替代历史版本的具体设置。
- 说明强人体先验可能使模型在图像证据较弱时仍给出解剖上合理的姿态，且它不像 SKT 那样使用显式双目一致性条件拒绝结果。因此高 coverage 只表示可用性，不自动代表正确性；不能写成 FastSAM3D“永远会输出结果”。
- 说明其输出与 SKT 不完全等价，以及现有 TRC 不包含 SKT 三角测量所需的逐视角二维中间结果，因而无法直接插入双目融合路径。

#### 3.3.3 Xsens：身体传感器与运动学模型

- 与前两种视觉方法保持同样的 input--processing--output 逻辑，但对商业系统内部方法保持精简：（1）穿戴式惯性信号和被试设置；（2）MVN sensor fusion 与生物力学重建；（3）MVNX 输出和本论文的解析。
- 增加一张简洁的方法边界图。输入包括安装在各身体节段上的传感器、身体尺寸、传感器分配和 calibration posture；MVN 将惯性传播、sensor-to-segment calibration、连接的人体模型和约束结合起来，输出带时间戳的 MVNX。本论文的贡献从解析 MVNX 和构造共同角度开始。
- 解释传感器测到的是传感器自身运动，而不是直接测量解剖关节中心。身体尺寸和 calibration 将传感器映射到人体节段坐标系，因此尺寸不准、传感器移位或 calibration posture 不理想都可能形成持续 offset。
- 同时说明优点和限制：Xsens 不依赖相机可见性，能够提供密集连续的轨迹；但这些位置和角度已经经过人体模型、sensor fusion 和厂商后处理，不是 raw IMU，也不是独立的光学真值。
- 记录实际文件属性：2025 Aitor 文件由 MVN 2024.0 导出，2026 Fanbo 文件由 MVN 2024.2 导出；两组均为 60 Hz、23 segments、22 joints 的 FullBody/HD 输出。以文件 metadata 为准，不套用当前产品默认值。
- 说明 parser 的实际处理：只保留 normal motion frames，将毫秒换算成秒、米换算成厘米；upper arm--forearm--hand 的 segment origins 映射为 shoulder--elbow--wrist，upper leg--lower leg--foot 映射为 hip--knee--ankle。
- 解释为什么主 reference 从 segment positions 通过共同几何角度定义重新计算。MVNX 中的 native joint angles 和 ergonomic angles 仅用于说明该选择并追溯历史分析，不形成第二套结果，也不构造“参考系统分辨率下限”。
- 结尾明确：Xsens 是 synchronized external reference 而不是 physical ground truth，因为 calibration、人体几何假设和内部融合始终包含在每一个输出值中。

### 3.4 共同关节表示与角度

- 先解释为什么不能直接比较三个系统“自身的角度”。SKT 输出的是三角化 COCO-17 三维点，本论文实际使用的 FastSAM3D 输入是 TRC marker trajectory；只有 Xsens 还直接提供厂商定义的 anatomical joint angles 和 ergonomic angles。历史 Xsens 提取中，肩部与肘/髋/膝使用的字段和分量规则也不同。若直接比较，会把姿态估计差异与角度定义差异混在一起。
- 增加一张简洁的系统边界表：每条路径实际提供什么、是否存在 native angle，以及论文最后构造什么共同比较量。
- 将 SKT 的 COCO-17 关键点、FastSAM3D/EasyErgo 的 TRC markers 和 Xsens 的 segment origins 映射到共同的肩、肘、腕、髋、膝和踝链，并说明单位、坐标约定和能够可靠对应的关节。
- 使用同一向量夹角函数处理三套系统。肘、髋和膝分别使用 shoulder--elbow--wrist、shoulder--hip--knee 和 hip--knee--ankle 三点链，并报告内部夹角的补角；肩部 elevation 则定义为上臂向量与向下躯干向量之间的夹角。这样系统差异不会来自不同的角度计算口径。
- 解释关节之间为何采用不同形式：肘/膝按近似单自由度的屈伸处理，髋部是 shoulder--hip--knee 构成的屈曲 proxy，肩部则是相对向下躯干方向的抬举角。规则随关节运动含义而不同，但对三个系统完全相同。
- 主要比较采用由 Xsens segment positions 用相同函数重算得到的 Xsens-derived geometric angle，而不是厂商 native angle；本节把全文“角度”的含义一次性固定，后续章节不再重复限定。
- 使用统一简洁记号：$m\in\{\mathrm{SKT},\mathrm{FS}\}$ 表示视觉方法，$\theta_{m,j,t}$ 表示方法 $m$ 在关节 $j$、时间 $t$ 的角度；$\theta_{X,j,t}$ 表示由 Xsens segment positions 重算得到的参考角。只有在必须区分 Xsens 两种输出时才临时使用 native/geometric 限定词。
- 记录参考口径一致性审计（2026-09-11）：受影响的 2026 结果已按统一几何定义重新计算，保存的视觉方法输出和时间 offset 保持不变。

### 3.5 有效性检查与缺失数据处理

- 明确区分三层处理：推理阶段的 SKT 三角测量接受规则、评估阶段的 SKT 质量掩码，以及在 5.1 进行开关实验的上肢深度一致性掩码。
- 按当前实现定义评估质量掩码：对肩、肘、腕关键点，双目置信度不足或极线误差过大时将该点标为无效。不能写成该评估掩码也使用重投影误差；重投影误差在当前实现中属于三角测量阶段的质量信号。
- 定义 shoulder--elbow--wrist 链上的深度一致性规则，解释它以降低 coverage 为代价，避免明显不合理的手臂深度；在第 5 章展示开关结果之前先说明其机制与作用范围。
- 说明 FastSAM3D 的缺失值来自所提供的 TRC 轨迹，Xsens 的可用范围由 MVNX 与相机时间重叠决定；不能为这两种系统虚构双目置信度或深度一致性规则。
- 角度计算后，只填补两端均有效且不超过 5 个相机采样点的短缺口，较长缺口继续保留为 missing。共同有效时间点和各方法 coverage 的统计定义放在第 4 章。

### 3.6 时间对齐与统一评估序列

- 根据硬件 frame ID 而不是视频行号或理想恒定帧率建立 stereo camera timeline：左右 metadata 按 frame ID 配对，以配对后的左相机 timestamp 作为时间，并保留硬件 ID 的缺失。
- 用 2025 recording 作为具体例子：左、右 metadata 分别有 3015 和 2882 行，配对后得到 2801 个同步双目帧，中位间隔约 80 ms。FastSAM3D 的 3015 行 TRC 通过保存的 left-frame index 选取对应的 2801 行，而不是直接裁剪。
- FastSAM3D 根据同步帧索引或 TRC 时间戳映射。每个 recording 使用 $t_X=t_C-\delta_r$，将 Xsens 的 60 Hz 信号插值到这些查询时间，而不是简单地每隔若干帧丢弃数据；时间重叠之外不外推。
- 说明每个 recording 的时间 offset 如何通过 coarse-to-fine search 确定：主要评分为 SKT 与 Xsens 短时间角度变化 Pearson correlation 的跨关节中位数，角度相关性和刚性对齐位置一致性作为辅助诊断。保存候选分数，并通过角度曲线、录制前后片段和持续时间检查弱峰、残余错位或时钟漂移。
- 写明实际搜索精度（coarse 0.10 s、fine 0.01 s），并以 2025 的 17.25 s offset 作为具体例子。说明对齐采用 recording-level mapping，不会局部扭曲动作曲线来提高一致性。
- 记录每个 recording 实际使用的系统输出来源和必要预处理。本文比较的是各系统在当前数据条件下产生的最终角度结果，不把不同 recording 中可能存在的上游处理差异解释为独立的滤波效果。
- 不在不同 recording 之间引入统一的 moving-average 说法。保留每项实验实际使用的 FastSAM3D、SKT 与 Xsens 输入来源及处理记录；当不同数据的上游处理并不统一或无法完全观察时，不把跨 recording 的差异解释成单一滤波因素造成的结果。
- 各系统输出经过关节映射、时间对齐、有效性检查和明确的缺失值处理后，形成每个 recording 可追溯的 evaluation-ready angle sequences。**绝对角度与 motion 从相应处理后的角度序列开始，motion 分支随后再进行 $K$-sample 差分；两条评价路径保留各自的有效性掩码，最终样本数不必完全相同。**

### 3.7 实验配置与可重复性

- 先交代开发背景：为提高不同距离、视角和困难检测条件下的准确性与 robustness，SKT pipeline 在项目中持续调整，而且对单一视频有效的修改不一定能推广到其他数据。V1/V2 是为比较而保留的两个相对稳定历史快照，不代表全部迭代过程。
- 区分 5.1--5.3 使用的 **canonical SKT pipeline** 与 2026-07-07 存档消融中的 **historical V1/V2 configurations**。V1/V2 只表示当时实际测试的历史版本，不能再作为最终主管线的名称。
- 定性解释两版差异：V1 是较早的逐帧 baseline，更依赖 full-frame detection、硬性接受/拒绝和缺口后的角度后处理；V2 是整组 revision，引入目标区域 tracking、置信度加权三角化与更柔性的极线处理、邻帧恢复，以及 temporal/depth/bone-length processing。这样可以解释升级方向，但不能声称消融独立验证了其中某一个组件。
- 说明 provenance 限制：部分 V1 单元读取早期实现保存的输出，其他单元来自当前仓库中的历史配置；历史 V2 实验把多项 reconstruction 和 post-processing 选择耦合在一起。因此该消融只能评价整组 pipeline revision，不能分离每个组件的独立贡献。YOLOv8m 与 YOLO11l 是 $2\times2$ 设计中的另一个独立 detector factor。
- 正文保留机制、处理顺序和直接决定物理有效性或样本纳入的参数；精确关节索引、各 recording offset、完整配置标识、软件版本及更广泛的模型尝试放入可重复性附录。只有在实际计算和核验后，才加入各数据集的 filter 触发率。

## 第四章：评估框架

这一章只回答一个问题：**如何公平、完整地评价 SKT 和 FastSAM3D？**

正文承接前文已定义的方法与缩写，不重新介绍三种系统或预告后续章节。开头放评估流程图：处理后的角度序列 → 绝对角度 / Delta 两条分支 → 各自共同有效样本上的指标，并结合各方法自身 coverage 解读。不要在计算 Delta 前删除缺失行。

**区分两种比较用途。** 主结果分别把两种视觉方法与 $\theta_X$ 比较，只有这种比较能支持“与外部参考的一致程度”这一类陈述。历史消融则把不同 SKT 配置与同一条不变的 FastSAM3D 轨迹比较；这种方法间差值可以在同一 recording 内对配置排序，但不是准确性指标，也不能被称为相对真值的误差。

### 4.1 角度维度

- 比较逐帧绝对关节角度，直接回答视觉方法给出的姿势与 Xsens-derived reference 有多接近。
- 以 Median absolute error 和 MAE 表示典型与整体差异，以 p95 和箱线图展示误差尾部与异常值；箱线图是展示方式，不是独立 metric。
- 使用 Bias 检查持续高估或低估，并在角度随时间曲线中保留不同系统之间的角度 offset，而不为了提高一致性而先将曲线归零。
- 使用人体工程学角度区间一致性作为应用层指标，但不称为完整 RULA/REBA 自动评分。
- 结合 Xsens 的 calibration 偏差解释结果，不把观察到的角度差异直接称为绝对物理误差。

### 4.2 运动维度

- 解释为什么 MAE 不能单独回答运动问题：绝对角度会受到 Xsens 初始偏差影响，因此还需要比较姿态随时间的变化；这不是放弃 MAE，而是增加一个互补视角。
- 以肘关节屈伸这一简化的单自由度动作为重点，使用 K-frame Delta Angle 表示一段明确时间间隔内的角度变化，并把 K 换算成实际时间。
- 说明选择单自由度肘屈伸的原因：角度定义直观、三种系统都能使用 shoulder–elbow–wrist 链计算，并且受控动作有助于先隔离时间对齐和抖动问题；同时说明结论不能直接推广到复杂多自由度动作。
- 以 Delta MAE 表示运动变化量相差多少，以 Pearson correlation 表示变化方向和趋势是否一致；二者必须结合解释，因为高相关性并不等于变化幅度准确。

### 4.3 公平比较与可靠性说明

- 明确当前不存在完美的比较标准：绝对角度容易暴露 Xsens 的 calibration offset，而 Delta Angle 又会放大视觉系统的逐帧 jitter；因此两类结果必须并列报告，不能用其中一种掩盖另一种的弱点。
- 对承担主要结论的 recording，尽量同时展示绝对角度与 motion 结果：前者说明姿势数值和基准偏差，后者说明变化趋势和动作幅度；不要求每个诊断、消融或鲁棒性 recording 都重复完整的两套分析。
- 所有方法使用统一时间范围、共同有效帧和统一时间对齐；同时分别报告各方法自身的有效帧比例，避免只在最容易的帧上比较精度。
- 报告缺失关键点、异常跳变和几何质量信息，用于解释结果是否稳定以及失败发生在哪里。
- 每项结果注明数据、关节、参考方式和配置，避免把不同 recording、不同 filter 或不同模型设置下的数字直接混合排名。
- 骨长、人体比例、度量尺度和坐标差异仅作为输出解释与部署讨论，不再构成独立的位置精度评价维度；MPJPE 等 3D 关节位置误差指标不作为本文的正式精度指标，理由见第二章。

**第五章写作备注（不属于第四章正文）：** 运动结果可使用三面板散点图比较 **Delta Xsens–Delta SKT**、**Delta Xsens–Delta FastSAM3D** 和 **Delta SKT–Delta FastSAM3D**。三幅图使用统一坐标范围，并显示理想一致线 `y = x`、线性拟合线和 Pearson correlation。前两组评价视觉方法与参考系统的运动一致性；第三组只说明两种视觉方法是否产生相似变化，不作为准确性排名依据。

## 第五章：实验结果

这一章只保留一条容易理解的主线：先在同一段 2025 长序列上回答绝对角度和运动一致性，再**在同一段录制内部**考察两种方法对距离的依赖、并用两段 2026 右肘视频交叉验证，最后用存档消融解释当时的 pipeline 和检测器选择。Xsens 仍然是外部参考系统，因此本章报告的是一致性，而不是已知的物理真值误差。主结果均分别将视觉方法与 $\theta_X$ 比较；5.4 的历史方法间消融会被明确标出。

| 小节 | 主要数据 | 在论文中的作用 |
|---|---|---|
| 5.1 | 2025 Aitor 长序列，8 个关节 | 回答绝对角度与 Xsens reference 的一致程度 |
| 5.2 | 与 5.1 相同的 2025 数据，左右肘 | 回答两种方法能否跟随动作变化 |
| 5.3 | 2025 录制内深度扫描（主证据）+ Fanbo7/Fanbo4 右肘（交叉验证） | 回答 RQ2 的距离依赖，并给出两种方法差异的机制解释 |
| 5.4 | 2026-07-07 的 2×2×5 消融实验 | 区分 pipeline 与检测器选择的影响 |
| 5.5 | 5.1--5.4 的结果 | 综合回答 RQ1 和 RQ2，不增加新的实验 |

### 5.1 绝对角度一致性：2025 主实验

- 只使用冻结的 **canonical SKT pipeline + YOLOv8m** 输出，在 2025 Aitor 长序列上比较 SKT、FastSAM3D 与 Xsens reference。
- 两种视觉方法**各自**与 $\theta_X$ 比较。并列表格限制在两种方法都有效的共同时间点上，避免任一方通过丢弃困难帧获得较低误差，并在同一张表里报告各自的 coverage。这里**不是**把 SKT 与 FastSAM3D 直接相减。
- 正文用一张按关节 MAE 表保留具体数值，配一组“误差分布 + coverage”图说明典型差异、尾部与可用性；不为每个指标分别制图。
- 人体工程学角度区间一致率作为补充结果用一句话报告，不扩展为完整 RULA/REBA 评分。
- 这一节只回答“角度水平与参考系统有多一致”。不能把与 Xsens 的差值写成已知的物理误差。

### 5.2 运动一致性：与 5.1 成对解读

- 使用与 5.1 相同的 2025 Aitor 数据，只评价左右肘，并固定使用第四章定义的 **K=6**；不再同时展示多个 K。
- 使用一张三面板相关性图：**Delta Xsens vs Delta SKT**、**Delta Xsens vs Delta FastSAM3D**、**Delta SKT vs Delta FastSAM3D**。各面板显示 `y = x`、线性拟合线和 Pearson correlation，并使用相同坐标范围。
- 用一张简短表格报告 SKT 和 FastSAM3D 相对于 Xsens-derived reference 的 Delta MAE 与 Pearson correlation。Delta MAE 回答变化量相差多少，Pearson correlation 回答变化趋势是否一致；第三组视觉方法间相关性只用于解释两种方法是否以相似方式响应运动。
- 必须将真实角度曲线和 Delta 曲线成对展示。保留 0511 报告曾查看的 22.5–43.8 s 右肘窗口，但用当前输出重绘；同时展示全序列背景，窗口仅用于解释，不据此选取定量结果。
- 本节结论只回答：哪种方法的变化幅度更接近参考、哪种方法更能跟随变化趋势。相关性不单独解释为准确性，也不替代 5.1 的绝对角度结果。

### 5.3 距离依赖与录制条件

本节回答 RQ2。主证据是**录制内**的距离扫描，因为这是本研究中唯一一处"被试、session、相机、标定全部固定、只有距离在变"的条件。

- **主证据 —— 2025 录制自身的深度扫描。**被试躯干深度在**5.1/5.2 已经在用的同一段录制内**从约 2.8 m 变化到 6.4 m（p10–p90），因此不引入任何新数据。把两种方法相对 $\theta_X$ 的逐帧差异按估计躯干深度分箱，报告每个箱的八关节平均值、帧数和 coverage。
- 结果按测量原理把两种方法分开了：**SKT 的差异随距离显著增大**（3–4 m 段约 16°，7 m 以外约 28°），而 **FastSAM3D 在同一区间基本持平**（两端都约 13°）。两者的差距因此从近距约 5° 扩大到远端约 14°。
- 把这解释为两种测量原理的必然结果，而不是笼统的优劣排名：立体深度不确定度随距离平方增长，且必须先进入三维位置估计才能传到角度；而带人体先验、由外部给定身体尺度的模型拟合，其结果基本与距离无关。**这就是 5.5 中总体排序背后的机制。**
- 如实说明横轴的局限：深度由被评估的 SKT 重建自身估计得到，因此不是独立测量。但这是唯一可得的口径——FastSAM3D 的尺度来自人工输入的身高，Xsens 没有相对相机的位置。使用躯干质心而非单个关节以提高稳定性，并说明第三章的物理有效性边界已剔除会干扰分箱的远端异常深度。
- **跨录制交叉验证 —— Fanbo7（约 1.8 m）与 Fanbo4（约 4.1 m）。**两段都含右肘屈伸，都使用 canonical SKT pipeline + YOLOv8m 并在共同有效帧上比较。用一组 0629 原始场景图、一组右肘时间曲线和一张紧凑表交代条件、偏差形态、数值与各方法自身 coverage；不增加其他 recording。
- 按第三章定义的参考口径审计修正后的数值：Fanbo7 为 SKT 6.68° / FastSAM3D 5.01°，Fanbo4 为 11.49° / 7.36°。来源和计算保存在 MasterThesis/figure/chapter6/evidence_manifest.json，原实验文件不覆盖。
- 两种方法在 Fanbo4 的差值都更大，方向与录制内扫描一致；但两段之间动作幅度、时长、视角和图像尺度也不同，因此这一对数据**只作为方向上的佐证**，不作为距离效应的独立测量。
- Viewpoint 和 occlusion 没有满足共同 reference 与受控变量要求的定量结果，在第七章作为限制说明。

### 5.4 Pipeline 与检测器消融实验

- 开头先说明目的：区分 SKT 的改善究竟来自 V1--V2 pipeline 更新，还是仅仅来自换用更大的姿态检测器，因此两个因素必须分别固定、一次只改变一个。
- 在出图前交代五个 2026 条件：Fanbo7 与 Fanbo4 比较两个距离下的右肘，Fanbo3 是 walking，Fanbo9 提供同一次录制的两个 stereo camera view。明确 V1/V2 是 3.7 定义的历史配置，不是 5.1--5.3 所用 canonical pipeline 的名称。
- 使用 2026-07-07 的 **V1/V2 pipeline × YOLOv8m/YOLO11l × 5 个场景** 矩阵，正文用两面板柱状图分别展示“固定 YOLOv8m 更换 pipeline”和“固定 V2 更换模型”。完整矩阵留在可追溯数据中，避免表图重复。
- 本节单独使用 SKT--FastSAM3D 差值。之所以可以用于消融排序，是因为四种配置面对的是同一条不变的比较轨迹，因此**排序**可解释，即使**绝对水平**不是精度数字。这些数值不得带入 5.1--5.3。
- 主动堵住循环论证的质疑：用 FastSAM3D 当标尺选出 SKT 的配置、再拿 SKT 与 FastSAM3D 比较，只有在「排序依赖于标尺选择」时才构成循环。实际并不依赖——在同时具备参考系比较的 Fanbo9 双相机 session 上，**八个配置 cell 在两种比较标尺下给出的检测器排序完全一致**。这一句要主动写出来，不要留给答辩老师来问。
- 只保留两个稳健结论：在相同 YOLOv8m 下，V2 在五个场景中都降低或基本保持该差值；在 V2 下，YOLOv8m 在 3/5 场景更好，YOLO11l 在另外两项只领先 0.05° 和 0.78°。
- 明确写出 trade-off：历史 V2 增加了处理步骤和配置复杂度，但在五个条件中表现出更一致的改善；YOLO11l 增加 detector 侧计算负担，却没有带来稳定的精度收益。因此 **historical V2 + YOLOv8m** 是该存档消融中的优选组合，并为后来冻结 YOLOv8m 的工程决定提供证据；它不等同于 5.1--5.3 的最终主管线，也不证明 YOLOv8m 在所有条件下都更准确。
- 开发过程中还测试过更多姿态 detector、stereo model、17 组 filter 以及融合路线。5.4 不写成模型清单，正文只保留受控的 V1/V2--YOLO 对比；其余结果可整理到附录或第七章作为开发过程与限制。

### 5.5 结果总结

- 开头直接给出本章总判断：在当前测试条件和以角度为核心的人体工程学评估任务中，FastSAM3D 是表现更强的方法；V2 虽然改善了 SKT，但没有消除主实验中的差距。
- 用同一段 2025 数据回答 RQ1：FastSAM3D 整体上更接近 Xsens、误差尾部更短；SKT 在肘部绝对角度上的小优势到了 Delta 比较中发生反转，原因是 FastSAM3D 的肘部差异主要表现为较稳定的 offset。仍然不能把 Xsens 称为物理 ground truth。
- 将 reference 边界压缩为一句：Xsens 是 comparison system 而不是物理 ground truth，因此结论是相对一致性，而不是两种视觉方法的真实物理误差。
- 在现有证据范围内回答 RQ2：coverage 与误差互相独立，depth filter 明确体现了二者的 trade-off；2025 场景分组和两段 2026 距离条件说明结果受到录制条件影响，但后者无法单独隔离 distance、viewpoint 或 occlusion，因为两段录制同时存在多项差异。
- 解释方法特点，但不再用未验证的潜在优势平衡已有结果：FastSAM3D 在当前测试中给出更稳定的角度；SKT 虽能输出 metric geometry，但本章没有验证其位置精度，这一能力也不能抵消已经观察到的角度差距。
- 最后收束消融结论：历史消融支持保留 YOLOv8m，并说明 pipeline revision 的贡献；5.1--5.3 使用的最终主管线已在第三章单独定义。两者都不支持“普遍最优”的声明。RQ3 的实时计算与部署可行性留到下一章回答。

## 第六章：GPU 加速与实时实现评估

- 先定义本章的实现目标和完整处理流程，并明确区分离线 stereo video throughput 与完整 live system 的实时性能。
- 将一次性的模型初始化时间、首帧时间、warm-up 后的稳态处理速度以及 p50/p95 latency 分开报告，避免把不同性质的耗时混在同一个 FPS 中。
- 对比 CPU 与 GPU throughput，同时说明两组测试使用不同硬件环境，因此结果用于展示部署量级，不能解释为严格的硬件对照实验。
- 报告 PyTorch FP32 的 GPU 结果；TensorRT FP32/FP16 因未通过输出等效性检查，作为有价值的 deployment negative results 讨论，而不把更快但输出不一致的结果称为成功加速。
- 完整记录视频 codec、pixel format、压缩参数和文件大小，并分析压缩对 2D 关键点、3D 重建、关节角度及人体工程学角度区间的影响。
- 总结当前 pipeline 是否达到既定实时门槛、哪些输入格式可以安全使用，以及从离线 GPU pipeline 到完整 live system 仍缺少哪些环节。

## 第七章：讨论

### 7.1 方法比较结果的解释

- 综合解释第五章的主要结果，而不重复逐表汇报数值。在当前以关节角度为核心的任务中，FastSAM3D 给出的轨迹整体更稳定；SKT 的特点则是先显式重建 metric 3D positions，再由局部关节点计算角度。
- 区分“具有 metric scale 的输出能力”和“已经验证的位置准确性”：本文没有独立的位置 Ground Truth，因此不能把 SKT 的 metric coordinates 写成已被证明的精度优势。
- 解释两类方法的根本差异：SKT 的结果依赖 2D 关键点、左右视图对应和三角测量；FastSAM3D 通过人体先验约束整体姿态。人体先验可以抑制独立关节的深度抖动，但也可能在视觉证据较弱时产生合理而未必完全符合真实观察的姿势。
- 预留对 SKT 误差来源的集中分析：按照“2D 定位或遮挡问题—左右观察不一致—深度与肢体向量误差—角度抖动或结果被拒绝”的链条组织，并将尚未由最终主管线直接验证的机制明确写成 possible explanation，而不是确定结论。

### 7.2 对工业人体工程学评估的意义

- 回到论文的实际目的，讨论视觉方法是否能够辅助发现关键或高风险姿势，而不是要求其替代专业人体工程学判断或达到医疗级测量精度。
- 结合 absolute-angle agreement、motion agreement 和 coverage 讨论结果是否可用；有效输出、较小误差和稳定运动趋势是不同条件，不能由单一指标代替。
- 说明在当前证据下，系统更适合作为姿势筛查和人工分析的辅助工具；在远距离、遮挡、复杂背景、长时间缺失或结果不连续时仍需要人工复核。
- 明确本文没有实现完整 RULA/REBA：负载、肌肉使用、手部耦合和其他表格判断没有由当前链路推断，因此不能把角度区间一致性扩大为完整风险评分能力。

### 7.3 工程与部署权衡

- 将第五章的算法结果与第六章的实现结果放在同一产品背景下讨论：detector 更大不一定改善双目角度，三角测量本身的计算成本很低，主要开销来自双视图检测和视频解码。
- 讨论关键点质量、工作距离、coverage、推理速度、GPU 需求、硬件成本和输入视频质量之间的关系，避免仅根据一个精度或 FPS 数字选择产品配置。
- 将 PyTorch FP32、TensorRT 和压缩实验解释为“速度必须以输出等效为前提”的工程约束；更快或更小的候选如果改变了重建结果，不能直接替代经过评价的主管线。
- 明确区分两层结论：核心离线计算链路在测试 GPU 上具备实时吞吐能力，但本文没有展示完整 live ergonomic-assessment system。

### 7.4 研究限制与有效性边界

- Reference 限制：Xsens 是外部比较系统而不是物理 Ground Truth，其 calibration、body model 和 registration 会影响绝对角度比较。
- 数据限制：被试和录制数量有限，动作覆盖、持续时间和可用模态不完全一致。
- 实验控制限制：distance、viewpoint、occlusion、动作和录制年份存在混杂，不能把跨 recording 的差异解释为单一因素的因果效应。
- 方法与追溯限制：部分历史 V1/V2 消融使用存档配置或以 FastSAM3D 为比较轨迹，只能支持配置选择，不能建立对真实精度的独立排名。
- 系统范围限制：正文验证的是保存视频上的处理核心，不包括实时采集、流传输、用户界面、风险提示、下游数据接口和长期现场稳定性。
- 对模型选择原因和未被实验完全验证的机制解释，统一标记为 hypothesis、possible cause 或 engineering judgement。

## 第八章：结论与未来工作

### 8.1 结论

- 用简短段落直接回答三个 Research Questions，不再逐节重复实验数字。RQ1 总结两种方法与 Xsens-derived reference 的相对一致性；RQ2 总结在现有场景和证据边界下观察到的条件敏感性；RQ3 只确认 SKT 核心计算链路的实时可行性，不把它扩大为完整实时系统已经完成。
- 总结项目价值：本文没有证明某一种方法在所有条件下绝对最优，而是建立了三种系统的统一比较方式，明确了两类视觉路线在工业人体工程学场景中的表现、限制、工程代价和后续开发依据。
- 再次说明本文在完整项目中的位置：所完成的是从已采集数据到姿态、角度、评价和 GPU 计算验证的视觉处理核心，完整现场系统仍需后续集成。

### 8.2 未来工作

#### A. 继续改进与验证 SKT pipeline

- 系统分析 2D 关键点误差如何通过左右视图对应和三角测量传播到 3D 位置与关节角度，并用最终主管线生成可追溯的典型失败案例。
- 继续研究更可靠的跨视图 person/joint association、uncertainty-aware triangulation、质量自适应处理，以及经过跨数据验证的人体或骨长约束；不预设某一种滤波或约束一定有效。
- 通过受控实验分别改变 distance、viewpoint 和 occlusion，并使用更强的独立参考系统验证 3D position 与 joint angle；扩展到更多被试、动作和真实工业任务。

#### B. 打通端到端实时系统

- 将当前保存视频处理链路扩展为 live stereo capture—hardware synchronization—validated lossless transport—online detection and triangulation—ergonomic analysis—visualization/warning/data export 的完整链路。
- 测试端到端 latency、长时间同步与运行稳定性、丢帧恢复和现场可维护性，而不仅是短窗口内的算法 throughput。
- 在目标 GPU、边缘或嵌入式硬件上重新测试；只有在通过输出等效性检查后，才采用 TensorRT、FP16、硬件解码或其他加速方案。
- 结合完整 RULA/REBA 所需的负载、肌肉使用和交互信息，并在需要时加入 activity recognition，使姿态估计能够进入完整的工业评估与决策流程。

## 建议附录

### Appendix A：标定、系统映射与共同定义

- 完整相机内外参数、标定搜索设置和标定验证数据。
- 三种系统的关节点映射、坐标约定和正文中未展开的完整角度公式。
- recording、设备、活动分段、时间偏移和数据用途的详细对照表。

### Appendix B：Pipeline 配置与补充实验

- 明确列出第五章主管线唯一采用的 canonical SKT 配置、run identifier 和复现入口，避免与 legacy 或名称相近的历史配置混淆。
- 保存 V1/V2、YOLOv8m/YOLO11l、filter、bone constraint、Stage C 和其他 detector/pipeline 测试的完整配置与结果。
- 收录正文未展开的典型成功/失败帧和 SKT 误差诊断；历史结果与最终主管线结果必须明确分开标注。

### Appendix C：完整数值结果

- 各关节的 MAE、median、percentiles、bias、coverage、Delta Angle 和 correlation 完整表格。
- 距离分组、场景分组、补充散点图、消融结果和正文为保持主线而省略的图表。

### Appendix D：部署与可重复性

- CPU/GPU 重复测试、初始化与 steady-state timing、TensorRT 等效性检查及详细输出。
- 视频 codec、pixel format、压缩参数、FFmpeg 命令和输入等效性结果。
- 硬件与软件版本、配置文件、输入与输出文件、source hashes、代码仓库、许可协议和完整复现步骤。
- 正文必须引用实际使用到的附录内容，避免把附录变成与论点无关的数据堆积。

## 写作与实验核查清单（不属于正式目录）

- 只要原因和适用范围解释清楚，positive 与 negative results 都具有研究价值。
- 有直接实验支持的内容称为 result；尚未完全验证的解释称为 hypothesis、possible cause 或 engineering judgement。
- 全文将 Xsens 描述为 external comparison system，而不是绝对 ground truth。
- 同时报告 absolute angle 与 Delta Angle，避免 motion 结果掩盖角度水平差异。
- 平均值应与 median、p95 和 box plot 共同使用，避免少数异常帧主导结论。
- 保留视频参数、相机标定、模型版本、run configuration、代码入口和图表原始数据，保证实验可重复。
