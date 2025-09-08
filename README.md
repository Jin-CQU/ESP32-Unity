# ESP32-Unity 实时脑电信号处理与 SSVEP 刺激系统

## 项目亮点
- 实时采集 ESP32 设备脑电数据，支持多通道
- Unity 可视化 EEG 波形，参数可调
- SSVEP 视觉刺激器，频率/颜色/时长可配置
- 集成 Sentis/ONNX 模型推理，实时 EEG 分类
- 推理后端可选（GPU/CPU），自动回退并显示实际后端
- 性能计时与诊断（Schedule/Download 分段耗时、Softmax 概率、Inspector 字段）
- 线程安全数据处理，避免 UI 卡顿
- 支持 YVR SDK、Oculus XR

## 快速安装
1. 克隆仓库并用 Unity 2022.3+ 打开
2. （可选）安装 YVR SDK，配置 VR 设备
3. 配置 ESP32 设备 IP/端口，确保数据包格式正确
4. 运行主场景，系统自动初始化 UDP 接收与可视化

## 主要模块
- `Assets/Algorithm/EEG_Classify_test.cs`：EEG 实时推理组件，支持 ONNX 模型、后端选择、性能统计
- `Assets/Comunication/UDP_1.cs`：UDP 通信，异步接收 ESP32 数据包，CRC 校验
- `Assets/Data_Display/EEGVisualizer.cs`：脑电波形实时显示，支持多通道
- `Assets/Stimulus/SSVEPStimulus.cs`：视觉刺激器，参数可调
- `Assets/Logic/MainLogic.cs`：主逻辑控制

## EEG 推理说明
- Inspector 可选 GPUCompute/CPU，实际后端显示于 `actualBackend`
- 推理输出自动 Softmax 归一化（等价 PyTorch `torch.softmax(outputs, dim=1)`）
- 输入为最新 UDP 缓冲区数据，非随机测试
- Inspector 显示 `lastInferenceTimeMs`、`averageInferenceTimeMs`、`lastScheduleTimeMs`、`lastDownloadTimeMs`，日志自动打印分段耗时
- UDP 回调与主线程通过队列和标志同步，避免 Unity API 跨线程调用
- Worker 创建与推理均可延迟/分帧执行，避免启动/推理卡顿

## 项目结构
```
ESP32-Unity/
├── Assets/
│   ├── Algorithm/           # EEG 推理与算法
│   ├── Comunication/        # UDP 通信
│   ├── Data_Display/        # EEG 可视化
│   ├── Stimulus/            # SSVEP 刺激
│   ├── Logic/               # 主逻辑
│   ├── Scenes/              # Unity 场景
│   ├── XR/                  # XR 支持
│   └── YVRProjectSettings/  # YVR 配置
├── ProjectSettings/         # Unity 项目设置
├── Packages/                # 包管理
└── README.md                # 项目说明
```

## 性能与诊断
- 推理总耗时、分段耗时、平均耗时均可在 Inspector 和日志中查看
- 控制台日志如：`耗时:0.39ms(S:0.1+D:0.3)`，便于定位瓶颈
- 后端不可用自动回退并提示

## 常见问题
- UDP 连接失败：检查 IP/端口、防火墙、ESP32 配置
- 数据显示异常：检查数据包格式、CRC 校验、显示参数
- VR 设备连接问题：确认 YVR SDK 安装、设备驱动、XR 设置
- 推理卡顿：建议使用分帧推理，检查后端选择

## 扩展开发
- 新算法：在 `Assets/Algorithm/` 新建脚本并集成到 UDP_1.cs
- 新可视化：参考 EEGVisualizer.cs 创建新组件
- 新刺激模式：参考 SSVEPStimulus.cs 实现自定义逻辑

## Git 换行符 (LF/CRLF) 说明
建议添加 `.gitattributes` 统一换行符：
```
* text=auto
*.cs text eol=lf
*.md text eol=lf
```
如需 Windows CRLF，可调整规则并设置 `core.autocrlf`。

## 贡献与许可证
- 欢迎提交 Issue 和 Pull Request
- 本项目采用 MIT 许可证，详见 LICENSE 文件

---
**注意**：本系统仅用于研究和教育目的，实际医疗应用请遵循相关法规。
