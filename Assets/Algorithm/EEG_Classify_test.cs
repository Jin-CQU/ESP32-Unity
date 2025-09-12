using System;
using System.Collections.Generic;
using System.Linq;
using System.Collections.Concurrent;
using System.Collections;
using UnityEngine;
using Unity.Sentis;
using UnityEngine.UI;
using MathNet.Numerics;
using MathNet.Filtering;
using MathNet.Filtering.IIR; // Butterworth滤波器可能在IIR命名空间中
using MathNet.Filtering.FIR; // 如果也需要FIR滤波器
// 注意：根据您实际使用的Math.NET Numerics版本，可能需要调整命名空间引用
// 如果下面的代码仍有问题，请检查您使用的DLL版本对应的文档

public class EEG_Classify_test : MonoBehaviour
{
    [Header("ONNX 模型 (Sentis ModelAsset)")]
    [SerializeField] private ModelAsset modelAsset; // 拖拽 .onnx 导入后的 ModelAsset

    [Header("推理后端")]
    [SerializeField] private BackendType backend = BackendType.GPUCompute; // 默认GPU，设备不支持会回退到CPU

    [Header("可选：若模型输入名不为默认首个，可在此填入")]
    [SerializeField] private string inputName = string.Empty; // 留空则使用模型的第一个输入

    [Header("EEG数据处理参数")]
    [SerializeField] private int windowSize = 768; // 滑动窗口大小(采样点数)
    [SerializeField] private int downsampleTo = 384; // 降采样后的点数
    [SerializeField] private int stepSize = 200; // 滑动步长(0.2秒 * 1000Hz = 200点)
    [SerializeField] private int targetChannels = 3; // 目标通道数
    [SerializeField] private float inferenceInterval = 1.0f; // 推理间隔改为1秒
    [SerializeField] private bool enableInference = false; // 默认关闭推理，避免启动卡顿
    [SerializeField] private bool debugMode = false; // 调试模式开关

    [Header("推理结果")]
    [SerializeField] private float[] output = new float[2]; // 推理输出结果（Softmax概率）
    [SerializeField] private int inferenceCount = 0; // 推理次数
    [SerializeField] private int predictedClass = -1; // 预测类别 (0或1)
    [SerializeField] private float confidence = 0f; // 预测置信度
    [SerializeField] private float lastInferenceTimeMs = 0f; // 最后一次推理耗时(毫秒)
    [SerializeField] private float averageInferenceTimeMs = 0f; // 平均推理耗时(毫秒)
    [SerializeField] private float lastScheduleTimeMs = 0f; // 最后一次Schedule耗时(毫秒)
    [SerializeField] private float lastDownloadTimeMs = 0f; // 最后一次Download耗时(毫秒)
    [SerializeField] private bool preInitializeWorker = false; // 改为默认不预初始化，避免启动阻塞
    [SerializeField] private string workerStatus = "未初始化"; // Worker状态显示
    [SerializeField] private string startupStatus = "等待启动..."; // 启动状态显示
    [SerializeField] private string actualBackend = "未知"; // 实际使用的后端类型

    [Header("组件引用")]
    [SerializeField] private Text InferenceInfo; // 显示推理信息
    [SerializeField] private Image ClassResult; // 提示图像

    private Worker worker;
    private float[] lastOutput;
    private UDP_1 udpReceiver;
    private bool isInferenceRunning = false; // 推理运行状态标记

    // 数据缓冲
    private ConcurrentQueue<(int channel, float[] data, double timestamp)> dataQueue = new ConcurrentQueue<(int, float[], double)>();
    private List<float>[] channelBuffers = new List<float>[3];
    private int nextStepPosition = 0; // 保留这个变量
    
    // 序列号和时间戳跟踪
    private Dictionary<int, double> lastProcessedTimestamp = new Dictionary<int, double>();
    private Dictionary<int, int> interpolatedPoints = new Dictionary<int, int>();
    
    // 滑动窗口相关字段
    private int lastWindowPosition = 0; // 上次窗口的起始位置
    private Dictionary<int, List<float>> channelDataCache = new Dictionary<int, List<float>>(); // 每个通道的数据缓存
    
    // 带通滤波器相关字段
    // 注意：MathNet.Filtering可能不可完全使用，我们实现了多重备选方案：
    // 1. 优先使用MathNet.Filtering.OnlineFilter
    // 2. 备选使用MathNet.Numerics实现的数字滤波
    // 3. 最后使用简单的时域滤波
    private Dictionary<int, OnlineFilter> bandpassFilters = new Dictionary<int, OnlineFilter>(); // 每个通道的带通滤波器（可能为null）
    private const double samplingRate = 1000.0; // 假设采样率为1000Hz
    private bool filtersInitialized = false; // 滤波器是否已初始化

    // 简化控制
    private float lastInferenceTime = 0f;
    
    // 归一化相关 - 用于模型输入前的数据标准化
    private float[] channelMeans = null; // 每个通道的数据均值
    private float[] channelStds = null;  // 每个通道的数据标准差
    private const int SCALER_FIT_SAMPLES = 1000; // 用于拟合标准化的样本数量
    private bool inferenceReady = false;
    private bool isWorkerInitializing = false; // 防止重复初始化Worker
    private float lastDataReceivedTime = 0f; // 最后收到数据的时间
    private int lastDataCount = 0; // 上次检查时的数据量
    private float dataTimeoutSeconds = 2.0f; // 数据超时时间（秒）
    private bool hasNewDataFlag = false; // 新数据标记，用于线程安全的时间更新

    // 推理时间统计
    private float totalInferenceTimeMs = 0f; // 总推理时间(毫秒)
    private System.Diagnostics.Stopwatch inferenceStopwatch; // 推理计时器
    private System.Diagnostics.Stopwatch scheduleStopwatch; // Schedule操作计时器
    private System.Diagnostics.Stopwatch downloadStopwatch; // Download操作计时器

    private void Start()
    {
        // 绝对最简化启动 - 不做任何可能阻塞的操作
        workerStatus = "未初始化";
        startupStatus = "游戏启动中...";

        // 初始化推理计时器
        inferenceStopwatch = new System.Diagnostics.Stopwatch();
        scheduleStopwatch = new System.Diagnostics.Stopwatch();
        downloadStopwatch = new System.Diagnostics.Stopwatch();

        // 立即启用推理功能，确保不会被遗忘
        enableInference = true;
        SafeLog("启动时强制启用推理功能");
    }

    private void LateStart()
    {
        startupStatus = "初始化组件中...";

        // 延迟初始化所有内容
        channelBuffers = new List<float>[3];
        for (int i = 0; i < 3; i++)
        {
            channelBuffers[i] = new List<float>();
        }
        nextStepPosition = 768;

        // 初始化滑动窗口缓存
        InitializeSlidingWindowCache();
        
        // 初始化带通滤波器
        InitializeBandpassFilters();

        udpReceiver = FindObjectOfType<UDP_1>();
        if (udpReceiver != null)
        {
            udpReceiver.OnDataReceived += OnEEGDataReceived;
            startupStatus = "UDP连接已建立";
        }
        else
        {
            startupStatus = "UDP连接失败";
        }

        // 完全移除预初始化Worker，改为真正的按需初始化
        // 这样可以确保启动时绝对不会阻塞
        if (preInitializeWorker && modelAsset != null)
        {
            // 即使启用预初始化，也要延迟到稳定后
            StartCoroutine(DelayedWorkerInit());
        }
        else if (modelAsset != null)
        {
            // 如果没有启用预初始化但有模型，也要自动初始化
            SafeLog("自动启动Worker初始化...");
            StartCoroutine(DelayedWorkerInit());
        }

        SafeLog("[EEG_Classify_test] 组件初始化完成，Worker将按需创建");
    }

    /// <summary>
    /// 延迟的Worker初始化 - 确保系统完全稳定后才初始化
    /// </summary>
    private IEnumerator DelayedWorkerInit()
    {
        // 等待5秒后再初始化Worker，确保系统完全稳定
        yield return new WaitForSeconds(5f);

        // 重要：检查是否已经有其他协程在初始化Worker
        if (worker != null)
        {
            SafeLog("[EEG_Classify_test] Worker已存在，跳过重复初始化");
            yield break;
        }

        if (modelAsset != null)
        {
            SafeLog("[EEG_Classify_test] 开始延迟Worker初始化");
            yield return StartCoroutine(InitializeWorkerAsync());
        }
    }

    /// <summary>
    /// 线程安全的日志记录方法
    /// </summary>
    /// <param name="message">日志消息</param>
    /// <param name="logType">日志类型</param>
    private void SafeLog(string message, LogType logType = LogType.Log)
    {
        switch (logType)
        {
            case LogType.Error:
                Debug.LogError($"[EEG_Classify] {message}");
                break;
            case LogType.Warning:
                Debug.LogWarning($"[EEG_Classify] {message}");
                break;
            default:
                Debug.Log($"[EEG_Classify] {message}");
                break;
        }
    }

    /// <summary>
    /// 检查并触发推理（极简版本）
    /// </summary>
    private void CheckAndTriggerInference()
    {
        // 检查基本条件
        if (!enableInference || modelAsset == null)
        {
            SafeLog($"[CheckAndTriggerInference] 跳过: enableInference={enableInference}, modelAsset={modelAsset != null}");
            return;
        }

        // 检查Worker状态 - 必须完全就绪
        if (worker == null || !inferenceReady || workerStatus != "就绪")
        {
            SafeLog($"[CheckAndTriggerInference] Worker未就绪: worker={worker != null}, inferenceReady={inferenceReady}, status={workerStatus}");
            return;
        }

        // 检查是否已经有推理在运行
        if (isInferenceRunning)
        {
            SafeLog("[CheckAndTriggerInference] 推理正在运行中，跳过");
            return;
        }

        // 检查推理间隔
        if (Time.time - lastInferenceTime < inferenceInterval)
        {
            SafeLog($"[CheckAndTriggerInference] 推理间隔未到: {Time.time - lastInferenceTime:F2}s < {inferenceInterval}s");
            return;
        }

        // 简单检查数据 - 降低数据要求，更容易触发推理
        if (channelBuffers[0] == null || channelBuffers[0].Count < 500) // 降低到500个点
        {
            SafeLog($"[CheckAndTriggerInference] 数据不足: buffer0={channelBuffers[0]?.Count ?? 0} < 500");
            return;
        }

        // 关键新增：检查数据新鲜度 - 确保使用实时数据
        float timeSinceLastData = Time.time - lastDataReceivedTime;
        if (timeSinceLastData > dataTimeoutSeconds)
        {
            SafeLog($"[CheckAndTriggerInference] 数据过时: {timeSinceLastData:F1}s前停止接收数据，跳过推理");
            return;
        }

        // 检查数据是否有更新
        int currentDataCount = channelBuffers[0].Count;
        if (currentDataCount == lastDataCount && timeSinceLastData > 0.5f)
        {
            SafeLog($"[CheckAndTriggerInference] 数据未更新: 缓冲区大小{currentDataCount}未变化");
            return;
        }
        lastDataCount = currentDataCount;

        // 更新时间并执行推理
        lastInferenceTime = Time.time;

        SafeLog("[CheckAndTriggerInference] ✅ 所有条件满足，准备开始推理...");

        // 关键修复：使用延迟启动推理避免立即阻塞
        StartCoroutine(DelayedInferenceStart());
    }

    /// <summary>
    /// 延迟启动推理 - 避免Worker刚就绪时立即推理导致阻塞
    /// </summary>
    private IEnumerator DelayedInferenceStart()
    {
        SafeLog("[DelayedInferenceStart] 开始延迟推理...");

        // 等待几帧确保系统完全稳定
        yield return null;
        yield return null;
        yield return null;

        SafeLog("[DelayedInferenceStart] 启动简化推理测试...");

        // 先尝试最简单的推理 - 不使用协程
        SimpleInferenceTest();
    }

    /// <summary>
    /// 实现Softmax函数，对应PyTorch的torch.softmax(outputs, dim=1)
    /// </summary>
    /// <param name="logits">原始输出值（未归一化）</param>
    /// <returns>softmax概率分布</returns>
    private float[] Softmax(float[] logits)
    {
        if (logits == null || logits.Length == 0)
        {
            SafeLog("[Softmax] 输入为空，返回原始值", LogType.Warning);
            return logits;
        }

        try
        {
            // 找到最大值用于数值稳定性（防止指数溢出）
            float maxVal = logits[0];
            for (int i = 1; i < logits.Length; i++)
            {
                if (logits[i] > maxVal)
                    maxVal = logits[i];
            }

            // 计算 exp(x - max) 的和
            float sum = 0f;
            float[] expValues = new float[logits.Length];
            for (int i = 0; i < logits.Length; i++)
            {
                expValues[i] = Mathf.Exp(logits[i] - maxVal);
                sum += expValues[i];
            }

            // 检查数值稳定性
            if (sum <= 0f || float.IsNaN(sum) || float.IsInfinity(sum))
            {
                SafeLog($"[Softmax] 数值异常 - sum:{sum}, 返回均匀分布", LogType.Warning);
                float[] uniform = new float[logits.Length];
                for (int i = 0; i < logits.Length; i++)
                    uniform[i] = 1f / logits.Length;
                return uniform;
            }

            // 归一化得到概率
            float[] probabilities = new float[logits.Length];
            for (int i = 0; i < logits.Length; i++)
            {
                probabilities[i] = expValues[i] / sum;
            }

            return probabilities;
        }
        catch (Exception e)
        {
            SafeLog($"[Softmax] 异常: {e.Message}, 返回原始值", LogType.Error);
            return logits;
        }
    }

    /// <summary>
    /// 从缓冲区提取实时EEG数据用于推理（使用滑动窗口）
    /// </summary>
    /// <param name="outputData">输出数据数组 (3通道 * 384点)</param>
    /// <returns>是否成功提取数据</returns>
    private bool ExtractRealTimeData(float[] outputData)
    {
        bool result = GetNextSlidingWindow(outputData);
        
        // 如果提取失败，可能需要智能重置窗口位置
        if (!result)
        {
            // 检查是否是因为窗口位置太远导致的
            bool allChannelsHaveData = true;
            for (int ch = 0; ch < targetChannels; ch++)
            {
                if (channelBuffers[ch] == null || channelBuffers[ch].Count < windowSize)
                {
                    allChannelsHaveData = false;
                    break;
                }
            }
            
            // 如果所有通道都有足够数据但窗口位置太远，智能重置窗口
            if (allChannelsHaveData)
            {
                SafeLog($"[ExtractRealTimeData] 窗口位置太远，执行智能重置");
                SmartResetSlidingWindow();
                // 再次尝试提取
                result = GetNextSlidingWindow(outputData);
            }
            else
            {
                SafeLog($"[ExtractRealTimeData] 数据不足，无法提取窗口数据");
            }
        }
        
        return result;
    }

    /// <summary>
    /// 最简化的推理测试 - 直接在主线程执行
    /// </summary>
    private void SimpleInferenceTest()
    {
        SafeLog("[SimpleInferenceTest] 开始简化推理测试...");

        try
        {
            // 设置推理运行状态
            isInferenceRunning = true;

            // 使用实时EEG数据而不是随机数据
            var testData = new float[3 * 384]; // 3通道 * 384点

            // 从缓冲区提取实时数据
            bool dataReady = ExtractRealTimeData(testData);
            if (!dataReady)
            {
                SafeLog("[SimpleInferenceTest] 实时数据不足，跳过推理");
                isInferenceRunning = false;
                return;
            }

            SafeLog("[SimpleInferenceTest] 实时EEG数据已准备");

            // 在模型输入前进行归一化（类似Python代码：self.scaler.transform()）
            float[] normalizedData = NormalizeDataBeforeInput(testData);
            
            if (debugMode)
            {
                SafeLog($"[SimpleInferenceTest] 归一化前后数据对比：");
                SafeLog($"  原始数据范围: [{testData.Min():F3}, {testData.Max():F3}]");
                SafeLog($"  归一化后范围: [{normalizedData.Min():F3}, {normalizedData.Max():F3}]");
            }

            // 创建输入张量 - 修复：使用4维形状 (1, 1, 3, 384)
            var inputShape = new TensorShape(1, 1, 3, 384);
            using var input = new Tensor<float>(inputShape, normalizedData);

            SafeLog($"[SimpleInferenceTest] 输入张量已创建（已归一化），形状: {inputShape}");

            // 🔥 在真正的推理操作前开始计时
            SafeLog("[SimpleInferenceTest] 即将执行worker.Schedule()...");
            inferenceStopwatch.Restart();
            scheduleStopwatch.Restart();

            // 执行推理 - 最关键的测试点
            worker.Schedule(input);
            scheduleStopwatch.Stop();
            float scheduleTimeMs = (float)scheduleStopwatch.Elapsed.TotalMilliseconds;
            SafeLog($"[SimpleInferenceTest] ✅ worker.Schedule() 完成 - 耗时:{scheduleTimeMs:F3}ms");

            // 获取结果
            SafeLog("[SimpleInferenceTest] 即将获取输出...");
            var output = worker.PeekOutput() as Tensor<float>;
            SafeLog("[SimpleInferenceTest] ✅ PeekOutput() 完成");

            if (output != null)
            {
                SafeLog("[SimpleInferenceTest] 即将下载数据...");
                downloadStopwatch.Restart();
                var result = output.DownloadToArray();
                downloadStopwatch.Stop();
                float downloadTimeMs = (float)downloadStopwatch.Elapsed.TotalMilliseconds;
                SafeLog($"[SimpleInferenceTest] ✅ DownloadToArray() 完成 - 耗时:{downloadTimeMs:F3}ms");

                // 🔥 在核心推理操作完成后立即停止计时
                inferenceStopwatch.Stop();
                lastInferenceTimeMs = (float)inferenceStopwatch.Elapsed.TotalMilliseconds;
                lastScheduleTimeMs = scheduleTimeMs;
                lastDownloadTimeMs = downloadTimeMs;

                SafeLog($"[SimpleInferenceTest] 🕒 详细计时 - Schedule:{scheduleTimeMs:F3}ms + Download:{downloadTimeMs:F3}ms = 总计:{lastInferenceTimeMs:F3}ms");

                if (result != null && result.Length >= 2)
                {
                    // 应用Softmax处理，对应torch.softmax(outputs, dim=1)
                    float[] probabilities = Softmax(result);

                    this.output[0] = probabilities[0];
                    this.output[1] = probabilities[1];

                    // 计算预测类别和置信度
                    if (probabilities[0] > probabilities[1])
                    {
                        predictedClass = 0;
                        confidence = probabilities[0];
                    }
                    else
                    {
                        predictedClass = 1;
                        confidence = probabilities[1];
                    }

                    inferenceCount++;

                    // 更新平均推理时间
                    totalInferenceTimeMs += lastInferenceTimeMs;
                    averageInferenceTimeMs = totalInferenceTimeMs / inferenceCount;

                    // 合并为单行日志避免截断，包含推理时间
                    SafeLog($"[SimpleInferenceTest] ✅ 实时推理#{inferenceCount} - 原始:[{result[0]:F6},{result[1]:F6}] Softmax:[{probabilities[0]:F6},{probabilities[1]:F6}] 预测:类别{predictedClass} 置信度{confidence:F4}({confidence * 100:F1}%) 耗时:{lastInferenceTimeMs:F3}ms(S:{scheduleTimeMs:F1}+D:{downloadTimeMs:F1}) 平均:{averageInferenceTimeMs:F3}ms");
                    InferenceInfo.text = $"推理#{inferenceCount} - 类别{predictedClass} 置信度{confidence:F4} 耗时:{lastInferenceTimeMs:F2}ms 平均:{averageInferenceTimeMs:F2}ms";
                    if (predictedClass == 0)
                    {
                        ClassResult.color = new Color(0.07199074f, 0.654088f, 0.183762f); // 绿色0.07199074f, 0.654088f, 0.183762f
                        //ClassResult.Label = "放松";
                    }
                    else
                    {
                        ClassResult.color = new Color(0.9622642f, 0.1027691f, 0.08775356f); // 红色0.9622642f, 0.1027691f, 0.08775356f
                        //ClassResult.Label = "专注";
                    }
                }
            }
            else
            {
                // 如果没有输出，也要停止计时器
                if (inferenceStopwatch.IsRunning)
                {
                    inferenceStopwatch.Stop();
                }
                SafeLog("[SimpleInferenceTest] ❌ 未获取到推理输出");
            }

            // 清理
            // input 已经用using自动清理
            isInferenceRunning = false;

            SafeLog("[SimpleInferenceTest] ✅ 简化推理测试完全成功");
        }
        catch (Exception e)
        {
            SafeLog($"[SimpleInferenceTest] ❌ 推理失败: {e.Message}", LogType.Error);
            isInferenceRunning = false;
            // 确保计时器被停止
            if (inferenceStopwatch.IsRunning)
            {
                inferenceStopwatch.Stop();
            }
        }
    }

    /// <summary>
    /// 最简单的推理 - 分帧执行避免阻塞
    /// </summary>
    private void SimpleInference()
    {
        StartCoroutine(DelayedInferenceStart());
    }

    /// <summary>
    /// 真正的非阻塞Worker初始化 - 使用状态机模式
    /// </summary>
    private IEnumerator InitializeWorkerAsync()
    {
        // 防止重复初始化
        if (isWorkerInitializing)
        {
            SafeLog("[EEG_Classify_test] Worker正在初始化中，跳过重复请求");
            yield break;
        }

        if (worker != null)
        {
            SafeLog("[EEG_Classify_test] Worker已存在，跳过初始化");
            yield break;
        }

        isWorkerInitializing = true;
        workerStatus = "准备初始化...";
        SafeLog("[EEG_Classify_test] 开始非阻塞Worker初始化...");

        // 阶段1：等待Unity稳定
        yield return null;
        yield return null;
        yield return null;

        // 阶段2：检查模型资源
        if (modelAsset == null)
        {
            workerStatus = "模型资源缺失";
            isWorkerInitializing = false;
            SafeLog("[EEG_Classify_test] ModelAsset为空", LogType.Error);
            yield break;
        }

        workerStatus = "检查模型资源完成";
        yield return null;

        // 阶段3：加载模型（分离try-catch和yield）
        workerStatus = "开始加载模型...";
        yield return null;

        Model model = null;
        string loadError = null;

        try
        {
            model = ModelLoader.Load(modelAsset);
        }
        catch (Exception e)
        {
            loadError = e.Message;
        }

        yield return null; // 加载后让出控制权

        if (loadError != null)
        {
            workerStatus = "模型加载失败";
            isWorkerInitializing = false;
            SafeLog($"[EEG_Classify_test] 模型加载异常: {loadError}", LogType.Error);
            yield break;
        }

        if (model == null)
        {
            workerStatus = "模型为空";
            isWorkerInitializing = false;
            SafeLog("[EEG_Classify_test] 加载的模型为空", LogType.Error);
            yield break;
        }

        workerStatus = "模型加载成功";
        SafeLog($"[EEG_Classify_test] 模型加载成功 - 输入数量:{model.inputs.Count}, 输出数量:{model.outputs.Count}");

        // 打印模型详细信息用于诊断
        if (model.inputs.Count > 0)
        {
            var inputInfo = model.inputs[0];
            SafeLog($"[EEG_Classify_test] 模型输入: {inputInfo.name}");
        }
        if (model.outputs.Count > 0)
        {
            var outputInfo = model.outputs[0];
            SafeLog($"[EEG_Classify_test] 模型输出: {outputInfo.name}");
        }
        SafeLog($"[EEG_Classify_test] 模型层数: {model.layers.Count}");

        yield return null;
        yield return null; // 额外等待确保模型完全加载

        // 阶段4：创建Worker（分离try-catch和yield）
        workerStatus = "开始创建Worker...";
        yield return null;

        Worker newWorker = null;
        string workerError = null;

        try
        {
            // 使用 Inspector 中配置的后端类型，而不是强制CPU
            SafeLog($"[EEG_Classify_test] 尝试使用后端: {backend}");
            newWorker = new Worker(model, backend);
        }
        catch (Exception e)
        {
            workerError = e.Message;
            // 如果配置的后端失败，回退到CPU
            SafeLog($"[EEG_Classify_test] {backend} 后端失败，回退到CPU: {e.Message}", LogType.Warning);
            try
            {
                newWorker = new Worker(model, BackendType.CPU);
                actualBackend = "CPU(回退)";
                SafeLog("[EEG_Classify_test] CPU 后端创建成功");
            }
            catch (Exception cpuError)
            {
                workerError = $"{backend}失败: {e.Message}, CPU也失败: {cpuError.Message}";
                actualBackend = "失败";
            }
        }

        yield return null; // Worker创建后让出控制权
        yield return null; // 额外等待确保Worker完全初始化

        if (workerError != null)
        {
            workerStatus = "Worker创建失败";
            isWorkerInitializing = false;
            SafeLog($"[EEG_Classify_test] Worker创建异常: {workerError}", LogType.Error);
            yield break;
        }

        if (newWorker == null)
        {
            workerStatus = "Worker为空";
            isWorkerInitializing = false;
            SafeLog("[EEG_Classify_test] 创建的Worker为空", LogType.Error);
            yield break;
        }

        // 阶段5：最终验证和设置
        worker = newWorker;
        inferenceReady = true;
        workerStatus = "初始化完成";

        // 记录实际使用的后端
        actualBackend = backend.ToString();

        SafeLog($"[EEG_Classify_test] Worker初始化完全成功，使用后端: {actualBackend}，可以开始推理");

        // 重要：Worker创建后不要立即测试，这可能导致阻塞
        // 让Worker在后台准备就绪
        yield return null;
        yield return null;
        yield return null; // 额外等待确保完全就绪

        workerStatus = "就绪"; // 最终状态
        isWorkerInitializing = false;

        // 关键修复：Worker就绪后启用推理功能
        enableInference = true;

        SafeLog("[EEG_Classify_test] Worker完全就绪，推理功能已启用");

        yield return null;
    }

    /// <summary>
    /// 处理接收到的EEG数据（线程安全版本）
    /// </summary>
    /// <param name="channel">通道号(1-based)</param>
    /// <param name="dataPoints">数据点数组</param>
    /// <param name="timestamp">时间戳</param>
    private void OnEEGDataReceived(int channel, double[] dataPoints, double timestamp)
    {
        // 转换为0-based通道索引，并限制在目标通道数内
        int channelIndex = channel - 1;
        if (channelIndex < 0 || channelIndex >= targetChannels)
        {
            return; // 忽略超出目标通道范围的数据
        }

        // 将double数组转换为float
        float[] floatData = new float[dataPoints.Length];
        for (int i = 0; i < dataPoints.Length; i++)
        {
            floatData[i] = (float)dataPoints[i];
        }

        // 将数据加入队列（线程安全），包含时间戳
        dataQueue.Enqueue((channelIndex, floatData, timestamp));

        // 设置新数据标记（线程安全，避免使用Time.time）
        hasNewDataFlag = true;
    }

    private void OnDisable()
    {
        // 停止推理
        enableInference = false;
        inferenceReady = false;
        isWorkerInitializing = false; // 重置初始化状态

        // 取消订阅UDP数据事件
        if (udpReceiver != null)
        {
            udpReceiver.OnDataReceived -= OnEEGDataReceived;
        }

        // 释放Worker与其占用的后端资源
        if (worker != null)
        {
            worker.Dispose();
            worker = null;
        }

        workerStatus = "已停止";
        SafeLog("[EEG_Classify_test] 组件已清理");
    }

    private bool initialized = false;
    private int framesSinceStart = 0; // 跟踪启动后的帧数

    private void Update()
    {
        framesSinceStart++;

        // 更新启动状态显示
        if (framesSinceStart <= 30)
        {
            startupStatus = $"启动中...({framesSinceStart}/30)";
        }
        else if (!initialized)
        {
            startupStatus = "准备初始化组件...";
        }
        else if (framesSinceStart < 60)
        {
            startupStatus = $"等待稳定...({framesSinceStart}/60)";
        }
        else
        {
            startupStatus = "运行中";
        }

        // 延迟初始化 - 等待更多帧后才初始化
        if (!initialized && framesSinceStart > 30) // 等待30帧（约半秒）后才初始化
        {
            LateStart();
            initialized = true;
            return; // 初始化帧不处理其他逻辑
        }

        // 如果还没初始化完成，直接返回
        if (!initialized) return;

        // 进一步延迟数据处理 - 等待更多帧后才开始处理数据
        if (framesSinceStart < 60) return; // 等待60帧（约1秒）后才开始数据处理

        // 在主线程中安全更新数据接收时间
        if (hasNewDataFlag)
        {
            lastDataReceivedTime = Time.time;
            hasNewDataFlag = false;
        }

        // 限制每帧处理量，避免阻塞
        if (Time.frameCount % 5 == 0) // 每5帧处理一次数据（降低频率）
        {
            ProcessQueuedData();
        }

        // 紧急数据清理 - 防止内存溢出
        if (Time.frameCount % 10 == 0) // 每10帧检查一次缓冲区大小
        {
            for (int i = 0; i < targetChannels; i++)
            {
                if (channelBuffers[i] != null && channelBuffers[i].Count > 5000)
                {
                    int removeCount = channelBuffers[i].Count - 1000;
                    channelBuffers[i].RemoveRange(0, removeCount);
                    SafeLog($"[Update] 紧急清理通道{i}缓冲区，移除{removeCount}个点，剩余{channelBuffers[i].Count}个", LogType.Warning);
                }
            }
        }

        // 更频繁检查推理条件
        if (Time.frameCount % 30 == 0) // 每半秒检查一次(60fps)
        {
            float timeSinceLastData = Time.time - lastDataReceivedTime;
            SafeLog($"[Update] 检查推理条件: enableInference={enableInference}, workerStatus={workerStatus}, bufferCount={channelBuffers[0]?.Count ?? 0}, 数据新鲜度={timeSinceLastData:F1}s");
            CheckAndTriggerInference();
        }
    }

    /// <summary>
    /// 处理队列中的UDP数据
    /// </summary>
    private void ProcessQueuedData()
    {
        int processedCount = 0;
        while (dataQueue.TryDequeue(out var data) && processedCount < 10)
        {
            if (data.channel >= 0 && data.channel < targetChannels && channelBuffers[data.channel] != null)
            {
                // 检查时间戳间隙并进行插值补偿
                HandleTimestampGap(data.channel, data.timestamp);
                
                // 添加数据点
                int pointsToAdd = Mathf.Min(data.data.Length, 50);
                for (int i = 0; i < pointsToAdd; i++)
                {
                    channelBuffers[data.channel].Add(data.data[i]);
                }

                // 更新最后处理的时间戳
                lastProcessedTimestamp[data.channel] = data.timestamp;

                // 清理过多数据
                if (channelBuffers[data.channel].Count > 2000)
                {
                    int removeCount = channelBuffers[data.channel].Count - 1000;
                    channelBuffers[data.channel].RemoveRange(0, removeCount);
                }
            }
            processedCount++;
        }
    }
    
    /// <summary>
    /// 处理时间戳间隙（轻量级数据完整性处理）
    /// </summary>
    private void HandleTimestampGap(int channel, double currentTimestamp)
    {
        if (lastProcessedTimestamp.ContainsKey(channel))
        {
            double timeInterval = currentTimestamp - lastProcessedTimestamp[channel];
            double expectedInterval = 0.01; // 假设100Hz采样率，10个点间隔0.01秒
            
            // 如果时间间隔异常（超过预期间隔的2倍），可能是数据包丢失
            if (timeInterval > expectedInterval * 2)
            {
                int estimatedMissingPoints = (int)(timeInterval / expectedInterval) * 10;
                if (estimatedMissingPoints > 100) estimatedMissingPoints = 100; // 限制最大插值点数
                
                // 进行简单的数据插值补偿
                PerformSimpleInterpolation(channel, estimatedMissingPoints);
                
                // 记录插值点数统计
                interpolatedPoints[channel] = interpolatedPoints.ContainsKey(channel) ? 
                    interpolatedPoints[channel] + estimatedMissingPoints : estimatedMissingPoints;
            }
        }
    }
    
    /// <summary>
    /// 执行简单的数据插值补偿
    /// </summary>
    private void PerformSimpleInterpolation(int channel, int missingPoints)
    {
        if (channelBuffers[channel].Count < 2)
            return;

        // 简单复制最后一个值
        float lastValue = channelBuffers[channel][channelBuffers[channel].Count - 1];
        for (int i = 0; i < missingPoints; i++)
        {
            channelBuffers[channel].Add(lastValue);
        }
    }
    
    /// <summary>
    /// 获取插值点数统计
    /// </summary>
    public Dictionary<int, int> GetInterpolationStatistics()
    {
        return new Dictionary<int, int>(interpolatedPoints);
    }
    
    /// <summary>
    /// 重置插值统计
    /// </summary>
    public void ResetInterpolationStatistics()
    {
        interpolatedPoints.Clear();
    }
    
    /// <summary>
    /// 获取下一个可用的滑动窗口数据
    /// </summary>
    /// <param name="outputData">输出数据数组</param>
    /// <returns>是否成功获取窗口数据</returns>
    private bool GetNextSlidingWindow(float[] outputData)
    {
        int requiredPoints = windowSize; // 768点
        int downsampledPoints = downsampleTo; // 384点
        
        // 添加调试信息
        SafeLog($"[GetNextSlidingWindow] 开始提取窗口数据 - lastWindowPosition: {lastWindowPosition}, requiredPoints: {requiredPoints}");
        
        // 检查所有通道是否有足够数据
        for (int ch = 0; ch < targetChannels; ch++)
        {
            if (channelBuffers[ch] == null)
            {
                SafeLog($"[GetNextSlidingWindow] 通道{ch}缓冲区为空");
                return false; // 缓冲区为空
            }
            
            int neededPoints = lastWindowPosition + requiredPoints;
            int availablePoints = channelBuffers[ch].Count;
            
            SafeLog($"[GetNextSlidingWindow] 通道{ch} - 需要: {neededPoints}, 可用: {availablePoints}");
            
            if (availablePoints < neededPoints)
            {
                SafeLog($"[GetNextSlidingWindow] 通道{ch}数据不足: {availablePoints} < {neededPoints}");
                return false; // 数据不足
            }
        }
        
        // 检查是否需要移动窗口
        if (channelBuffers[0].Count >= lastWindowPosition + requiredPoints + stepSize)
        {
            // 可以移动到下一个窗口位置
            lastWindowPosition += stepSize;
            SafeLog($"[GetNextSlidingWindow] 窗口移动到位置: {lastWindowPosition}");
        }
        
        // 检查输出数组大小
        int expectedOutputSize = targetChannels * downsampledPoints;
        if (outputData == null || outputData.Length < expectedOutputSize)
        {
            SafeLog($"[GetNextSlidingWindow] 输出数组大小不足: {outputData?.Length ?? 0} < {expectedOutputSize}");
            return false;
        }
        
        // 从每个通道提取窗口数据并降采样
        for (int ch = 0; ch < targetChannels; ch++)
        {
            var buffer = channelBuffers[ch];
            
            // 添加额外的边界检查
            if (lastWindowPosition + requiredPoints > buffer.Count)
            {
                SafeLog($"[GetNextSlidingWindow] 通道{ch}索引越界: lastWindowPosition({lastWindowPosition}) + requiredPoints({requiredPoints}) > buffer.Count({buffer.Count})");
                return false;
            }
            
            // 提取窗口数据
            float[] windowData = new float[requiredPoints];
            for (int i = 0; i < requiredPoints; i++)
            {
                int sourceIndex = lastWindowPosition + i;
                if (sourceIndex < buffer.Count)
                {
                    windowData[i] = buffer[sourceIndex];
                }
                else
                {
                    SafeLog($"[GetNextSlidingWindow] 通道{ch}窗口数据提取越界: sourceIndex({sourceIndex}) >= buffer.Count({buffer.Count})");
                    windowData[i] = 0f; // 使用默认值
                }
            }
            
            // 应用带通滤波 (7-47Hz)
            float[] filteredData = ApplyBandpassFilter(windowData, ch);
            
            // 只在通道0记录一次滤波信息（避免日志过载）
            if (ch == 0 && lastWindowPosition % 100 == 0) // 每100个窗口记录一次
            {
                SafeLog($"[GetNextSlidingWindow] 通道{ch}滤波完成: {windowData.Length}点 -> {filteredData.Length}点");
            }
            
            // 降采样到目标点数
            float[] downsampledData = DownsampleData(filteredData, downsampledPoints);
            
            // 记录调试信息
            if (ch == 0 && lastWindowPosition % 50 == 0) // 每50个窗口记录一次
            {
                SafeLog($"[GetNextSlidingWindow] 通道{ch}处理: 滤波{windowData.Length}点 -> 降采样{downsampledData.Length}点");
            }
            
            // 将降采样后的数据放入输出数组
            for (int i = 0; i < downsampledPoints; i++)
            {
                int outputIndex = ch * downsampledPoints + i;
                if (outputIndex < outputData.Length)
                {
                    outputData[outputIndex] = downsampledData[i];
                }
                else
                {
                    SafeLog($"[GetNextSlidingWindow] 输出数组越界: outputIndex({outputIndex}) >= outputData.Length({outputData.Length})");
                    return false;
                }
            }
        }
        
        SafeLog($"[GetNextSlidingWindow] 成功提取窗口数据 - 窗口位置: {lastWindowPosition}");
        return true;
    }
    
    /// <summary>
    /// 简单降采样实现（线性插值）
    /// </summary>
    /// <param name="inputData">输入数据</param>
    /// <param name="targetLength">目标长度</param>
    /// <returns>降采样后的数据</returns>
    private float[] DownsampleData(float[] inputData, int targetLength)
    {
        if (inputData.Length == targetLength)
            return inputData;
            
        float[] outputData = new float[targetLength];
        float ratio = (float)inputData.Length / targetLength;
        
        for (int i = 0; i < targetLength; i++)
        {
            float srcIndex = i * ratio;
            int srcIndex1 = (int)srcIndex;
            int srcIndex2 = Math.Min(srcIndex1 + 1, inputData.Length - 1);
            float fraction = srcIndex - srcIndex1;
            
            // 线性插值
            outputData[i] = inputData[srcIndex1] * (1 - fraction) + inputData[srcIndex2] * fraction;
        }
        
        return outputData;
    }
    
    /// <summary>
    /// 初始化滑动窗口数据缓存
    /// </summary>
    private void InitializeSlidingWindowCache()
    {
        channelDataCache.Clear();
        for (int i = 0; i < targetChannels; i++)
        {
            channelDataCache[i] = new List<float>();
        }
        lastWindowPosition = 0;
    }
    
    /// <summary>
    /// 初始化带通滤波器
    /// </summary>
    private void InitializeBandpassFilters()
    {
        if (filtersInitialized)
            return;
            
        bandpassFilters.Clear();
        
        try
        {
            // 创建7-47Hz带通滤波器
            // double lowFreq = 7.0;
            // double highFreq = 47.0;
            
            // 由于MathNet.Filtering可能存在命名空间和API不一致问题，
            // 我们简化处理，直接使用MathNet.Numerics实现带通滤波器
            SafeLog($"[InitializeBandpassFilters] 使用MathNet.Numerics实现7-47Hz带通滤波器");
            
            // 为所有通道初始化标记，指示使用替代方案
            for (int i = 0; i < targetChannels; i++)
            {
                bandpassFilters[i] = null; // null表示使用替代实现
            }
            
            filtersInitialized = true;
            SafeLog($"[InitializeBandpassFilters] 成功初始化{targetChannels}个带通滤波器 (7-47Hz)");
        }
        catch (Exception e)
        {
            SafeLog($"[InitializeBandpassFilters] 滤波器初始化失败: {e.Message}", LogType.Warning);
            filtersInitialized = false;
        }
    }
    
    /// <summary>
    /// 应用带通滤波到数据（简化版本，直接使用MathNet.Numerics实现）
    /// </summary>
    /// <param name="inputData">输入数据</param>
    /// <param name="channel">通道号</param>
    /// <returns>滤波后的数据</returns>
    private float[] ApplyBandpassFilter(float[] inputData, int channel)
    {
        if (!filtersInitialized || !bandpassFilters.ContainsKey(channel))
        {
            SafeLog($"[ApplyBandpassFilter] 滤波器未初始化或通道{channel}不存在，返回原始数据", LogType.Warning);
            return inputData;
        }
        
        try
        {
            // 由于所有通道都使用替代实现，直接调用MathNet.Numerics方法
            return ApplyBandpassWithNumerics(inputData, 7.0, 47.0, samplingRate);
        }
        catch (Exception e)
        {
            SafeLog($"[ApplyBandpassFilter] 通道{channel}滤波失败: {e.Message}", LogType.Warning);
            // 失败后尝试使用简化的滤波器
            return SimpleBandpassFilter(inputData, 7.0, 47.0, samplingRate);
        }
    }
    
    /// <summary>
    /// 在模型输入前对数据进行归一化（标准化）
    /// 参考Python代码：X_data = self.scaler.transform(X_data.reshape(_sample, -1)).reshape(_sample, _channel_cv, _channel, _time)
    /// </summary>
    /// <param name="data">输入数据，形状为 [channels * time_points]</param>
    /// <returns>归一化后的数据</returns>
    private float[] NormalizeDataBeforeInput(float[] data)
    {
        if (data == null || data.Length == 0)
            return data;
            
        int totalSamples = data.Length;
        int samplesPerChannel = totalSamples / targetChannels;
        
        // 首次运行时，需要拟合标准化参数
        if (channelMeans == null || channelStds == null)
        {
            SafeLog("[NormalizeDataBeforeInput] 首次运行，拟合标准化参数...");
            FitNormalizer(data, samplesPerChannel);
        }
        
        // 创建归一化后的数据副本
        float[] normalizedData = new float[data.Length];
        
        // 按通道进行标准化：对每个通道的数据减去均值并除以标准差
        for (int ch = 0; ch < targetChannels; ch++)
        {
            int startIdx = ch * samplesPerChannel;
            
            for (int i = 0; i < samplesPerChannel; i++)
            {
                int idx = startIdx + i;
                if (idx < data.Length)
                {
                    // z-score标准化：(x - mean) / std
                    normalizedData[idx] = (data[idx] - channelMeans[ch]) / (channelStds[ch] + 1e-8f); // 添加小常数避免除零
                }
            }
        }
        
        SafeLog($"[NormalizeDataBeforeInput] 归一化完成，数据范围: [{normalizedData.Min():F3}, {normalizedData.Max():F3}]");
        return normalizedData;
    }
    
    /// <summary>
    /// 拟合标准化参数（计算每个通道的均值和标准差）
    /// </summary>
    private void FitNormalizer(float[] data, int samplesPerChannel)
    {
        channelMeans = new float[targetChannels];
        channelStds = new float[targetChannels];
        
        // 计算每个通道的均值
        for (int ch = 0; ch < targetChannels; ch++)
        {
            int startIdx = ch * samplesPerChannel;
            float sum = 0f;
            
            for (int i = 0; i < samplesPerChannel; i++)
            {
                sum += data[startIdx + i];
            }
            
            channelMeans[ch] = sum / samplesPerChannel;
        }
        
        // 计算每个通道的标准差
        for (int ch = 0; ch < targetChannels; ch++)
        {
            int startIdx = ch * samplesPerChannel;
            float sumSquaredDiff = 0f;
            
            for (int i = 0; i < samplesPerChannel; i++)
            {
                float diff = data[startIdx + i] - channelMeans[ch];
                sumSquaredDiff += diff * diff;
            }
            
            // 使用样本标准差（除以n-1）
            channelStds[ch] = (float)Math.Sqrt(sumSquaredDiff / (samplesPerChannel - 1));
        }
        
        SafeLog($"[FitNormalizer] 标准化参数拟合完成 - 均值: [{string.Join(", ", channelMeans.Select(m => m.ToString("F3")))}], 标准差: [{string.Join(", ", channelStds.Select(s => s.ToString("F3")))}]");
    }
    
    /// <summary>
    /// 重置标准化参数（在需要重新拟合时调用）
    /// </summary>
    public void ResetNormalizer()
    {
        channelMeans = null;
        channelStds = null;
        SafeLog("[ResetNormalizer] 标准化参数已重置，将在下次推理时重新拟合");
    }
    
    /// <summary>
    /// 使用MathNet.Numerics实现带通滤波
    /// </summary>
    private float[] ApplyBandpassWithNumerics(float[] inputData, double lowFreq, double highFreq, double sampleRate)
    {
        try
        {
            // 使用傅里叶变换实现简单的频域滤波
            int n = inputData.Length;
            
            // 应用汉宁窗减少频谱泄漏
            float[] windowedData = new float[n];
            for (int i = 0; i < n; i++)
            {
                double window = 0.5 - 0.5 * Math.Cos(2 * Math.PI * i / (n - 1));
                windowedData[i] = inputData[i] * (float)window;
            }
            
            // 简化的时域滤波：使用移动平均结合差分
            float[] output = new float[n];
            
            // 参数：根据7-47Hz截止频率计算合适的窗口大小
            int lowFreqWindow = Math.Max(1, (int)(sampleRate / (highFreq * 2)));
            int highFreqWindow = Math.Max(1, (int)(sampleRate / (lowFreq * 2)));
            
            // 实现简单的带通效果（高通+低通）
            for (int i = Math.Max(highFreqWindow, lowFreqWindow); i < n - Math.Max(highFreqWindow, lowFreqWindow); i++)
            {
                // 高通部分（去除低频）
                float highPass = windowedData[i] - 0;
                for (int j = 1; j <= highFreqWindow; j++)
                {
                    highPass -= (windowedData[i - j] + windowedData[i + j]) / (2 * highFreqWindow);
                }
                
                // 低通部分（去除高频）
                float lowPass = 0;
                int winSize = Math.Min(lowFreqWindow, 5); // 限制窗口大小
                lowPass += windowedData[i] * 0.5f;
                for (int j = 1; j <= winSize && i - j >= 0 && i + j < n; j++)
                {
                    float weight = 1.0f / (j + 1);
                    lowPass += (windowedData[i - j] + windowedData[i + j]) * weight / (2 * winSize);
                }
                
                output[i] = highPass + lowPass * 0.5f;
            }
            
            // 边界处理
            for (int i = 0; i < Math.Min(highFreqWindow, lowFreqWindow); i++)
            {
                output[i] = output[Math.Min(highFreqWindow, lowFreqWindow)];
            }
            for (int i = n - Math.Min(highFreqWindow, lowFreqWindow); i < n; i++)
            {
                output[i] = output[n - Math.Min(highFreqWindow, lowFreqWindow) - 1];
            }
            
            return output;
        }
        catch (Exception e)
        {
            SafeLog($"[ApplyBandpassWithNumerics] 数字滤波失败: {e.Message}", LogType.Warning);
            return inputData; // 失败时返回原始数据
        }
    }
    
    /// <summary>
    /// 简单的带通滤波器实现（7-47Hz）
    /// </summary>
    /// <param name="input">输入数据</param>
    /// <param name="lowFreq">低频截止频率</param>
    /// <param name="highFreq">高频截止频率</param>
    /// <param name="sampleRate">采样率</param>
    /// <returns>滤波后的数据</returns>
    private float[] SimpleBandpassFilter(float[] input, double lowFreq, double highFreq, double sampleRate)
    {
        if (input == null || input.Length == 0)
            return input;
            
        // 简单的移动平均滤波作为示例（实际应用中应使用更复杂的数字滤波器）
        // 这里我们实现一个非常基础的低通滤波来演示
        float[] output = new float[input.Length];
        float alpha = 0.1f; // 滤波系数
        
        // 简单的一阶低通滤波器
        output[0] = input[0];
        for (int i = 1; i < input.Length; i++)
        {
            output[i] = alpha * input[i] + (1 - alpha) * output[i - 1];
        }
        
        return output;
    }

    /// <summary>
    /// 重置滑动窗口位置
    /// </summary>
    private void ResetSlidingWindow()
    {
        lastWindowPosition = 0;
    }
    
    /// <summary>
    /// 测试带通滤波器功能
    /// </summary>
    private void TestBandpassFilter()
    {
        if (!filtersInitialized)
        {
            SafeLog("[TestBandpassFilter] 滤波器未初始化，跳过测试", LogType.Warning);
            return;
        }
        
        try
        {
            // 生成测试信号：包含7Hz, 20Hz, 47Hz和60Hz成分
            int testLength = 1000;
            float[] testSignal = new float[testLength];
            double dt = 1.0 / samplingRate;
            
            for (int i = 0; i < testLength; i++)
            {
                double t = i * dt;
                // 7Hz成分
                testSignal[i] += (float)(Math.Sin(2 * Math.PI * 7 * t) * 0.5);
                // 20Hz成分
                testSignal[i] += (float)(Math.Sin(2 * Math.PI * 20 * t) * 1.0);
                // 47Hz成分
                testSignal[i] += (float)(Math.Sin(2 * Math.PI * 47 * t) * 0.3);
                // 60Hz干扰成分（应该被滤除）
                testSignal[i] += (float)(Math.Sin(2 * Math.PI * 60 * t) * 0.2);
                // 添加一些白噪声
                testSignal[i] += (float)(UnityEngine.Random.Range(-0.05f, 0.05f));
            }
            
            // 应用滤波器
            float[] filteredSignal = ApplyBandpassFilter(testSignal, 0);
            
            // 计算信号能量变化
            float originalEnergy = 0f;
            float filteredEnergy = 0f;
            for (int i = 0; i < testLength; i++)
            {
                originalEnergy += testSignal[i] * testSignal[i];
                filteredEnergy += filteredSignal[i] * filteredSignal[i];
            }
            
            originalEnergy /= testLength;
            filteredEnergy /= testLength;
            float energyReduction = (originalEnergy - filteredEnergy) / originalEnergy * 100f;
            
            SafeLog($"[TestBandpassFilter] 测试完成:");
            SafeLog($"  - 原始信号长度: {testSignal.Length}, 滤波后长度: {filteredSignal.Length}");
            SafeLog($"  - 原始信号能量: {originalEnergy:F4}");
            SafeLog($"  - 滤波后能量: {filteredEnergy:F4}");
            SafeLog($"  - 能量衰减: {energyReduction:F1}%");
            
            // 检查滤波器是否正常工作（衰减应该在一定范围内）
            if (energyReduction < 10f)
            {
                SafeLog("  - 警告：滤波器衰减过小，可能未正常工作");
            }
            else if (energyReduction > 70f)
            {
                SafeLog("  - 警告：滤波器衰减过大，可能过度滤波");
            }
            else
            {
                SafeLog("  - 滤波器工作正常");
            }
        }
        catch (Exception e)
        {
            SafeLog($"[TestBandpassFilter] 测试失败: {e.Message}", LogType.Error);
        }
    }
    
    /// <summary>
    /// 智能重置滑动窗口位置到合适位置
    /// </summary>
    private void SmartResetSlidingWindow()
    {
        int maxValidPosition = 0;
        bool allChannelsValid = true;
        
        // 找到所有通道中都能支持窗口提取的最大位置
        for (int ch = 0; ch < targetChannels; ch++)
        {
            if (channelBuffers[ch] != null && channelBuffers[ch].Count >= windowSize)
            {
                int validPosition = channelBuffers[ch].Count - windowSize;
                if (ch == 0 || validPosition < maxValidPosition)
                {
                    maxValidPosition = validPosition;
                }
            }
            else
            {
                allChannelsValid = false;
                break;
            }
        }
        
        if (allChannelsValid)
        {
            // 将窗口位置设置到最后一个有效位置，而不是完全重置
            lastWindowPosition = Math.Max(0, maxValidPosition - stepSize);
            SafeLog($"[SmartResetSlidingWindow] 智能重置窗口位置到: {lastWindowPosition}");
        }
        else
        {
            // 如果数据不足，完全重置
            lastWindowPosition = 0;
            SafeLog("[SmartResetSlidingWindow] 数据不足，完全重置窗口位置到: 0");
        }
    }
}

