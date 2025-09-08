using System;
using System.Collections.Generic;
using System.Linq;
using System.Collections.Concurrent;
using System.Collections;
using UnityEngine;
using Unity.Sentis;

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
    
    private Worker worker;
    private float[] lastOutput;
    private UDP_1 udpReceiver;
    private bool isInferenceRunning = false; // 推理运行状态标记
    
    // 数据缓冲
    private ConcurrentQueue<(int channel, float[] data)> dataQueue = new ConcurrentQueue<(int, float[])>();
    private List<float>[] channelBuffers = new List<float>[3];
    private int nextStepPosition = 0; // 保留这个变量
    
    // 简化控制
    private float lastInferenceTime = 0f;
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
    
    // 日志
    private ConcurrentQueue<string> logQueue = new ConcurrentQueue<string>();

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
    /// 初始化数据缓冲区
    /// </summary>
    private void InitializeBuffers()
    {
        for (int i = 0; i < targetChannels; i++)
        {
            channelBuffers[i] = new List<float>();
        }
        nextStepPosition = windowSize; // 第一次推理需要收集完整窗口
        SafeLog($"[EEG_Classify_test] 缓冲区初始化完成 - 窗口大小:{windowSize}, 降采样到:{downsampleTo}, 步长:{stepSize}");
    }
    
    /// <summary>
    /// 线程安全的日志记录方法
    /// </summary>
    /// <param name="message">日志消息</param>
    /// <param name="logType">日志类型</param>
    private void SafeLog(string message, LogType logType = LogType.Log)
    {
        // 直接输出日志，不使用队列
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
    /// 检查是否可以执行推理
    /// </summary>
    private bool CanPerformInference()
    {
        if (worker == null) return false;
        
        // 检查所有通道是否都有足够的数据
        int minDataLength = int.MaxValue;
        for (int i = 0; i < targetChannels; i++)
        {
            if (channelBuffers[i] == null) return false;
            minDataLength = Mathf.Min(minDataLength, channelBuffers[i].Count);
        }
        
        return minDataLength >= nextStepPosition;
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
    /// 从缓冲区提取实时EEG数据用于推理
    /// </summary>
    /// <param name="outputData">输出数据数组 (3通道 * 384点)</param>
    /// <returns>是否成功提取数据</returns>
    private bool ExtractRealTimeData(float[] outputData)
    {
        // 检查所有通道是否有足够数据
        int requiredPoints = 384; // 每个通道需要384个点
        for (int ch = 0; ch < targetChannels; ch++)
        {
            if (channelBuffers[ch] == null || channelBuffers[ch].Count < requiredPoints)
            {
                SafeLog($"[ExtractRealTimeData] 通道{ch}数据不足: {channelBuffers[ch]?.Count ?? 0} < {requiredPoints}");
                return false;
            }
        }
        
        // 从每个通道的最新数据中提取384个点
        for (int ch = 0; ch < targetChannels; ch++)
        {
            var buffer = channelBuffers[ch];
            int startIndex = buffer.Count - requiredPoints; // 从最新的384个点开始
            
            for (int i = 0; i < requiredPoints; i++)
            {
                int outputIndex = ch * requiredPoints + i; // 数据排列：ch0[0-383], ch1[384-767], ch2[768-1151]
                outputData[outputIndex] = buffer[startIndex + i];
            }
        }
        
        SafeLog($"[ExtractRealTimeData] 成功提取实时数据: 3通道 × {requiredPoints}点, 数据新鲜度: {Time.time - lastDataReceivedTime:F1}s前");
        return true;
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
            
            // 创建输入张量 - 修复：使用4维形状 (1, 1, 3, 384)
            var inputShape = new TensorShape(1, 1, 3, 384);
            using var input = new Tensor<float>(inputShape, testData);
            
            SafeLog($"[SimpleInferenceTest] 输入张量已创建，形状: {inputShape}");
            
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
                    SafeLog($"[SimpleInferenceTest] ✅ 实时推理#{inferenceCount} - 原始:[{result[0]:F6},{result[1]:F6}] Softmax:[{probabilities[0]:F6},{probabilities[1]:F6}] 预测:类别{predictedClass} 置信度{confidence:F4}({confidence*100:F1}%) 耗时:{lastInferenceTimeMs:F3}ms(S:{scheduleTimeMs:F1}+D:{downloadTimeMs:F1}) 平均:{averageInferenceTimeMs:F3}ms");
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
        StartCoroutine(InferenceCoroutine());
    }
    
    /// <summary>
    /// 分帧推理协程 - 包含异步Worker初始化
    /// </summary>
    private IEnumerator InferenceCoroutine()
    {
        SafeLog("[EEG_Classify_test] 开始推理流程...");
        
        // 设置推理运行状态
        isInferenceRunning = true;
        
        // 如果Worker未初始化，异步初始化
        if (worker == null && modelAsset != null)
        {
            SafeLog("[EEG_Classify_test] Worker未初始化，开始初始化...");
            yield return StartCoroutine(InitializeWorkerAsync());
            if (worker == null)
            {
                SafeLog("[EEG_Classify_test] Worker初始化失败", LogType.Error);
                isInferenceRunning = false;
                yield break;
            }
        }
        
        // 确保Worker完全就绪后再继续
        if (workerStatus != "就绪")
        {
            SafeLog("[EEG_Classify_test] Worker未就绪，跳过推理", LogType.Warning);
            yield break;
        }
        
        // 额外等待确保Worker稳定
        yield return null;
        yield return null;
        
        // 准备输入数据
        var shape = new TensorShape(1, 1, targetChannels, downsampleTo);
        float[] data = new float[targetChannels * downsampleTo];
        
        // 使用实时EEG数据
        bool dataExtracted = ExtractRealTimeData(data);
        if (!dataExtracted)
        {
            SafeLog("[InferenceCoroutine] 实时数据提取失败，跳过推理", LogType.Warning);
            isInferenceRunning = false;
            yield break;
        }
        
        yield return null; // 让出一帧
        
        // 创建张量 - 分离异常处理
        Tensor<float> input = null;
        bool tensorSuccess = false;
        string tensorError = null;
        
        try
        {
            input = new Tensor<float>(shape);
            input.Upload(data);
            tensorSuccess = true;
        }
        catch (Exception e)
        {
            tensorError = e.Message;
        }
        
        if (!tensorSuccess)
        {
            SafeLog($"[EEG_Classify_test] 张量创建错误: {tensorError}", LogType.Error);
            yield break;
        }
        
        yield return null; // 让出一帧
        
        // 执行推理 - 分离异常处理，关键修复：分帧执行避免阻塞
        bool scheduleSuccess = false;
        string scheduleError = null;
        
        // 执行前先让出控制权
        yield return null;
        
        // 🔥 在真正的推理操作前开始计时
        inferenceStopwatch.Restart();
        
        try
        {
            SafeLog("[EEG_Classify_test] 执行Sentis推理...");
            worker.Schedule(input);
            scheduleSuccess = true;
        }
        catch (Exception e)
        {
            scheduleError = e.Message;
        }
        
        // 执行后立即让出控制权
        yield return null;
        yield return null; // 额外等待确保推理完成
        
        if (!scheduleSuccess)
        {
            SafeLog($"[EEG_Classify_test] 推理执行错误: {scheduleError}", LogType.Error);
            // 推理失败时停止计时器
            if (inferenceStopwatch.IsRunning)
            {
                inferenceStopwatch.Stop();
            }
            input?.Dispose();
            yield break;
        }
        
        yield return null; // 让出一帧
        
        // 获取结果 - 关键修复：完全分离阻塞操作和异常处理
        bool resultSuccess = false;
        string resultError = null;
        Tensor<float> outputTensor = null;
        float[] resultArray = null;
        
        // 第一步：获取输出张量
        yield return null; // 获取前让出控制权
        
        try
        {
            outputTensor = worker.PeekOutput() as Tensor<float>;
        }
        catch (Exception e)
        {
            resultError = e.Message;
        }
        
        yield return null; // 获取后让出控制权
        
        // 第二步：下载数据（最可能阻塞的操作）
        if (outputTensor != null && resultError == null)
        {
            yield return null; // 下载前额外等待
            
            try
            {
                resultArray = outputTensor.DownloadToArray();
            }
            catch (Exception e)
            {
                resultError = e.Message;
            }
            
            // 🔥 在核心推理操作完成后立即停止计时
            if (inferenceStopwatch.IsRunning)
            {
                inferenceStopwatch.Stop();
                lastInferenceTimeMs = (float)inferenceStopwatch.Elapsed.TotalMilliseconds;
            }
            
            yield return null; // 下载后立即让出控制权
        }
        else
        {
            // 如果无法获取输出张量，也要停止计时器
            if (inferenceStopwatch.IsRunning)
            {
                inferenceStopwatch.Stop();
            }
        }
        
        // 第三步：处理结果
        if (resultArray != null && resultArray.Length >= 2 && resultError == null)
        {
            // 应用Softmax处理
            float[] probabilities = Softmax(resultArray);
            
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
            
            // 更新平均推理时间（时间已在数据下载后计算）
            totalInferenceTimeMs += lastInferenceTimeMs;
            averageInferenceTimeMs = totalInferenceTimeMs / inferenceCount;
            
            SafeLog($"[InferenceCoroutine] 推理#{inferenceCount} - 类别{predictedClass} 置信度{confidence:F4} Softmax:[{probabilities[0]:F6},{probabilities[1]:F6}] 耗时:{lastInferenceTimeMs:F2}ms 平均:{averageInferenceTimeMs:F2}ms");
            resultSuccess = true;
        }
        
        // 清理资源
        input?.Dispose();
        
        // 最终状态设置和错误处理
        if (!resultSuccess)
        {
            // 推理失败时也要停止计时器
            inferenceStopwatch.Stop();
            
            if (resultError != null)
            {
                SafeLog($"[EEG_Classify_test] 推理失败: {resultError}");
            }
            else
            {
                SafeLog("[EEG_Classify_test] 推理失败: 无输出结果");
            }
        }
        
        isInferenceRunning = false;
        yield return null; // 最后让出一帧
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
        
        // 将数据加入队列（线程安全）
        dataQueue.Enqueue((channelIndex, floatData));
        
        // 设置新数据标记（线程安全，避免使用Time.time）
        hasNewDataFlag = true;
    }
    
    /// <summary>
    /// 降采样函数：从原始数据中等间隔选取目标数量的点
    /// </summary>
    /// <param name="originalData">原始数据</param>
    /// <param name="targetLength">目标长度</param>
    /// <returns>降采样后的数据</returns>
    private float[] Downsample(float[] originalData, int targetLength)
    {
        if (originalData.Length <= targetLength)
        {
            return originalData; // 如果原始数据不够长，直接返回
        }
        
        float[] result = new float[targetLength];
        float step = (float)originalData.Length / targetLength;
        
        for (int i = 0; i < targetLength; i++)
        {
            int index = Mathf.RoundToInt(i * step);
            index = Mathf.Clamp(index, 0, originalData.Length - 1);
            result[i] = originalData[index];
        }
        
        return result;
    }
    
    /// <summary>
    /// 清理旧数据（简化版本）
    /// </summary>
    private void CleanupOldData()
    {
        // 简单保留最新1000个点
        for (int ch = 0; ch < targetChannels; ch++)
        {
            if (channelBuffers[ch].Count > 2000)
            {
                int removeCount = channelBuffers[ch].Count - 1000;
                channelBuffers[ch].RemoveRange(0, removeCount);
                nextStepPosition = windowSize; // 重置位置
            }
        }
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
        
        if (Time.frameCount % 20 == 0) // 每20帧处理一次日志（降低频率）
        {
            ProcessLogQueue();
        }
        
        // 更频繁检查推理条件
        if (Time.frameCount % 30 == 0) // 每半秒检查一次(60fps)
        {
            // 添加状态调试信息，包括数据新鲜度
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
        while (dataQueue.TryDequeue(out var data) && processedCount < 10) // 大幅减少每帧处理数量
        {
            // 确保通道索引有效
            if (data.channel >= 0 && data.channel < targetChannels && channelBuffers[data.channel] != null)
            {
                // 限制每次添加的数据点数量
                int pointsToAdd = Mathf.Min(data.data.Length, 50); // 最多添加50个点
                for (int i = 0; i < pointsToAdd; i++)
                {
                    channelBuffers[data.channel].Add(data.data[i]);
                }
                
                // 关键修复：立即检查并清理过多的数据
                if (channelBuffers[data.channel].Count > 2000)
                {
                    int removeCount = channelBuffers[data.channel].Count - 1000;
                    channelBuffers[data.channel].RemoveRange(0, removeCount);
                    SafeLog($"[ProcessQueuedData] 紧急清理通道{data.channel}，移除{removeCount}个点，剩余{channelBuffers[data.channel].Count}个");
                }
            }
            processedCount++;
        }
        
        // 减少同步频率
        if (Time.frameCount % 180 == 0) // 每3秒同步一次
        {
            SynchronizeChannelData();
        }
    }
    
    /// <summary>
    /// 同步通道数据，确保所有通道长度一致（简化版本）
    /// </summary>
    private void SynchronizeChannelData()
    {
        // 找到最短通道的长度
        int minLength = int.MaxValue;
        for (int i = 0; i < targetChannels; i++)
        {
            if (channelBuffers[i] != null)
            {
                minLength = Mathf.Min(minLength, channelBuffers[i].Count);
            }
        }
        
        // 如果长度差异不大，不需要同步
        bool needSync = false;
        for (int i = 0; i < targetChannels; i++)
        {
            if (channelBuffers[i] != null && channelBuffers[i].Count > minLength + 50)
            {
                needSync = true;
                break;
            }
        }
        
        if (!needSync) return;
        
        // 简单截断过长的通道，但不要太频繁
        if (Time.frameCount % 60 == 0) // 每秒检查一次
        {
            for (int i = 0; i < targetChannels; i++)
            {
                if (channelBuffers[i] != null && channelBuffers[i].Count > minLength + 100)
                {
                    int removeCount = channelBuffers[i].Count - minLength - 50;
                    channelBuffers[i].RemoveRange(0, removeCount);
                    SafeLog($"[EEG_Classify_test] 同步通道{i}数据，移除{removeCount}个点", LogType.Warning);
                }
            }
        }
    }
    
    /// <summary>
    /// 处理日志队列
    /// </summary>
    private void ProcessLogQueue()
    {
        int logCount = 0;
        while (logQueue.TryDequeue(out string logMessage) && logCount < 10) // 限制每帧日志数量
        {
            var parts = logMessage.Split(':');
            if (parts.Length >= 2)
            {
                string logTypeStr = parts[0];
                string message = string.Join(":", parts.Skip(1));
                
                if (Enum.TryParse<LogType>(logTypeStr, out LogType logType))
                {
                    switch (logType)
                    {
                        case LogType.Error:
                            Debug.LogError(message);
                            break;
                        case LogType.Warning:
                            Debug.LogWarning(message);
                            break;
                        default:
                            Debug.Log(message);
                            break;
                    }
                }
                else
                {
                    Debug.Log(logMessage);
                }
            }
            logCount++;
        }
    }
}
