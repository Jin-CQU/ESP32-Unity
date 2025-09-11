using System;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Collections.Concurrent;
using UnityEngine;

/// <summary>
/// TCP 客户端，读取 EEG_Classify_test 的推理结果并发送给机器人。
/// 直接发送 UTF-8 编码的 JSON 数据，后台线程处理网络 I/O。
/// </summary>
public class TCP_Client : MonoBehaviour
{
    [Header("Connection")]
    /// <summary>机器人的 IP 地址</summary>
    public string host = "192.168.1.22";
    /// <summary>机器人的 TCP 端口号</summary>
    public int port = 5000;
    /// <summary>启动时是否自动连接</summary>
    public bool autoConnect = true;

    [Header("EEG Integration")]
    /// <summary>EEG 结果发送间隔（秒）</summary>
    public float eegSendInterval = 0.5f;
    /// <summary>最低置信度阈值，低于此值的结果不会发送</summary>
    public float minConfidence = 0.3f;

    // 内部网络连接相关
    /// <summary>TCP 客户端实例</summary>
    private TcpClient _client;
    /// <summary>TCP 网络流</summary>
    private NetworkStream _stream;
    /// <summary>用于取消异步任务的取消令牌源</summary>
    private CancellationTokenSource _cts;
    /// <summary>线程安全的发送消息队列</summary>
    private readonly ConcurrentQueue<string> _outgoing = new ConcurrentQueue<string>();
    /// <summary>连接操作的线程锁</summary>
    private readonly object _connectLock = new object();

    // EEG相关
    /// <summary>EEG 分类组件的引用</summary>
    private EEG_Classify_test _eegClassifier;
    /// <summary>上次发送 EEG 结果的时间</summary>
    private DateTime _lastEEGSendTime = DateTime.MinValue;
    /// <summary>上次发送的分类结果</summary>
    private int _lastSentClass = -1;
    /// <summary>上次发送的置信度</summary>
    private float _lastSentConfidence = 0f;
    /// <summary>线程安全的 EEG 结果队列</summary>
    private readonly ConcurrentQueue<EEGResult> _eegResultQueue = new ConcurrentQueue<EEGResult>();
    /// <summary>EEG 后台读取任务</summary>
    private Task _eegProcessingTask;
    /// <summary>预测类别字段的反射信息（缓存）</summary>
    private System.Reflection.FieldInfo _predictedClassField;
    /// <summary>置信度字段的反射信息（缓存）</summary>
    private System.Reflection.FieldInfo _confidenceField;
    /// <summary>推理计数字段的反射信息（缓存）</summary>
    private System.Reflection.FieldInfo _inferenceCountField;

    // EEG结果数据结构
    /// <summary>
    /// EEG 推理结果数据结构
    /// 用于在后台线程和主线程之间传递 EEG 分类结果
    /// </summary>
    private struct EEGResult
    {
        /// <summary>预测的类别 (0=放松, 1=专注)</summary>
        public int predictedClass;
        /// <summary>推理置信度 (0.0-1.0)</summary>
        public float confidence;
        /// <summary>累计推理次数</summary>
        public int inferenceCount;
        /// <summary>Unix 时间戳（毫秒）</summary>
        public long timestamp;
    }

    /// <summary>
    /// JSON 消息格式 - 用于序列化发送的 EEG 数据
    /// </summary>
    [System.Serializable]
    public class EEGMessage
    {
        public string type;
        public long timestamp;
        public EEGData data;
    }

    /// <summary>
    /// EEG 数据部分 - JSON 消息的 data 字段
    /// </summary>
    [System.Serializable]
    public class EEGData
    {
        public int predicted_class;
        public float confidence;
        public string class_name;
        public int inference_count;
    }

    /// <summary>
    /// Unity 生命周期 Start - 初始化 TCP 客户端和 EEG 数据读取
    /// 查找 EEG 分类组件，设置反射字段缓存，启动后台处理任务
    /// 如果 autoConnect 为 true，自动开始连接循环
    /// </summary>
    private void Start()
    {
        // 查找EEG分类组件
        _eegClassifier = FindObjectOfType<EEG_Classify_test>();
        if (_eegClassifier != null)
        {
            Debug.Log("TCP: 已找到EEG分类组件");
            CacheEEGReflectionFields();
        }
        else
        {
            Debug.LogWarning("TCP: 未找到EEG分类组件");
        }

        if (autoConnect)
        {
            _ = StartConnectLoopAsync();
        }
    }

    /// <summary>
    /// Unity 生命周期 Update - 主线程处理 EEG 结果队列
    /// 从后台线程的 EEG 结果队列中取出结果并发送给机器人
    /// 限制每帧最多处理 3 个结果，避免主线程阻塞
    /// </summary>
    private void Update()
    {
        // 只处理EEG结果发送队列
        ProcessEEGResultQueue();
    }

    /// <summary>
    /// 异步连接循环 - 主要的网络连接管理方法
    /// 持续尝试连接到指定的 host:port，连接成功后启动发送循环
    /// 连接断开时自动重连（2秒间隔），直到取消或组件销毁
    /// 使用 CancellationToken 支持优雅的任务取消
    /// </summary>
    /// <returns>异步任务</returns>
    private async Task StartConnectLoopAsync()
    {
        lock (_connectLock)
        {
            if (_cts != null) return;
            _cts = new CancellationTokenSource();
        }

        // 启动EEG后台读取任务（在_cts创建后）
        if (_eegClassifier != null && _eegProcessingTask == null)
        {
            StartEEGBackgroundProcessing();
        }

        var token = _cts.Token;
        while (!token.IsCancellationRequested)
        {
            try
            {
                Debug.Log($"TCP: 尝试连接 {host}:{port}");
                _client = new TcpClient();
                await _client.ConnectAsync(host, port);
                _stream = _client.GetStream();
                Debug.Log("TCP: 已连接");

                // 只启动发送循环
                var sendTask = Task.Run(() => SendLoopAsync(token), token);
                await sendTask;
                
                Debug.LogWarning("TCP: 连接断开，准备重连");
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"TCP: 连接错误：{ex.Message}");
            }

            SafeCloseClient();
            await Task.Delay(2000, token); // 2秒后重连
        }

        lock (_connectLock)
        {
            _cts?.Dispose();
            _cts = null;
        }
    }

    /// <summary>
    /// 后台发送循环 - 处理发送队列中的消息
    /// 使用 4 字节 little-endian 长度前缀 + UTF-8 负载的协议格式
    /// 从发送队列中取出消息并通过 TCP 流发送给接收端
    /// 发送失败或连接断开时自动退出循环，触发重连机制
    /// </summary>
    /// <param name="token">取消令牌，用于优雅停止发送循环</param>
    /// <returns>异步任务</returns>
    private async Task SendLoopAsync(CancellationToken token)
    {
        var stream = _stream;
        try
        {
            while (!token.IsCancellationRequested && stream != null && _client != null && _client.Connected)
            {
                if (_outgoing.TryDequeue(out var msg))
                {
                    try
                    {
                        var payload = Encoding.UTF8.GetBytes(msg);
                        // 直接发送JSON数据，不添加长度前缀
                        await stream.WriteAsync(payload, 0, payload.Length, token);
                        await stream.FlushAsync(token);
                    }
                    catch (Exception ex)
                    {
                        Debug.LogWarning($"TCP: 发送异常：{ex.Message}");
                        break;
                    }
                }
                else
                {
                    await Task.Delay(10, token);
                }
            }
        }
        catch (OperationCanceledException) { }
        catch (Exception ex)
        {
            Debug.LogWarning($"TCP: 发送循环异常：{ex.Message}");
        }
    }

    /// <summary>
    /// 将消息入列，实际发送在后台线程。
    /// 消息会以 UTF-8 编码直接发送，不添加长度前缀。
    /// </summary>
    public void Send(string message)
    {
        if (string.IsNullOrEmpty(message)) return;
        // 简单防护：防止队列无限增长
        if (_outgoing.Count > 1000)
        {
            Debug.LogWarning("TCP: 发送队列过大，丢弃消息");
            return;
        }
        _outgoing.Enqueue(message);
    }

    /// <summary>
    /// 主动触发连接（如果 autoConnect=false）
    /// </summary>
    public void Connect()
    {
        if (_cts != null) return;
        _ = StartConnectLoopAsync();
    }

    /// <summary>
    /// 立即断开并停止所有后台任务
    /// </summary>
    public void Disconnect()
    {
        lock (_connectLock)
        {
            if (_cts == null) return;
            _cts.Cancel();
        }
        SafeCloseClient();
    }

    /// <summary>
    /// 安全关闭 TCP 客户端连接
    /// 依次关闭网络流和客户端连接，并清理相关资源
    /// 使用 try-catch 确保即使某个步骤失败也能继续清理其他资源
    /// </summary>
    private void SafeCloseClient()
    {
        try
        {
            try { _stream?.Close(); } catch { }
            try { _client?.Close(); } catch { }
        }
        finally
        {
            _stream = null;
            _client = null;
        }
    }

    /// <summary>
    /// 缓存 EEG 分类组件的反射字段信息
    /// 预先获取 predictedClass、confidence、inferenceCount 等私有字段的反射信息
    /// 避免每次读取时重复进行反射查找，提高性能
    /// 使用 NonPublic | Instance 标志访问私有实例字段
    /// </summary>
    private void CacheEEGReflectionFields()
    {
        try
        {
            var bindingFlags = System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance;
            _predictedClassField = typeof(EEG_Classify_test).GetField("predictedClass", bindingFlags);
            _confidenceField = typeof(EEG_Classify_test).GetField("confidence", bindingFlags);
            _inferenceCountField = typeof(EEG_Classify_test).GetField("inferenceCount", bindingFlags);
            
            if (_predictedClassField == null || _confidenceField == null || _inferenceCountField == null)
            {
                Debug.LogWarning("TCP: 无法缓存EEG反射字段");
            }
            else
            {
                Debug.Log("TCP: EEG反射字段缓存成功");
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"TCP: 缓存EEG反射字段异常: {e.Message}");
        }
    }

    /// <summary>
    /// 启动 EEG 后台处理任务
    /// 在后台线程中定期读取 EEG 分类结果，避免阻塞主线程
    /// 确保 CancellationTokenSource 已初始化后才启动任务
    /// 只能启动一次，重复调用会被忽略
    /// </summary>
    private void StartEEGBackgroundProcessing()
    {
        if (_eegProcessingTask != null) return;
        
        lock (_connectLock)
        {
            if (_cts == null)
            {
                Debug.LogWarning("TCP: 无法启动EEG后台读取");
                return;
            }
            
            _eegProcessingTask = Task.Run(() => EEGBackgroundReadingLoop(_cts.Token), _cts.Token);
            Debug.Log("TCP: EEG后台读取任务已启动");
        }
    }

    /// <summary>
    /// EEG 后台读取循环 - 在后台线程中定期读取 EEG 推理结果
    /// 按照 eegSendInterval 间隔读取 EEG 分类组件的推理结果
    /// 只有在 TCP 连接正常时才读取，读取到的结果放入队列供主线程处理
    /// 使用 realtimeSinceStartup 确保时间计算不受 Time.timeScale 影响
    /// </summary>
    /// <param name="token">取消令牌，用于停止后台读取循环</param>
    /// <returns>异步任务</returns>
    private async Task EEGBackgroundReadingLoop(CancellationToken token)
    {
        var lastReadTime = DateTime.Now;
        
        try
        {
            while (!token.IsCancellationRequested)
            {
                var currentTime = DateTime.Now;
                if ((currentTime - lastReadTime).TotalSeconds >= eegSendInterval)
                {
                    if (_eegClassifier != null)
                    {
                        var eegResult = ReadEEGResultsSafely();
                        if (eegResult.HasValue)
                        {
                            _eegResultQueue.Enqueue(eegResult.Value);
                            Debug.Log($"TCP: 读取到EEG结果 - 类别:{eegResult.Value.predictedClass} 置信度:{eegResult.Value.confidence:F3}");
                        }
                    }
                    lastReadTime = currentTime;
                }

                await Task.Delay(100, token);
            }
        }
        catch (OperationCanceledException)
        {
            Debug.Log("TCP: EEG读取任务已取消");
        }
        catch (Exception e)
        {
            Debug.LogError($"TCP: EEG读取异常: {e.Message}");
        }
    }

    /// <summary>
    /// 安全读取 EEG 推理结果
    /// 使用反射从 EEG_Classify_test 组件中读取私有字段的值
    /// 只有当结果有效且发生变化时才返回结果，避免重复发送相同数据
    /// 应用置信度阈值和变化检测逻辑，提高数据传输效率
    /// </summary>
    /// <returns>有效的 EEG 结果或 null</returns>
    private EEGResult? ReadEEGResultsSafely()
    {
        try
        {
            if (_predictedClassField == null || _confidenceField == null || _inferenceCountField == null)
            {
                Debug.LogWarning("TCP: 反射字段为空");
                return null;
            }

            int currentClass = (int)_predictedClassField.GetValue(_eegClassifier);
            float currentConfidence = (float)_confidenceField.GetValue(_eegClassifier);
            int currentInferenceCount = (int)_inferenceCountField.GetValue(_eegClassifier);

            Debug.Log($"TCP: 原始EEG数据 - 类别:{currentClass} 置信度:{currentConfidence:F3} 推理次数:{currentInferenceCount} 阈值:{minConfidence}");

            if (currentClass < 0 || currentConfidence < minConfidence)
            {
                Debug.LogWarning($"TCP: EEG数据不合格 - 类别:{currentClass} 置信度:{currentConfidence:F3} < 阈值:{minConfidence}");
                return null;
            }

            bool hasChanged = (currentClass != _lastSentClass) || 
                             (Mathf.Abs(currentConfidence - _lastSentConfidence) > 0.1f);
            
            var currentTime = DateTime.Now;
            if (!hasChanged && (currentTime - _lastEEGSendTime).TotalSeconds < eegSendInterval * 2)
            {
                Debug.Log($"TCP: EEG数据未变化，跳过发送");
                return null;
            }

            _lastSentClass = currentClass;
            _lastSentConfidence = currentConfidence;
            _lastEEGSendTime = currentTime;

            Debug.Log($"TCP: 生成有效EEG结果 - 类别:{currentClass} 置信度:{currentConfidence:F3}");

            return new EEGResult
            {
                predictedClass = currentClass,
                confidence = currentConfidence,
                inferenceCount = currentInferenceCount,
                timestamp = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds()
            };
        }
        catch (Exception e)
        {
            Debug.LogWarning($"TCP: 读取EEG结果异常: {e.Message}");
            return null;
        }
    }

    /// <summary>
    /// 处理 EEG 结果队列 - 在主线程中处理后台读取的 EEG 结果
    /// 从线程安全的队列中取出 EEG 结果，构造 JSON 消息并发送
    /// 限制每帧最多处理 3 个结果，避免主线程卡顿
    /// 将 EEG 分类结果转换为标准的 JSON 格式，包含类型、时间戳和详细数据
    /// </summary>
    private void ProcessEEGResultQueue()
    {
        int processedCount = 0;
        while (_eegResultQueue.TryDequeue(out var eegResult) && processedCount < 3)
        {
            try
            {
                if (!IsConnected())
                {
                    Debug.LogWarning($"TCP: 未连接，丢弃EEG结果 - 类别:{eegResult.predictedClass} 置信度:{eegResult.confidence:F3}");
                    processedCount++;
                    continue;
                }

                var eegMessage = new EEGMessage
                {
                    type = "eeg_classification",
                    timestamp = eegResult.timestamp,
                    data = new EEGData
                    {
                        predicted_class = eegResult.predictedClass,
                        confidence = (float)Math.Round(eegResult.confidence, 4),
                        class_name = eegResult.predictedClass == 0 ? "relaxed" : "focused",
                        inference_count = eegResult.inferenceCount
                    }
                };

                string jsonMessage = JsonUtility.ToJson(eegMessage);
                Send(jsonMessage);

                Debug.Log($"TCP: 发送EEG结果 - 类别:{eegResult.predictedClass} 置信度:{eegResult.confidence:F3} JSON:{jsonMessage}");
                processedCount++;
            }
            catch (Exception e)
            {
                Debug.LogError($"TCP: 处理EEG结果异常: {e.Message}");
                processedCount++;
            }
        }
    }

    /// <summary>
    /// 读取并发送EEG推理结果（主线程模式 - 保留兼容性）
    /// 这是一个兼容性方法，提供主线程模式的 EEG 结果处理
    /// 与后台模式相比，这个方法在主线程中直接读取和发送，可能影响帧率
    /// 包含完整的验证逻辑：间隔检查、字段访问、置信度阈值、变化检测
    /// </summary>
    private void CheckAndSendEEGResults()
    {
        // 检查基本条件
        if (_eegClassifier == null || !IsConnected())
        {
            return;
        }

        // 检查发送间隔
        if ((DateTime.Now - _lastEEGSendTime).TotalSeconds < eegSendInterval)
        {
            return;
        }

        // 获取当前推理结果（通过反射访问private字段）
        var predictedClassField = typeof(EEG_Classify_test).GetField("predictedClass", 
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
        var confidenceField = typeof(EEG_Classify_test).GetField("confidence", 
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
        var inferenceCountField = typeof(EEG_Classify_test).GetField("inferenceCount", 
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);

        if (predictedClassField == null || confidenceField == null || inferenceCountField == null)
        {
            Debug.LogWarning("TCP: 无法访问EEG推理结果字段");
            return;
        }

        int currentClass = (int)predictedClassField.GetValue(_eegClassifier);
        float currentConfidence = (float)confidenceField.GetValue(_eegClassifier);
        int currentInferenceCount = (int)inferenceCountField.GetValue(_eegClassifier);

        // 检查是否有新的有效结果
        if (currentClass < 0 || currentConfidence < minConfidence)
        {
            return; // 没有有效推理结果或置信度太低
        }

        // 检查结果是否有变化或足够时间间隔
        bool hasChanged = (currentClass != _lastSentClass) || 
                         (Mathf.Abs(currentConfidence - _lastSentConfidence) > 0.1f);
        
        if (!hasChanged && (DateTime.Now - _lastEEGSendTime).TotalSeconds < eegSendInterval * 2)
        {
            return; // 结果没变化且时间间隔不够
        }

        // 构造发送消息（JSON格式）
        var eegMessage = new EEGMessage
        {
            type = "eeg_classification",
            timestamp = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds(),
            data = new EEGData
            {
                predicted_class = currentClass,
                confidence = (float)Math.Round(currentConfidence, 4),
                class_name = currentClass == 0 ? "relaxed" : "focused",
                inference_count = currentInferenceCount
            }
        };

        string jsonMessage = JsonUtility.ToJson(eegMessage);
        
        // 发送消息
        Send(jsonMessage);
        
        // 更新发送记录
        _lastEEGSendTime = DateTime.Now;
        _lastSentClass = currentClass;
        _lastSentConfidence = currentConfidence;

        Debug.Log($"TCP: [主线程] 发送EEG结果 - 类别:{currentClass}({(currentClass == 0 ? "放松" : "专注")}) 置信度:{currentConfidence:F3} 推理次数:{currentInferenceCount}");
    }

    /// <summary>
    /// 检查TCP连接状态
    /// </summary>
    public bool IsConnected()
    {
        return _client != null && _client.Connected && _stream != null;
    }

    /// <summary>
    /// 手动发送EEG分类结果（供外部调用）
    /// </summary>
    /// <param name="predictedClass">预测类别 (0=放松, 1=专注)</param>
    /// <param name="confidence">置信度 (0-1)</param>
    /// <param name="inferenceCount">推理次数</param>
    public void SendEEGResult(int predictedClass, float confidence, int inferenceCount = 0)
    {
        if (!IsConnected())
        {
            Debug.LogWarning("TCP: 未连接，无法发送EEG结果");
            return;
        }

        var eegMessage = new EEGMessage
        {
            type = "eeg_classification",
            timestamp = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds(),
            data = new EEGData
            {
                predicted_class = predictedClass,
                confidence = (float)Math.Round(confidence, 4),
                class_name = predictedClass == 0 ? "relaxed" : "focused",
                inference_count = inferenceCount
            }
        };

        string jsonMessage = JsonUtility.ToJson(eegMessage);
        Send(jsonMessage);
        
        Debug.Log($"TCP: 手动发送EEG结果 - 类别:{predictedClass} 置信度:{confidence:F3}");
    }

    /// <summary>
    /// Unity 生命周期 OnDisable - 清理资源和停止所有任务
    /// 在组件被禁用或销毁时调用，确保所有后台任务正确停止
    /// 等待 EEG 后台读取任务完成（最多 1 秒），防止资源泄漏
    /// 断开 TCP 连接并清理相关资源
    /// </summary>
    private void OnDisable()
    {
        Disconnect();
        
        // 等待EEG后台读取任务完成
        if (_eegProcessingTask != null)
        {
            try
            {
                _eegProcessingTask.Wait(1000); // 最多等待1秒
            }
            catch (Exception e)
            {
                Debug.LogWarning($"TCP: 等待EEG后台读取任务结束异常: {e.Message}");
            }
            finally
            {
                _eegProcessingTask = null;
            }
        }
    }
}
