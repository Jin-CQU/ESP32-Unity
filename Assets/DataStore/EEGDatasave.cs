using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class EEGDatasave : MonoBehaviour
{
    // UI组件（在Inspector拖拽赋值）
    public Toggle[] channelToggles; // 每个通道一个Toggle
    public InputField dataCountInput; // 输入保存点数
    public Button recordButton; // 记录按钮

    // 数据缓存
    private Dictionary<int, List<(double value, double timestamp)>> channelData = new();
    private HashSet<int> selectedChannels = new();
    private HashSet<int> completedChannels = new();
    private HashSet<int> activeChannels = new();
    private readonly object dataLock = new();
    private int saveDataCount = 30000; // 默认保存点数
    private bool isRecording = false;
    private DateTime recordStartTime;
    
    // AR设备存储路径（预留，需要根据实际设备确定）
    private string arDeviceStoragePath = "/storage/emulated/0/Download/Data"; // Android设备默认下载目录

    // 后台线程标记完成，由主线程在下一帧一次性更新UI
    private volatile bool _pendingCompletedUI = false;

    void Start()
    {
        // 订阅UDP_1的数据事件
        UDP_1 udp = FindObjectOfType<UDP_1>();
        if (udp != null)
        {
            udp.OnDataReceived += OnEEGDataReceived;
        }

        if (recordButton != null)
        {
            recordButton.onClick.AddListener(OnRecordButtonClicked);
        }
    }

    // 记录按钮事件
    public void OnRecordButtonClicked()
    {
        // 获取当前按钮文字
        TextMeshProUGUI btnText = recordButton.GetComponentInChildren<TextMeshProUGUI>();
        string currentText = btnText != null ? btnText.text : "";
        
        // 如果状态为"Completed"，重置按钮状态
        if (currentText == "Completed")
        {
            ResetButtonState();
            return;
        }
        
        // 如果已经在记录中，则停止记录
        if (isRecording)
        {
            StopRecording();
            ResetButtonState();
            return;
        }
        
        // 开始记录
        StartRecording();
    }

    private void Update()
    {
        if (_pendingCompletedUI)
        {
            _pendingCompletedUI = false;
            var btnText = recordButton != null ? recordButton.GetComponentInChildren<TextMeshProUGUI>() : null;
            if (btnText != null)
            {
                btnText.text = "Completed";
                btnText.color = Color.red;
            }
        }
    }
    
    private void StartRecording()
    {
        // 更新按钮文字
        TextMeshProUGUI btnText = recordButton.GetComponentInChildren<TextMeshProUGUI>();
        if (btnText != null)
        {
            btnText.text = "Recording";
            btnText.color = Color.green;
        }

        // 获取选中的通道与保存点数
        lock (dataLock)
        {
            selectedChannels.Clear();
            if (channelToggles != null)
            {
                for (int i = 0; i < channelToggles.Length; i++)
                {
                    if (channelToggles[i] != null && channelToggles[i].isOn)
                    {
                        selectedChannels.Add(i + 1); // 通道号从1开始
                    }
                }
            }

            if (dataCountInput != null && int.TryParse(dataCountInput.text, out int count))
            {
                saveDataCount = count;
            }
            else
            {
                saveDataCount = 30000; // 默认值
            }

            // 初始化数据缓存
            channelData.Clear();
            completedChannels.Clear();
            activeChannels.Clear();
            foreach (var channel in selectedChannels)
            {
                channelData[channel] = new List<(double, double)>();
            }
        }
        
        isRecording = true;
        recordStartTime = DateTime.Now;
    }
    
    private void StopRecording(bool setCompletedUI = false)
    {
        bool wasRecording = isRecording;
        isRecording = false;

        if (setCompletedUI && wasRecording)
        {
            _pendingCompletedUI = true;
        }
    }
    
    // 重置按钮状态
    private void ResetButtonState()
    {
        TextMeshProUGUI btnText = recordButton.GetComponentInChildren<TextMeshProUGUI>();
        if (btnText != null)
        {
            btnText.text = "Start Record";
            btnText.color = Color.black;
        }
    }

    // 数据接收回调
    // EEG数据接收回调
    // channel: 通道号（从1开始）
    // data: 当前批次的数据数组
    // timestamp: ESP32传来的时间戳（单位毫秒）
    private void OnEEGDataReceived(int channel, double[] data, double timestamp)
    {
        // 如果未处于记录状态，直接返回
        if (!isRecording)
            return;

        bool shouldSave = false; // 标记是否需要保存数据
        bool shouldStop = false; // 标记是否需要停止记录

        lock (dataLock)
        {
            // 如果该通道未被选中，忽略
            if (!selectedChannels.Contains(channel))
            {
                return;
            }

            // 获取或初始化该通道的数据列表
            if (!channelData.TryGetValue(channel, out var dataList))
            {
                dataList = new List<(double, double)>();
                channelData[channel] = dataList;
            }

            if (data != null && data.Length > 0)
            {
                activeChannels.Add(channel);
            }

            // 添加数据到缓存
            foreach (var value in data)
            {
                // 达到保存点数上限则跳出
                if (dataList.Count >= saveDataCount)
                {
                    break;
                }

                dataList.Add((value, timestamp));

                // 达到保存点数，标记保存并记录完成通道
                if (dataList.Count >= saveDataCount)
                {
                    shouldSave = true;
                    completedChannels.Add(channel);
                    activeChannels.Remove(channel);
                    // 所有正在传输的数据通道都完成则停止
                    shouldStop = activeChannels.Count == 0 && completedChannels.Count > 0;
                    break;
                }
            }
        }

        // 保存数据到文件
        if (shouldSave)
        {
            SaveChannelData(channel);
        }

        // 所有通道完成后，停止记录并更新UI
        if (shouldStop)
        {
            StopRecording(setCompletedUI: true);
        }
    }
    
    // 保存单个通道的数据
    private void SaveChannelData(int channel)
    {
        if (!channelData.ContainsKey(channel) || channelData[channel].Count == 0)
        {
            return;
        }

        string timeStr = recordStartTime.ToString("yyyyMMdd_HHmmss");
        string fileName = $"{timeStr}_Ch{channel}_{channelData[channel].Count}.csv";
        
        // 尝试保存到AR设备目录
        string arPath = Path.Combine(arDeviceStoragePath, fileName);
        bool savedToAR = false;
        
        try
        {
            // 检查AR设备目录是否存在
            if (Directory.Exists(arDeviceStoragePath))
            {
                using (StreamWriter sw = new StreamWriter(arPath))
                {
                    sw.WriteLine("Time,Value");
                    foreach (var (value, timestamp) in channelData[channel])
                    {
                        // 使用ESP32传来的真实时间戳
                        DateTime dataTime = DateTime.FromOADate(timestamp / 86400000.0); // 转换为OADate格式
                        sw.WriteLine($"{dataTime:yyyy-MM-dd HH:mm:ss.fff},{value}");
                    }
                }
                savedToAR = true;
            }
        }
        catch (Exception ex)
        {
            // 保存失败，继续尝试本地保存
            Debug.LogWarning($"保存到AR设备目录失败: {ex.Message}");
        }
        
        // 如果AR设备保存失败，保存到本地
        if (!savedToAR)
        {
            string localPath = Path.Combine(Application.persistentDataPath, fileName);
            try
            {
                using (StreamWriter sw = new StreamWriter(localPath))
                {
                    sw.WriteLine("Time,Value");
                    foreach (var (value, timestamp) in channelData[channel])
                    {
                        // 使用ESP32传来的真实时间戳
                        DateTime dataTime = DateTime.FromOADate(timestamp / 86400000.0); // 转换为OADate格式
                        sw.WriteLine($"{dataTime:yyyy-MM-dd HH:mm:ss.fff},{value}");
                    }
                }
            }
            catch (Exception ex)
            {
                // 本地保存也失败
                Debug.LogWarning($"保存到本地失败: {ex.Message}");
            }
        }
    }
    
    // 设置AR设备存储路径的方法
    public void SetARDeviceStoragePath(string path)
    {
        arDeviceStoragePath = path;
    }
}
