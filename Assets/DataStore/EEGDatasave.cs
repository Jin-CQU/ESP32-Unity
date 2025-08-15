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
    private int saveDataCount = 1000; // 默认保存点数
    private bool isRecording = false;
    private DateTime recordStartTime;
    
    // AR设备存储路径（预留，需要根据实际设备确定）
    private string arDeviceStoragePath = "/storage/emulated/0/Download/Data"; // Android设备默认下载目录

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
    
    private void StartRecording()
    {
        // 更新按钮文字
        TextMeshProUGUI btnText = recordButton.GetComponentInChildren<TextMeshProUGUI>();
        if (btnText != null)
        {
            btnText.text = "Recording...";
        }

        // 获取选中的通道
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

        // 获取保存点数
        if (dataCountInput != null && int.TryParse(dataCountInput.text, out int count))
        {
            saveDataCount = count;
        }
        else
        {
            saveDataCount = 1000; // 默认值
        }

        // 初始化数据缓存
        channelData.Clear();
        foreach (var channel in selectedChannels)
        {
            channelData[channel] = new List<(double, double)>();
        }
        
        isRecording = true;
        recordStartTime = DateTime.Now;
    }
    
    private void StopRecording()
    {
        isRecording = false;
    }
    
    // 重置按钮状态
    private void ResetButtonState()
    {
        TextMeshProUGUI btnText = recordButton.GetComponentInChildren<TextMeshProUGUI>();
        if (btnText != null)
        {
            btnText.text = "Start Record";
        }
    }

    // 数据接收回调
    private void OnEEGDataReceived(int channel, double[] data, double timestamp)
    {
        if (!isRecording || !selectedChannels.Contains(channel))
            return;

        if (!channelData.ContainsKey(channel))
            channelData[channel] = new List<(double, double)>();

        foreach (var value in data)
        {
            if (channelData[channel].Count < saveDataCount)
            {
                channelData[channel].Add((value, timestamp));
                
                // 检查是否达到指定点数
                if (channelData[channel].Count >= saveDataCount)
                {
                    // 立即保存该通道的数据
                    SaveChannelData(channel);
                    
                    // 检查所有通道是否都完成
                    bool allChannelsComplete = true;
                    foreach (var ch in selectedChannels)
                    {
                        if (channelData.ContainsKey(ch) && channelData[ch].Count < saveDataCount)
                        {
                            allChannelsComplete = false;
                            break;
                        }
                    }
                    
                    if (allChannelsComplete)
                    {
                        StopRecording();
                        
                        // 更新按钮文字为完成状态
                        TextMeshProUGUI btnText = recordButton.GetComponentInChildren<TextMeshProUGUI>();
                        if (btnText != null)
                        {
                            btnText.text = "Completed";
                        }
                    }
                }
            }
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
            }
        }
    }
    
    // 设置AR设备存储路径的方法
    public void SetARDeviceStoragePath(string path)
    {
        arDeviceStoragePath = path;
    }
}
