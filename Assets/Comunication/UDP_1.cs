using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using System.Linq;
using UnityEngine.UI;
using UnityEngine;

public class UDP_1 : MonoBehaviour
{
    [Header("UDP配置")]
    [SerializeField] private string listenIP = "192.168.1.109";
    [SerializeField] private int listenPort = 30300;
    
    [Header("设备配置")]
    [SerializeField] private string serialNumber = "000000"; // 6位十六进制序列号
    [SerializeField] private WORK_MODE workMode = WORK_MODE.EEG;
    
    [Header("调试信息")]
    [SerializeField] private bool showDebugInfo = true;

    [Header("运行时显示")]
    public Text ipText; // 拖拽UI Text到此字段

    // 工作模式枚举
    public enum WORK_MODE
    {
        SEMG_LOW_FRE,
        SEMG_HIGH_FRE,
        EEG,
    }
    
    private UdpClient udpClient;
    private Thread receiveThread;
    private bool isRunning = false;
    
    // CRC校验表
    private byte[] crcTable = new byte[256];
    
    // 数据包格式相关
    private byte[] frameHeader = new byte[2];
    private int timestampOffset = 0;
    private int crcOffset = 0;
    private int frameLength = 0;
    private byte[] serialNumBytes = new byte[3];
    
    // 统计信息
    private int receivedPacketCount = 0;
    private int totalBytesReceived = 0;
    private int validPacketCount = 0;
    private int crcErrorCount = 0;
    private int serialErrorCount = 0;
    
    // RMS计算相关
    private List<double[]> slideSqrtBuf = new List<double[]>();
    private int[] curIdx = new int[4];
    private double[] winBuf = new double[50];
    // private int winPos = 0;
    // private double winSum = 0;
    
    // 数据回调事件
    public event Action<int, double[], double> OnDataReceived; // 通道号, 数据数组, 时间戳
    public event Action<int, double, double> OnRMSAMPReceived; // 通道号, RMS值, AMP值
    
    void Start()
    {
        listenIP = GetLocalIPAddress(); // 自动获取本机IP
        if (ipText != null)
        {
            ipText.text = "本机IP: " + listenIP;
        }
        InitializeCRC(); // 初始化CRC校验
        InitializeFrameFormat(); // 初始化数据帧格式
        InitializeRMSBuffers(); // 初始化RMS缓冲区
        StartUDPReceiver();  // 启动UDP接收
    }
    
    void OnDestroy()
    {
        StopUDPReceiver();
    }
    
    void Update()
    {
        // 更新IP显示
        if (ipText != null)
        {
            ipText.text = "本机IP: " + listenIP;
        }
        // 在Update中显示调试信息
        if (showDebugInfo && Time.frameCount % 60 == 0) // 每秒显示一次
        {
            Debug.Log($"UDP状态: {(isRunning ? "运行" : "停止")}, " +
                     $"总包数: {receivedPacketCount}, 有效包: {validPacketCount}, " +
                     $"CRC错误: {crcErrorCount}, 序列号错误: {serialErrorCount}");
        }
    }

    // 获取本机IP地址
    private string GetLocalIPAddress()
    {
        string localIP = "";
        try
        {
            var host = Dns.GetHostEntry(Dns.GetHostName());
            foreach (var ip in host.AddressList)
            {
                if (ip.AddressFamily == AddressFamily.InterNetwork)
                {
                    localIP = ip.ToString();
                    break;
                }
            }
        }
        catch (Exception e)
        {
            Debug.LogError("获取本机IP失败: " + e.Message);
        }
        return localIP;
    }

    // 初始化CRC校验表
    private void InitializeCRC()
    {
        for (int i = 0; i < 256; i++)
        {
            int cur = i;
            for (int j = 0; j < 8; j++)
            {
                if ((cur & 0x80) != 0)
                {
                    cur = (cur << 1) ^ 0xD5;
                }
                else
                {
                    cur <<= 1;
                }
            }
            crcTable[i] = (byte)cur;
        }
    }
    
    // 计算CRC校验值
    private byte CalculateCRC(byte[] data, int length)
    {
        byte crc = 0;
        for (int i = 0; i < length; i++)
        {
            crc = crcTable[crc ^ data[i]];
        }
        return crc;
    }
    
    // 初始化数据包格式
    private void InitializeFrameFormat()
    {
        // 解析序列号
        if (serialNumber.Length == 6)
        {
            for (int i = 0; i < 3; i++)
            {
                serialNumBytes[i] = Convert.ToByte(serialNumber.Substring(i * 2, 2), 16);
            }
        }
        
        // 根据工作模式设置帧格式
        switch (workMode)
        {
            case WORK_MODE.SEMG_LOW_FRE:
                frameHeader[0] = 0xAA;
                frameHeader[1] = 0xAA;
                timestampOffset = 38;
                crcOffset = 46;
                frameLength = 47;
                break;
                
            case WORK_MODE.SEMG_HIGH_FRE:
                frameHeader[0] = 0xAB;
                frameHeader[1] = 0xAB;
                timestampOffset = 308;
                crcOffset = 316;
                frameLength = 317;
                break;
                
            case WORK_MODE.EEG:
                frameHeader[0] = 0xAD;
                frameHeader[1] = 0xAD;
                // 根据采样率确定帧格式
                timestampOffset = 38;
                crcOffset = 46;
                frameLength = 47;
                break;
        }
        
        Debug.Log($"帧格式初始化: 帧头[{frameHeader[0]:X2} {frameHeader[1]:X2}], " +
                 $"时间戳偏移:{timestampOffset}, CRC偏移:{crcOffset}, 帧长:{frameLength}");
    }
    
    // 初始化RMS计算缓冲区
    private void InitializeRMSBuffers()
    {
        slideSqrtBuf.Clear();
        for (int i = 0; i < 4; i++)
        {
            slideSqrtBuf.Add(new double[1000]); // 默认1000点窗口
            curIdx[i] = 0;
        }
    }
    
    public void StartUDPReceiver()
    {
        if (isRunning) return;
        
        try
        {
            udpClient = new UdpClient();
            udpClient.Client.SetSocketOption(SocketOptionLevel.Socket, SocketOptionName.ReuseAddress, true);
            udpClient.Client.Bind(new IPEndPoint(IPAddress.Parse(listenIP), listenPort));
            
            isRunning = true;
            receiveThread = new Thread(ReceiveData);
            receiveThread.IsBackground = true;
            receiveThread.Start();
            
            Debug.Log($"UDP接收器已启动 - 监听 {listenIP}:{listenPort}");
        }
        catch (Exception e)
        {
            Debug.LogError($"启动UDP接收器失败: {e.Message}");
            isRunning = false;
        }
    }
    
    public void StopUDPReceiver()
    {
        isRunning = false;
        
        if (receiveThread != null && receiveThread.IsAlive)
        {
            receiveThread.Join(1000);
        }
        
        if (udpClient != null)
        {
            udpClient.Close();
            udpClient = null;
        }
        
        Debug.Log("UDP接收器已停止");
    }
    
    private void ReceiveData()
    {
        while (isRunning)
        {
            try
            {
                IPEndPoint remoteEndPoint = new IPEndPoint(IPAddress.Any, 0);
                byte[] receivedData = udpClient.Receive(ref remoteEndPoint);
                
                if (receivedData != null && receivedData.Length > 0)
                {
                    receivedPacketCount++;
                    totalBytesReceived += receivedData.Length;
                    
                    // 处理接收到的数据
                    ProcessReceivedData(receivedData, receivedData.Length);
                }
            }
            catch (SocketException e)
            {
                if (isRunning)
                {
                    Debug.LogError($"UDP接收错误: {e.Message}");
                }
            }
            catch (Exception e)
            {
                if (isRunning)
                {
                    Debug.LogError($"UDP接收异常: {e.Message}");
                }
            }
        }
    }
    
    // 处理接收到的数据
    private void ProcessReceivedData(byte[] buffer, int length)
    {
        // 查找帧头
        for (int i = 0; i < length - frameLength; i++)
        {
            if (buffer[i] == frameHeader[0] && buffer[i + 1] == frameHeader[1])
            {
                // 提取完整帧
                byte[] frame = new byte[frameLength];
                Array.Copy(buffer, i, frame, 0, frameLength);
                
                // 处理单帧数据
                ProcessFrame(frame);
            }
        }
    }
    
    // 处理单帧数据
    private void ProcessFrame(byte[] frame)
    {
        // CRC校验
        byte calculatedCRC = CalculateCRC(frame, crcOffset);
        if (calculatedCRC != frame[crcOffset])
        {
            crcErrorCount++;
            if (showDebugInfo)
            {
                Debug.LogWarning($"CRC校验失败: 期望{frame[crcOffset]:X2}, 实际{calculatedCRC:X2}");
            }
            return;
        }
        
        // 序列号验证
        if (frame[2] != serialNumBytes[0] || frame[3] != serialNumBytes[1] || frame[4] != serialNumBytes[2])
        {
            serialErrorCount++;
            if (showDebugInfo)
            {
                Debug.LogWarning($"序列号不匹配: 期望[{serialNumBytes[0]:X2} {serialNumBytes[1]:X2} {serialNumBytes[2]:X2}], " +
                               $"实际[{frame[2]:X2} {frame[3]:X2} {frame[4]:X2}]");
            }
            return;
        }
        
        // 解析数据
        ParseFrameData(frame);
        validPacketCount++;
    }
    
    // 解析帧数据
    private void ParseFrameData(byte[] frame)
    {
        try
        {
            // 获取数据长度
            int dataLength = (frame[6] << 8) | frame[7];
            
            // 获取通道号
            int channel = frame[5];
            if (workMode == WORK_MODE.EEG)
            {
                channel += 1; // 脑电模式通道号+1
            }
            
            // 获取时间戳
            double timestamp = 0;
            if (timestampOffset + 8 <= frame.Length)
            {
                timestamp = (double)((long)frame[timestampOffset] << 56 |
                                   (long)frame[timestampOffset + 1] << 48 |
                                   (long)frame[timestampOffset + 2] << 40 |
                                   (long)frame[timestampOffset + 3] << 32 |
                                   (long)frame[timestampOffset + 4] << 24 |
                                   (long)frame[timestampOffset + 5] << 16 |
                                   (long)frame[timestampOffset + 6] << 8 |
                                   (long)frame[timestampOffset + 7]);
            }
            
            // 解析数据点
            int dataPointCount = dataLength / 3;
            double[] dataPoints = new double[dataPointCount];
            
            for (int i = 0; i < dataPointCount; i++)
            {
                int dataOffset = 8 + i * 3;
                if (dataOffset + 2 < frame.Length)
                {
                    int rawValue;
                    if ((frame[dataOffset] & 0x80) != 0)
                    {
                        // 负数
                        rawValue = (frame[dataOffset] << 16) | (frame[dataOffset + 1] << 8) | frame[dataOffset + 2] | (0xFF << 24);
                    }
                    else
                    {
                        // 正数
                        rawValue = (frame[dataOffset] << 16) | (frame[dataOffset + 1] << 8) | frame[dataOffset + 2];
                    }
                    
                    // 转换为微伏值
                    dataPoints[i] = rawValue / 1000.0;
                }
            }
            
                         // 触发数据接收事件
             OnDataReceived?.Invoke(channel, dataPoints, timestamp);
             
             // 计算RMS和AMP
             double rms = CalculateSlideSqrtRMS(dataPoints, channel - 1);
             double amp = Math.Sqrt(2) * rms;
             
             // 触发RMS和AMP事件
             OnRMSAMPReceived?.Invoke(channel, rms, amp);
             
             if (showDebugInfo)
             {
                 Debug.Log($"解析成功: 通道{channel}, 数据点{dataPointCount}, 时间戳{timestamp}, RMS:{rms:F3}, AMP:{amp:F3}");
             }
        }
        catch (Exception e)
        {
            Debug.LogError($"解析帧数据失败: {e.Message}");
        }
    }
    
    // 设置工作模式
    public void SetWorkMode(WORK_MODE mode)
    {
        workMode = mode;
        InitializeFrameFormat();
    }
    
    // 设置序列号
    public void SetSerialNumber(string serialNum)
    {
        if (serialNum.Length == 6)
        {
            serialNumber = serialNum;
            InitializeFrameFormat();
        }
        else
        {
            Debug.LogError("序列号必须是6位十六进制字符串");
        }
    }
    
    // 获取统计信息
    public (int totalPackets, int validPackets, int crcErrors, int serialErrors, int totalBytes) GetStatistics()
    {
        return (receivedPacketCount, validPacketCount, crcErrorCount, serialErrorCount, totalBytesReceived);
    }
    
    // 重置统计信息
    public void ResetStatistics()
    {
        receivedPacketCount = 0;
        validPacketCount = 0;
        crcErrorCount = 0;
        serialErrorCount = 0;
        totalBytesReceived = 0;
    }
    
    // 计算滑动窗口RMS值
    private double CalculateSlideSqrtRMS(double[] data, int channelIdx)
    {
        double ret = 0;
        double validSum = 0;
        
        // 将数据添加到滑动窗口缓冲区
        for (int i = 0; i < data.Length; i++)
        {
            slideSqrtBuf[channelIdx][curIdx[channelIdx]] = data[i];
            curIdx[channelIdx] += 1;
            if (slideSqrtBuf[channelIdx].Length == curIdx[channelIdx])
            {
                curIdx[channelIdx] = 0;
            }
        }
        
        // 计算有效值的平方和
        validSum = slideSqrtBuf[channelIdx].Select(x => Math.Pow(x, 2)).Sum();
        ret = Math.Sqrt(validSum / slideSqrtBuf[channelIdx].Length);
        
        return ret;
    }
}


