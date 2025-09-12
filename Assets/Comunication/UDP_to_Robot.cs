using System;
using System.Net.Sockets;
using System.Text;
using UnityEngine;
using UnityEngine.UI;

public class UDP_to_Robot : MonoBehaviour
{
    [Header("UDP Configuration")]
    public string broadcastAddress = "255.255.255.255"; // 广播地址
    public int broadcastPort = 31001; // 发送端口
    
    [Header("Message Configuration")]
    public float broadcastInterval = 1.0f; // 广播间隔（秒）
    public float messageIdChangeInterval = 15.0f; // MessageId变化间隔（秒）
    public float eegDataTimeout = 1.0f; // EEG数据超时时间（秒）

    public Text isWalking;
    
    [Header("Component References")]
    public EEG_Classify_test eegClassifier; // EEG分类器引用
    public UDP_1 udpReceiver; // UDP接收器引用
    
    private UdpClient udpClient;
    private float lastBroadcastTime = 0f;
    private float lastMessageIdChangeTime = 0f;
    private int messageId = 1;
    private bool messageIdFlag = false;
    private bool isAccomplish = false; // 是否想象成功
    private float lastEegDataTime = 0f; // 最后一次接收到EEG数据的时间
    private int lastPacketCount = 0; // 上一次的数据包计数
    private int lastMessageId = 0; // 记录上一次的MessageId用于比较
    
    // JSON消息结构
    [Serializable]
    public class RobotMessage
    {
        public string GameState = "playing"; // 游戏状态
        public bool isAccomplish; // 是否想象成功
        public int MessageId; // 消息ID
        public int EegAttention = 87; // 脑电注意力分数
    }
    
    void Start()
    {
        // 初始化UDP客户端
        try
        {
            udpClient = new UdpClient();
            udpClient.EnableBroadcast = true;
            
            // 如果没有指定组件，尝试自动查找
            if (eegClassifier == null)
            {
                eegClassifier = FindObjectOfType<EEG_Classify_test>();
            }
            
            if (udpReceiver == null)
            {
                udpReceiver = FindObjectOfType<UDP_1>();
            }
            
            Debug.Log("UDP to Robot broadcaster initialized");
        }
        catch (Exception e)
        {
            Debug.LogError("Failed to initialize UDP client: " + e.Message);
        }
    }
    
    void Update()
    {
        // 更新最后EEG数据接收时间
        UpdateLastEegDataTime();
        
        // 更新MessageId（每15秒变化一次）
        if (Time.time - lastMessageIdChangeTime >= messageIdChangeInterval)
        {
            messageId++;
            
            lastMessageIdChangeTime = Time.time;
            Debug.Log("MessageId updated to: " + messageId);
        }
        
        // 广播消息（每秒一次）
        if (Time.time - lastBroadcastTime >= broadcastInterval)
        {
            BroadcastMessageToRobot();
            lastBroadcastTime = Time.time;
        }
    }
    
    void UpdateLastEegDataTime()
    {
        // 检查UDP接收器是否存在并且正在运行
        if (udpReceiver != null)
        {
            // 通过反射获取私有字段值
            var receivedPacketCountField = typeof(UDP_1).GetField("receivedPacketCount", 
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            
            if (receivedPacketCountField != null)
            {
                int currentPacketCount = (int)receivedPacketCountField.GetValue(udpReceiver);
                
                // 如果接收到新的数据包，更新最后数据接收时间
                if (currentPacketCount > lastPacketCount)
                {
                    lastEegDataTime = Time.time;
                    lastPacketCount = currentPacketCount;
                }
            }
        }
    }
    
    void BroadcastMessageToRobot()
    {
        try
        {
            string gameState = "stop"; // 默认为stop状态
            bool currentIsAccomplish = false;
            
            // 检查是否有最近的EEG数据
            if (Time.time - lastEegDataTime < eegDataTimeout)
            {
                // 有最近的EEG数据，设置为playing状态
                gameState = "playing";
                
                // 获取EEG分类结果来确定isAccomplish
                if (eegClassifier != null)
                {
                    // 通过反射获取私有字段值
                    var predictedClassField = typeof(EEG_Classify_test).GetField("predictedClass", 
                        System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                    var confidenceField = typeof(EEG_Classify_test).GetField("confidence", 
                        System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                    
                    if (predictedClassField != null && confidenceField != null)
                    {
                        int predictedClass = (int)predictedClassField.GetValue(eegClassifier);
                        float confidence = (float)confidenceField.GetValue(eegClassifier);
                        
                        // 根据分类结果和置信度判断是否想象成功
                        // 这里假设类别1（专注）且置信度>0.5为成功
                        currentIsAccomplish = (predictedClass == 1 && confidence > 0.5f);
                    }
                }
            }
            
            // 创建消息对象
            RobotMessage message = new RobotMessage
            {
                GameState = gameState,
                isAccomplish = currentIsAccomplish,
                MessageId = messageId,
                EegAttention = 87 // 固定值，可根据实际EEG数据调整
            };
            
            // 序列化为JSON（不换行）
            string jsonMessage = JsonUtility.ToJson(message);
            
            // 转换为字节数组
            byte[] data = Encoding.UTF8.GetBytes(jsonMessage);

            // 只在messageId发生变化时更新isWalking显示
            if (messageId != lastMessageId)
            {
                if (message.isAccomplish == true && message.GameState == "playing")
                {
                    isWalking.text = "Walking";
                    isWalking.color = new Color(0.9622642f, 0.1027691f, 0.08775356f);
                }
                else
                {
                    isWalking.text = "Rest";
                    isWalking.color = new Color(0.07199074f, 0.654088f, 0.183762f);
                }
                
                lastMessageId = messageId; // 更新记录的消息ID
            }
            // 发送UDP广播
            udpClient.Send(data, data.Length, broadcastAddress, broadcastPort);
            
            Debug.Log("Broadcasted message: " + jsonMessage);
        }
        catch (Exception e)
        {
            Debug.LogError("Failed to broadcast message: " + e.Message);
        }
    }
    
    void OnDisable()
    {
        // 清理UDP客户端
        if (udpClient != null)
        {
            try
            {
                udpClient.Close();
            }
            catch (Exception e)
            {
                Debug.LogError("Error closing UDP client: " + e.Message);
            }
        }
    }
}
