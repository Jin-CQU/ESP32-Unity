using System.Collections.Generic;
using System.Collections.Concurrent;
using UnityEngine.UI;
using UnityEngine;

public class EEGVisualizer : MonoBehaviour
{
    [Header("显示配置")]
    [SerializeField] private int channelNumber = 1;
    [SerializeField] private int maxDataPoints = 1000;
    [SerializeField] private float yScale = 1.0f;
    [SerializeField] private Color lineColor = Color.green;
    [SerializeField] private bool useWorldSpace = false;
    [SerializeField] private float lineWidth = 1.0f; // 线条宽度，可在Inspector中调整
    
    [Header("组件引用")]
    [SerializeField] private LineRenderer lineRenderer;
    [SerializeField] private Text rmsAmpLabel; // 在Inspector中拖拽Text组件

    private ConcurrentQueue<float> dataQueue = new ConcurrentQueue<float>();
    private UDP_1 udpReceiver;
    private double[] channelRMS = new double[4];
    private double[] channelAMP = new double[4];

    void Start()
    {
        InitializeLineRenderer();
        SubscribeToUDPEvents();
    }
    
    void Update()
    {
        UpdateLineRenderer();
        UpdateRMSAMP();
    }
    
    void OnDestroy()
    {
        UnsubscribeFromUDPEvents();
    }
    
    private void InitializeLineRenderer()
    {
        if (lineRenderer == null)
        {
            lineRenderer = GetComponent<LineRenderer>();
        }
        
        if (lineRenderer == null)
        {
            lineRenderer = gameObject.AddComponent<LineRenderer>();
        }
        
        if (lineRenderer != null)
        {
            // 创建材质并设置正确的shader
            Material material = new Material(Shader.Find("Sprites/Default"));
            if (material == null)
            {
                // 如果找不到Sprites/Default shader，使用Unlit/Color
                material = new Material(Shader.Find("Unlit/Color"));
            }
            
            material.color = lineColor;
            lineRenderer.material = material;
            
            // 设置LineRenderer属性
            lineRenderer.startWidth = lineWidth;
            lineRenderer.endWidth = lineWidth;
            lineRenderer.positionCount = 0;
            lineRenderer.useWorldSpace = useWorldSpace;
            
            // 关键设置：确保在UI中正确渲染
            lineRenderer.sortingOrder = 100; // 提高排序顺序，确保在UI元素之上
            lineRenderer.sortingLayerName = "Default"; // 使用Default排序层
            
            // 确保LineRenderer可见
            lineRenderer.enabled = true;
            
            // 设置渲染队列，确保在UI之后渲染
            if (material != null)
            {
                material.renderQueue = 3000; // UI渲染队列之后
            }
            
            Debug.Log($"LineRenderer初始化完成 - 排序顺序: {lineRenderer.sortingOrder}, 排序层: {lineRenderer.sortingLayerName}");
        }
        else
        {
            Debug.LogError("无法创建或找到LineRenderer组件");
        }
    }
    
    private void SubscribeToUDPEvents()
    {
        udpReceiver = FindObjectOfType<UDP_1>();
        if (udpReceiver != null)
        {
            udpReceiver.OnDataReceived += OnDataReceived;
            udpReceiver.OnRMSAMPReceived += OnRMSAMPReceivedHandler;
        }
    }

    private void UdpReceiver_OnRMSAMPReceived(int arg1, double arg2, double arg3)
    {
        throw new System.NotImplementedException();
    }

    private void UnsubscribeFromUDPEvents()
    {
        if (udpReceiver != null)
        {
            udpReceiver.OnDataReceived -= OnDataReceived;
            udpReceiver.OnRMSAMPReceived -= OnRMSAMPReceivedHandler;
        }
    }
    
    private void OnDataReceived(int channel, double[] data, double timestamp)
    {
        if (channel == channelNumber)
        {
            foreach (double value in data)
            {
                dataQueue.Enqueue((float)value);
            }
        }
    }

    private void OnRMSAMPReceivedHandler(int channel, double rms, double amp)
    {
        // channel 从1开始，数组下标从0开始
        int idx = channel - 1; //通道索引
        if (idx >= 0 && idx < channelRMS.Length)
        {
            channelRMS[idx] = rms;
            channelAMP[idx] = amp;
        }
    }

    private void UpdateLineRenderer()
    {
        if (lineRenderer == null) return;
        
        // 安全地获取数据
        if (dataQueue.Count == 0) return;
        
        // 限制数据点数量
        while (dataQueue.Count > maxDataPoints)
        {
            float temp;
            dataQueue.TryDequeue(out temp);
        }
        
        // 再次检查数据量
        if (dataQueue.Count == 0) return;
        
        // 安全地转换为数组
        float[] dataArray;
        try
        {
            dataArray = dataQueue.ToArray();
        }
        catch
        {
            return; // 如果转换失败，直接返回
        }
        
        // 检查数组长度
        if (dataArray == null || dataArray.Length == 0) return;
        
        Vector3[] positions = new Vector3[dataArray.Length];
        
        // 获取当前GameObject的RectTransform
        RectTransform rectTransform = GetComponent<RectTransform>();
        if (rectTransform != null)
        {
            // 使用UI坐标系统
            float width = rectTransform.rect.width;
            float height = rectTransform.rect.height;
            
            for (int i = 0; i < dataArray.Length; i++)
            {
                // X坐标：从左到右填充面板宽度
                float x = (float)i / dataArray.Length * width - width / 2f;
                // Y坐标：数据值映射到面板高度，居中显示
                float y = dataArray[i] * yScale;
                // 限制Y值在面板范围内
                y = Mathf.Clamp(y, -height / 2f, height / 2f);
                
                positions[i] = new Vector3(x, y, 0);
            }
        }
        else
        {
            // 如果没有RectTransform，使用原来的世界坐标
            for (int i = 0; i < dataArray.Length; i++)
            {
                float x = (float)i / dataArray.Length * 10f - 5f;
                float y = dataArray[i] * yScale;
                positions[i] = new Vector3(x, y, 0);
            }
        }
        
        // 安全地设置LineRenderer
        try
        {
            lineRenderer.positionCount = positions.Length;
            lineRenderer.SetPositions(positions);
            
            // 确保LineRenderer在运行时可见
            if (!lineRenderer.enabled)
            {
                lineRenderer.enabled = true;
            }
        }
        catch (System.Exception e)
        {
            Debug.LogWarning($"LineRenderer更新失败: {e.Message}");
        }
    }

    private void UpdateRMSAMP()
    {
        if (rmsAmpLabel != null && channelNumber >= 0 && channelNumber < channelRMS.Length)
        {
            rmsAmpLabel.text = $"Ch{channelNumber} RMS: {channelRMS[channelNumber - 1]:F2}  AMP: {channelAMP[channelNumber - 1]:F2}";
        }
    }
    
}
