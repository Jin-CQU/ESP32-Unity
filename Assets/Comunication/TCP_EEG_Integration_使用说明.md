# TCP_Sever EEG 结果传输功能

## 🎯 职责定位
**TCP_Sever 的唯一职责：读取 `EEG_Classify_test` 的推理结果并通过 TCP 发送给机器人**

- ✅ **纯数据传输**: 不进行任何推理计算，只负责结果传输
- ✅ **结果读取**: 通过反射读取 `EEG_Classify_test` 中的 `predictedClass`、`confidence`、`inferenceCount`
- ✅ **网络发送**: 将读取的结果封装为 JSON 并通过 TCP 发送
- ❌ **不做推理**: 推理计算完全在 `EEG_Classify_test` 中进行

## 🚀 线程优化特性
- **后台读取**: EEG 结果读取在后台线程执行，不影响主线程推理
- **反射缓存**: 启动时缓存反射字段，避免运行时开销
- **队列限制**: 限制每帧处理的消息数量，防止帧率下降
- **智能发送**: 结果变化时立即发送，无变化时延长间隔

## 线程架构
```
主线程 (Unity)              后台线程 (TCP)
│                          │
├─ EEG 实时推理            ├─ TCP 连接/发送/接收
├─ UI 渲染                 ├─ 读取推理结果
├─ 处理发送队列            ├─ 结果变化检测
└─ 游戏逻辑                └─ 心跳维护
     ↑                          │
     └─── ConcurrentQueue ──────┘
```

## 配置参数

### EEG Integration 设置
- **Auto Connect EEG**: 自动查找并连接EEG推理组件（默认启用）
- **EEG Send Interval**: 发送EEG结果的间隔（秒，默认0.5秒）
- **Min Confidence**: 只发送置信度高于此值的结果（默认0.6）
- **Use Background Reading**: 使用后台线程读取模式（默认启用，推荐）

### 读取模式选择
1. **后台读取模式** (推荐): 
   - ✅ 最低主线程占用
   - ✅ 不影响 EEG 推理性能
   - ✅ 后台线程读取结果
   
2. **主线程读取模式** (兼容模式):
   - ⚠️ 每帧在主线程执行反射
   - ⚠️ 可能轻微影响推理性能
   - ✅ 兼容性最好

## 工作流程
```
EEG_Classify_test                    TCP_Sever
│                                   │
├─ 接收 UDP EEG 数据                ├─ 后台线程:
├─ 实时推理 (Sentis)                │  ├─ 读取推理结果
├─ 更新 predictedClass               │  ├─ 检测结果变化  
├─ 更新 confidence                   │  └─ 加入发送队列
└─ 更新 inferenceCount               │
                                    ├─ 主线程:
                                    │  ├─ 处理发送队列
                                    │  ├─ 构造 JSON 消息
                                    │  └─ 通过 TCP 发送
                                    │
                                    └─ 机器人接收 JSON
```

## 发送的消息格式

### JSON 格式
```json
{
  "type": "eeg_classification",
  "timestamp": 1693737600000,
  "data": {
    "predicted_class": 0,
    "confidence": 0.8567,
    "class_name": "relaxed",
    "inference_count": 42
  }
}
```

### 字段说明
- `type`: 消息类型，固定为 "eeg_classification"
- `timestamp`: Unix 时间戳（毫秒）
- `predicted_class`: 预测类别（0=放松，1=专注）
- `confidence`: 置信度（0-1之间的浮点数）
- `class_name`: 类别名称（"relaxed" 或 "focused"）
- `inference_count`: 推理次数

## 使用方法

### 1. 自动发送（推荐）
1. 在场景中添加 `TCP_Sever` 组件
2. 确保场景中有 `EEG_Classify_test` 组件
3. 配置 TCP 连接参数（host, port）
4. 启用 `Auto Connect EEG`
5. 调整 `EEG Send Interval` 和 `Min Confidence`

### 2. 手动发送
```csharp
// 获取TCP组件引用
TCP_Sever tcpSever = FindObjectOfType<TCP_Sever>();

// 手动发送EEG结果
tcpSever.SendEEGResult(
    predictedClass: 1,     // 专注状态
    confidence: 0.92f,     // 92% 置信度
    inferenceCount: 100    // 第100次推理
);
```

## 发送策略

### 智能发送
- 只发送置信度 >= `minConfidence` 的结果
- 结果有变化时立即发送
- 结果无变化时按 `eegSendInterval * 2` 间隔发送
- 避免发送过于频繁的重复结果

### 发送条件
1. TCP 连接正常
2. EEG 分类组件存在且工作正常
3. 分类结果有效（predicted_class >= 0）
4. 置信度达到最低要求
5. 满足时间间隔或结果变化条件

## 调试信息

### 控制台日志
- "TCP: 已找到并连接EEG分类组件" - 成功连接EEG组件
- "TCP: 发送EEG分类结果 - 类别:X 置信度:Y" - 成功发送分类结果
- "TCP: 未连接，无法发送EEG结果" - TCP 连接断开

### 检查连接状态
```csharp
TCP_Sever tcpSever = FindObjectOfType<TCP_Sever>();
bool isConnected = tcpSever.IsConnected();
```

## 机器人端接收示例

### Python 接收代码
```python
import socket
import json
import struct

def receive_eeg_data():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('0.0.0.0', 5000))
    sock.listen(1)
    
    conn, addr = sock.accept()
    print(f"Unity 连接来自: {addr}")
    
    while True:
        # 读取4字节长度头
        length_data = conn.recv(4)
        if not length_data:
            break
            
        length = struct.unpack('<I', length_data)[0]
        
        # 读取消息内容
        message_data = conn.recv(length)
        message = message_data.decode('utf-8')
        
        # 解析JSON
        data = json.loads(message)
        if data['type'] == 'eeg_classification':
            eeg_data = data['data']
            print(f"收到EEG分类: {eeg_data['class_name']} (置信度: {eeg_data['confidence']})")
            
            # 根据分类结果控制机器人
            if eeg_data['predicted_class'] == 0:  # 放松
                print("机器人执行放松动作")
            else:  # 专注
                print("机器人执行专注动作")
```

## 注意事项
1. 确保机器人端监听正确的IP和端口
2. 机器人端需要实现长度前缀帧协议（4字节little-endian长度 + UTF-8消息）
3. 建议在机器人端添加消息队列和错误处理
4. 可以根据需要调整 `minConfidence` 来过滤低置信度的结果
