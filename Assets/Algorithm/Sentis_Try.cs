using UnityEngine; // 引入Unity引擎核心库
using Unity.Sentis; // 引入Unity Sentis机器学习库

// 手写数字分类器类，继承自MonoBehaviour以便在Unity中使用
public class ClassifyHandwrittenDigit : MonoBehaviour
{
    public Texture2D inputTexture; // 输入纹理（图片），需要在Inspector中赋值
    public ModelAsset modelAsset; // 模型资源，需要在Inspector中赋值

    Model runtimeModel; // 运行时模型对象
    Worker worker; // 模型推理执行器
    public float[] results; // 存储模型输出结果的数组

    void Start() // Unity生命周期函数，游戏开始时执行一次
    {
        Model sourceModel = ModelLoader.Load(modelAsset); // 从模型资源加载模型

        // 创建一个函数图，用于在原模型后添加softmax激活函数
        FunctionalGraph graph = new FunctionalGraph();
        FunctionalTensor[] inputs = graph.AddInputs(sourceModel); // 获取模型的输入张量
        FunctionalTensor[] outputs = Functional.Forward(sourceModel, inputs); // 通过模型前向传播获取输出
        FunctionalTensor softmax = Functional.Softmax(outputs[0]); // 对输出应用softmax函数，用于概率分布

        // 编译函数图，创建包含softmax的完整运行时模型
        runtimeModel = graph.Compile(softmax);

        // 创建输入数据张量 - 匹配模型输入形状 (1, 3, 299, 299) NCHW格式
        // 1=批次大小, 3=颜色通道数(RGB), 299x299=图片尺寸
        //using Tensor<float> inputTensor = new Tensor<float>(new TensorShape(1, 3, 299, 299));
        //using Tensor<float> inputTensor = TextureConverter.ToTensor(inputTexture, width: 28, height: 28, channels: 1);

        using Tensor<float> inputTensor = new Tensor<float>(new TensorShape(1, 1, 28, 28));
        // 设置纹理转换参数，指定NCHW布局（批次-通道-高度-宽度）
        TextureTransform transform = new TextureTransform()
            .SetTensorLayout(TensorLayout.NCHW);
        
        // 将输入纹理转换为张量数据
        TextureConverter.ToTensor(inputTexture, inputTensor, transform);

        // 创建模型推理引擎，使用GPU计算后端
        worker = new Worker(runtimeModel, BackendType.GPUCompute);

        // 使用输入数据运行模型推理
        worker.Schedule(inputTensor);

        // 获取模型输出结果张量
        Tensor<float> outputTensor = worker.PeekOutput() as Tensor<float>;

        // 输出张量可能仍在GPU上计算中
        // 执行阻塞下载调用，将结果从GPU复制到CPU内存
        results = outputTensor.DownloadToArray();
    }

    void OnDisable() // Unity生命周期函数，对象被禁用时执行
    {
        // 释放GPU内存，清理推理引擎占用的资源
        worker.Dispose();
    }
}