using UnityEngine; // 引入Unity引擎核心库
using Unity.Sentis; // 引入Unity Sentis机器学习库

// 手写数字分类器类，继承自MonoBehaviour以便在Unity中使用
public class ClassifyHandwrittenDigit_b : MonoBehaviour
{
    public Texture2D inputTexture; // 输入纹理（图片），需要在Inspector中赋值
    public ModelAsset modelAsset; // 模型资源，需要在Inspector中赋值

    Model runtimeModel; // 运行时模型对象
    Worker worker; // 模型推理执行器
    public float[] results; // 存储模型输出结果的数组

    void Start() // Unity生命周期函数，游戏开始时执行一次
    {
        Model sourceModel = ModelLoader.Load(modelAsset); // 从模型资源加载模型

        // Create a functional graph that runs the input model and then applies softmax to the output.
        // 创建一个函数图，用于在原模型后添加softmax激活函数
        FunctionalGraph graph = new FunctionalGraph(); // 创建函数图对象
        FunctionalTensor[] inputs = graph.AddInputs(sourceModel); // 获取模型的输入张量
        FunctionalTensor[] outputs = Functional.Forward(sourceModel, inputs); // 通过模型前向传播获取输出
        FunctionalTensor softmax = Functional.Softmax(outputs[0]); // 对输出应用softmax函数，用于概率分布

        // Create a model with softmax by compiling the functional graph.
        // 编译函数图，创建包含softmax的完整运行时模型
        runtimeModel = graph.Compile(softmax); // 编译函数图生成最终模型

        // Create input data as a tensor
        // 创建输入数据张量
        using Tensor inputTensor = TextureConverter.ToTensor(inputTexture, width: 28, height: 28, channels: 1); // 将输入纹理转换为28x28单通道张量

        // Create an engine
        // 创建推理引擎
        worker = new Worker(runtimeModel, BackendType.GPUCompute); // 创建GPU计算后端的推理工作器

        // Run the model with the input data
        // 使用输入数据运行模型
        worker.Schedule(inputTensor); // 安排模型推理任务执行

        // Get the result
        // 获取推理结果
        Tensor<float> outputTensor = worker.PeekOutput() as Tensor<float>; // 获取模型输出张量

        // outputTensor is still pending
        // Either read back the results asynchronously or do a blocking download call
        // 输出张量可能仍在GPU上计算中，执行阻塞下载调用
        results = outputTensor.DownloadToArray(); // 将结果从GPU下载到CPU内存数组中
    }

    void OnDisable() // Unity生命周期函数，对象被禁用时执行
    {
        // Tell the GPU we're finished with the memory the engine used
        // 释放GPU内存，清理推理引擎占用的资源
        worker.Dispose(); // 释放工作器资源
    }
}