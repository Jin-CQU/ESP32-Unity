using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Net.Sockets;
using System.Threading;              //用于多线程
using System.Collections.Concurrent; //用于线程安全的集合
using UnityEngine;

public class Unity语句csharp联网功能客户端 : MonoBehaviour
{
    public TcpClient 客户端实例;
    public string ip地址;
    public int 端口;
    public bool 连接成功;

    private  StreamReader 数据读取器;

    // 用于线程安全的队列
    private ConcurrentQueue<string> 消息队列 = new ConcurrentQueue<string>();
    private Thread 接收线程;
    private bool 线程运行 = false;

    public void 连接服务器()
    {
        客户端实例 = new TcpClient();
        try
        {
            客户端实例.Connect(ip地址, 端口);
            连接成功 = true;
            Debug.Log("已连接到服务器: " + ip地址 + ":" + 端口);

            // 初始化数据读取器
            数据读取器 = new StreamReader(客户端实例.GetStream());

            // StartCoroutine(接收数据());
            // 启动数据接收线程
            线程运行 = true;
            接收线程 = new Thread(接收数据线程);
            接收线程.Start();

        }
        catch(SocketException 错误)
        {
            print(错误);
            连接成功 = false;
        }
        
    }

    //private IEnumerator 接收数据()
    //{
    //    while (连接成功)
    //    {
    //        if (客户端实例.Available > 0)
    //        {
    //            try
    //            {
    //                // string 接收到的信息 = 数据读取器.ReadLine();
    //                char[] buffer = new char[客户端实例.Available];
    //                数据读取器.Read(buffer, 0, buffer.Length);
    //                string 接收到的信息 = new string(buffer);
    //                if (!string.IsNullOrEmpty(接收到的信息))
    //                {
    //                    Debug.Log("收到服务器信息: " + 接收到的信息);
    //                }
    //            }
    //            catch (IOException 错误)
    //            {
    //                Debug.LogError("接收数据时发生错误: " + 错误.Message);
    //                连接成功 = false;
    //            }
    //        }
    //        yield return null; // 等待下一帧
    //    }
    //}

    private void 接收数据线程()
    {
        while (线程运行 && 客户端实例 != null && 客户端实例.Connected)
        {
            try
            {
                if (客户端实例.Available > 0)
                {
                    char[] buffer = new char[客户端实例.Available];
                    数据读取器.Read(buffer, 0, buffer.Length);
                    string 接收到的信息 = new string(buffer);
                    if (!string.IsNullOrEmpty(接收到的信息))
                    {
                        消息队列.Enqueue(接收到的信息);
                    }
                }
            }
            catch (IOException 错误)
            {
                Debug.LogError("接收数据时发生错误: " + 错误.Message);
                连接成功 = false;
                线程运行 = false;
            }
            Thread.Sleep(10); // 降低CPU占用
        }
    }

    // Start is called before the first frame update
    void Start()
    {
        连接服务器();
    }

    // Update is called once per frame
    void Update()
    {
        // 主线程处理队列中的消息
        while (消息队列.TryDequeue(out string 接收到的信息))
        {
            Debug.Log("收到服务器信息: " + 接收到的信息);
        }
    }

    // 确保在脚本销毁时关闭连接和线程
    void OnDestroy()
    {
        线程运行 = false;
        if (接收线程 != null && 接收线程.IsAlive)
        {
            接收线程.Join();
        }
        if (数据读取器 != null)
        {
            数据读取器.Close();
        }
        if (客户端实例 != null)
        {
            客户端实例.Close();
        }
    }
}
