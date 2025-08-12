using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MainLogic : MonoBehaviour
{
    private void Awake()
    {
        Debug.Log("** Awake() ,初始化111");
        Application.targetFrameRate = 60; // 尽量使更新帧率为60，即deltatime=1000/60~=16.7ms
    }
    // Start is called before the first frame update
    void Start()
    {
        Debug.Log("** 我的第一个脚本");
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
