using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using YVR.Core;

public class MainLogic : MonoBehaviour
{
    private void Awake()
    {
        Debug.Log("** Awake() ,初始化111");
        Application.targetFrameRate = 60; // 尽量使更新帧率为60，即deltatime=1000/60~=16.7ms
        YVRManager.instance.hmdManager.SetPassthrough(true);
    }
    // Start is called before the first frame update
    void Start()
    {
        Debug.Log("** 我的第一个脚本");
        Debug.Log(Application.persistentDataPath);
        
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
