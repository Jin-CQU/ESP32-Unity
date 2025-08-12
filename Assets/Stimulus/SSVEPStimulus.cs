using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

/// <summary>
/// SSVEP刺激器 - 用于生成视觉稳态诱发电位刺激
/// </summary>
public class SSVEPStimulus : MonoBehaviour
{
    public float frequency = 10f; // 闪烁频率 Hz
    public float duration = 10f;  // 持续时间 秒
    public Color onColor = Color.white;
    public Color offColor = Color.black;

    private float timer = 0f;
    private bool isOn = false;
    private Image img;
    private float elapsed = 0f;
    
    // Start is called before the first frame update
    void Start()
    {
        img = GetComponent<Image>();
        if (img == null)
            Debug.LogError("SSVEPStimulus需要挂载在有Image组件的物体上！");

        onColor = Color.white;      // 开启颜色
        offColor = Color.gray;    // 关闭颜色
        img.color = offColor;
    }

    // Update is called once per frame
    void Update()
    {
        if (elapsed > duration)
        {
            img.color = offColor;
            return;
        }

        timer += Time.deltaTime;
        elapsed += Time.deltaTime;

        // 控制闪烁（正弦波方式，或简单的方波）
        float period = 1f / frequency;
        if (timer >= period / 2f)
        {
            isOn = !isOn;
            img.color = isOn ? onColor : offColor;
            timer = 0f;
        }
    }
}
