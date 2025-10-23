using UnityEngine;
using UnityEngine.UI;
using TMPro;

/// <summary>
/// 简单的图像交替闪烁刺激（例如黑白棋盘反转）。
/// 默认含义：flickerHz 表示“图像切换次数/秒”。
/// 也就是说 flickerHz=10 -> 每秒切换 10 次（状态 A/B/A/B...），等价于 10Hz 反转刺激。
/// 如果你想让“一次 A+B 视为 1 个周期”的 10Hz（即每 0.05s 切换一次），把 flickerHz 设成 20 即可。
/// 将脚本挂到拥有 Image（UI）或 SpriteRenderer（2D/3D 物体）的对象上。
/// 1. 若提供两张 Sprite，则在它们之间切换。
/// 2. 若未提供 Sprite 或 useColorOnly=true，则使用颜色 A / B 切换。
/// </summary>
public class FlashStimulus : MonoBehaviour
{
    [Header("闪烁配置")]
    [Tooltip("图像切换频率(Hz)，即每秒切换次数。10 表示每 0.1 秒换一次图像。")]
    [SerializeField] private float flickerHz = 10f;
    [Tooltip("开始前可选延迟(秒)")] [SerializeField] private float startDelay = 0f;
    [Tooltip("持续时间(秒)，<=0 表示无限。")]
    [SerializeField] private float duration = 0f;

    [Header("图像 / 颜色切换")]
    [Tooltip("状态 A 的 Sprite（可为黑白棋盘）")] public Sprite stateA;
    [Tooltip("状态 B 的 Sprite（可为反相棋盘）")] public Sprite stateB;
    [Tooltip("强制使用颜色切换（忽略 Sprite）")] public bool useColorOnly = false;
    [Tooltip("状态 A 颜色（无 Sprite 时使用）")] public Color colorA = Color.white;
    [Tooltip("状态 B 颜色（无 Sprite 时使用）")] public Color colorB = Color.black;

    [Header("可选：同步事件")]
    [Tooltip("每次图像切换时回调（参数为当前状态 true=stateA,false=stateB）")]
    public System.Action<bool> OnFlip;

    [Header("控制")]
    [Tooltip("是否启用按键控制闪烁")] public bool enableKeyControl = true;
    [Tooltip("触发开始/停止切换的按键")] public KeyCode toggleKey = KeyCode.Space;
    [Tooltip("Stop 时是否恢复到状态A (true=回到A; false=保持当前帧)")] public bool resetToAOnStop = true;
    [Tooltip("组件启用(Enable)时是否立即开始闪烁；关闭则等待手动 StartFlash/Toggle")] public bool autoStartOnEnable = true;
    [Tooltip("可选：点击此 UI 按钮触发 ToggleFlash")] [SerializeField] public Button toggleButton;

    [Header("频率控制")]
    [Tooltip("用于输入新频率(Hz)的文本框，停止状态按下按钮生效")] public TMP_InputField frequencyInputField;
    [Tooltip("停止状态下应用输入频率的按钮")] public Button applyFrequencyButton;
    // 组件缓存
    private Image _uiImage;
    private SpriteRenderer _spriteRenderer;

    // 运行时
    private float _period;          // 单次切换周期（秒）= 1 / flickerHz
    private float _timer;           // 计时器
    private bool _useSprites;       // 是否使用 Sprite 切换
    private bool _currentState;     // true 使用 stateA / colorA; false 使用 stateB / colorB
    private float _startTime;       // 记录开始时间
    private bool _started;          // 是否已过 startDelay 开始闪烁
    private bool _ended;            // 是否已结束（duration>0）
    private bool _manuallyStopped;  // 用户手动停止标志

    void Awake()
    {
        _uiImage = GetComponent<Image>();
        if (_uiImage == null)
            _spriteRenderer = GetComponent<SpriteRenderer>();
    }

    void OnEnable()
    {
        if (_uiImage == null) _uiImage = GetComponent<Image>();
        if (_spriteRenderer == null && _uiImage == null) _spriteRenderer = GetComponent<SpriteRenderer>();
        if (toggleButton != null)
        {
            toggleButton.onClick.RemoveListener(OnToggleButtonClicked);
            toggleButton.onClick.AddListener(OnToggleButtonClicked);
        }
        if (applyFrequencyButton != null)
        {
            applyFrequencyButton.onClick.RemoveListener(OnApplyFrequencyButtonClicked);
            applyFrequencyButton.onClick.AddListener(OnApplyFrequencyButtonClicked);
        }
        if (autoStartOnEnable)
        {
            InitFlicker();
        }
        else
        {
            // 仅做一次初始外观渲染（状态A），但不进入计时循环
            _manuallyStopped = true; // 标记为“未运行”
            _started = false;
            _ended = false;
            _currentState = true;
            _useSprites = !useColorOnly && stateA != null && stateB != null && (_uiImage != null || _spriteRenderer != null);
            ApplyState(_currentState, force:true);
        }
    }

    void OnDisable()
    {
        if (toggleButton != null)
        {
            toggleButton.onClick.RemoveListener(OnToggleButtonClicked);
        }
        if (applyFrequencyButton != null)
        {
            applyFrequencyButton.onClick.RemoveListener(OnApplyFrequencyButtonClicked);
        }
    }

    void InitFlicker()
    {
        if (flickerHz <= 0f) flickerHz = 10f; // 防止除零
        _period = 1f / flickerHz;
        _timer = 0f;
        _currentState = true; // 初始为 A
        _startTime = Time.time;
        _started = (startDelay <= 0f);
        _ended = false;
        _manuallyStopped = false; // 关键：重新开始时清除手动停止标志
        _useSprites = !useColorOnly && stateA != null && stateB != null && (_uiImage != null || _spriteRenderer != null);
        ApplyState(_currentState, force:true);
    }

    void Update()
    {
        // 按键控制
        if (enableKeyControl && Input.GetKeyDown(toggleKey))
        {
            ToggleFlash();
        }

        if (_ended || _manuallyStopped) return;

        // 处理延迟开始
        if (!_started)
        {
            if (Time.time - _startTime >= startDelay)
            {
                _started = true;
                // 重置基线，避免延迟造成第一段时间过长
                _timer = 0f;
            }
            else return;
        }

        // 持续时间控制
        if (duration > 0f && Time.time - _startTime - startDelay >= duration)
        {
            _ended = true;
            return;
        }

        _timer += Time.deltaTime;
        // 避免长帧跳过多个翻转，用 while 保证连续补齐
        while (_timer >= _period)
        {
            _timer -= _period;
            _currentState = !_currentState;
            ApplyState(_currentState);
        }
    }

    private void ApplyState(bool state, bool force = false)
    {
        if (_useSprites)
        {
            if (_uiImage)
            {
                if (force || _uiImage.sprite != (state ? stateA : stateB))
                    _uiImage.sprite = state ? stateA : stateB;
            }
            else if (_spriteRenderer)
            {
                if (force || _spriteRenderer.sprite != (state ? stateA : stateB))
                    _spriteRenderer.sprite = state ? stateA : stateB;
            }
        }
        else // 颜色切换
        {
            if (_uiImage)
            {
                var target = state ? colorA : colorB;
                if (force || _uiImage.color != target)
                    _uiImage.color = target;
            }
            else if (_spriteRenderer)
            {
                var target = state ? colorA : colorB;
                if (force || _spriteRenderer.color != target)
                    _spriteRenderer.color = target;
            }
        }

        OnFlip?.Invoke(state);
    }

    /// <summary>
    /// 运行时动态修改频率（立即生效，不重置相位）。
    /// </summary>
    public void SetFrequency(float hz)
    {
        if (hz <= 0f) return;
        flickerHz = hz;
        _period = 1f / flickerHz;
    }

    /// <summary>
    /// 强制重启（重新计时 & 回到初始 A 状态）。
    /// </summary>
    public void Restart()
    {
        InitFlicker();
    }

    /// <summary>
    /// 开始闪烁（若之前 Stop 则重新初始化，不改变频率设置）。
    /// </summary>
    public void StartFlash()
    {
        if (!_manuallyStopped && _started && !_ended) return; // 已在运行
        InitFlicker();
    }

    /// <summary>
    /// 停止闪烁。
    /// </summary>
    public void StopFlash()
    {
        _manuallyStopped = true;
        _started = false;
        if (resetToAOnStop)
        {
            _currentState = true;
            ApplyState(_currentState, force: true);
        }
    }

    /// <summary>
    /// 切换开始 / 停止。
    /// </summary>
    public void ToggleFlash()
    {
        if (_manuallyStopped || _ended)
        {
            StartFlash();
        }
        else
        {
            StopFlash();
        }
    }

    private void OnToggleButtonClicked()
    {
        ToggleFlash();
    }

    private void OnApplyFrequencyButtonClicked()
    {
        ApplyFrequencyFromInput();
    }

    private void TryApplyFrequencyFromInput()
    {
        if (frequencyInputField == null) return;

        string text = frequencyInputField.text;
        if (float.TryParse(text, out float newHz) && newHz > 0f)
        {
            SetFrequency(newHz);
            // 停止状态下调整频率后保持在状态A以便观察
            if (resetToAOnStop)
            {
                _currentState = true;
                ApplyState(_currentState, force: true);
            }
        }
        else
        {
            Debug.LogWarning($"无效的频率输入: '{text}'");
        }
    }

    /// <summary>
    /// 可供 UI Button / InputField 事件调用，停止状态下应用当前输入框频率。
    /// </summary>
    public void ApplyFrequencyFromInput()
    {
        if (_manuallyStopped || _ended)
        {
            TryApplyFrequencyFromInput();
        }
    }

}
