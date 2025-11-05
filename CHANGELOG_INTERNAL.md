# 内部变更记录 (自动生成)

日期: 2025-11-05
分支: prefrontal-2-class-YVR2

此文件总结了我在本次会话中对仓库所做的修改、辅助操作以及后续建议，便于你审阅或回退。

## 概要
- 目标：修复 YVR2 SDK 与工程的 AndroidManifest 合并问题（导致 editor 报错与设备上被识别为 2D 窗口的黑屏），并消除预处理器在处理 manifest 时的异常。
- 主要修复：添加/调整 `AndroidManifest.xml`、修正 `YVRSDKSettingAsset.asset` 中的 manifest 条目；清理 Gradle 缓存以强制重建。

## 已修改/新增的文件（按重要性排序）

- `Assets/Plugins/Android/AndroidManifest.xml` (新增/多次编辑)
  - 添加了一个最小且完整的 manifest 骨架（包含 `<queries>` 与 `<application>/<activity>/<intent-filter>` 基本结构），并最终写入了完整的 OpenXR `<queries>` 条目（包含 `<provider>` 与 `<intent><action>`），以避免 YVR 预处理器尝试创建嵌套节点失败。
  - 目的：确保 Android manifest 在 Gradle 合并阶段包含 OpenXR 查询和必要结构，避免 YVR 的 ManifestPreprocessor 抛出 `NullReferenceException` 或 `XmlException`。

- `Assets/XR/Resources/YVRSDKSettingAsset.asset` (修改)
  - 修正了原资产中错误的 `ManifestTagInfo`：把 `tag: intent/action`（非法，包含 `/`）拆分/移除，删除会导致冲突的 `queries` 条目。
  - 目的：防止 YVR 的预处理器尝试用包含 `/` 的字符串创建 XML 元素，从而导致 `XmlException` 或 `NullReferenceException`。

- `Packages/manifest.json` (修改)
  - 从依赖里移除了 `com.unity.xr.oculus`（Oculus 包），以避免与 YVR 插件冲突（Oculus/其它 XR 插件可能造成 XR Loader 抢先初始化，阻止 YVR 注入清单）。

- 其它（显示为已修改但主要为工程/设置变动）
  - `Assets/Scenes/SampleScene.unity`、`ProjectSettings.asset`、`PackageVersion.asset`、`packages-lock.json` 等在会话中被读取/改动（有可能是 Unity 自动写入或 asset 序列化变动）。请以 git diff/提交为准核对。

## 在磁盘/环境上执行的操作（非文件修改）

- 删除并清理了 Gradle/Library 相关缓存目录：
  - `Library/Bee/Android/Prj/IL2CPP/Gradle`（被删除过以强制 Gradle 重新生成）
  - `Library` / `obj` / `Logs` 多次建议或执行清理（目的是强制 Unity 重新导入并重触发 manifest 合并预处理器），注意这些是不可逆的操作，会触发全量重新导入。

## 出现的问题与修复理由（简明）

- 问题1：YVR 的 ManifestPreprocessor 在尝试修改 manifest 时抛出 NullReference 或 XmlException（如找不到 `/manifest/queries` 或 `tag` 包含 `/`）
  - 原因：YVR 的配置资产有错误条目（`tag: intent/action`），并且工程里没有预先存在 `queries` 或 `intent` 节点，导致预处理器在创建节点时出错。
  - 修复：在 manifest 中新增必要节点（或将这些 queries 写死），并修正资产里的非法 tag 条目，避免预处理器进入错误路径。

- 问题2：APK 在设备上被识别为 2D 窗口（黑屏），日志里出现 `isVr3dApp: false`、`2D_YvrXRService` 等
  - 原因：最终合并的 AndroidManifest 中缺少 YVR 需要注入的 activity/intent-filter/meta-data，从而系统无法识别为 YVR 3D/VR 应用。
  - 修复：保证 manifest 包含 YVR 期待的 activity/intent-filter 和必要 meta-data（通过让插件注入或手动写入）。

## 验证步骤（你可以按此执行并把结果贴回）

1. 重新打开 Unity（已删除 Library 后必须重开）并构建（推荐先 Development Build）。
2. 若构建失败，抓取 gradle 构建日志与 manifest 合并输出：
   - 在 Unity 构建输出里或 `Library/Bee/.../unityLibrary/src/main/AndroidManifest.xml` 查看合并后的 manifest（报错会写到 gradle 输出）。
3. 在设备上安装 APK 后执行：
```
adb shell dumpsys package com.SCUT_Liz.ESP32_Unity | findstr /i "queries meta-data activity intent yvr"
adb logcat -d | findstr /i "isVr3dApp 2D_YvrXRService Force stopping"
```
4. 若仍为 2D 窗口，贴 logcat 与合并后的 manifest 给我，我会继续定位。

## 回退与注意事项

- 如果你想回退我的改动，最好用 Git 回退 `Assets/Plugins/Android/AndroidManifest.xml` 与 `Assets/XR/Resources/YVRSDKSettingAsset.asset`、`Packages/manifest.json` 这三个文件。其它文件为 Unity 自动修改，可由版本控制恢复。
- 我对 `BuildAPK.cs` 的临时新增已被用户撤销（不在变更列表）。

## 后续建议

- 使用 YVR 官方提供的 Sample 工程先做一次“纯净”构建，确认设备上能进入 3D 模式。
- 尽量不要在工程根放置自定义 `Assets/Plugins/Android/AndroidManifest.xml` 覆盖插件注入，除非知道要合并的确切条目；若必须自定义，请先用 apkanalyzer 对比合并结果。
- 构建效率：在开发期间使用 Development Build / Script Debugging 可以大幅降低单次构建时间。

---
如需我把这些变更 revert 成具体的 git commit 或生成回退补丁（patch），我可以继续操作。现在我已把本日志放到仓库根目录：`CHANGELOG_INTERNAL.md`。
