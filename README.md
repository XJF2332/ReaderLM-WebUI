# ReaderLM-WebUI

适用于 ReaderLM 的 Gradio WebUI

![img.png](img.png)

注意：本仓库的代码部分采用 MIT 协议，但模型仍然保留 CC-BY-NC-4.0 （ReaderLM 的许可证）

## 开发计划

- [ ] 英文本地化 / Localize (English)
- [x] V2 模型的完整支持
- [x] 在转换前清理 HTML

## 部署

### 依赖

你需要以下库：

```text
charset_normalizer
llama-cpp-python
gradio
pyperclip
```

### 下载模型

只支持 GGUF 模型  
请把下载好的模型放置到`models`文件夹中

### 开始使用

```commandline
python .\GUI.py
```
