# ReaderLM-WebUI

Gradio WebUI for ReaderLM

![img.png](img.png)

[中文版](README.md)

Note: codes in this repo uses MIT license, but the model uses CC-BY-NC-4.0 (License used by ReaderLM)

## Dev Plans

- [ ] Localize (English)
- [x] Automatically detect model version
- [x] Full support for V2 model
- [x] Clean HTML before converting

## Get Started

### Requirements

You need these packages:

```text
charset_normalizer
llama-cpp-python
gradio
pyperclip
```

### Download Models

Only GGUF models are supported  
Put downloaded models into `models` folder

### Run

```commandline
python .\GUI.py
```
