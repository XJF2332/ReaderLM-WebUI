import os
import gradio as gr
import pyperclip

from backend import config, save_config

def copy(content: str, remove_markdown_block: bool):
    try:
        if remove_markdown_block:
            content = md_deliver(content)
        pyperclip.copy(content)
        gr.Info("Markdown 已复制")
    except Exception as e:
        raise gr.Error(f"复制失败：{str(e)}")

def md_deliver(text: str) -> str:
    """
    移除最外层的markdown代码块（如果有的话）
    :param text: 要处理的文本
    :return: 处理后的文本
    """
    lines = text.split("\n")
    if lines[0] == "```markdown" and lines[-2] == "```" and len(lines) >= 2:
        return "\n".join(lines[1:-2])
    else:
        return text

def save_md(text: str, remove_codeblock: bool, path: str = "", max_filename_length: int = 40) -> None:
    config["remove_codeblock"] = remove_codeblock
    config["max_filename_length"] = max_filename_length
    save_config()

    if remove_codeblock:
        text = md_deliver(text)

    if not path:
        if not os.path.exists("saved_files"):
            os.mkdir("saved_files")
        path = os.path.join("saved_files", f"{text.splitlines()[0][0:max_filename_length]}.md")
    else:
        ext = os.path.splitext(path)[1]
        if ext == ".md":
            pass
        else:
            if ext != "":
                path = path.rpartition('.')[0] + '.md'
            else:
                path += ".md"

    try:
        with open(path, 'w') as f:
            f.write(text)
            gr.Info(f"Markdown 已保存到 {path}")
    except Exception as e:
        raise gr.Error(f"保存失败：{str(e)}")
