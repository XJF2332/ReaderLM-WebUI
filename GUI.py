import os
import re
from typing import Any, Generator

import charset_normalizer
import gradio as gr
import pyperclip
from llama_cpp import Llama

model = None
stop_gen = False

theme = gr.themes.Base(
    primary_hue="violet",
    secondary_hue="indigo",
    radius_size="sm",
).set(
    background_fill_primary='*neutral_50',
    border_color_accent='*neutral_50',
    color_accent_soft='*neutral_50',
    shadow_drop='none',
    shadow_drop_lg='none',
    shadow_inset='none',
    shadow_spread='none',
    shadow_spread_dark='none',
    layout_gap='*spacing_xl',
    checkbox_background_color='*primary_50',
    checkbox_background_color_focus='*primary_200'
)


def stop_generate():
    global stop_gen
    stop_gen = True


def load_html_file(file_path):
    with open(file_path, 'rb') as file:
        content_bytes = file.read()
        encoding = charset_normalizer.detect(content_bytes)
    with open(file_path, 'r', encoding=encoding['encoding']) as f:
        return f.read()


def unload_model():
    global model
    model = None
    return "模型已卸载"


def load_model(model_path, n_gpu_layers, n_ctx):
    global model
    model = None
    model = Llama(model_path=model_path, n_gpu_layers=n_gpu_layers, n_ctx=n_ctx)
    return f"模型 '{model_path}' 已成功加载"


def clean_html(html: str, repl_svg: bool = False, repl_base64: bool = False, new_svg: str = "this is a placeholder",
               new_img: str = "#") -> str:
    # 匹配模式
    script = r"<[ ]*script.*?\/[ ]*script[ ]*>"
    style = r"<[ ]*style.*?\/[ ]*style[ ]*>"
    meta = r"<[ ]*meta.*?>"
    comment = r"<[ ]*!--.*?--[ ]*>"
    link = r"<[ ]*link.*?>"
    svg = r"(<svg[^>]*>)(.*?)(<\/svg>)"
    base64_img = r'<img[^>]+src="data:image/[^;]+;base64,[^"]+"[^>]*>'

    def replace_svg(html: str, new_content: str) -> str:
        return re.sub(
            svg,
            lambda match: f"{match.group(1)}{new_content}{match.group(3)}",
            html,
            flags=re.DOTALL,
        )

    def replace_base64_images(html: str, new_image_src) -> str:
        return re.sub(base64_img, f'<img src="{new_image_src}"/>', html)

    html = re.sub(
        script, "", html, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL
    )
    html = re.sub(
        style, "", html, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL
    )
    html = re.sub(
        meta, "", html, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL
    )
    html = re.sub(
        comment, "", html, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL
    )
    html = re.sub(
        link, "", html, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL
    )

    if repl_svg:
        html = replace_svg(html, new_svg)
    if repl_base64:
        html = replace_base64_images(html, new_img)

    return html


def generate_response(html_path: str, max_tokens: int, temperature: float, top_p: float, model_gen: str,
                      instruction: str, schema: str, html_clean: bool, repl_svg: bool, repl_base64: bool, new_svg: str,
                      new_img: str) -> Generator[str | Any, Any, str | Any]:
    """
    最重要的部分，生成 Markdown

    :param html_path: 将要转换的 HTML 路径
    :param max_tokens: 最大 token 数量
    :param temperature: 温度
    :param top_p: top_p
    :param model_gen: 模型代数
    :param instruction: 自定义提示词，仅适用于第二代模型
    :param schema: 自定义输出 JSON 格式
    :param html_clean: 是否预清理 HTML 内容
    :param repl_svg: 是否替换 SVG
    :param repl_base64: 是否替换 base64 形式的图片
    :param new_svg: 新的 SVG
    :param new_img: 新的图片

    :return: output: Markdown
    """
    global model, stop_gen
    stop_gen = False
    html_path = os.path.join('html', html_path)
    html_loaded = load_html_file(html_path)

    if html_clean:
        html_loaded = clean_html(html=html_loaded, repl_svg=repl_svg, repl_base64=repl_base64, new_svg=new_svg,
                                 new_img=new_img)

    if model is None:
        return "模型未加载"

    # 构建提示词
    if model_gen == "2":
        if not instruction:
            instruction = "Extract the main content from the given HTML and convert it to Markdown format."
        if schema:
            instruction = "Extract the specified information from a list of news threads and present it in a structured JSON format."
            prompt = f"{instruction}\n```html\n{html_loaded}\n```\nThe JSON schema is as follows:```json\n{schema}\n```"
        else:
            prompt = f"{instruction}\n```html\n{html_loaded}\n```"
        input_text = prompt
    else:
        input_text = f"{html_loaded}"
    message = [
        {
            "role":"user",
            "content":input_text
        }
    ]
    # 流式生成
    temp = model.create_chat_completion(messages=message, max_tokens=max_tokens, temperature=temperature, top_p=top_p, stream=True)
    output = ""
    for chunk in temp:
        if not "content" in chunk["choices"][0]["delta"]:
            continue
        output += chunk["choices"][0]["delta"]["content"]
        if stop_gen:  # 检测stop_gen是否为真
            break
        yield output
    return output


def md_deliver(text):
    return text

def html_deliver(text):
    return text


def update_html_prev(html_file):
    try:
        html_path = os.path.join('html', html_file)
        html_loaded = load_html_file(html_path)
        html_prev = gr.Markdown(html_loaded)
        return html_prev
    except:
        html_prev = gr.Markdown("")
        return html_prev


def scan_models():
    file_list = []  # 创建一个空列表来存储文件路径

    for item in os.listdir('models'):
        if item.endswith('.gguf'):  # 如果发现目标文件扩展名，直接添加到列表中
            file_list.append(os.path.join(item))

    return file_list  # 函数结束时返回完整的文件路径列表


def copy(content):
    pyperclip.copy(content)


def show_repl_svg(repl):
    return gr.Textbox(interactive=True, label="替换后的 SVG", visible=True) if repl else gr.Textbox(interactive=True,
                                                                                                    label="替换后的 SVG",
                                                                                                    visible=False)


def show_repl_img(repl):
    return gr.Textbox(interactive=True, label="替换后的图片", visible=True) if repl else gr.Textbox(interactive=True,
                                                                                                    label="替换后的图片",
                                                                                                    visible=False)

with gr.Blocks(theme=theme) as demo:
    gr.Markdown("## ReaderLM WebUI")

    with gr.Tab("生成"):
        with gr.Row():
            with gr.Column():
                html_file = gr.File(label="选择 HTML 文件", file_count="single", file_types=[".html"], type="filepath")
                html_preview = gr.Markdown()
                html_file.change(update_html_prev, inputs=html_file, outputs=html_preview)
            with gr.Column():
                with gr.Row():
                    generate_button = gr.Button("转换为 Markdown", variant="primary")
                    stop_button = gr.Button("停止生成")
                    copy_button = gr.Button("复制")
                output_text = gr.Textbox(label="Markdown", interactive=False, lines=20)
    with gr.Tab("Markdown"):
        render_button = gr.Button("渲染 Markdown")
        output_md = gr.Markdown("")
    with gr.Tab("HTML"):
        html_render_button = gr.Button("渲染 HTML")
        output_html = gr.HTML("")
    with gr.Tab("设置"):
        gr.Markdown("模型设置")
        with gr.Row():
            n_gpu_layers_input = gr.Number(label="GPU 层数", value=-1)
            model_files = scan_models()
            model_file_dropdown = gr.Dropdown(label="选择模型", choices=model_files)
            model_type = gr.Dropdown(label="模型代数", choices=["1", "2"], value="1", interactive=True)
        with gr.Row():
            unload_model_button = gr.Button("卸载模型")
            load_model_button = gr.Button("加载模型", variant="primary")
        model_load_info = gr.Markdown("")
        gr.Markdown("生成设置")
        with gr.Row():
            n_ctx_input = gr.Number(label="上下文长度", value=204800)
            max_tokens_input = gr.Number(label="最大新分配 token 数量", value=102400)
        with gr.Row():
            temperature_input = gr.Number(label="Temperature", value=0.8)
            top_p_input = gr.Number(label="Top P", value=0.95)
        gr.Markdown("HTML 设置 - 正在施工，暂时不可用")
        with gr.Row():
            clean_html_cbox = gr.Checkbox(interactive=True, value=True, label="清理 HTML")
            repl_svg = gr.Checkbox(interactive=True, value=False, label="替换 SVG")
            repl_img = gr.Checkbox(interactive=True, value=False, label="替换 Base64 形式的图片")
        with gr.Row():
            new_svg = gr.Textbox(interactive=True, label="替换后的 SVG", visible=False)
            new_img = gr.Textbox(interactive=True, label="替换后的图片", visible=False)
        gr.Markdown("指令设置 - 只对第二代模型生效，两个设置互斥，同时只有一个生效")
        with gr.Row():
            custom_instruction = gr.Textbox(interactive=True, label="自定义提示词")
            json_schema = gr.Textbox(interactive=True, label="自定义输出 JSON 格式")



    repl_svg.change(
        fn=show_repl_svg,
        inputs=repl_svg,
        outputs=new_svg
    )

    repl_img.change(
        fn=show_repl_img,
        inputs=repl_img,
        outputs=new_img
    )

    load_model_button.click(
        fn=lambda model_file, n_gpu_layers, n_ctx: load_model(
            os.path.join('models', model_file), n_gpu_layers, n_ctx
        ),
        inputs=[model_file_dropdown, n_gpu_layers_input, n_ctx_input],
        outputs=model_load_info
    )

    generate_button.click(
        fn=generate_response,
        inputs=[html_file, max_tokens_input, temperature_input, top_p_input, model_type, custom_instruction,
                json_schema, clean_html_cbox, repl_svg, repl_img, new_svg, new_img],
        outputs=output_text
    )

    render_button.click(
        fn=md_deliver,
        inputs=output_text,
        outputs=output_md
    )

    stop_button.click(
        fn=stop_generate,
        inputs=None,
        outputs=None
    )

    html_render_button.click(
        fn=html_deliver,
        inputs=html_preview,
        outputs=output_html
    )

    unload_model_button.click(
        fn=unload_model,
        inputs=None,
        outputs=model_load_info
    )

    copy_button.click(
        fn=copy,
        inputs=output_text,
        outputs=None
    )

demo.launch()
