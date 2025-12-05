import os
from typing import Any, Generator

import gradio as gr
import pyperclip
from llama_cpp import Llama

from backend import HTML

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


def unload_model() -> str:
    global model
    model = None
    return "模型已卸载"


def load_model(model_path: str,
               n_gpu_layers: int,
               n_ctx: int) -> tuple[str, gr.components.dropdown.Dropdown]:
    global model
    model = None
    model = Llama(model_path=model_path, n_gpu_layers=n_gpu_layers, n_ctx=n_ctx)
    metadata = model.metadata
    if "general.version" in metadata.keys():
        version = metadata["general.version"]
        if version == "v2":
            return f"模型 '{model_path}' 已成功加载", gr.Dropdown(
                label="模型代数", choices=["1", "2"], value="2", interactive=True)
        else:
            return f"模型 '{model_path}' 已成功加载，但它看起来不像一代模型，也不像二代模型", gr.Dropdown(
                label="模型代数", choices=["1", "2"], value="1", interactive=True)
    else:
        return f"模型 '{model_path}' 已成功加载", gr.Dropdown(
            label="模型代数", choices=["1", "2"], value="1", interactive=True)


def cal_token_count(html: str, max_tokens: int) -> str:
    global model
    if model is None:
        return "未加载模型，无法计算 Token 数量"
    else:
        if html is not None:
            tokens = model.tokenize(html.encode('utf-8'))
            tokens_cleaned = model.tokenize(HTML.clean_html(html).encode('utf-8'))
            tokens_count = len(tokens)
            tokens_count_cleaned = len(tokens_cleaned)
            if tokens_count_cleaned > max_tokens:
                return \
                    f"""⚠️HTML 过长，尝试减少文件长度或增加上下文长度⚠️  
    Token 数量：{tokens_count}  
    预清理 HTML 后的预计 Token 数量：{tokens_count_cleaned}"""
            elif tokens_count > max_tokens >= tokens_count_cleaned:
                return \
                    f"""⚠️HTML 过长，需要预清理⚠️  
    Token 数量：{tokens_count}  
    预清理 HTML 后的预计 Token 数量：{tokens_count_cleaned}"""
            else:
                return \
                    f"""
    Token 数量：{tokens_count}  
    预清理 HTML 后的预计 Token 数量：{tokens_count_cleaned}
    """
        else:
            return "文本为空"


def generate_response(html_content: str, max_tokens: int,
                      temperature: float, top_p: float,
                      model_gen: str, instruction: str,
                      schema: str, html_clean: bool,
                      repl_svg: bool, repl_base64: bool,
                      new_svg: str, new_img: str) -> Generator[str | Any, Any, str | Any]:
    """
    最重要的部分，生成 Markdown

    :param html_content: 将要转换的 HTML 内容
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

    if html_clean:
        html_content = HTML.clean_html(
            html=html_content,
            repl_svg=repl_svg,
            repl_base64=repl_base64,
            new_svg=new_svg,
            new_img=new_img
        )

    if model is None:
        return "模型未加载"

    # 构建提示词
    if model_gen == "2":
        if not instruction:
            instruction = "Extract the main content from the given HTML and convert it to Markdown format."
        if schema:
            instruction = "Extract the specified information from a list of news threads and present it in a structured JSON format."
            prompt = f"{instruction}\n```html\n{html_content}\n```\nThe JSON schema is as follows:```json\n{schema}\n```"
        else:
            prompt = f"{instruction}\n```html\n{html_content}\n```"
        input_text = prompt
    else:
        input_text = f"{html_content}"
    message = [
        {
            "role": "user",
            "content": input_text
        }
    ]
    # 流式生成
    temp = model.create_chat_completion(messages=message, max_tokens=max_tokens,
                                        temperature=temperature, top_p=top_p, stream=True)
    output = ""
    for chunk in temp:
        if not "content" in chunk["choices"][0]["delta"]:
            continue
        output += chunk["choices"][0]["delta"]["content"]
        if stop_gen:  # 检测stop_gen是否为真
            break
        yield output
    return output


def md_deliver(text: str) -> str:
    lines = text.split("\n")
    if lines[0] == "```markdown" and lines[-2] == "```" and len(lines) >= 2:
        return "\n".join(lines[1:-2])
    else:
        return text


def update_html_prev(html_file: str, html_url: str) -> tuple[gr.components.markdown.Markdown, str]:
    html_content = ""
    if html_file and not html_url:
        html_path = os.path.join('html', html_file)
        html_content = HTML.load_html_file(html_path)
    elif html_url:
        gr.Info("正在尝试读取 HTML，具体时间依网络状况而定")
        html_content = HTML.get_html(html_url)
    return gr.Markdown(html_content), html_content


def scan_models() -> list:
    return [f for f in os.listdir("models") if f.lower().endswith(".gguf")]


def refresh_model_list(current_selection: str) -> gr.components.dropdown.Dropdown:
    file_list = scan_models()
    if current_selection in file_list:
        new_selection = current_selection
    elif file_list:
        new_selection = file_list[0]
    else:
        new_selection = None

    return gr.Dropdown(label="选择模型", choices=file_list, interactive=True, value=new_selection)


def copy(content: str, remove_code_block: bool):
    if remove_code_block:
        content = md_deliver(content)
    pyperclip.copy(content)


def toggle_repl_svg(repl: bool) -> gr.components.textbox.Textbox:
    return gr.Textbox(interactive=True,
                      label="替换后的 SVG",
                      visible=True) if repl else gr.Textbox(interactive=True,
                                                            label="替换后的 SVG",
                                                            visible=False)


def toggle_repl_img(repl: bool) -> gr.components.textbox.Textbox:
    return gr.Textbox(interactive=True,
                      label="替换后的图片",
                      visible=True) if repl else gr.Textbox(interactive=True,
                                                            label="替换后的图片",
                                                            visible=False)


with gr.Blocks(theme=theme) as demo:
    gr.Markdown("## ReaderLM WebUI")
    html_content_store = gr.State()

    with gr.Tab("生成"):
        with gr.Row():
            with gr.Column():
                token_count = gr.Markdown()
                html_url = gr.Textbox(label="输入 URL")
                commit_url = gr.Button("提交 URL")
                html_file = gr.File(label="或选择 HTML 文件", file_count="single", file_types=[".html"],
                                    type="filepath")
                html_preview = gr.Markdown()
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
        html_render_warning = gr.Markdown("HTML 中的 CSS 可能会对 UI 产生意料之外的影响，请谨慎加载")
        html_render_button = gr.Button("渲染 HTML")
        output_html = gr.HTML("")
    with gr.Tab("设置"):
        gr.Markdown("模型设置")
        with gr.Row():
            n_gpu_layers_input = gr.Number(label="GPU 层数", value=-1, maximum=128, minimum=-1)
            model_files = scan_models()
            model_file_dropdown = gr.Dropdown(label="选择模型", choices=model_files)
            model_type = gr.Dropdown(label="模型代数", choices=["1", "2"], value="1", interactive=True)
        with gr.Row():
            load_model_button = gr.Button("加载模型", variant="primary", scale=10)
            refresh_models_list_btn = gr.Button("🔄", min_width=10, scale=1)
            unload_model_button = gr.Button("卸载模型", scale=10)
        model_load_info = gr.Markdown("")
        gr.Markdown("生成设置")
        with gr.Row():
            n_ctx_input = gr.Number(label="上下文长度", value=204800, minimum=1)
            max_tokens_input = gr.Number(label="最大新分配 token 数量", value=102400, minimum=1)
        with gr.Row():
            temperature_input = gr.Number(label="Temperature", value=0.8, minimum=0)
            top_p_input = gr.Number(label="Top P", value=0.95, minimum=0, maximum=1)
        remove_code_block = gr.Checkbox(interactive=True, value=True,
                                        label="移除最外层的代码块（通常出现于 V2 模型）")
        gr.Markdown("HTML 设置")
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
        fn=toggle_repl_svg,
        inputs=repl_svg,
        outputs=new_svg
    )

    repl_img.change(
        fn=toggle_repl_img,
        inputs=repl_img,
        outputs=new_img
    )

    html_file.change(
        update_html_prev,
        inputs=[html_file, html_url],
        outputs=[html_preview, html_content_store]
    )

    commit_url.click(
        update_html_prev,
        inputs=[html_file, html_url],
        outputs=[html_preview, html_content_store]
    )

    html_content_store.change(
        fn=cal_token_count,
        inputs=[html_content_store, n_ctx_input],
        outputs=token_count
    )

    load_model_button.click(
        fn=lambda model_file, n_gpu_layers, n_ctx: load_model(
            os.path.join('models', model_file), n_gpu_layers, n_ctx
        ),
        inputs=[model_file_dropdown, n_gpu_layers_input, n_ctx_input],
        outputs=[model_load_info, model_type]
    )

    generate_button.click(
        fn=generate_response,
        inputs=[html_preview, max_tokens_input, temperature_input, top_p_input, model_type, custom_instruction,
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
        fn=HTML.html_deliver,
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
        inputs=[output_text, remove_code_block],
        outputs=None
    )

    refresh_models_list_btn.click(
        fn=refresh_model_list,
        inputs=model_file_dropdown,
        outputs=model_file_dropdown
    )

demo.launch()