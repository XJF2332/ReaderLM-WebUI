import gradio as gr
import os
from llama_cpp import Llama
import pyperclip
import re

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
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


def unload_model():
    global model
    model = None
    return "模型已卸载"


def load_model(model_path, n_gpu_layers, n_ctx):
    global model
    model = None
    model = Llama(model_path=model_path, n_gpu_layers=n_gpu_layers, n_ctx=n_ctx)
    return f"模型 '{model_path}' 已成功加载"



def generate_response(html_file, max_tokens, temperature, top_p):
    global model, stop_gen
    stop_gen = False
    html_path = os.path.join('html', html_file)
    html_loaded = load_html_file(html_path)
    if model is None:
        return "模型未加载"
    input_text = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user: \n{html_loaded}<|im_end|>\nassistant:"
    temp = model(input_text, max_tokens=max_tokens, temperature=temperature, top_p=top_p, stream=True)
    output = ""
    for chunk in temp:
        output += chunk["choices"][0]["text"]
        if stop_gen:  # 检测stop_gen是否为真
            break
        yield output
    return output


def deliver(text):
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

    load_model_button.click(
        fn=lambda model_file, n_gpu_layers, n_ctx: load_model(
            os.path.join('models', model_file), n_gpu_layers, n_ctx
        ),
        inputs=[model_file_dropdown, n_gpu_layers_input, n_ctx_input],
        outputs=model_load_info
    )

    generate_button.click(
        fn=generate_response,
        inputs=[
            html_file, max_tokens_input, temperature_input, top_p_input
        ],
        outputs=output_text
    )

    render_button.click(
        fn=deliver,
        inputs=output_text,
        outputs=output_md
    )

    stop_button.click(
        fn=stop_generate,
        inputs=None,
        outputs=None
    )

    html_render_button.click(
        fn=deliver,
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
