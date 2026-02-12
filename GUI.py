import os

import gradio as gr

import backend.markdown as markdown
import backend.model as model
from backend import HTML, config

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


def update_html_prev(html_path: str, url: str) -> tuple[gr.components.markdown.Markdown, str]:
    html_content = ""
    if html_path and not url:
        html_path = os.path.join('html', html_path)
        html_content = HTML.load_html_file(html_path)
    elif url:
        gr.Info("正在尝试读取 HTML，具体时间依网络状况而定")
        html_content = HTML.get_html(url)
    return gr.Markdown(html_content), html_content


def refresh_model_list(current_selection: str) -> gr.components.dropdown.Dropdown:
    file_list = model.scan_models()
    if current_selection in file_list:
        new_selection = current_selection
    elif file_list:
        new_selection = file_list[0]
    else:
        new_selection = None

    return gr.Dropdown(label="选择模型", choices=file_list, interactive=True, value=new_selection)


with gr.Blocks() as demo:
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
        with gr.Row():
            md_save_path = gr.Textbox(label="将 Markdown 保存到", interactive=True,
                                      placeholder="留空以在 saved_files 文件夹中生成文件", scale=4)
            max_filename_length = gr.Number(value=config.get("max_filename_length", 40), interactive=True,
                                            label="自动生成文件名的长度限制")
        with gr.Row():
            save_md_button = gr.Button("保存 Markdown", variant="primary")
            render_button = gr.Button("渲染 Markdown")
        output_md = gr.Markdown("")
    with gr.Tab("HTML"):
        html_render_warning = gr.Markdown("HTML 中的 CSS 可能会对 UI 产生意料之外的影响，请谨慎加载")
        with gr.Row():
            html_render_button = gr.Button("渲染 HTML")
            clear_html_button = gr.Button("清除 HTML")
        output_html = gr.HTML("")
    with gr.Tab("设置"):
        with gr.Tab("模型设置"):
            with gr.Row():
                n_gpu_layers_input = gr.Number(label="GPU 层数", value=config.get("n_gpu_layers", -1), maximum=128,
                                               minimum=-1)
                model_files = model.scan_models()
                model_file_dropdown = gr.Dropdown(label="选择模型", choices=model_files,
                                                  value=config.get("model_path", None))
                model_type = gr.Dropdown(label="模型代数", choices=["1", "2"], value="1", interactive=True)
            with gr.Row():
                load_model_button = gr.Button("加载模型", variant="primary", scale=10)
                refresh_models_list_btn = gr.Button("🔄", min_width=10, scale=1)
                unload_model_button = gr.Button("卸载模型", scale=10)
            model_load_info = gr.Markdown("")
        with gr.Tab("生成设置"):
            with gr.Row():
                n_ctx_input = gr.Number(label="上下文长度", value=config.get("n_ctx", 204800), minimum=1)
                max_tokens_input = gr.Number(label="最大新分配 token 数量", value=config.get("max_new_tokens", 102400),
                                             minimum=1)
                temperature_input = gr.Number(label="Temperature", value=config.get("temperature", 0.8), minimum=0)
                top_p_input = gr.Number(label="Top P", value=config.get("top_p", 0.95), minimum=0, maximum=1)
        with gr.Tab("预处理设置"):
            with gr.Accordion("清理 HTML"):
                clean_html_cbox = gr.Checkbox(interactive=True, value=config.get("html_clean", True), label="启用")
                with gr.Row():
                    remove_script = gr.Checkbox(interactive=True, value=config.get("remove_script", True), label="删除脚本 (script)")
                    remove_style = gr.Checkbox(interactive=True, value=config.get("remove_style", True), label="删除样式 (style)")
                    remove_meta = gr.Checkbox(interactive=True, value=config.get("remove_meta", True), label="删除元标签 (meta)")
                    remove_comment = gr.Checkbox(interactive=True, value=config.get("remove_comment", True), label="删除注释 (comment)")
                    remove_link = gr.Checkbox(interactive=True, value=config.get("remove_link", True), label="删除链接标签 (link)")
                    remove_svg = gr.Checkbox(interactive=True, value=config.get("remove_svg", True), label="删除 SVG")
                    remove_img = gr.Checkbox(interactive=True, value=config.get("remove_base64", True), label="删除 Base64 图片")
        with gr.Tab("后处理设置"):
            remove_code_block = gr.Checkbox(interactive=True, value=config.get("remove_codeblock", True),
                                            label="移除最外层的代码块（通常出现于 V2 模型）")
        with gr.Tab("指令设置"):
            gr.Markdown("如果设置了自定义格式，则自定义指令不会生效")
            with gr.Row():
                custom_instruction = gr.Textbox(interactive=True, label="自定义提示词", lines=5,
                                                value=config.get("instruction", ""))
                json_schema = gr.Textbox(interactive=True, label="自定义输出 JSON 格式", lines=5,
                                         value=config.get("schema", ""))

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
        fn=model.cal_token_count,
        inputs=[html_content_store, n_ctx_input],
        outputs=token_count
    )

    load_model_button.click(
        fn=lambda model_file, n_gpu_layers, n_ctx: model.load_model(
            os.path.join('models', model_file), n_gpu_layers, n_ctx
        ),
        inputs=[model_file_dropdown, n_gpu_layers_input, n_ctx_input],
        outputs=[model_load_info, model_type]
    )

    generate_button.click(
        fn=model.generate_response,
        inputs=[html_preview, max_tokens_input, temperature_input, top_p_input, model_type, custom_instruction,
                json_schema, clean_html_cbox, remove_script, remove_style, remove_meta, remove_comment, remove_link,
                remove_svg, remove_img],
        outputs=output_text
    )

    render_button.click(
        fn=markdown.md_deliver,
        inputs=output_text,
        outputs=output_md
    )

    stop_button.click(
        fn=model.stop_generate,
        inputs=None,
        outputs=None
    )

    html_render_button.click(
        fn=HTML.html_deliver,
        inputs=[html_preview, gr.State("render")],
        outputs=output_html
    )

    clear_html_button.click(
        fn=HTML.html_deliver,
        inputs=[html_preview, gr.State("clear")],
        outputs=output_html
    )

    unload_model_button.click(
        fn=model.unload_model,
        inputs=None,
        outputs=model_load_info
    )

    copy_button.click(
        fn=markdown.copy,
        inputs=[output_text, remove_code_block],
        outputs=None
    )

    refresh_models_list_btn.click(
        fn=refresh_model_list,
        inputs=model_file_dropdown,
        outputs=model_file_dropdown
    )

    save_md_button.click(
        fn=markdown.save_md,
        inputs=[output_text, remove_code_block,
                md_save_path,max_filename_length],
        outputs=None
    )

demo.launch(inbrowser=True, theme=theme)
