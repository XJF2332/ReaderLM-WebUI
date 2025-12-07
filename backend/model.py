from typing import Any, Generator

from llama_cpp import Llama

from backend import HTML

model = None
stop_gen = False


def load_model(model_path: str,
               n_gpu_layers: int,
               n_ctx: int) -> tuple[str, str]:
    global model
    model = None
    model = Llama(model_path=model_path, n_gpu_layers=n_gpu_layers, n_ctx=n_ctx)
    metadata = model.metadata
    if "general.version" in metadata.keys():
        version = metadata["general.version"]
        if version == "v2":
            return f"模型 '{model_path}' 已成功加载", "2"
        else:
            return f"模型 '{model_path}' 已成功加载，但无法确定模型代数，回滚到一代", "1"
    else:
        return f"模型 '{model_path}' 已成功加载", "1"


def unload_model() -> str:
    global model
    model = None
    return "模型已卸载"


def stop_generate():
    global stop_gen
    stop_gen = True


def cal_token_count(html: str, max_tokens: int) -> str:
    global model
    if model is None:
        return "未加载模型，无法计算 Token 数量"
    else:
        if html is not None:
            tokens = model.tokenize(html.encode('utf-8'))  # type: ignore
            # 使用默认的清理参数计算清理后的token数量
            tokens_cleaned = model.tokenize(HTML.clean_html(html).encode('utf-8'))  # type: ignore
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
                      repl_script: bool, repl_style: bool,
                      repl_meta: bool, repl_comment: bool,
                      repl_link: bool, repl_svg: bool,
                      repl_base64: bool) -> Generator[str | Any, Any, str | Any]:
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
    :param repl_script: 是否删除script标签
    :param repl_style: 是否删除style标签
    :param repl_meta: 是否删除meta标签
    :param repl_comment: 是否删除注释
    :param repl_link: 是否删除link标签
    :param repl_svg: 是否删除 SVG
    :param repl_base64: 是否删除 base64 形式的图片

    :return: output: Markdown
    """
    global model, stop_gen
    stop_gen = False

    if html_clean:
        html_content = HTML.clean_html(
            html=html_content,
            remove_script=repl_script,
            remove_style=repl_style,
            remove_meta=repl_meta,
            remove_comment=repl_comment,
            remove_link=repl_link,
            remove_svg=repl_svg,
            remove_base64=repl_base64
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
    temp = model.create_chat_completion(messages=message, max_tokens=max_tokens,  # type: ignore
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
