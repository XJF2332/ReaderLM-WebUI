import re
import charset_normalizer
import requests


def get_html(url: str) -> str:
    """
    从URL获取HTML内容

    Args:
        url: 目标网页URL

    Returns:
        str: HTML内容或错误信息
    """
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        # 发送带请求头的GET请求
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        html = response.text
        return html
    except requests.exceptions.HTTPError as e:
        return f"HTTP错误: 状态码 {e.response.status_code}"
    except requests.exceptions.RequestException as e:
        return f"请求失败: {e}"
    except Exception as e:
        return f"其他错误: {e}"


def load_html_file(file_path: str) -> str:
    """
    从文件加载HTML内容

    Args:
        file_path: HTML文件路径

    Returns:
        str: HTML内容
    """
    with open(file_path, 'rb') as file:
        content_bytes = file.read()
        encoding = charset_normalizer.detect(content_bytes)
    with open(file_path, 'r', encoding=encoding['encoding']) as f:
        return f.read()


def clean_html(html: str,
               repl_script: bool = True,
               repl_style: bool = True,
               repl_meta: bool = True,
               repl_comment: bool = True,
               repl_link: bool = True,
               repl_svg: bool = True,
               repl_base64: bool = True) -> str:
    """
    清理HTML内容，移除脚本、样式、注释、SVG和base64图片等

    Args:
        html: 原始HTML内容
        repl_script: 是否删除script标签
        repl_style: 是否删除style标签
        repl_meta: 是否删除meta标签
        repl_comment: 是否删除注释
        repl_link: 是否删除link标签
        repl_svg: 是否删除SVG
        repl_base64: 是否删除base64图片

    Returns:
        str: 清理后的HTML
    """
    # 匹配模式
    script = r"<[ ]*script.*?\/[ ]*script[ ]*>"
    style = r"<[ ]*style.*?\/[ ]*style[ ]*>"
    meta = r"<[ ]*meta.*?>"
    comment = r"<[ ]*!--.*?--[ ]*>"
    link = r"<[ ]*link.*?>"
    svg = r"<svg[^>]*>.*?<\/svg>"
    base64_img = r'<img[^>]+src="data:image/[^;]+;base64,[^"]+"[^>]*>'

    if repl_script:
        html = re.sub(
            script, "", html, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL
        )
    if repl_style:
        html = re.sub(
            style, "", html, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL
        )
    if repl_meta:
        html = re.sub(
            meta, "", html, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL
        )
    if repl_comment:
        html = re.sub(
            comment, "", html, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL
        )
    if repl_link:
        html = re.sub(
            link, "", html, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL
        )

    if repl_svg:
        html = re.sub(svg, "", html, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
    if repl_base64:
        html = re.sub(base64_img, "", html, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)

    return html


def html_deliver(text: str, mode: str = "clear") -> str:
    """
    传递HTML内容（原样返回）

    :param text: 输入文本
    :param mode: 返回模式 - render为原样返回， clear返回空字符串
    """
    if mode == "render":
        return text
    else:
        return ""