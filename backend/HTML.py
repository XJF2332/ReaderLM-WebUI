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


def clean_html(html: str, repl_svg: bool = False,
               repl_base64: bool = False,
               new_svg: str = "",
               new_img: str = "") -> str:
    """
    清理HTML内容，移除脚本、样式、注释等

    Args:
        html: 原始HTML内容
        repl_svg: 是否替换SVG
        repl_base64: 是否替换base64图片
        new_svg: 替换后的SVG内容
        new_img: 替换后的图片路径

    Returns:
        str: 清理后的HTML
    """
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


def html_deliver(text: str) -> str:
    """
    传递HTML内容（原样返回）

    Args:
        text: HTML内容

    Returns:
        str: 相同的HTML内容
    """
    return text