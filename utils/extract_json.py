import json, re, logging

logger = logging.getLogger(__name__)

def extract_json_block(text: str, warn_on_fail: bool = True):
    """
    从模型输出中提取第一个合法 JSON 对象。
    """
    if not text or not isinstance(text, str):
        return text
    
    try:
        # 预处理：先尝试清理
        text = text.strip()
        
        # 如果整个文本就是 JSON，直接解析
        if text.startswith('{') and text.endswith('}'):
            try:
                cleaned = _clean_json_string(text)
                return json.loads(cleaned)
            except:
                pass
        
        # 提取第一个 { ... } 块 (非贪婪，找到第一个完整的JSON对象)
        # 使用平衡括号匹配
        json_str = _extract_balanced_json(text)
        if not json_str:
            # 回退到简单正则
            match = re.search(r'\{[\s\S]*\}', text)
            if not match:
                # 尝试查找JSON数组
                match = re.search(r'\[[\s\S]*\]', text)
                if not match:
                    raise ValueError("No JSON object found in output.")
            json_str = match.group(0)
        
        # 预处理：多层清理策略
        json_str = _clean_json_string(json_str)

        # 尝试解析为 JSON
        return json.loads(json_str)

    except json.JSONDecodeError as e:
        # 如果还是失败，尝试更激进的清理
        try:
            cleaned = _aggressive_clean(text)
            match = re.search(r'\{[^{}]*\}', cleaned)  # 只匹配简单的单层 JSON
            if match:
                return json.loads(match.group(0))
        except:
            pass
        if warn_on_fail:
            logger.warning(f"Failed to parse JSON: {e}")
            logger.debug(f"Raw output (first 500 chars): {text[:500]}")
        return text
    except Exception as e:
        if warn_on_fail:
            logger.warning(f"Failed to parse JSON: {e}")
            logger.debug(f"Raw output (first 500 chars): {text[:500]}")
        return text


def _extract_balanced_json(text: str) -> str:
    """提取平衡括号的 JSON 对象"""
    start = text.find('{')
    if start == -1:
        return ""
    
    depth = 0
    in_string = False
    escape = False
    
    for i, char in enumerate(text[start:], start):
        if escape:
            escape = False
            continue
        if char == '\\':
            escape = True
            continue
        if char == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == '{':
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0:
                return text[start:i+1]
    
    return ""


def _clean_json_string(s: str) -> str:
    """多层清理 JSON 字符串"""
    # 1. 移除控制字符
    s = re.sub(r'[\x00-\x1f\x7f]', '', s)
    
    # 2. 处理常见的 LaTeX/数学表达式中的反斜杠
    # 将 \frac, \sin, \cos, \tan, \log, \sqrt 等替换为安全形式
    latex_patterns = [
        r'\\frac', r'\\sin', r'\\cos', r'\\tan', r'\\log', r'\\sqrt',
        r'\\alpha', r'\\beta', r'\\gamma', r'\\theta', r'\\pi',
        r'\\sum', r'\\prod', r'\\int', r'\\lim', r'\\infty',
        r'\\left', r'\\right', r'\\cdot', r'\\times', r'\\div',
        r'\\leq', r'\\geq', r'\\neq', r'\\approx',
        r'\\mathbb', r'\\mathbf', r'\\mathrm', r'\\text',
        r'\\boxed', r'\\quad', r'\\qquad', r'\\sec', r'\\csc', r'\\cot',
        r'\\pm', r'\\mp', r'\\le', r'\\ge', r'\\ne', r'\\to', r'\\in',
        r'\\subset', r'\\cup', r'\\cap', r'\\forall', r'\\exists',
        r'\\partial', r'\\nabla', r'\\Delta', r'\\Sigma', r'\\Pi',
        r'\\over', r'\\binom', r'\\choose', r'\\atop',
        r'\\begin', r'\\end', r'\\item', r'\\hline', r'\\vline'
    ]
    for pat in latex_patterns:
        s = s.replace(pat, pat.replace('\\', '\\\\'))
    
    # 3. 修复剩余的无效转义序列
    # 匹配 \ 后面不是有效 JSON 转义字符的情况
    s = re.sub(r'\\(?!["\\/bfnrtu\\])', r'\\\\', s)
    
    # 4. 修复双重转义
    s = s.replace('\\\\\\\\', '\\\\')
    
    # 5. 移除 markdown 代码块标记
    s = re.sub(r'```json\s*', '', s)
    s = re.sub(r'```\s*', '', s)
    
    return s


def _aggressive_clean(s: str) -> str:
    """激进清理：移除所有可能有问题的反斜杠"""
    # 移除控制字符
    s = re.sub(r'[\x00-\x1f\x7f]', '', s)
    
    # 移除所有非标准转义的反斜杠
    s = re.sub(r'\\(?!["\\/bfnrtu])', '', s)
    
    # 移除 markdown 代码块标记
    s = re.sub(r'```json\s*', '', s)
    s = re.sub(r'```\s*', '', s)
    
    return s

