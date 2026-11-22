# -*- coding: utf-8 -*-
# File   : nodes/lora_folder_loader.py
# Description : 扫描指定文件夹批量加载 LoRA（仅作用于 MODEL），复用官方 LoraLoader 加载逻辑
import contextlib
import json
import os
import threading

import folder_paths
import nodes

# 官方加载器依赖全局 loras 搜索路径，串行化“临时注册 + 加载 + 还原”避免并发互相干扰
_LORA_LOCK = threading.RLock()


def _notify(message, level="info"):
    """向前端发送一条提示，便于在画布上直接看到执行结果。"""
    try:
        from server import PromptServer
        if getattr(PromptServer, "instance", None) is not None:
            PromptServer.instance.send_sync(
                "a_my_nodes.lora_folder_loader",
                {"message": message, "level": level},
            )
    except Exception:
        pass


@contextlib.contextmanager
def temporary_lora_root(root):
    """临时把目录提到 loras 搜索路径首位，只改变本次官方加载器的路径解析，退出时整体还原。"""
    entry = folder_paths.folder_names_and_paths.get("loras")
    if not entry or not root:
        yield
        return
    roots = entry[0]
    snapshot = list(roots)
    try:
        roots.insert(0, root)
        yield
    finally:
        roots[:] = snapshot


def resolve_lora_folder(folder):
    """解析文件夹路径：支持绝对路径，以及相对 loras / models / ComfyUI 根目录的路径。"""
    folder = str(folder or "").strip().strip('"').strip("'")
    if not folder:
        return ""
    expanded = os.path.expanduser(folder)
    if os.path.isabs(expanded):
        return os.path.abspath(expanded)
    for root in folder_paths.get_folder_paths("loras"):
        candidate = os.path.join(root, expanded)
        if os.path.isdir(candidate):
            return os.path.abspath(candidate)
    for root in (getattr(folder_paths, "models_dir", ""), getattr(folder_paths, "base_path", "")):
        if not root:
            continue
        candidate = os.path.join(root, expanded)
        if os.path.isdir(candidate):
            return os.path.abspath(candidate)
    return os.path.abspath(expanded)


def list_lora_files(folder, recursive=True):
    """列出文件夹内的 LoRA 文件，返回 (解析后的目录, [{name, path}])。

    recursive=False 时只列当前目录，不进入子文件夹。
    """
    resolved = resolve_lora_folder(folder)
    if not resolved:
        raise FileNotFoundError("未填写 LoRA 文件夹路径")
    if not os.path.isdir(resolved):
        raise FileNotFoundError(f"文件夹不存在: {resolved}")

    # LoRA 一般只使用 safetensors，这里只统计该后缀
    extensions = {".safetensors"}
    candidates = []
    if recursive:
        for dirpath, dirnames, filenames in os.walk(resolved):
            dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
            for filename in sorted(filenames):
                candidates.append((dirpath, filename))
    else:
        for filename in sorted(os.listdir(resolved)):
            if os.path.isfile(os.path.join(resolved, filename)):
                candidates.append((resolved, filename))

    items = []
    for dirpath, filename in candidates:
        if os.path.splitext(filename)[1].lower() not in extensions:
            continue
        full_path = os.path.join(dirpath, filename)
        relative = os.path.relpath(full_path, resolved).replace("\\", "/")
        items.append({"name": relative, "path": full_path})
    items.sort(key=lambda item: item["name"].lower())
    return resolved, items


# 关键字过滤里视为连接词的干扰词（不参与匹配）
FILTER_IGNORED_TOKENS = {"存在"}


def parse_filter_expression(keyword_filter):
    """把过滤表达式解析为 [[ [词1, 词2], [词3] ], ...]。

    结构含义：外层是“并(OR)”，内层每个 [词...] 是“且(AND)”，
    而每个 [词...] 内部按空格视为“并(OR)”。即“且”优先于“并”。
    """
    normalized = str(keyword_filter or "").replace("且", " 且 ").replace("并", " 并 ")
    tokens = [token for token in normalized.split() if token]
    or_groups = []
    and_groups = []
    terms = []

    def flush_terms():
        if terms:
            and_groups.append(list(terms))
            terms.clear()

    def flush_and_group():
        flush_terms()
        if and_groups:
            or_groups.append([list(group) for group in and_groups])
            and_groups.clear()

    for token in tokens:
        if token == "并":
            flush_and_group()
        elif token == "且":
            flush_terms()
        elif token not in FILTER_IGNORED_TOKENS:
            terms.append(token.lower())
    flush_and_group()
    return or_groups


def matches_keyword_filter(name, keyword_filter):
    """按关键字过滤规则匹配名称；空表达式视为全部匹配。"""
    groups = parse_filter_expression(keyword_filter)
    if not groups:
        return True
    haystack = str(name or "").lower()
    return any(
        all(any(term in haystack for term in terms) for terms in and_groups)
        for and_groups in groups
    )


class AnyType(str):
    """与任意类型都比较相等，用于兼容前端动态传入的参数。"""

    def __ne__(self, __value: object) -> bool:
        return False


any_type = AnyType("*")


class FlexibleOptionalInputType(dict):
    """允许接收前端动态生成的 LORA_x 等未声明参数。"""

    def __init__(self, data=None):
        super().__init__()
        if data:
            self.update(data)

    def __getitem__(self, key):
        if key in self.keys():
            return super().__getitem__(key)
        return (any_type,)

    def __contains__(self, key):
        return True


class LoraFolderLoader:
    """按文件夹批量加载 LoRA，支持逐个开关、强度调整、多选/单选。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "folder": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "LoRA 所在文件夹，支持绝对路径或相对 ComfyUI models/loras 的路径",
                }),
                "mode": (["多选", "单选"], {
                    "default": "多选",
                    "tooltip": "单选时只应用列表中第一个开启的 LoRA",
                }),
                "keyword_filter": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "关键字过滤：空格=并(OR)，且=AND，并=OR；且优先于并。例：15 16 且 hand r16 = (15 或 16) 且 (hand 或 r16)",
                }),
            },
            "optional": FlexibleOptionalInputType({
                # 前端写入的 LoRA 行状态（名称、开关、强度），随工作流保存
                "loras_info": ("STRING", {"default": "{}", "multiline": False}),
                # 是否递归列出子文件夹；关闭后只列当前目录
                "include_subfolders": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "刷新时是否列出子文件夹里的 LoRA",
                }),
            }),
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "A_my_nodes/loaders"

    @staticmethod
    def _parse_rows(loras_info, kwargs):
        """优先读取 loras_info，其次兼容前端直接传入的 LORA_x 参数。"""
        rows = []
        try:
            data = json.loads(loras_info) if loras_info else {}
            if isinstance(data, dict):
                rows = data.get("rows", [])
            elif isinstance(data, list):
                rows = data
        except (json.JSONDecodeError, TypeError):
            rows = []
        if not isinstance(rows, list) or not rows:
            rows = [value for key, value in kwargs.items()
                    if key.startswith("LORA_") and isinstance(value, dict)]
        return [row for row in rows if isinstance(row, dict)]

    @staticmethod
    def _resolve_file(row, folder):
        """把行数据还原为磁盘上的 LoRA 文件绝对路径。"""
        path = str(row.get("path") or "").strip()
        if path and os.path.isfile(path):
            return os.path.abspath(path)

        name = str(row.get("lora") or row.get("name") or "").strip()
        if not name:
            return ""
        base = resolve_lora_folder(folder)
        if base:
            candidate = os.path.join(base, name)
            if os.path.isfile(candidate):
                return os.path.abspath(candidate)
        return folder_paths.get_full_path("loras", name) or ""

    @staticmethod
    def _lora_reference(lora_path):
        """返回 (搜索根目录, 官方加载器使用的相对名)。

        优先使用包含该文件的、最具体的已注册 loras 根目录；否则用文件所在目录。
        """
        absolute = os.path.abspath(lora_path)
        best_root = ""
        for root in folder_paths.get_folder_paths("loras"):
            root_abs = os.path.abspath(root)
            try:
                inside = os.path.commonpath([root_abs, absolute]) == root_abs
            except ValueError:
                inside = False
            if inside and len(root_abs) > len(best_root):
                best_root = root_abs
        if best_root:
            return best_root, os.path.relpath(absolute, best_root).replace("\\", "/")
        return os.path.dirname(absolute), os.path.basename(absolute)

    def apply(self, model, folder, mode="多选", keyword_filter="", loras_info="{}", **kwargs):
        rows = self._parse_rows(loras_info, kwargs)
        print(
            f"[LoraFolderLoader] ▶ 执行: folder={folder!r}, mode={mode!r}, "
            f"filter={keyword_filter!r}, rows={len(rows)}"
        )
        if not rows:
            lora_keys = [key for key in kwargs if key.startswith("LORA_")]
            print(
                "[LoraFolderLoader] ⚠️ 没有收到任何 LoRA 行数据：loras_info 为空，"
                f"且未收到 LORA_x 参数（当前 keys={lora_keys}）"
            )
            _notify("LoRA 文件夹：没有收到行数据，请点刷新并确认已保存", "error")
            return (model,)

        keyword_filter = str(keyword_filter or "").strip()
        if not keyword_filter:
            try:
                data = json.loads(loras_info) if loras_info else {}
            except (json.JSONDecodeError, TypeError):
                data = {}
            if isinstance(data, dict):
                keyword_filter = str(data.get("filter", "") or "").strip()

        matched_rows = [
            row for row in rows
            if matches_keyword_filter(row.get("lora") or row.get("name") or row.get("path"), keyword_filter)
        ]
        enabled = [row for row in matched_rows if row.get("on")]
        if mode == "单选":
            enabled = enabled[:1]

        current_model = model
        applied = 0
        for row in enabled:
            try:
                strength = float(row.get("strength", 1.0))
            except (TypeError, ValueError):
                strength = 1.0
            if strength == 0:
                continue

            lora_path = self._resolve_file(row, folder)
            if not lora_path:
                print(f"[LoraFolderLoader] ❌ 找不到 LoRA 文件: {row.get('lora') or row.get('path')}")
                continue

            # 只替换“文件名解析”的搜索根，加载仍完全交给官方 LoraLoader
            lora_root, lora_name = self._lora_reference(lora_path)
            print(f"[LoraFolderLoader] 解析: {lora_path} -> {lora_root} / {lora_name}")
            try:
                with _LORA_LOCK, temporary_lora_root(lora_root):
                    current_model, _ = nodes.LoraLoader().load_lora(
                        current_model, None, lora_name, strength, 0
                    )
                applied += 1
                print(f"[LoraFolderLoader] ✅ {lora_name} (强度 {strength})")
            except Exception as error:
                print(f"[LoraFolderLoader] ❌ 加载失败 {lora_name}: {error}")

        if applied == 0:
            print(
                "[LoraFolderLoader] ⚠️ 没有应用任何 LoRA："
                f"匹配 {len(matched_rows)} 行，开启 {len(enabled)} 行"
                "（检查开关、关键字过滤、单选模式、强度是否为 0）"
            )
        summary = f"匹配 {len(matched_rows)}/{len(rows)}，应用 {applied}/{len(enabled)} 个 LoRA"
        _notify(summary, "success" if applied else "warning")
        print(f"[LoraFolderLoader] 完成，{summary}")
        return (current_model,)
