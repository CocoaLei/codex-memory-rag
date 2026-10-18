"""Parse source spans; embedding decoration never changes their line positions."""
import hashlib
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from markdown_it import MarkdownIt

from .embedding import format_document
from .schemas import Chunk, CmrError, SourceDocument, chunk_id, identity_hash, source_key
from .sources import derive_retrieval_state

CHUNKER_VERSION = "chunker-v7"
# General identifier/documentation terms, not project paths or answer labels.
# These decorate only the derived FTS field; source and embedding stay intact.
BILINGUAL_TERMS = {
    "normalize": "规范化 标准化", "normalization": "规范化 标准化",
    "carrier": "物流商 承运商", "tracking": "追踪号 面单号",
    "shipping": "发货", "record": "记录", "date": "日期",
    "token": "标记", "extract": "提取 抽取", "resolve": "解析 推导",
    "format": "格式化", "summary": "汇总 统计", "count": "数量 计数",
    "group": "分组", "order": "订单 顺序", "sequence": "顺序 序列",
    "path": "路径", "filename": "文件名", "file": "文件",
    "template": "模板", "table": "表格", "capacity": "容量",
    "row": "行", "column": "列", "ensure": "确保",
    "validate": "校验 验证", "verify": "校验 验证", "missing": "缺失",
    "unique": "唯一 去重", "duplicate": "重复", "clean": "清洗 清理 去除",
    "whitespace": "空白 空格", "selected": "选定", "save": "保存 另存",
    "atomic": "原子", "atomically": "原子", "replace": "替换",
    "image": "图片", "canvas": "画布", "grid": "格子 网格",
    "geometry": "几何 布局", "layout": "布局 排版", "block": "区块",
    "annotate": "标注", "label": "标签 面单 信息栏", "pixel": "像素",
    "px": "像素", "minimum": "最小 下限", "physical": "物理",
    "round": "取整", "upward": "向上", "center": "居中",
    "incomplete": "未满 不完整", "width": "宽度", "height": "高度",
    "margin": "边距", "spacing": "间距", "quantity": "数量",
    "small": "小图", "large": "大图", "split": "拆分 切分",
    "chunk": "分组 片段", "resize": "调整尺寸 缩放", "upscale": "放大",
    "source": "源 输入", "target": "目标", "transparent": "透明",
    "transparency": "透明度", "ratio": "比例", "fallback": "回退",
    "preserve": "保留", "lossless": "无损", "crop": "裁剪",
    "padding": "填充", "render": "渲染 绘制", "publish": "发布",
    "staging": "暂存", "backup": "备份", "rollback": "回滚",
    "docx": "Word", "pdf": "PDF", "png": "PNG",
    "cm": "厘米", "inch": "英寸", "inches": "英寸",
    "one": "一 1", "two": "二 两 2", "three": "三 3",
    "four": "四 4", "five": "五 5", "six": "六 6", "seven": "七 7",
}
LANGUAGES = {".py": "python", ".ts": "typescript", ".tsx": "tsx", ".js": "javascript",
             ".jsx": "javascript", ".kt": "kotlin", ".java": "java"}


def _spans(document: SourceDocument):
    lines = document.text.splitlines(keepends=True)
    suffix = Path(document.relative_path).suffix
    if suffix == ".md":
        start = 0
        if document.text.startswith("---"):
            for i in range(1, len(lines)):
                if lines[i].strip() == "---":
                    start = i + 1
                    break
        body = "".join(lines[start:])
        tokens = MarkdownIt().parse(body)
        headings = [t.map[0] + start for t in tokens if t.type == "heading_open" and t.map]
        bounds = sorted(set([start] + headings + [len(lines)]))
        return [(a, b, "", "parsed") for a, b in zip(bounds, bounds[1:]) if a < b]
    language = LANGUAGES.get(suffix)
    if language:
        try:
            from tree_sitter_language_pack import PackConfig, configure, downloaded_languages, get_parser
            configure(PackConfig(cache_dir=os.environ.get("CMR_GRAMMAR_CACHE", str(Path.home() / ".cache/codex-memory-rag/grammars"))))
            if language not in downloaded_languages():
                return [(0, len(lines), "", "fallback")]
            parser = get_parser(language)
            tree = parser.parse(document.text.encode("utf-8"))
            if tree.root_node.has_error:
                return [(0, len(lines), "", "fallback")]
            # A large class must not bisect its small methods.
            declarations = []
            function_types = {"function_definition", "function_declaration", "method_definition", "method_declaration", "constructor_declaration"}
            def visit(node, parent=""):
                name = node.child_by_field_name("name")
                name_text = name.text.decode("utf-8") if name else ""
                if node.type in function_types:
                    wrapper = node.parent if node.parent and node.parent.type == "decorated_definition" else node
                    # Point.row/column descriptors in the pinned native binding
                    # corrupt references for line numbers beyond cached integers.
                    # Tuple access preserves owned Python integers and exact spans.
                    start_point, end_point = wrapper.start_point, wrapper.end_point
                    end = end_point[0] + (1 if end_point[1] else 0)
                    declarations.append((start_point[0], end, ".".join(x for x in [parent, name_text] if x)))
                    return
                owner = ".".join(x for x in [parent, name_text] if x) if node.type in {"class_definition", "class_declaration"} else parent
                for child in node.named_children:
                    visit(child, owner)
            visit(tree.root_node)
            declarations.sort()
            spans = []
            cursor = 0
            for start, end, symbol in declarations:
                if start > cursor:
                    spans.append((cursor, start, "", "parsed"))
                spans.append((start, min(end, len(lines)), symbol, "parsed"))
                cursor = end
            if cursor < len(lines):
                spans.append((cursor, len(lines), "", "parsed"))
            return spans
        except (ImportError, RuntimeError, ValueError):
            return [(0, len(lines), "", "fallback")]
    return [(0, len(lines), "", "fallback")]


def chunk_document(document: SourceDocument, tokenizer) -> list[Chunk]:
    if not document.text.strip():
        return []
    lines = document.text.splitlines(keepends=True)
    source = document.source
    state, reasons = derive_retrieval_state(document)
    title = str(document.metadata.get("title") or document.relative_path)
    language = LANGUAGES.get(Path(document.relative_path).suffix, "markdown" if document.relative_path.endswith(".md") else "text")
    hard_limit = 1600 if source.kind == "code" else 1200
    key = source_key(source.project_id, source.source_id, source.worktree_id, document.relative_path)
    result = []
    spans = _spans(document)
    # Short session notes describe one fact across goal/result/verification
    # sections. Keep those together without changing the source span or budget.
    if source.kind == "memory" and spans:
        start, end = spans[0][0], spans[-1][1]
        if tokenizer.token_count(format_document(title, "".join(lines[start:end]))) <= hard_limit:
            spans = [(start, end, "", "parsed")]
    for start, end, symbol, quality in spans:
        if not "".join(lines[start:end]).strip():
            continue
        while start < end:
            stop = start
            for candidate in range(start + 1, end + 1):
                text = "".join(lines[start:candidate])
                if tokenizer.token_count(format_document(title, text)) > hard_limit:
                    break
                stop = candidate
            if stop == start:
                # Splitting a single physical line would break inclusive line provenance.
                raise CmrError("EMBEDDING_CONTRACT_MISMATCH")
            text = "".join(lines[start:stop])
            if not text.strip():
                start = stop
                continue
            identifiers = sorted(set(re.findall(r"[A-Za-z_$][A-Za-z0-9_$]*", text)))
            search_text = "\n".join([title, document.relative_path, symbol, text])
            splits = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", search_text).replace("_", " ")
            splits = re.sub(r"(?<=\d)(?=[A-Za-z])|(?<=[A-Za-z])(?=\d)", " ", splits)
            git = document.git_state or {}
            result.append(Chunk(chunk_id=chunk_id(key, document.file_sha256, CHUNKER_VERSION, len(result)),
                                source_key=key, source_id=source.source_id, project_id=source.project_id,
                                worktree_id=source.worktree_id, relative_path=document.relative_path, title=title,
                                heading=next((l.strip() for l in lines[start:stop] if l.startswith("#")), ""),
                                symbol=symbol, language=language, source_kind=source.kind,
                                memory_type=document.metadata.get("type"), status=document.metadata.get("status"),
                                retrieval_state=state, retrieval_reasons=reasons,
                                valid_until=str(document.metadata["valid_until"]) if document.metadata.get("valid_until") else None,
                                supersedes=str(document.metadata["supersedes"]) if document.metadata.get("supersedes") else None,
                                text=text, lexical_text=text + "\n" + splits + "\n" + " ".join(
                                    sorted({BILINGUAL_TERMS[token.lower()] for token in
                                            re.findall(r"[A-Za-z]+", splits)
                                            if token.lower() in BILINGUAL_TERMS})), identifiers=identifiers,
                                line_start=start + 1, line_end=stop, file_sha256=document.file_sha256,
                                text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                                git_commit=git.get("head"), branch=git.get("branch"), source_epoch=identity_hash(git),
                                indexed_at=datetime.now(timezone.utc).isoformat(), parse_quality=quality))
            start = stop
    return result
