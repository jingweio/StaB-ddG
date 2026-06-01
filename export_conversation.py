#!/usr/bin/env python3
"""Render the current Claude Code conversation transcript (JSONL) into a
human-readable Markdown file.

Picks the most-recently-modified *.jsonl transcript for this project and
writes it to claude_conversation.md. Designed to be run periodically (e.g.
via cron every 20 min) so the Markdown stays in sync with the live chat.
"""
import os
import glob
import json
import datetime

PROJECT_DIR = "/home/guoj0f/.claude/projects/-home-guoj0f-repos-StaB-ddG"
OUT = "/home/guoj0f/repos/StaB-ddG/claude_conversation.md"


def newest_transcript():
    files = glob.glob(os.path.join(PROJECT_DIR, "*.jsonl"))
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def text_of(content):
    """Extract plain text from a message 'content' (str or list of blocks)."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for b in content:
            if not isinstance(b, dict):
                continue
            t = b.get("type")
            if t == "text":
                parts.append(b.get("text", ""))
            elif t == "tool_use":
                parts.append(f"_[调用工具: {b.get('name', '?')}]_")
            # skip thinking / tool_result / images to keep it readable
        return "\n".join(p for p in parts if p).strip()
    return ""


def main():
    src = newest_transcript()
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# Claude Code 对话记录",
        "",
        f"> 自动导出,最近更新:**{now}**  ",
        f"> 来源 transcript:`{os.path.basename(src) if src else '(无)'}`  ",
        "> 该文件由 `export_conversation.py` 每 20 分钟刷新一次(cron),内容为只读快照。",
        "",
        "---",
        "",
    ]

    if src:
        for raw in open(src, encoding="utf-8"):
            try:
                o = json.loads(raw)
            except Exception:
                continue
            role = (o.get("message", {}) or {}).get("role") or o.get("type", "")
            if role not in ("user", "assistant"):
                continue
            body = text_of((o.get("message", {}) or {}).get("content", ""))
            if not body:
                continue
            ts = o.get("timestamp", "")
            tag = "🧑 User" if role == "user" else "🤖 Claude"
            header = f"## {tag}" + (f"  · {ts}" if ts else "")
            lines += [header, "", body, "", "---", ""]

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[{now}] wrote {OUT} from {src}")


if __name__ == "__main__":
    main()
