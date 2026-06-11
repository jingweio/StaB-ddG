#!/usr/bin/env python3
"""Split the Claude Code conversation into per-day Markdown files.

Reads the same transcript source as export_conversation.py (the newest JSONL
for this project) and writes one file per calendar day (KAUST local time,
UTC+3) into claude-conversation-records/{YYYY-MM-DD}.md.

Run anytime to (re)generate; safe to re-run (overwrites the day files).
"""
import os
import glob
import json
import datetime

PROJECT_DIR = "/home/guoj0f/.claude/projects/-home-guoj0f-repos-StaB-ddG"
OUT_DIR = "/home/guoj0f/repos/StaB-ddG/claude-conversation-records"
TZ = datetime.timezone(datetime.timedelta(hours=3))  # Asia/Riyadh (KAUST), no DST


def newest_transcript():
    files = glob.glob(os.path.join(PROJECT_DIR, "*.jsonl"))
    return max(files, key=os.path.getmtime) if files else None


def text_of(content):
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for b in content:
            if not isinstance(b, dict):
                continue
            if b.get("type") == "text":
                parts.append(b.get("text", ""))
            elif b.get("type") == "tool_use":
                parts.append(f"_[调用工具: {b.get('name', '?')}]_")
        return "\n".join(p for p in parts if p).strip()
    return ""


def local_dt(ts_str):
    """'2026-06-01T07:51:11.720Z' -> aware datetime in UTC+3, or None."""
    if not ts_str:
        return None
    try:
        dt = datetime.datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        return dt.astimezone(TZ)
    except Exception:
        return None


def main():
    src = newest_transcript()
    if not src:
        print("no transcript found")
        return
    os.makedirs(OUT_DIR, exist_ok=True)

    days = {}          # 'YYYY-MM-DD' -> list of rendered blocks
    last_date = "unknown-date"
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
        dt = local_dt(o.get("timestamp", ""))
        if dt:
            last_date = dt.strftime("%Y-%m-%d")
            tlabel = dt.strftime("%H:%M:%S")
        else:
            tlabel = ""
        tag = "🧑 User" if role == "user" else "🤖 Claude"
        header = f"## {tag}" + (f"  · {tlabel}" if tlabel else "")
        days.setdefault(last_date, []).append(f"{header}\n\n{body}")

    gen = datetime.datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")
    for date in sorted(days):
        path = os.path.join(OUT_DIR, f"{date}.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# Claude Code 对话记录 — {date} (KAUST/UTC+3)\n\n")
            f.write(f"> 按天拆分,自动生成于 {gen};来源 transcript `{os.path.basename(src)}`;"
                    f"本日 {len(days[date])} 条消息。\n\n---\n\n")
            f.write("\n\n---\n\n".join(days[date]))
            f.write("\n")
    print(f"wrote {len(days)} day files to {OUT_DIR}:")
    for date in sorted(days):
        print(f"  {date}.md  ({len(days[date])} msgs)")


if __name__ == "__main__":
    main()
