"""Export a readable transcript and recovery report from a Codex rollout JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo


AMBIENT_RE = re.compile(
    r"\s*<in-app-browser-context\b[^>]*>.*?</in-app-browser-context>\s*",
    re.DOTALL,
)
INSTRUCTION_RE = re.compile(
    r"\s*<(?:developer|system|skills|permissions|collaboration_mode|apps_instructions|plugins_instructions)\b[^>]*>.*?</(?:developer|system|skills|permissions|collaboration_mode|apps_instructions|plugins_instructions)>\s*",
    re.DOTALL,
)


def parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def content_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    chunks: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        value = item.get("text")
        if isinstance(value, str):
            chunks.append(value)
    return "\n".join(chunks).strip()


def clean_user_text(text: str) -> str:
    text = AMBIENT_RE.sub("\n", text)
    text = INSTRUCTION_RE.sub("\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--date", required=True, help="Local date in YYYY-MM-DD")
    parser.add_argument("--timezone", default="Asia/Shanghai")
    args = parser.parse_args()

    local_zone = ZoneInfo(args.timezone)
    target_date = datetime.fromisoformat(args.date).date()
    messages: list[tuple[datetime, int, str, str]] = []
    file_changes: list[tuple[datetime, int, list[str]]] = []
    malformed = 0
    ordinals: list[int] = []
    record_types: Counter[str] = Counter()

    with args.source.open("r", encoding="utf-8") as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            timestamp_raw = row.get("timestamp")
            if not isinstance(timestamp_raw, str):
                continue
            timestamp = parse_timestamp(timestamp_raw).astimezone(local_zone)
            ordinal = row.get("ordinal")
            if isinstance(ordinal, int):
                ordinals.append(ordinal)
            row_type = str(row.get("type", "unknown"))
            record_types[row_type] += 1
            if timestamp.date() != target_date:
                continue

            payload = row.get("payload")
            if not isinstance(payload, dict):
                continue
            if row_type == "response_item" and payload.get("type") == "message":
                role = payload.get("role")
                if role not in {"user", "assistant"}:
                    continue
                text = content_text(payload.get("content"))
                if role == "user":
                    text = clean_user_text(text)
                if text:
                    messages.append((timestamp, ordinal or -1, role, text))
            elif row_type == "event_msg" and payload.get("type") == "item_completed":
                item = payload.get("item")
                if isinstance(item, dict) and item.get("type") == "FileChange":
                    paths: list[str] = []
                    changes = item.get("changes", {})
                    if isinstance(changes, dict):
                        paths.extend(str(path) for path in changes)
                    elif isinstance(changes, list):
                        for change in changes:
                            if isinstance(change, dict) and isinstance(change.get("path"), str):
                                paths.append(change["path"])
                    file_changes.append((timestamp, ordinal or -1, paths))

    messages.sort(key=lambda value: (value[0], value[1]))
    file_changes.sort(key=lambda value: (value[0], value[1]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    digest = sha256(args.source)

    transcript = [
        f"# Codex 会话恢复稿 — {target_date.isoformat()}",
        "",
        "> 本文档从原始 rollout JSONL 只读导出。时间均为 Asia/Shanghai。",
        f"> 原任务 ID：`01a0323c-7acc-7de3-9977-50df9d6b0885`",
        f"> 原始备份 SHA-256：`{digest}`",
        "",
    ]
    labels = {"user": "用户", "assistant": "Codex"}
    for timestamp, ordinal, role, text in messages:
        transcript.extend(
            [
                f"## {timestamp:%H:%M:%S} · {labels[role]} · ordinal {ordinal}",
                "",
                text,
                "",
            ]
        )
    args.output.write_text("\n".join(transcript), encoding="utf-8")

    role_counts = Counter(role for _, _, role, _ in messages)
    unique_paths = sorted({path for _, _, paths in file_changes for path in paths})
    ordinal_gaps: list[str] = []
    for previous, current in zip(sorted(set(ordinals)), sorted(set(ordinals))[1:]):
        if current != previous + 1:
            ordinal_gaps.append(f"{previous} → {current}")

    report_lines = [
        "# Codex 会话恢复报告",
        "",
        f"- 任务 ID：`01a0323c-7acc-7de3-9977-50df9d6b0885`",
        f"- 恢复日期：{target_date.isoformat()}（{args.timezone}）",
        f"- 原始备份：`{args.source}`",
        f"- 文件大小：{args.source.stat().st_size:,} bytes",
        f"- SHA-256：`{digest}`",
        f"- 恢复用户消息：{role_counts['user']}",
        f"- 恢复助手消息：{role_counts['assistant']}",
        f"- 当日 FileChange 事件：{len(file_changes)}",
        f"- 当日涉及的唯一文件：{len(unique_paths)}",
        f"- 无法解析的 JSONL 行：{malformed}",
        f"- 检测到的 ordinal 跳跃：{', '.join(ordinal_gaps) if ordinal_gaps else '无'}",
        "",
        "## 已知 Codex 投影故障",
        "",
        "尝试通过受支持的任务分叉接口恢复时，Codex 返回：",
        "",
        "```text",
        "thread history projection for 01a0323c-7acc-7de3-9977-50df9d6b0885 expected ordinal 1485, got 1484",
        "```",
        "",
        "这表明任务存储的历史投影存在序号不一致；本工具没有修改 Codex 内部数据库。",
        "",
        "## 当日代码修改涉及文件",
        "",
    ]
    report_lines.extend(f"- `{path}`" for path in unique_paths)
    args.report.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "messages": len(messages),
        "user_messages": role_counts["user"],
        "assistant_messages": role_counts["assistant"],
        "file_change_events": len(file_changes),
        "unique_changed_files": len(unique_paths),
        "malformed_lines": malformed,
        "ordinal_gaps": ordinal_gaps,
        "sha256": digest,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
