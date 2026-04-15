#!/usr/bin/env python3
"""Build a speaker-attribution index for transcript files.

The generated index intentionally stores offsets and labels, not duplicated
full transcript text. Consumers can resolve the text from transcripts/*.json
when they need to audit a specific segment.
"""

from __future__ import annotations

import collections
import datetime as dt
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "references" / "core-video-sources" / "transcripts"
OUT_JSON = ROOT / "references" / "core-video-sources" / "speaker-segments.json"
OUT_MD = ROOT / "references" / "core-video-sources" / "speaker-segments.md"


CONSULTANT_PATTERNS = [
    ("push_question", re.compile(r"你先|那我问|我问你|我跟你说|我告诉你|你告诉我")),
    ("direct_address", re.compile(r"[你您]")),
    ("imperative", re.compile(r"你别|别绕|别|不要|先|直接|记住|发个照片|别呲牙")),
    ("judgment", re.compile(r"适合|不适合|能找|找不了|看不上|不甘心|没有问题|问题就在|判断|结论")),
    ("market_frame", re.compile(r"门当户对|高不成低不就|错位|路子|池子|盘面|市场|婚介|生活当中|体制内")),
    ("rating_frame", re.compile(r"颜值|几分|打分|四点|五点|六分|七分|八点|A[0-9]|a\s*[0-9]")),
    ("consultant_self", re.compile(r"我建议|我判断|我给大家|我见过|我办了|我推荐")),
]

CLIENT_PATTERNS = [
    ("self_intro", re.compile(r"我是|本人|我今年|我身高|我学历|我工作|我妈妈|我妈|我爸爸|我爸|我的|我们家")),
    ("family_profile", re.compile(r"父母|爸爸|妈妈|家里|家里面|独生|有一个|房子|车子|年收入")),
    ("profile_fields", re.compile(r"本科|硕士|博士|公务员|国企|银行|年薪|身高|体重|一米|零零年|九九年|九八年")),
    ("client_question", re.compile(r"我想知道|想问一下|我想问|我可以|能不能|我能|我不想|我之前|我发|我给你发|我单纯|我自评|您觉得我|我适合|适合什么样|我大概能|我能找到|我想找|我可能|我保证|老师|月老")),
    ("acknowledge", re.compile(r"明白了|好的|好好好|对对对|嗯|行")),
]

STRONG_CLIENT_RE = re.compile(
    r"我是|本人|我今年|我身高|我学历|我工作|我妈妈|我妈|我爸爸|我爸|"
    r"我想知道|我想问|我不想|假如说我|您觉得我|我适合|适合什么样|我大概能|我能找到|"
    r"我感觉我|我觉得我|我自己的|我一直|我可能|我保证|学历本科|工作的话|老师|月老"
)

STRONG_CONSULTANT_RE = re.compile(
    r"你先|那我问|我问你|我跟你说|我告诉你|我建议|我判断|我给大家|"
    r"你别|别绕|别呲牙|我推荐|你就|你得|你应该|这路子|门当户对"
)

STYLE_TERMS = [
    "门当户对",
    "高不成低不就",
    "错位",
    "路子",
    "池子",
    "盘面",
    "体制内",
    "生活当中",
    "市场",
    "婚介",
    "情绪价值",
    "看不上",
    "不甘心",
    "适合",
    "不适合",
    "颜值",
    "打分",
    "你先",
    "那我问",
    "你别",
]


def compact(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def split_segments(text: str, max_len: int = 140):
    for match in re.finditer(r"[^。！？!?；;]+[。！？!?；;]?", text):
        raw = match.group(0)
        cleaned = compact(raw)
        if not cleaned:
            continue

        if len(cleaned) <= max_len:
            yield match.start(), match.end(), cleaned
            continue

        base = match.start()
        for sub in re.finditer(r"[^，,]+[，,]?", raw):
            sub_cleaned = compact(sub.group(0))
            if sub_cleaned:
                yield base + sub.start(), base + sub.end(), sub_cleaned


def score_patterns(text: str, patterns):
    score = 0
    markers = []
    for name, pattern in patterns:
        hits = pattern.findall(text)
        if hits:
            score += min(len(hits), 3)
            markers.append(name)
    return score, markers


def classify(text: str):
    consultant_score, consultant_markers = score_patterns(text, CONSULTANT_PATTERNS)
    client_score, client_markers = score_patterns(text, CLIENT_PATTERNS)
    has_strong_client = bool(STRONG_CLIENT_RE.search(text))
    has_strong_consultant = bool(STRONG_CONSULTANT_RE.search(text))

    if re.search(r"我建议|我判断|我给大家|我见过|我办了|我推荐", text):
        consultant_score += 2
    if re.search(r"我想知道|我想问|我可以|能不能|我能|我之前|我是|本人|您觉得我|我适合|我感觉我|我觉得我", text):
        client_score += 2
    if re.search(r"[你您]", text) and re.search(r"适合|不适合|能找|找不了|看不上|别|先|应该|路子|门当户对", text):
        consultant_score += 2
    if re.fullmatch(r"(嗯|对|好的|行|明白了|好好好)[。！!？?，, ]*", text):
        client_score += 1

    if has_strong_client and has_strong_consultant:
        return "uncertain", "medium", consultant_score, client_score, sorted(set(consultant_markers + client_markers))
    if has_strong_client and consultant_score < client_score + 5:
        return "client", "high", consultant_score, client_score, client_markers

    diff = consultant_score - client_score
    if diff >= 4 and has_strong_consultant:
        return "consultant", "high", consultant_score, client_score, consultant_markers
    if diff >= 2:
        return "consultant", "medium", consultant_score, client_score, consultant_markers
    if diff <= -4:
        return "client", "high", consultant_score, client_score, client_markers
    if diff <= -2:
        return "client", "medium", consultant_score, client_score, client_markers
    return "uncertain", "low", consultant_score, client_score, sorted(set(consultant_markers + client_markers))


def load_transcripts():
    for path in sorted(SOURCE_DIR.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        transcript = data.get("transcript", "")
        if transcript:
            yield path, data, transcript


def build():
    items = []
    speaker_counts = collections.Counter()
    confidence_counts = collections.Counter()
    char_counts = collections.Counter()
    style_term_counts = collections.Counter()
    videos = 0

    for path, data, transcript in load_transcripts():
        videos += 1
        segment_index = 0
        for start, end, text in split_segments(transcript):
            speaker, confidence, consultant_score, client_score, markers = classify(text)
            speaker_counts[speaker] += 1
            confidence_counts[f"{speaker}:{confidence}"] += 1
            char_counts[f"{speaker}:{confidence}"] += len(text)

            if speaker == "consultant" and confidence == "high":
                for term in STYLE_TERMS:
                    style_term_counts[term] += text.count(term)

            items.append(
                {
                    "videoId": data["id"],
                    "transcriptFile": f"transcripts/{path.name}",
                    "segmentIndex": segment_index,
                    "startOffset": start,
                    "endOffset": end,
                    "chars": len(text),
                    "speaker": speaker,
                    "confidence": confidence,
                    "consultantScore": consultant_score,
                    "clientScore": client_score,
                    "markers": sorted(set(markers)),
                }
            )
            segment_index += 1

    payload = {
        "schema": "speaker-segments-v1",
        "generatedAt": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "source": "core-video-v1 transcripts",
        "method": "automatic heuristic speaker attribution from ASR text; not human-verified diarization",
        "textPolicy": "full transcript text is not duplicated here; use offsets against transcriptFile for audit",
        "videos": videos,
        "segments": len(items),
        "counts": {
            "bySpeaker": dict(sorted(speaker_counts.items())),
            "bySpeakerConfidence": dict(sorted(confidence_counts.items())),
            "charsBySpeakerConfidence": dict(sorted(char_counts.items())),
            "styleTermCountsFromHighConfidenceConsultant": {
                key: value for key, value in style_term_counts.most_common() if value
            },
        },
        "items": items,
    }

    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    OUT_MD.write_text(render_markdown(payload), encoding="utf-8")


def render_markdown(payload: dict) -> str:
    counts = payload["counts"]
    rows = []
    for key, value in counts["bySpeakerConfidence"].items():
        chars = counts["charsBySpeakerConfidence"].get(key, 0)
        rows.append(f"| `{key}` | {value} | {chars} |")

    terms = []
    for key, value in counts["styleTermCountsFromHighConfidenceConsultant"].items():
        terms.append(f"| {key} | {value} |")
    if not terms:
        terms.append("| none | 0 |")

    return "\n".join(
        [
            "# 说话人切分证据",
            "",
            "本文件汇总 50 条文稿的自动说话人归属结果。",
            "为了便于复核，JSON 只保存片段位置和标签，不重复复制完整文稿正文。",
            "",
            "## 方法边界",
            "",
            "- 来源：`core-video-v1` 文稿文件。",
            "- 方法：基于连续 ASR 文本的启发式说话人归属。",
            "- 这不是人工逐句校对后的说话人分离。",
            "- 语言风格提取只使用 `speaker=consultant` 且 `confidence=high` 的片段。",
            "- `client` 和 `uncertain` 片段不得用于语言风格提取。",
            "",
            "## 统计",
            "",
            f"- 处理视频数：{payload['videos']}",
            f"- 生成片段数：{payload['segments']}",
            "",
            "| 分组 | 片段数 | 字符数 |",
            "|---|---:|---:|",
            *rows,
            "",
            "## 高置信咨询师片段中的风格词",
            "",
            "| 词语 | 次数 |",
            "|---|---:|",
            *terms,
            "",
            "## 使用规则",
            "",
            "更新技能语言风格时，按这个优先级使用：",
            "",
            "1. 高置信咨询师片段。",
            "2. 中置信咨询师片段只能作为弱辅助证据。",
            "3. 不得把咨询者自述、确认语或不确定混合片段当成风格证据。",
            "",
        ]
    )


if __name__ == "__main__":
    build()
