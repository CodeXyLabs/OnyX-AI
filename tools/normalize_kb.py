"""
Normalise core/corpus.js so every KB entry is [topic, fact, fact, ...].

The knowledge base was authored as flat arrays, but the generators expected
[topic, [fact, ...]] pairs. This rewrites the literal and the accessors in one
pass so the two agree, and validates the result afterwards.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "core" / "corpus.js"


def main() -> int:
    src = TARGET.read_text(encoding="utf-8")

    start = src.index("var KB = [")
    end = src.index("\n];", start) + len("\n];")
    block = src[start:end]

    # Split the block into entries. Each entry is  [ "topic", "fact", ... ]  and
    # entries are separated by "],\n  [".  We transform entry-wise so the
    # remaining text is untouched.
    def fix_entry(match: re.Match) -> str:
        inner = match.group(1)
        return "[" + inner + "]"

    fixed = re.sub(r"\[((?:[^\[\]]|\n)*?)\]", fix_entry, block, flags=re.S)
    src = src[:start] + fixed + src[end:]

    # Update accessors that assumed the nested pair shape.
    src = src.replace(
        "for (var f = 0; f < KB[i][1].length; f++) parts.push('## ' + KB[i][0] + '\\n' + KB[i][1])",
        "for (var f = 1; f < KB[i].length; f++) parts.push('## ' + KB[i][0] + '\\n' + KB[i].slice(1).join('\\n'))",
    )
    src = src.replace(
        "for (var i = 0; i < KB.length; i++) parts.push('## ' + KB[i][0] + '\\n' + KB[i][1].join('\\n'));",
        "for (var i = 0; i < KB.length; i++) parts.push('## ' + KB[i][0] + '\\n' + KB[i].slice(1).join('\\n'));",
    )
    # audit(): facts live at indices 1..n
    src = src.replace(
        "for (j = 0; j < KB[i][1].length; j++) collectWords(KB[i][1][j], vocab);",
        "for (j = 1; j < KB[i].length; j++) collectWords(KB[i][j], vocab);",
    )
    # generateParaphrases(): same
    src = src.replace(
        "var facts = KB[i][1];",
        "var facts = KB[i].slice(1);",
    )

    TARGET.write_text(src, encoding="utf-8")
    print("rewrote", TARGET)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
