"""Turn README.md into a Hugging Face article (submission/hf_article.md).

The HF article editor has no repo around it, so relative links become GitHub
links, and the Mermaid diagram becomes a plain text diagram.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GITHUB = "https://github.com/prathameshk421/lexsi_hackathon/blob/main/"

DIAGRAM = """```
gen_data.py --seed 0  ->  SQLite DBs, tasks, oracle traces
        |
        v
AgentTune Project.distill (LoRA SFT)  ->  AgentTune agentic GRPO (tools + reward.py)
        |                                          |
        +-------------------+----------------------+
                            v
           runner.py (AgentTune rollout engine)  ->  trace jsonl, one line per episode
                            |
              +-------------+--------------+
              v                            v
   AuditKit ak.evaluate,          AuditKit AgentEvalRunner
   custom scorers, ak.compare     (tool-call F1 vs oracle)
              |                            |
              +-------------+--------------+
                            v
                     docs/results/*.md
```"""


def main() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    title, body = text.split("\n", 1)
    body = re.sub(r"```mermaid\n.*?```", DIAGRAM, body, count=1, flags=re.S)

    def fix(m: re.Match) -> str:
        label, target = m.group(1), m.group(2)
        if re.match(r"^(https?:|mailto:|#)", target):
            return m.group(0)
        return f"[{label}]({GITHUB}{target})"

    body = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", fix, body)
    out = ROOT / "submission" / "hf_article.md"
    out.write_text(body.lstrip(), encoding="utf-8")
    print("title:", title.lstrip("# ").strip())
    print("wrote", out, len(body.split()), "words")


if __name__ == "__main__":
    main()
