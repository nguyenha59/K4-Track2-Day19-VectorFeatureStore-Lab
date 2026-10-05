"""5-query demo for HybridMemoryAgent.  Run:  python bonus/demo.py

Profile features come from the Feast repo applied in NB4 (run NB4 first for
the full picture; without it the agent still runs, profile shows '?').
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent import HybridMemoryAgent  # noqa: E402

MEMORIES_U001 = [
    "Hôm nay tôi đọc tài liệu về Kubernetes. Pod là đơn vị triển khai nhỏ nhất, "
    "còn Deployment quản lý số replica và rolling update.",
    "Ghi chú: Horizontal Pod Autoscaler tự động mở rộng số pod theo CPU hoặc "
    "lưu lượng request. Cần đặt resource requests thì HPA mới tính đúng.",
    "Đọc bài về cloud security: nguyên tắc least privilege cho IAM role, "
    "mã hoá dữ liệu at-rest bằng KMS, và không để S3 bucket ở chế độ public.",
    "Tôi lưu lại checklist bảo mật đám mây: bật MFA cho root account, "
    "xoay vòng access key định kỳ, bật CloudTrail để audit.",
    "Học về vector database: Qdrant dùng HNSW, filter theo payload nằm bên "
    "trong vòng duyệt index nên không bị mất recall như post-filter.",
    "Meeting notes: team muốn chuyển batch ETL sang streaming với Kafka để "
    "feature freshness dưới 1 phút.",
]
# Another user's memory: must NEVER appear in u_001's context.
MEMORIES_U002 = [
    "Bí mật của u_002: kế hoạch tăng giá gói Kubernetes managed vào quý 4.",
]

QUERIES = [
    ("1. Vector hit",          "Tôi đã đọc gì về Kubernetes?"),
    ("2. Cần profile",         "Recommend đọc gì tiếp"),
    ("3. Cần fresh activity",  "Tôi đang quan tâm gì gần đây?"),
    ("4. Paraphrase",          "Tài liệu về tự động mở rộng hạ tầng?"),
    ("5. Mixed + profile",     "Cho tôi summary cloud security"),
]


def main() -> int:
    agent = HybridMemoryAgent()
    for m in MEMORIES_U001:
        agent.remember(m, user_id="u_001")
    for m in MEMORIES_U002:
        agent.remember(m, user_id="u_002")
    print(f"Feast profile: {'connected' if agent.fs else 'not found (run NB4)'}\n")

    for label, q in QUERIES:
        ctx = agent.recall(q, user_id="u_001")
        print(f"=== {label}: {q!r}")
        print(ctx, "\n")
        assert "u_002" not in ctx.split("Top memories:")[1], "cross-user leak!"

    print("Isolation check: no u_002 memory leaked into u_001 context. OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
