"""Local operator CLI for the same audited review service as the API."""

import argparse
import asyncio
import json

from app.flywheel.review import list_review_questions, publish_approved, review_question


async def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    listing = sub.add_parser("list")
    listing.add_argument("--offset", type=int, default=0)
    listing.add_argument("--limit", type=int, default=20)
    decision = sub.add_parser("decide")
    decision.add_argument("canonical_id", type=int)
    decision.add_argument("action", choices=("reject", "defer", "merge", "approve"))
    decision.add_argument("--request-id", required=True)
    decision.add_argument("--reason")
    decision.add_argument("--category")
    decision.add_argument("--approved-answer")
    decision.add_argument("--merge-target-id", type=int)
    publication = sub.add_parser("publish")
    publication.add_argument("canonical_id", type=int)
    publication.add_argument("--request-id", required=True)
    args = parser.parse_args()
    if args.command == "list":
        result = await list_review_questions(offset=args.offset, limit=args.limit)
        print(json.dumps(result, ensure_ascii=False))
    elif args.command == "publish":
        result = await publish_approved(args.canonical_id, args.request_id)
        print(json.dumps({"canonical_id": result.canonical_id,
                          "knowledge_chunk_id": result.knowledge_chunk_id,
                          "status": result.status}, ensure_ascii=False))
    else:
        result = await review_question(
            args.canonical_id, args.action, args.request_id, reason=args.reason,
            category=args.category, approved_answer=args.approved_answer,
            merge_target_id=args.merge_target_id,
        )
        print(json.dumps({"canonical_id": result.canonical_id, "status": result.status,
                          "merged_into_id": result.merged_into_id}, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
