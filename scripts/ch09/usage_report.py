"""Print token usage by day, intent and model from local MySQL."""

import argparse
import asyncio
import json
from datetime import date

from app.db.observability import usage_by_day
from app.observability.dashboard import summarize_usage


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    today = date.today()
    parser.add_argument("--from", dest="start", type=date.fromisoformat,
                        default=today.replace(day=1))
    parser.add_argument("--to", dest="end", type=date.fromisoformat, default=today)
    parser.add_argument("--format", choices=("json", "table"), default="table")
    parser.add_argument('--summary', action='store_true', help='print the same summary as the dashboard')
    args = parser.parse_args(argv)
    if args.start > args.end:
        parser.error("--from must be on or before --to")
    rows = asyncio.run(usage_by_day(args.start, args.end))
    if args.summary:
        print(json.dumps(summarize_usage(rows), ensure_ascii=False, indent=2))
        return
    if args.format == "json":
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return
    columns = ("day", "intent", "model_name", "calls", "available_calls",
               "unavailable_calls", "input_tokens", "output_tokens")
    if not rows:
        print("No usage events in selected range")
        return
    rendered = [[str(row[key]) if row[key] is not None else "unavailable" for key in columns]
                for row in rows]
    widths = [max(len(key), *(len(row[index]) for row in rendered))
              for index, key in enumerate(columns)]
    print("  ".join(key.ljust(width) for key, width in zip(columns, widths)))
    for row in rendered:
        print("  ".join(value.ljust(width) for value, width in zip(row, widths)))


if __name__ == "__main__":
    main()
