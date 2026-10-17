"""CLI: python -m metso_assistant [--message ...] or interactive chat."""
import argparse
from pathlib import Path
import sys

from . import WORK_DIR
from .agent import Assistant, BASELINE_RECOVERY
from .model import LocalModel
from .services import json_numbers


def _engine_argument(value):
    if value == "baseline":
        raise argparse.ArgumentTypeError(BASELINE_RECOVERY)
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--message", help="Single message; omit for interactive chat")
    parser.add_argument("--store", type=Path, default=WORK_DIR / "prepared/knowledge.sqlite")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--raw-dir", type=Path, default=WORK_DIR / "raw")
    parser.add_argument("--endpoint", default="http://127.0.0.1:11435")
    parser.add_argument("--model", default="qwen3-vl:8b-instruct-q4_K_M")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--context", type=int, default=16384)
    parser.add_argument("--trace-dir", type=Path, default=WORK_DIR / "runs")
    parser.add_argument("--retrieval", choices=("keyword", "semantic", "hybrid"), default="keyword")
    parser.add_argument("--semantic-index", type=Path, default=WORK_DIR / "brochure-semantic.json")
    parser.add_argument("--embedding-endpoint", help="Local embedding runtime; defaults to --endpoint")
    parser.add_argument("--json", action="store_true", help="Print complete structured result and tool trace")
    parser.add_argument("--engine", type=_engine_argument, choices=("adaptive",), default="adaptive", help="Current evidence engine; retained for command compatibility")
    parser.add_argument("--service-index", type=Path, help="Optional derived service-search index location")
    parser.add_argument("--service-semantic-index", type=Path, default=WORK_DIR / "service-semantic.json")
    args = parser.parse_args(argv)
    if not args.store.is_file():
        parser.error("Prepared SQLite store is missing; run corpus preparation first")
    if args.retrieval != "keyword" and not args.semantic_index.is_file():
        parser.error("Build the optional semantic index before selecting semantic/hybrid retrieval")
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(encoding="utf-8")
    assistant = Assistant(store_path=args.store, data_dir=args.data_dir, raw_dir=args.raw_dir,
                          model=LocalModel(args.endpoint, args.model, timeout=args.timeout, context=args.context), trace_dir=args.trace_dir,
                          retrieval_mode=args.retrieval, semantic_index=args.semantic_index, embedding_endpoint=args.embedding_endpoint or args.endpoint,
                          engine=args.engine, service_index=args.service_index, service_semantic_index=args.service_semantic_index)
    def show(message):
        result = assistant.turn(message)
        print(json_numbers(result) if args.json else result["answer"], flush=True)
        return result["status"] != "error"
    if args.message is not None:
        return 0 if show(args.message) else 1
    print("Metso assistant. /new starts a fresh conversation; /quit exits.", flush=True)
    while True:
        try:
            message = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); return 0
        if message.casefold() in {"/quit", "/exit"}:
            return 0
        if message.casefold() == "/new":
            assistant.reset(); print("Started a fresh conversation."); continue
        if message:
            try:
                show(message)
            except ValueError as exc:
                print(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
