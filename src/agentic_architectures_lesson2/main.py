"""CLI entry point for all five architecture demos."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from architectures import (SingleAgent, SupervisorAgent, HandoffArchitecture,
                           ParallelAgents, CriticArchitecture)
from core import create_model

DEFAULT_REQUEST = "Explain PLC basics and estimate cost for 3 slides at $0.05 per slide"


def run_pattern(name: str, model, workers: int = 2):
    if name == "single":
        return SingleAgent(model).run(DEFAULT_REQUEST)
    if name == "supervisor":
        return SupervisorAgent(model).run(DEFAULT_REQUEST)
    if name == "handoff":
        return HandoffArchitecture(model).run("Create a short PLC safety training slide")
    if name == "parallel":
        return ParallelAgents(model, max_workers=workers).run(["PLC", "HMI", "SCADA"])
    if name == "critic":
        return CriticArchitecture(model).run("PLC")
    raise ValueError(f"Unknown architecture: {name}")


def main():
    parser = argparse.ArgumentParser(description="Lesson 2 — five agent architectures")
    parser.add_argument("architecture", choices=["single", "supervisor", "handoff", "parallel", "critic", "all"])
    parser.add_argument("--mode", choices=["mock", "ollama"], default="mock",
                        help="mock=offline deterministic; ollama=actual local model")
    parser.add_argument("--model", default=None, help="Local Ollama model, e.g. qwen3:4b")
    parser.add_argument("--workers", type=int, default=2, help="Parallel worker count 1..4")
    args = parser.parse_args()
    model = create_model(args.mode, args.model)
    patterns = ["single", "supervisor", "handoff", "parallel", "critic"] if args.architecture == "all" else [args.architecture]
    output = {}
    for name in patterns:
        print(f"\n===== {name.upper()} ({args.mode}) =====", flush=True)
        try:
            output[name] = run_pattern(name, model, args.workers)
        except Exception as exc:
            output[name] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(output[name], ensure_ascii=False, indent=2), flush=True)
    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(exist_ok=True)
    file_path = out_dir / f"results_{args.mode}.json"
    file_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved results to {file_path}")


if __name__ == "__main__":
    main()
