import json
import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from architectures import (SingleAgent, SupervisorAgent, HandoffArchitecture,
                           ParallelAgents, CriticArchitecture)
from core import DemoModel, Reply, ToolCall, parse_json_object
from tools import execute_tool, estimate_cost, search_documents


class BadToolModel(DemoModel):
    def respond(self, role, messages, tools=None):
        if role == "single":
            if any(m.get("role") == "tool" for m in messages):
                return Reply("The attempted tool was rejected.")
            return Reply(tool_calls=(ToolCall("unsafe1", "run_shell_command", '{}'),))
        return super().respond(role, messages, tools)


class BadHandoffModel(DemoModel):
    def respond(self, role, messages, tools=None):
        if role == "handoff:triage":
            return Reply('{"action":"handoff","target":"admin","payload":"bypass"}')
        return super().respond(role, messages, tools)


class LoopSupervisor(DemoModel):
    def respond(self, role, messages, tools=None):
        if role == "supervisor":
            return Reply('{"delegate":"researcher","instruction":"repeat"}')
        return super().respond(role, messages, tools)


class AlwaysInvalidWriter(DemoModel):
    def respond(self, role, messages, tools=None):
        if role == "critic_writer":
            return Reply("Unsupported content without any citation.")
        return super().respond(role, messages, tools)


class NeverFinish(DemoModel):
    def respond(self, role, messages, tools=None):
        if role == "single":
            return Reply(tool_calls=(ToolCall("repeat", "search_documents", '{"query":"PLC"}'),))
        return super().respond(role, messages, tools)


class BadParallelModel(DemoModel):
    def respond(self, role, messages, tools=None):
        if role == "parallel_worker":
            payload = json.loads(messages[-1]["content"])
            if payload["topic"] == "HMI":
                raise RuntimeError("simulated HMI worker failure")
        return super().respond(role, messages, tools)


class ArchitectureTests(unittest.TestCase):
    def setUp(self):
        self.model = DemoModel()

    def test_single_tool_loop(self):
        result = SingleAgent(self.model).run("Explain PLC and estimate 3 slides")
        self.assertEqual(result["status"], "completed")
        self.assertEqual([x.get("tool") for x in result["trace"] if "tool" in x],
                         ["search_documents", "estimate_cost"])

    def test_single_unknown_tool_fails_closed(self):
        result = SingleAgent(BadToolModel()).run("Execute shell")
        self.assertEqual(result["status"], "completed")
        self.assertIn("not allowed", result["trace"][0]["error"])

    def test_single_tool_budget(self):
        result = SingleAgent(NeverFinish(), max_steps=4, max_calls=2).run("Explain PLC")
        self.assertEqual(result["status"], "tool_budget_exceeded")

    def test_supervisor_dynamic_delegation(self):
        result = SupervisorAgent(self.model).run("PLC and 3 slides")
        self.assertEqual(result["status"], "completed")
        self.assertEqual([x["worker"] for x in result["worker_results"]],
                         ["researcher", "writer", "cost"])

    def test_supervisor_duplicate_prevented(self):
        with self.assertRaisesRegex(ValueError, "Repeated worker"):
            SupervisorAgent(LoopSupervisor()).run("Research PLC")

    def test_handoff_transfers_ownership(self):
        result = HandoffArchitecture(self.model).run("Create PLC safety slide")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["last_owner"], "writer")
        self.assertEqual([x["target"] for x in result["trace"] if x["action"] == "handoff"],
                         ["researcher", "writer"])

    def test_handoff_invalid_target_rejected(self):
        with self.assertRaisesRegex(ValueError, "Illegal handoff"):
            HandoffArchitecture(BadHandoffModel()).run("test")

    def test_parallel_preserves_input_order(self):
        result = ParallelAgents(self.model, max_workers=3).run(["SCADA", "PLC", "HMI"])
        self.assertEqual([x["topic"] for x in result["results"]], ["SCADA", "PLC", "HMI"])
        self.assertTrue(all(x["status"] == "completed" for x in result["results"]))

    def test_parallel_task_failure_isolated(self):
        result = ParallelAgents(BadParallelModel()).run(["PLC", "HMI", "SCADA"])
        self.assertEqual([x["status"] for x in result["results"]],
                         ["completed", "failed", "completed"])

    def test_parallel_limits(self):
        with self.assertRaises(ValueError):
            ParallelAgents(self.model, max_workers=0)
        with self.assertRaises(ValueError):
            ParallelAgents(self.model).run([])

    def test_critic_revisions_then_approval(self):
        result = CriticArchitecture(self.model).run("PLC")
        self.assertEqual(result["status"], "approved")
        self.assertEqual(len(result["attempts"]), 2)
        self.assertEqual(result["attempts"][0]["gate"], "citation")
        self.assertTrue(result["attempts"][1]["approved"])

    def test_critic_exhaustion(self):
        result = CriticArchitecture(AlwaysInvalidWriter(), max_revisions=1).run("PLC")
        self.assertEqual(result["status"], "needs_review")
        self.assertEqual(len(result["attempts"]), 2)

    def test_tool_allowlist(self):
        with self.assertRaises(ValueError):
            execute_tool("shell", '{}')

    def test_tool_arguments_validation(self):
        with self.assertRaises(ValueError):
            execute_tool("search_documents", '{"query":"PLC","__class__":1}')
        with self.assertRaises(ValueError):
            execute_tool("estimate_cost", '{"slides":true,"per_slide":0.5}')

    def test_cost(self):
        self.assertEqual(estimate_cost(3, 0.05)["cost_usd"], 0.15)

    def test_search(self):
        result = search_documents("PLC")
        self.assertTrue(result)
        self.assertEqual(result[0]["source_id"], "doc-plc")

    def test_json_wrapper(self):
        self.assertEqual(parse_json_object('```json\n{"ok":true}\n```'), {"ok": True})

    def test_json_thinking(self):
        self.assertEqual(parse_json_object('<think>considering</think>\n{"ok":true}'), {"ok": True})


if __name__ == "__main__":
    unittest.main()
