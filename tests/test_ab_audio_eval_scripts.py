from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = ROOT / "tests"


class _AbEvalHandler(BaseHTTPRequestHandler):
    server_version = "AbEvalTest/1.0"

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        _ = self.rfile.read(length)

        if self.path == "/api/v1/auth/login":
            self._send_json(200, {"token": "test-token"})
            return

        if self.path == "/api/v1/process-call":
            self._send_json(
                200,
                {
                    "status": "completed",
                    "call_id": "call-123",
                    "total_time": 5.5,
                    "processing_time": {
                        "transcription": 4.1,
                        "routing": 0.7,
                        "entity_extraction": 0.4,
                        "ticket_creation": 0.3,
                    },
                    "routing": {
                        "intent_id": "consulting",
                        "suggested_group": "sales",
                        "priority": "high",
                        "intent_confidence": 0.91,
                    },
                    "ticket": {
                        "external_id": "T-1",
                        "url": "https://example.test/t/T-1",
                    },
                },
            )
            return

        self._send_json(404, {"error": "not_found"})

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return

    def _send_json(self, status: int, payload) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class AbAudioEvalScriptsTest(unittest.TestCase):
    def _run(self, script_name: str, *args: str) -> subprocess.CompletedProcess[str]:
        script_path = TESTS_DIR / script_name
        return subprocess.run(
            [sys.executable, str(script_path), *args],
            check=True,
            capture_output=True,
            text=True,
        )

    def test_prepare_merge_and_evaluate_ab_from_audio(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            audio_dir = tmp / "audio"
            out_dir = tmp / "out"
            audio_dir.mkdir()
            (audio_dir / "call_01.wav").write_bytes(b"fake wav bytes")

            server = ThreadingHTTPServer(("127.0.0.1", 0), _AbEvalHandler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base_url = f"http://127.0.0.1:{server.server_address[1]}"

            try:
                self._run(
                    "prepare_ab_eval_audio.py",
                    "--audio-dir",
                    str(audio_dir),
                    "--base-url",
                    base_url,
                    "--username",
                    "admin",
                    "--password",
                    "secret",
                    "--out-dir",
                    str(out_dir),
                )

                system_csv = out_dir / "system_results.csv"
                manual_csv = out_dir / "manual_template.csv"
                self.assertTrue(system_csv.exists())
                self.assertTrue(manual_csv.exists())

                with manual_csv.open("r", encoding="utf-8-sig", newline="") as file:
                    rows = list(csv.DictReader(file))
                self.assertEqual(len(rows), 1)
                rows[0]["manual_time_sec"] = "42"
                rows[0]["final_intent_id"] = "consulting"
                rows[0]["final_group_id"] = "sales"
                rows[0]["final_priority"] = "high"
                with manual_csv.open("w", encoding="utf-8-sig", newline="") as file:
                    writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
                    writer.writeheader()
                    writer.writerows(rows)

                self._run(
                    "merge_ab_eval_audio.py",
                    "--system-csv",
                    str(system_csv),
                    "--manual-csv",
                    str(manual_csv),
                )

                merged_csv = out_dir / "ab_eval_merged.csv"
                self.assertTrue(merged_csv.exists())

                self._run("evaluate_ab_test.py", "--csv", str(merged_csv))
                report = json.loads((out_dir / "ab_test_metrics.json").read_text(encoding="utf-8"))
                self.assertAlmostEqual(report["manual"]["avg_time_sec"], 42.0, places=6)
                self.assertAlmostEqual(report["system"]["avg_time_sec"], 5.5, places=6)
                self.assertAlmostEqual(report["agreement"]["intent"]["accuracy"], 1.0, places=6)
                self.assertAlmostEqual(report["agreement"]["group"]["accuracy"], 1.0, places=6)
                self.assertAlmostEqual(report["agreement"]["priority"]["accuracy"], 1.0, places=6)
            finally:
                server.shutdown()
                server.server_close()


if __name__ == "__main__":
    unittest.main()
