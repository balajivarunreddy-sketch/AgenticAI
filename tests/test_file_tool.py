"""Focused tests for workspace path restrictions and hidden-file blocking."""

import tempfile
import unittest
from pathlib import Path

from tools.file_tool import create_file_tools


class FileToolTests(unittest.TestCase):
    def test_reads_workspace_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "notes.txt").write_text("practice plan", encoding="utf-8")
            _, read_tool = create_file_tools(root)
            result = read_tool.invoke({"relative_path": "notes.txt"})
            self.assertIn("practice plan", result)
            self.assertIn('"source": "WORKSPACE FILE"', result)

    def test_redacts_credentials_from_workspace_file_content(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            secret = "nvapi-" + "x" * 24
            (root / "notes.txt").write_text(
                f"Do not expose API_KEY={secret}", encoding="utf-8"
            )
            _, read_tool = create_file_tools(root)

            result = read_tool.invoke({"relative_path": "notes.txt"})

        self.assertNotIn(secret, result)
        self.assertIn("[REDACTED]", result)

    def test_rejects_parent_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "workspace"
            root.mkdir()
            outside_file = Path(temporary_directory) / "secret.txt"
            outside_file.write_text("not available", encoding="utf-8")
            _, read_tool = create_file_tools(root)
            with self.assertRaises(ValueError):
                read_tool.invoke({"relative_path": "../secret.txt"})

    def test_blocks_environment_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / ".env").write_text("SECRET=value", encoding="utf-8")
            _, read_tool = create_file_tools(root)
            with self.assertRaises(ValueError):
                read_tool.invoke({"relative_path": ".env"})


if __name__ == "__main__":
    unittest.main()
