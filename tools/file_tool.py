"""Read-only file tools confined to a configured workspace directory."""

from pathlib import Path

from langchain_core.tools import BaseTool, tool

from config.guardrails import untrusted_data_block

_BLOCKED_PARTS = {".env", ".git", ".venv", "venv", "__pycache__"}
_MAX_FILE_BYTES = 100_000
_MAX_LISTED_FILES = 100


def create_file_tools(workspace_root: Path) -> list[BaseTool]:
    """Create file tools whose paths cannot escape the configured workspace."""
    root = workspace_root.resolve()
    root.mkdir(parents=True, exist_ok=True)

    @tool
    def list_workspace_files() -> str:
        """List up to 100 readable files in the configured workspace."""
        files = []
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root)
            if path.is_symlink() or any(
                part.lower() in _BLOCKED_PARTS or part.startswith(".")
                for part in relative.parts
            ):
                continue
            if path.is_file():
                files.append(relative.as_posix())
                if len(files) >= _MAX_LISTED_FILES:
                    break
        listing = "\n".join(files) if files else "No workspace files found."
        return untrusted_data_block("WORKSPACE FILE LIST", listing)

    @tool
    def read_workspace_file(relative_path: str) -> str:
        """Read a UTF-8 text file from the workspace using a relative path."""
        candidate = Path(relative_path)
        if candidate.is_absolute():
            raise ValueError("Only relative paths inside the workspace are allowed.")
        if any(
            part.lower() in _BLOCKED_PARTS or part.startswith(".")
            for part in candidate.parts
        ):
            raise ValueError("That path is not available to the file tool.")

        resolved = (root / candidate).resolve()
        if not resolved.is_relative_to(root):
            raise ValueError("The requested path is outside the configured workspace.")
        resolved_relative = resolved.relative_to(root)
        if any(
            part.lower() in _BLOCKED_PARTS or part.startswith(".")
            for part in resolved_relative.parts
        ):
            raise ValueError("That path is not available to the file tool.")
        if not resolved.is_file():
            raise FileNotFoundError(f"Workspace file not found: {relative_path}")
        if resolved.stat().st_size > _MAX_FILE_BYTES:
            raise ValueError("File exceeds the 100 KB read limit.")
        try:
            content = resolved.read_text(encoding="utf-8")
            return untrusted_data_block("WORKSPACE FILE", content)
        except UnicodeDecodeError as exc:
            raise ValueError("Only UTF-8 text files can be read.") from exc

    return [list_workspace_files, read_workspace_file]
