import sys
from pathlib import Path

# Принудительно добавляем src в sys.path
src_path = str(Path(__file__).parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from mnemo.mcp.server import mcp_app  # noqa: E402

if __name__ == "__main__":
    mcp_app.run(transport="stdio")
