"""让 `python scripts/xxx.py` 能 import src.*（把项目根目录加入 sys.path）。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
