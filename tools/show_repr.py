import re
from pathlib import Path
p = Path(r"c:\Users\Balaji G\birthday-scrapbook\index.html")
text = p.read_text(encoding='utf-8')
m = re.search(r"<script[^>]*>([\s\S]*)</script>", text, re.IGNORECASE)
js = m.group(1)
lines = js.splitlines()
for i in range(584, 668):
    if i-1 < len(lines):
        print(f'{i:4}: {repr(lines[i-1])}')
    else:
        print(f'{i:4}: (no line)')
