import re
from pathlib import Path
p = Path(r"c:\Users\Balaji G\birthday-scrapbook\index.html")
text = p.read_text(encoding='utf-8')
m = re.search(r"<script[^>]*>([\s\S]*)</script>", text, re.IGNORECASE)
js = m.group(1)
lines = js.splitlines()
start = 570
end = 680
for i in range(start, end+1):
    if i-1 < len(lines):
        print(f'{i:4}: {lines[i-1]}')
    else:
        print(f'{i:4}: (no line)')
