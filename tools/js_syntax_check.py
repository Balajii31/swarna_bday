import re
from pathlib import Path
p = Path(r"c:\Users\Balaji G\birthday-scrapbook\index.html")
text = p.read_text(encoding='utf-8')
# extract first <script> ... </script>
m = re.search(r"<script[^>]*>([\s\S]*)</script>", text, re.IGNORECASE)
if not m:
    print('No <script> tag found')
    raise SystemExit(1)
js = m.group(1)
lines = js.splitlines()
# naive balance checks
pairs = {'(': ')', '{': '}', '[': ']'}
open_stack = []
for i, line in enumerate(lines, start=1):
    for ch in line:
        if ch in pairs:
            open_stack.append((ch, i, line))
        elif ch in pairs.values():
            if not open_stack:
                print(f'Unmatched closing {ch} at line {i}: {line.strip()}')
            else:
                last, lnum, ltext = open_stack[-1]
                if pairs[last] == ch:
                    open_stack.pop()
                else:
                    print(f'Mismatched {last} opened at {lnum} but closed by {ch} at line {i}')
# report remaining opens
if open_stack:
    print('Remaining unclosed delimiters:')
    for ch, lnum, ltext in open_stack[-10:]:
        print(f"  {ch} opened at line {lnum}: {ltext.strip()[:120]}")
else:
    print('All parens/braces/brackets appear balanced (naive check).')
# Check for unclosed quotes per type
for quote in ("'", '"', '`'):
    opened = False
    escape = False
    for i, line in enumerate(lines, start=1):
        for j, ch in enumerate(line):
            if ch == '\\' and not escape:
                escape = True
                continue
            if ch == quote and not escape:
                opened = not opened
            escape = False
    print(f"Quote {quote}: {'open' if opened else 'closed'} (naive) ")

# Print a small window around lines that contain the replacement char
for i, line in enumerate(lines, start=1):
    if '\uFFFD' in line or '\ufffd' in line or '�' in line:
        start = max(1, i-3)
        end = min(len(lines), i+3)
        print('\nFound replacement char around line', i)
        for k in range(start, end+1):
            print(f'{k:4}: {lines[k-1].rstrip()}')

# Heuristic: find lines with 'console.' and long lines that might be truncated
for i, line in enumerate(lines, start=1):
    if 'console.' in line and (len(line) > 200):
        print('\nLong console line at', i)
        print(line)

print('\nDone analysis')
