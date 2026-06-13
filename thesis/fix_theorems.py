"""One-shot fix: convert \\textbf{Theorem N ...} to proper \\begin{theorem} environments."""
import re

path = r"E:\基于深度学习的遥感影像光谱分类\project1\thesis\chapters\02-math-foundations.tex"
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# Count before
before = len(re.findall(r'\\textbf\{Theorem \d', content))
print(f'Theorems before: {before}')

# Strategy: split by theorem markers and rebuild
lines = content.split('\n')
output_lines = []
i = 0
in_theorem = False
theorem_body = []
theorem_title = ""

while i < len(lines):
    line = lines[i]

    # Match \textbf{Theorem N (Title).}
    m = re.match(r'\\textbf\{Theorem (\d+) \(([^)]+)\)\.\}', line)
    if m:
        title = m.group(2)
        label = re.sub(r'[^a-z0-9-]', '', title.lower())[:50]
        output_lines.append(f'\\begin{{theorem}}[{title}]\\label{{thm:{label}}}')
        in_theorem = True
        theorem_lines = []
        i += 1
        continue

    # Match \textbf{Corollary N (Title).}
    m = re.match(r'\\textbf\{Corollary (\d+) \(([^)]+)\)\.\}', line)
    if m:
        title = m.group(2)
        label = re.sub(r'[^a-z0-9-]', '', title.lower())[:50]
        output_lines.append(f'\\begin{{corollary}}[{title}]\\label{{cor:{label}}}')
        in_theorem = True
        theorem_lines = []
        i += 1
        continue

    # End theorem when we hit next section/subsection or next theorem
    if in_theorem:
        if (re.match(r'\\textbf\{Theorem \d', line) or
            re.match(r'\\textbf\{Corollary \d', line) or
            re.match(r'\\subsubsection\{', line) or
            re.match(r'\\subsection\{', line) or
            re.match(r'\\section\{', line)):
            output_lines.append('\\end{' + ('theorem' if 'theorem' in output_lines[-1] else 'corollary') + '}')
            output_lines.append('')
            in_theorem = False
            # Reprocess this line
            continue
        else:
            output_lines.append(line)
    else:
        output_lines.append(line)

    i += 1

# Close any unclosed theorem
if in_theorem:
    output_lines.append('\\end{theorem}')

result = '\n'.join(output_lines)

# Count after
after = len(re.findall(r'\\begin\{theorem\}', result))
print(f'Theorems after: {after}')

with open(path, 'w', encoding='utf-8') as f:
    f.write(result)
print('Done.')
