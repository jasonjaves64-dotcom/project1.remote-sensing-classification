"""Final thesis integrity check."""
import os

all_text = ''
bib_text = ''
for f in sorted(os.listdir('chapters')):
    if f.endswith('.tex'):
        path = os.path.join('chapters', f)
        with open(path, encoding='utf-8') as fh:
            all_text += fh.read()
with open('references.bib', encoding='utf-8') as fh:
    bib_text = fh.read()

# Count balanced environments
environments = [
    'theorem', 'equation', 'align', 'table', 'algorithm',
    'enumerate', 'itemize', 'figure'
]

print('=== FINAL THESIS INTEGRITY ===')
print()

issues = []
for env in environments:
    label = 'begin{' + env + '}'
    elabel = 'end{' + env + '}'
    b = all_text.count(label)
    e = all_text.count(elabel)
    status = 'BALANCED' if b == e else 'FAIL'
    if b + e > 0:
        print(f'  {env:<15}  begin={b}  end={e}  {status}')
        if b != e:
            issues.append(f'{env}: {b} begins vs {e} ends')

print()
print(f'  Total environments: {sum(all_text.count("begin{") for _ in [1])}')

# Count lines per file
print()
print('--- File Inventory ---')
total = 0
files = ['00-abstract.tex', '01-introduction.tex', '02-math-foundations.tex',
         '02b-cross-module-bridges.tex', '03-v6-architecture.tex',
         '03b-geographic-analysis.tex', '04-experiments.tex',
         '05-conclusion.tex', '06-acknowledgments.tex', 'appendix.tex']
for f in files:
    path = os.path.join('chapters', f)
    if os.path.exists(path):
        lines = len(open(path, encoding='utf-8').readlines())
        total += lines
        print(f'  {f:<40} {lines:>4} lines')
main_lines = len(open('main.tex', encoding='utf-8').readlines())
bib_lines = len(open('references.bib', encoding='utf-8').readlines())
total += main_lines + bib_lines
print(f'  main.tex                                   {main_lines:>4} lines')
print(f'  references.bib                             {bib_lines:>4} lines')
print(f'  {"TOTAL":<40} {total:>4} lines')

print()
if issues:
    print('ISSUES:')
    for i in issues:
        print(f'  ! {i}')
else:
    print('ALL ENVIRONMENTS BALANCED - ready to compile')
