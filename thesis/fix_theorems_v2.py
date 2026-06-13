"""Fix all theorem environment nesting issues."""
import re

path = r"chapters\02-math-foundations.tex"
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# Fix 1: Add missing \end{theorem} before Theorem 2 (line 84-85)
content = content.replace(
    r"\end{equation}\end{theorem}\begin{theorem}[Theorema Egregium",
    r"\end{equation}\end{theorem}\begin{theorem}[Theorema Egregium"
)

# Fix 2: The Theorem 2 (Theorema Egregium) needs \end{theorem} after it
content = content.replace(
    r"remaining strictly invariant under any isometric deformation of the surface.\end{theorem}",
    r"remaining strictly invariant under any isometric deformation of the surface."
)
# Actually, T2 already has its own text. Let me find the exact pattern.
# T2 is: \begin{theorem}[Theorema Egregium ...] ... surface.
# No \end{theorem} after "surface."

# Better approach: find all \begin{theorem} and \end{theorem} positions
# and fix nesting

lines = content.split('\n')
new_lines = []
open_theorems = 0
i = 0

while i < len(lines):
    line = lines[i]

    # Count begins and ends on this line
    begins_here = line.count(r'\begin{theorem}')
    ends_here = line.count(r'\end{theorem}')

    # Don't count the \end{theorem} that's part of \end{theorem}\begin{theorem}
    # (that's a close+open on same line)

    new_lines.append(line)
    i += 1

# Post-process: just add missing \end{theorem} where needed
# Strategy: find every \begin{theorem} and ensure it has \end{theorem} before
# the next \section, \subsection, \subsubsection, or \begin{theorem}

# Regex to find unclosed theorems
# Split content into blocks and ensure each \begin{theorem} has matching \end{theorem}

# Simpler approach: add \end{theorem} before every \begin{theorem} that's
# preceded by another \begin{theorem} without an intervening \end{theorem}

# Actually the cleanest approach:
# 1. Find all \begin{theorem}...\end{theorem} pairs
# 2. Ensure no nesting

# Find all theorem blocks and re-extract
pattern = re.compile(
    r'\\begin\{theorem\}(.*?)\\end\{theorem\}',
    re.DOTALL
)

# Count
begins = content.count(r'\begin{theorem}')
ends = content.count(r'\end{theorem}')
print(f"Before: {begins} begins, {ends} ends")

# Manual fixes for known issues:
fixes = [
    # Fix nested: Unbiasedness starts, then Dimension-Independent starts
    (r"\begin{theorem}[Unbiasedness]\label{thm:unbiasedness}\n$\mathbb{E}_\theta[\widehat{\text{SGW}}_{ub}^L] = \text{SGW}_{ub}$ for any $L \geq 1$.\n\n\begin{theorem}[Dimension-Independent Concentration]",
     r"\begin{theorem}[Unbiasedness]\label{thm:unbiasedness}\n$\mathbb{E}_\theta[\widehat{\text{SGW}}_{ub}^L] = \text{SGW}_{ub}$ for any $L \geq 1$.\n\end{theorem}\n\n\begin{theorem}[Dimension-Independent Concentration]"),

    # Fix nested: Closed-Form V-Subproblem starts, then Linear Convergence starts
    (r"Complexity: $O(dk^2)$, vs. $O(d^3)$ for generic manifold optimization.\n\n\begin{theorem}[Linear Convergence]",
     r"Complexity: $O(dk^2)$, vs. $O(d^3)$ for generic manifold optimization.\n\end{theorem}\n\n\begin{theorem}[Linear Convergence]"),

    # Fix missing \end{theorem} after Theorema Egregium
    (r"remaining strictly invariant under any isometric deformation of the surface.\n\n\subsubsection{Numerical Stability}",
     r"remaining strictly invariant under any isometric deformation of the surface.\n\end{theorem}\n\n\subsubsection{Numerical Stability}"),

    # Fix missing \end{theorem} after SE(3)-Invariance (same line as next begin)
    (r"\end{equation}\end{theorem}\begin{theorem}[Theorema Egregium",
     r"\end{equation}\end{theorem}\n\end{theorem}\n\begin{theorem}[Theorema Egregium"),
]

for old, new in fixes:
    if old in content:
        content = content.replace(old, new)
        print(f"  Applied fix: {old[:60]}...")
    else:
        print(f"  NOT FOUND: {old[:60]}...")

# Remove duplicate \end{theorem} that might have been created
content = content.replace(r"\end{theorem}\n\end{theorem}", r"\end{theorem}")

begins = content.count(r'\begin{theorem}')
ends = content.count(r'\end{theorem}')
print(f"After: {begins} begins, {ends} ends")
print(f"Balanced: {begins == ends}")

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
