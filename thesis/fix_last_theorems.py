"""Fix the last 2 unconverted \\textbf{Theorem} instances."""
import re

path = r"chapters\02-math-foundations.tex"
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# Fix Theorem 1 (SE(3)-Invariance)
old1 = r"\textbf{Theorem 1 (SE(3)-Invariance).}"
new1 = r"\begin{theorem}[SE(3)-Invariance]\label{thm:se3-invariance}"
content = content.replace(old1, new1)

# Fix Theorem 2 (Theorema Egregium)
old2 = r"\textbf{Theorem 2 (Theorema Egregium — Isometry Invariance).}"
new2 = r"\begin{theorem}[Theorema Egregium — Isometry Invariance]\label{thm:egregium}"

# Check if the old text exists before replacing
if old2 in content:
    content = content.replace(old2, new2)
else:
    # Try alternative encoding of the em-dash
    old2b = r"\textbf{Theorem 2 (Theorema Egregium --- Isometry Invariance).}"
    new2b = r"\begin{theorem}[Theorema Egregium --- Isometry Invariance]\label{thm:egregium}"
    if old2b in content:
        content = content.replace(old2b, new2b)
    else:
        print("WARNING: Could not find Theorem 2 text")
        # Show surrounding context
        idx = content.find("Theorema Egregium")
        if idx >= 0:
            print(content[idx-20:idx+200])

# Add \end{theorem} after each
# For Theorem 1: it ends at the equation, before the blank line and next \begin{theorem}
# Insert \end{theorem} before "\begin{theorem}[Theorema Egregium"
content = content.replace(
    r"\end{equation}\end{theorem}",
    r"\end{equation}"
)
# Actually we need to insert \end{theorem} after the equation
# Find: |I(p) - I(g \cdot p)| \leq 2.3... \end{equation}
# Insert \end{theorem} after it

# Simpler approach: add \end{theorem} before the next theorem begins
content = content.replace(
    r"\begin{theorem}[Theorema Egregium",
    r"\end{theorem}\begin{theorem}[Theorema Egregium"
)

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)

# Verify
b = content.count('\\begin{theorem}')
e = content.count('\\end{theorem}')
print(f"Theorem begins: {b}")
print(f"Theorem ends: {e}")
print(f"Balanced: {b == e}")
