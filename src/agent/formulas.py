
import re
from typing import List, Dict, Tuple


def extract_equations(text: str) -> List[str]:
    """Extract mathematical equations from text using regex patterns."""
    # Common equation patterns
    patterns = [
        r'\$[^$]+\$',  # LaTeX inline math
        r'\\\[[^\]]+\\\]',  # LaTeX display math
        r'\\\([^)]+\\\)',  # LaTeX inline math alternative
        r'[A-Za-z]\s*=\s*[^=\n]+',  # Variable assignments
        r'[A-Za-z]\s*\([^)]+\)\s*=\s*[^=\n]+',  # Function definitions
        r'[A-Za-z]\s*[+\-*/]\s*[A-Za-z0-9\s+\-*/()]+',  # Basic expressions
    ]
    
    equations = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.MULTILINE)
        equations.extend(matches)
    
    # Clean and deduplicate
    cleaned = []
    for eq in equations:
        eq = eq.strip()
        if len(eq) > 3 and eq not in cleaned:
            cleaned.append(eq)
    
    return cleaned


def guess_symbol_definitions(text: str, equation: str) -> Dict[str, str]:
    """Guess symbol definitions from context around the equation."""
    definitions = {}
    
    # Extract symbols from equation
    symbols = re.findall(r'[A-Za-z]+', equation)
    
    # Look for definitions in surrounding text
    lines = text.split('\n')
    for i, line in enumerate(lines):
        if equation in line:
            # Check lines before and after for definitions
            context_lines = lines[max(0, i-3):i+4]
            context = ' '.join(context_lines)
            
            for symbol in symbols:
                if len(symbol) > 1:  # Skip single letters
                    # Look for patterns like "where X is..." or "X represents..."
                    patterns = [
                        rf'{symbol}\s+is\s+([^.]*)',
                        rf'{symbol}\s+represents?\s+([^.]*)',
                        rf'{symbol}\s+denotes?\s+([^.]*)',
                        rf'{symbol}\s+=\s*([^=\n]*)',
                    ]
                    
                    for pattern in patterns:
                        match = re.search(pattern, context, re.IGNORECASE)
                        if match:
                            definition = match.group(1).strip()
                            if len(definition) > 3:
                                definitions[symbol] = definition
                                break
    
    return definitions


def format_equation_latex(equation: str) -> str:
    """Format equation for LaTeX display."""
    # Clean up the equation
    equation = equation.strip()
    
    # Remove LaTeX delimiters if present
    equation = re.sub(r'^\$|\$$', '', equation)
    equation = re.sub(r'^\\\[|\\\]$', '', equation)
    equation = re.sub(r'^\\\(|\\\)$', '', equation)
    
    # Basic LaTeX formatting
    equation = equation.replace('*', r'\cdot')
    equation = equation.replace('^', '^')
    equation = equation.replace('_', '_')
    
    # Ensure proper spacing around operators
    equation = re.sub(r'([+\-*/=])([A-Za-z0-9])', r'\1 \2', equation)
    equation = re.sub(r'([A-Za-z0-9])([+\-*/=])', r'\1 \2', equation)
    
    return equation