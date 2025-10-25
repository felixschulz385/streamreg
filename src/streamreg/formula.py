import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import logging
from itertools import combinations
import warnings
import re

logger = logging.getLogger(__name__)

class FormulaParser:
    """
    Parser for R-style formulas with support for:
    - Basic terms: y ~ x1 + x2
    - Interactions: x1:x2 or x1*x2 (includes main effects)
    - Polynomials: I(x^2) or poly(x, 2)
    - Instruments: y ~ x1 + x2 | z1 + z2 (for 2SLS)
    """
    
    def __init__(self, formula: str):
        """
        Initialize formula parser.
        
        Parameters:
        -----------
        formula : str
            R-style formula string (e.g., "y ~ x1 + x2 + I(x1^2) | z1")
        """
        self.formula = formula.strip()
        self.target = None
        self.features = []
        self.instruments = []
        self.transformations = []
        self.has_intercept = True

        # New: demeaning specification parsed from formula (if present)
        self.demean_groups: List[List[str]] = []
        self.demean_except: List[str] = []

        self._parse_formula()
    
    def _parse_formula(self):
        """Parse the formula string."""
        # Split by ~ to separate target from features
        if '~' not in self.formula:
            raise ValueError(f"Formula must contain '~': {self.formula}")
        
        parts = self.formula.split('~')
        if len(parts) != 2:
            raise ValueError(f"Formula must have exactly one '~': {self.formula}")
        
        self.target = parts[0].strip()
        right_side = parts[1].strip()
        
        # Split by | to separate features from instruments / demean spec
        if '|' in right_side:
            feature_part, instrument_part = right_side.split('|', 1)
            instrument_part = instrument_part.strip()
            # If instrument part contains demeaning syntax (parentheses or 'except'),
            # parse as demean specification. Otherwise treat as instruments.
            if '(' in instrument_part or re.search(r'\bexcept\b', instrument_part, flags=re.IGNORECASE):
                groups, except_list = self._parse_demean_spec(instrument_part)
                self.demean_groups = groups
                self.demean_except = except_list
                # Instruments remain empty in this case
                self.instruments = []
            else:
                self.instruments = self._parse_term_list(instrument_part)
        else:
            feature_part = right_side

        # Parse features and transformations
        self._parse_features(feature_part)
    
    def _parse_features(self, feature_part: str):
        """Parse feature part of formula and extract transformations."""
        # Check for intercept removal
        if '-1' in feature_part or '- 1' in feature_part:
            self.has_intercept = False
            feature_part = re.sub(r'-\s*1', '', feature_part)
        
        # Split by + but respect parentheses
        terms = self._split_respecting_parens(feature_part, '+')
        
        base_features = set()
        
        for term in terms:
            term = term.strip()
            if not term:
                continue
            
            # Handle I() notation for arbitrary transformations
            if term.startswith('I(') and term.endswith(')'):
                self._parse_I_notation(term, base_features)
            # Handle poly() notation
            elif term.startswith('poly(') and term.endswith(')'):
                self._parse_poly_notation(term, base_features)
            # Handle interactions with *
            elif '*' in term:
                self._parse_interaction_star(term, base_features)
            # Handle interactions with :
            elif ':' in term:
                self._parse_interaction_colon(term, base_features)
            # Simple feature
            else:
                base_features.add(term)
        
        self.features = sorted(base_features)
    
    def _parse_I_notation(self, term: str, base_features: set):
        """Parse I() notation for transformations like I(x^2) or I(x*y)."""
        inner = term[2:-1].strip()
        
        # Handle powers: x^2, x^3, etc.
        if '^' in inner:
            match = re.match(r'(\w+)\s*\^\s*(\d+)', inner)
            if match:
                var_name = match.group(1)
                power = int(match.group(2))
                base_features.add(var_name)
                
                if power == 2:
                    self.transformations.append({
                        'type': 'quadratic',
                        'features': [var_name]
                    })
                elif power > 2:
                    self.transformations.append({
                        'type': 'polynomial',
                        'features': [var_name],
                        'degree': power
                    })
        # Handle products: x*y
        elif '*' in inner:
            vars_in_product = [v.strip() for v in inner.split('*')]
            for var in vars_in_product:
                base_features.add(var)
            
            if len(vars_in_product) == 2:
                self.transformations.append({
                    'type': 'interaction',
                    'feature_pairs': [vars_in_product]
                })
    
    def _parse_poly_notation(self, term: str, base_features: set):
        """Parse poly() notation like poly(x, 2) or poly(x, 3)."""
        inner = term[5:-1].strip()
        parts = [p.strip() for p in inner.split(',')]
        
        if len(parts) != 2:
            logger.warning(f"Invalid poly() notation: {term}")
            return
        
        var_name = parts[0]
        try:
            degree = int(parts[1])
        except ValueError:
            logger.warning(f"Invalid degree in poly(): {term}")
            return
        
        base_features.add(var_name)
        
        if degree > 1:
            self.transformations.append({
                'type': 'polynomial',
                'features': [var_name],
                'degree': degree
            })
    
    def _parse_interaction_star(self, term: str, base_features: set):
        """Parse interaction with * (includes main effects): x1*x2 = x1 + x2 + x1:x2."""
        vars_in_interaction = [v.strip() for v in term.split('*')]
        
        # Add all main effects
        for var in vars_in_interaction:
            base_features.add(var)
        
        # Add all pairwise interactions
        if len(vars_in_interaction) == 2:
            self.transformations.append({
                'type': 'interaction',
                'feature_pairs': [vars_in_interaction]
            })
        elif len(vars_in_interaction) > 2:
            pairs = list(combinations(vars_in_interaction, 2))
            self.transformations.append({
                'type': 'interaction',
                'feature_pairs': pairs
            })
    
    def _parse_interaction_colon(self, term: str, base_features: set):
        """Parse interaction with : (no main effects): x1:x2."""
        vars_in_interaction = [v.strip() for v in term.split(':')]
        
        # Add variables to base features (needed for computation)
        for var in vars_in_interaction:
            base_features.add(var)
        
        # Add interaction term only
        if len(vars_in_interaction) == 2:
            self.transformations.append({
                'type': 'interaction',
                'feature_pairs': [vars_in_interaction]
            })
    
    def _parse_term_list(self, term_list: str) -> List[str]:
        """Parse a simple list of terms separated by +."""
        return [t.strip() for t in term_list.split('+') if t.strip()]

    # Restore missing helper: split text by delimiter while respecting parentheses
    def _split_respecting_parens(self, text: str, delimiter: str) -> List[str]:
        """Split text by delimiter while respecting parentheses."""
        parts = []
        current = []
        paren_depth = 0

        for char in text:
            if char == '(':
                paren_depth += 1
                current.append(char)
            elif char == ')':
                paren_depth -= 1
                current.append(char)
            elif char == delimiter and paren_depth == 0:
                parts.append(''.join(current))
                current = []
            else:
                current.append(char)

        if current:
            parts.append(''.join(current))

        return parts

    # New helper: parse the compact demean spec syntax
    def _parse_demean_spec(self, spec: str) -> Tuple[List[List[str]], List[str]]:
        """
        Parse demeaning specification such as:
          "(tile_ix, tile_iy, pixel_id) + year, except = HDI_ME + HDI_HI"
        Returns:
          (groups, except_list) where groups is List[List[str]] and except_list is List[str]
        """
        # Split off 'except = ...' if present
        except_match = re.search(r',\s*except\s*=\s*(.+)$', spec, flags=re.IGNORECASE)
        if except_match:
            except_part = except_match.group(1).strip()
            before = spec[:except_match.start()].strip()
        else:
            except_part = ""
            before = spec.strip()

        # Parse except list (allow "+" or "," separators)
        except_list = []
        if except_part:
            except_list = [t.strip() for t in re.split(r'\s*\+\s*|\s*,\s*', except_part) if t.strip()]

        # Parse groups: split by '+' at top level respecting parentheses
        group_terms = self._split_respecting_parens(before, '+')
        groups: List[List[str]] = []
        for term in group_terms:
            term = term.strip()
            if not term:
                continue
            # Parenthesized tuple -> list of columns separated by commas
            if term.startswith('(') and term.endswith(')'):
                inner = term[1:-1].strip()
                cols = [c.strip() for c in re.split(r'\s*,\s*|\s*\+\s*', inner) if c.strip()]
                groups.append(cols)
            else:
                # Single-column group
                groups.append([term])

        return groups, except_list
    
    def get_feature_config(self) -> Dict[str, Any]:
        """
        Get feature engineering configuration from parsed formula.
        
        Returns:
        --------
        config : dict
            Configuration dictionary for FeatureTransformer
        """
        return {
            'transformations': self.transformations
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert parsed formula to dictionary representation."""
        return {
            'formula': self.formula,
            'target': self.target,
            'features': self.features,
            'instruments': self.instruments,
            'has_intercept': self.has_intercept,
            'transformations': self.transformations,
            # New fields
            'demean_groups': self.demean_groups,
            'demean_except': self.demean_except
        }
    
    @classmethod
    def parse(cls, formula: str) -> 'FormulaParser':
        """Convenience method to parse a formula."""
        return cls(formula)