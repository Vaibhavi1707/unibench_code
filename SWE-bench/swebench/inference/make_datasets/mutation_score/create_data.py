from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Any
from tqdm import tqdm

import json
import random
from typing import List, Dict, Optional

class MutationCategory(Enum):
    """High-level mutation categories for research taxonomy"""
    OPERATOR_BASED = "operator"      # Change operators
    STATEMENT_BASED = "statement"    # Modify/delete statements
    CONSTANT_BASED = "constant"      # Change literals/constants
    DECISION_BASED = "decision"      # Alter control flow
    VARIABLE_BASED = "variable"      # Modify variable usage

class MutationOperatorType(Enum):
    """Specific mutation operators from Table IV in RepoMasterEval"""
    # Operator-based
    AOR = "Arithmetic Operator Replacement"
    ROR = "Relational Operator Replacement"
    LOR = "Logical Operator Replacement"
    LCR = "Logical Connector Replacement"
    ASR = "Assignment Operator Replacement"
    BOR = "Bitwise Operator Replacement"

    # Statement-based
    SDL = "Statement Deletion"
    EHD = "Exception Handler Deletion"
    DDL = "Decorator Deletion"

    # Constant-based
    CRP = "Constant Replacement"

    # Decision-based
    COD = "Conditional Operator Deletion"
    COI = "Conditional Operator Insertion"
    BCR = "Break/Continue Replacement"
    OIL = "One Iteration Loop"

    # Variable-based
    BOD = "Binary Operator Deletion"

@dataclass
class MutationMetadata:
    """Metadata for scientific analysis"""
    operator_type: MutationOperatorType
    category: MutationCategory
    complexity: str  # 'syntactic', 'semantic', 'behavioral'
    detection_difficulty: str  # 'easy', 'medium', 'hard'
    equivalent_mutant_risk: str  # 'low', 'medium', 'high'

@dataclass
class GeneratedMutant:
    """Complete mutant representation for research"""
    mutant_id: int
    original_code: str
    mutated_code: str
    mutation_type: MutationOperatorType
    category: MutationCategory
    metadata: MutationMetadata
    location: Dict[str, Any]  # line, col, function_name, etc.
    diff: str  # Unified diff for visualization

import ast
import copy
import difflib
from typing import List, Tuple, Dict, Any, Optional

class CompleteMutationOperators:
    """All mutation operators from RepoMasterEval Table IV"""

    # ==================== OPERATOR METADATA ====================

    OPERATOR_METADATA = {
        MutationOperatorType.AOR: MutationMetadata(
            operator_type=MutationOperatorType.AOR,
            category=MutationCategory.OPERATOR_BASED,
            complexity='syntactic',
            detection_difficulty='easy',
            equivalent_mutant_risk='low'
        ),
        MutationOperatorType.ROR: MutationMetadata(
            operator_type=MutationOperatorType.ROR,
            category=MutationCategory.OPERATOR_BASED,
            complexity='semantic',
            detection_difficulty='medium',
            equivalent_mutant_risk='medium'
        ),
        MutationOperatorType.LOR: MutationMetadata(
            operator_type=MutationOperatorType.LOR,
            category=MutationCategory.OPERATOR_BASED,
            complexity='semantic',
            detection_difficulty='medium',
            equivalent_mutant_risk='low'
        ),
        MutationOperatorType.SDL: MutationMetadata(
            operator_type=MutationOperatorType.SDL,
            category=MutationCategory.STATEMENT_BASED,
            complexity='behavioral',
            detection_difficulty='easy',
            equivalent_mutant_risk='low'
        ),
        MutationOperatorType.CRP: MutationMetadata(
            operator_type=MutationOperatorType.CRP,
            category=MutationCategory.CONSTANT_BASED,
            complexity='semantic',
            detection_difficulty='medium',
            equivalent_mutant_risk='high'
        ),
        MutationOperatorType.COD: MutationMetadata(
            operator_type=MutationOperatorType.COD,
            category=MutationCategory.DECISION_BASED,
            complexity='behavioral',
            detection_difficulty='easy',
            equivalent_mutant_risk='low'
        ),
        MutationOperatorType.COI: MutationMetadata(
            operator_type=MutationOperatorType.COI,
            category=MutationCategory.DECISION_BASED,
            complexity='behavioral',
            detection_difficulty='medium',
            equivalent_mutant_risk='medium'
        ),
        MutationOperatorType.BCR: MutationMetadata(
            operator_type=MutationOperatorType.BCR,
            category=MutationCategory.DECISION_BASED,
            complexity='behavioral',
            detection_difficulty='medium',
            equivalent_mutant_risk='low'
        ),
        MutationOperatorType.EHD: MutationMetadata(
            operator_type=MutationOperatorType.EHD,
            category=MutationCategory.STATEMENT_BASED,
            complexity='behavioral',
            detection_difficulty='hard',
            equivalent_mutant_risk='low'
        ),
        MutationOperatorType.DDL: MutationMetadata(
            operator_type=MutationOperatorType.DDL,
            category=MutationCategory.STATEMENT_BASED,
            complexity='syntactic',
            detection_difficulty='medium',
            equivalent_mutant_risk='medium'
        ),
    }

# ==================== INDIVIDUAL OPERATORS ====================

class ArithmeticOperatorReplacement(ast.NodeTransformer):
    """AOR: + ↔ -, * ↔ /, etc."""

    REPLACEMENTS = {
        ast.Add: [ast.Sub, ast.Mult, ast.Div],
        ast.Sub: [ast.Add, ast.Mult, ast.Div],
        ast.Mult: [ast.Add, ast.Sub, ast.Div],
        ast.Div: [ast.Add, ast.Sub, ast.Mult],
        ast.Mod: [ast.Add, ast.Sub],
        ast.FloorDiv: [ast.Div, ast.Mult],
        ast.Pow: [ast.Mult, ast.Div]
    }

    def __init__(self):
        self.mutations = []

    def visit_BinOp(self, node):
        self.generic_visit(node)
        op_type = type(node.op)

        if op_type in self.REPLACEMENTS:
            for replacement in self.REPLACEMENTS[op_type]:
                self.mutations.append({
                    'type': MutationOperatorType.AOR,
                    'node': node,
                    'original_op': op_type,
                    'mutated_op': replacement,
                    'line': node.lineno,
                    'col': node.col_offset,
                })
        return node

class RelationalOperatorReplacement(ast.NodeTransformer):
    """ROR: < ↔ >, == ↔ !=, etc."""

    REPLACEMENTS = {
        ast.Lt: [ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq],
        ast.LtE: [ast.Lt, ast.Gt, ast.GtE, ast.Eq, ast.NotEq],
        ast.Gt: [ast.GtE, ast.Lt, ast.LtE, ast.Eq, ast.NotEq],
        ast.GtE: [ast.Gt, ast.Lt, ast.LtE, ast.Eq, ast.NotEq],
        ast.Eq: [ast.NotEq, ast.Lt, ast.Gt],
        ast.NotEq: [ast.Eq, ast.Lt, ast.Gt],
        ast.Is: [ast.IsNot, ast.Eq],
        ast.IsNot: [ast.Is, ast.NotEq],
        ast.In: [ast.NotIn],
        ast.NotIn: [ast.In]
    }

    def __init__(self):
        self.mutations = []

    def visit_Compare(self, node):
        self.generic_visit(node)

        for i, op in enumerate(node.ops):
            op_type = type(op)
            if op_type in self.REPLACEMENTS:
                for replacement in self.REPLACEMENTS[op_type]:
                    self.mutations.append({
                        'type': MutationOperatorType.ROR,
                        'node': node,
                        'original_op': op_type,
                        'mutated_op': replacement,
                        'position': i,
                        'line': node.lineno,
                        'col': node.col_offset,
                    })
        return node

class LogicalOperatorReplacement(ast.NodeTransformer):
    """LOR: and ↔ or"""

    def __init__(self):
        self.mutations = []

    def visit_BoolOp(self, node):
        self.generic_visit(node)

        if isinstance(node.op, ast.And):
            replacement = ast.Or
        elif isinstance(node.op, ast.Or):
            replacement = ast.And
        else:
            return node

        self.mutations.append({
            'type': MutationOperatorType.LOR,
            'node': node,
            'original_op': type(node.op),
            'mutated_op': replacement,
            'line': node.lineno,
            'col': node.col_offset,
        })
        return node

class LogicalConnectorReplacement(ast.NodeTransformer):
    """LCR: & ↔ |, ^ etc. (bitwise)"""

    REPLACEMENTS = {
        ast.BitAnd: [ast.BitOr, ast.BitXor],
        ast.BitOr: [ast.BitAnd, ast.BitXor],
        ast.BitXor: [ast.BitAnd, ast.BitOr],
    }

    def __init__(self):
        self.mutations = []

    def visit_BinOp(self, node):
        self.generic_visit(node)
        op_type = type(node.op)

        if op_type in self.REPLACEMENTS:
            for replacement in self.REPLACEMENTS[op_type]:
                self.mutations.append({
                    'type': MutationOperatorType.LCR,
                    'node': node,
                    'original_op': op_type,
                    'mutated_op': replacement,
                    'line': node.lineno,
                    'col': node.col_offset,
                })
        return node

class AssignmentOperatorReplacement(ast.NodeTransformer):
    """ASR: += ↔ -=, *= ↔ /=, etc."""

    REPLACEMENTS = {
        ast.Add: [ast.Sub, ast.Mult],
        ast.Sub: [ast.Add, ast.Mult],
        ast.Mult: [ast.Add, ast.Div],
        ast.Div: [ast.Mult, ast.Sub],
    }

    def __init__(self):
        self.mutations = []

    def visit_AugAssign(self, node):
        self.generic_visit(node)
        op_type = type(node.op)

        if op_type in self.REPLACEMENTS:
            for replacement in self.REPLACEMENTS[op_type]:
                self.mutations.append({
                    'type': MutationOperatorType.ASR,
                    'node': node,
                    'original_op': op_type,
                    'mutated_op': replacement,
                    'line': node.lineno,
                    'col': node.col_offset,
                })
        return node

class ConstantReplacement(ast.NodeTransformer):
    """CRP: 0→1, True→False, ""→"x", etc."""

    BOUNDARY_VALUES = {
        0: [1, -1],
        1: [0, 2, -1],
        -1: [0, 1],
        True: [False],
        False: [True],
        '': ['x', 'mutant'],
        None: [0, False, ''],
    }

    def __init__(self):
        self.mutations = []

    def visit_Constant(self, node):
        self.generic_visit(node)
        value = node.value

        # Determine replacements
        if value in self.BOUNDARY_VALUES:
            replacements = self.BOUNDARY_VALUES[value]
        elif isinstance(value, int) and value not in self.BOUNDARY_VALUES:
            replacements = [0, 1, -1, value + 1, value - 1]
        elif isinstance(value, float):
            replacements = [0.0, 1.0, value + 1.0, value - 1.0]
        elif isinstance(value, str) and value not in ['', None]:
            replacements = ['', 'x', value[:-1] if len(value) > 1 else '']
        else:
            replacements = []

        for replacement in replacements:
            if replacement != value:  # Don't create identity mutations
                self.mutations.append({
                    'type': MutationOperatorType.CRP,
                    'node': node,
                    'original_value': value,
                    'mutated_value': replacement,
                    'line': node.lineno,
                    'col': node.col_offset,
                })
        return node

class StatementDeletion(ast.NodeTransformer):
    """SDL: Delete individual statements"""

    def __init__(self):
        self.mutations = []
        self.current_function = None

    def visit_FunctionDef(self, node):
        old_func = self.current_function
        self.current_function = node.name
        self.generic_visit(node)

        # Delete each statement in function body
        for i, stmt in enumerate(node.body):
            # Skip docstrings
            if i == 0 and isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                continue
            # Skip final return (makes function invalid)
            if i == len(node.body) - 1 and isinstance(stmt, ast.Return):
                continue

            self.mutations.append({
                'type': MutationOperatorType.SDL,
                'parent_node': node,
                'deleted_stmt': stmt,
                'position': i,
                'line': stmt.lineno,
                'col': stmt.col_offset,
                'function_name': self.current_function,
            })

        self.current_function = old_func
        return node

class ConditionalOperatorDeletion(ast.NodeTransformer):
    """COD: if x: ... → if True: ..."""

    def __init__(self):
        self.mutations = []

    def visit_If(self, node):
        self.generic_visit(node)

        # Replace condition with True (always execute if-branch)
        self.mutations.append({
            'type': MutationOperatorType.COD,
            'node': node,
            'original_test': node.test,
            'mutated_test': ast.Constant(value=True),
            'line': node.lineno,
            'col': node.col_offset,
            'mutation_variant': 'force_true'
        })

        # Replace condition with False (always execute else-branch)
        self.mutations.append({
            'type': MutationOperatorType.COD,
            'node': node,
            'original_test': node.test,
            'mutated_test': ast.Constant(value=False),
            'line': node.lineno,
            'col': node.col_offset,
            'mutation_variant': 'force_false'
        })

        return node

class ConditionalOperatorInsertion(ast.NodeTransformer):
    """COI: x → not x in conditionals"""

    def __init__(self):
        self.mutations = []

    def visit_If(self, node):
        self.generic_visit(node)

        # Negate the condition
        self.mutations.append({
            'type': MutationOperatorType.COI,
            'node': node,
            'original_test': node.test,
            'mutated_test': ast.UnaryOp(op=ast.Not(), operand=node.test),
            'line': node.lineno,
            'col': node.col_offset,
        })

        return node

class BreakContinueReplacement(ast.NodeTransformer):
    """BCR: break ↔ continue"""

    def __init__(self):
        self.mutations = []

    def visit_Break(self, node):
        self.mutations.append({
            'type': MutationOperatorType.BCR,
            'node': node,
            'original': 'break',
            'mutated': 'continue',
            'line': node.lineno,
            'col': node.col_offset,
        })
        return node

    def visit_Continue(self, node):
        self.mutations.append({
            'type': MutationOperatorType.BCR,
            'node': node,
            'original': 'continue',
            'mutated': 'break',
            'line': node.lineno,
            'col': node.col_offset,
        })
        return node

# class ExceptionHandlerDeletion(ast.NodeTransformer):
#     """EHD: Remove exception handlers"""

#     def __init__(self):
#         self.mutations = []

#     def visit_Try(self, node):
#         self.generic_visit(node)

#         # Delete each exception handler
#         for i, handler in enumerate(node.handlers):
#             self.mutations.append({
#                 'type': MutationOperatorType.EHD,
#                 'node': node,
#                 'handler': handler,
#                 'position': i,
#                 'line': handler.lineno,
#                 'col': handler.col_offset,
#                 'exception_type': handler.type.id if handler.type else 'all'
#             })

#         return node

class ExceptionHandlerDeletion(ast.NodeTransformer):
    """EHD: Remove exception handlers"""
    
    def __init__(self):
        self.mutations = []
    
    def visit_Try(self, node):
        self.generic_visit(node)
        
        # Delete each exception handler
        for i, handler in enumerate(node.handlers):
            # Extract exception type name safely
            exception_type = self._get_exception_type_name(handler.type)
            
            self.mutations.append({
                'type': MutationOperatorType.EHD,
                'node': node,
                'handler': handler,
                'position': i,
                'line': handler.lineno,
                'col': handler.col_offset,
                'exception_type': exception_type
            })
        
        return node
    
    def _get_exception_type_name(self, type_node):
        """
        Safely extract exception type name from handler.type
        
        Handles:
        - Name: ValueError → "ValueError"
        - Attribute: module.Error → "module.Error"
        - Tuple: (ValueError, TypeError) → "(ValueError, TypeError)"
        - None: except: → "all"
        """
        if type_node is None:
            return 'all'
        elif isinstance(type_node, ast.Name):
            return type_node.id
        elif isinstance(type_node, ast.Attribute):
            # module.Error or obj.Error
            return ast.unparse(type_node)
        elif isinstance(type_node, ast.Tuple):
            # (ValueError, TypeError)
            return ast.unparse(type_node)
        else:
            # Fallback for any other node type
            try:
                return ast.unparse(type_node)
            except:
                return 'unknown'


class DecoratorDeletion(ast.NodeTransformer):
    """DDL: Remove decorators (@staticmethod, @property, etc.)"""

    def __init__(self):
        self.mutations = []

    def visit_FunctionDef(self, node):
        self.generic_visit(node)

        for i, decorator in enumerate(node.decorator_list):
            self.mutations.append({
                'type': MutationOperatorType.DDL,
                'node': node,
                'decorator': decorator,
                'position': i,
                'line': node.lineno,
                'col': node.col_offset,
                'function_name': node.name
            })

        return node

class BinaryOperatorDeletion(ast.NodeTransformer):
    """BOD: a + b → a or b"""

    def __init__(self):
        self.mutations = []

    def visit_BinOp(self, node):
        self.generic_visit(node)

        # Delete left operand (keep right)
        self.mutations.append({
            'type': MutationOperatorType.BOD,
            'node': node,
            'kept_operand': 'right',
            'line': node.lineno,
            'col': node.col_offset,
        })

        # Delete right operand (keep left)
        self.mutations.append({
            'type': MutationOperatorType.BOD,
            'node': node,
            'kept_operand': 'left',
            'line': node.lineno,
            'col': node.col_offset,
        })

        return node


class ResearchGradeMutationGenerator:
    """
    Modified mutation generator that samples one mutant per operator type
    """

    def __init__(
        self,
        source_code: str,
        operators: Optional[List[str]] = None,
        max_mutants_per_location: int = None,
        filter_equivalent: bool = False,
        one_per_operator: bool = False  # NEW parameter
    ):
        """
        Args:
            source_code: Complete source file as string
            operators: List of operator names (e.g., ['AOR', 'ROR'])
                      If None, uses all operators
            max_mutants_per_location: Limit mutations per code location
                      If None, generates all possible mutations
            filter_equivalent: Attempt to filter equivalent mutants (experimental)
            one_per_operator: If True, generate only 1 mutant per operator type
        """
        self.source_code = source_code
        self.original_ast = ast.parse(source_code)
        self.source_lines = source_code.split('\n')
        self.max_mutants_per_location = max_mutants_per_location
        self.filter_equivalent = filter_equivalent
        self.one_per_operator = one_per_operator  # Store the flag

        # Initialize operator map
        self.all_operators = {
            'AOR': ArithmeticOperatorReplacement,
            'ROR': RelationalOperatorReplacement,
            'LOR': LogicalOperatorReplacement,
            'LCR': LogicalConnectorReplacement,
            'ASR': AssignmentOperatorReplacement,
            'CRP': ConstantReplacement,
            'SDL': StatementDeletion,
            'COD': ConditionalOperatorDeletion,
            'COI': ConditionalOperatorInsertion,
            'BCR': BreakContinueReplacement,
            'EHD': ExceptionHandlerDeletion,
            'DDL': DecoratorDeletion,
            'BOD': BinaryOperatorDeletion,
        }

        # Select operators
        if operators is None:
            self.active_operators = self.all_operators
        else:
            self.active_operators = {
                k: v for k, v in self.all_operators.items()
                if k in operators
            }

    def generate_all_mutants(self) -> List[GeneratedMutant]:
        """Generate mutants (one per operator if one_per_operator=True)"""
        all_mutants = []
        mutant_id = 0

        for op_name, op_class in self.active_operators.items():
            # Collect all mutation points for this operator
            operator = op_class()
            tree_copy = copy.deepcopy(self.original_ast)
            operator.visit(tree_copy)
            mutations = operator.mutations

            # print(f"[{op_name}] Found {len(mutations)} mutation points")

            # NEW: If one_per_operator, randomly sample 1 mutation
            if self.one_per_operator and len(mutations) > 0:
                mutations = [random.choice(mutations)]
                # print(f"  → Randomly selected 1 mutation point")
            elif not self.one_per_operator and self.max_mutants_per_location:
                # Original sampling logic
                mutations = self._sample_mutations(mutations)

            # Generate actual mutants
            for mutation_info in mutations:
                try:
                    mutated_ast = self._apply_mutation(mutation_info)
                    mutated_code = ast.unparse(mutated_ast)

                    # Skip if equivalent (experimental)
                    if self.filter_equivalent and self._is_likely_equivalent(
                        self.source_code, mutated_code
                    ):
                        continue

                    # Create mutant with metadata
                    mutant = GeneratedMutant(
                        mutant_id=mutant_id,
                        original_code=self.source_code,
                        mutated_code=mutated_code,
                        mutation_type=mutation_info['type'],
                        category=CompleteMutationOperators.OPERATOR_METADATA[
                            mutation_info['type']
                        ].category,
                        metadata=CompleteMutationOperators.OPERATOR_METADATA[
                            mutation_info['type']
                        ],
                        location=self._extract_location(mutation_info),
                        diff=self._generate_diff(self.source_code, mutated_code)
                    )

                    all_mutants.append(mutant)
                    mutant_id += 1

                except Exception as e:
                    print(f"  ⚠ Skipping invalid mutant: {e}")
                    continue

        return all_mutants

    # ... (keep all other methods the same: _apply_mutation, _sample_mutations, etc.)
    # I'll include the key ones below for completeness

    def _apply_mutation(self, mutation_info: Dict) -> ast.Module:
        """Apply a single mutation to AST"""
        mutated_ast = copy.deepcopy(self.original_ast)

        mutation_type = mutation_info['type']

        # Handle different mutation types
        if mutation_type in [MutationOperatorType.AOR, MutationOperatorType.ROR,
                            MutationOperatorType.LOR, MutationOperatorType.LCR,
                            MutationOperatorType.ASR]:
            mutated_ast = self._replace_operator(mutated_ast, mutation_info)

        elif mutation_type == MutationOperatorType.CRP:
            mutated_ast = self._replace_constant(mutated_ast, mutation_info)

        elif mutation_type == MutationOperatorType.SDL:
            mutated_ast = self._delete_statement(mutated_ast, mutation_info)

        elif mutation_type in [MutationOperatorType.COD, MutationOperatorType.COI]:
            mutated_ast = self._modify_conditional(mutated_ast, mutation_info)

        elif mutation_type == MutationOperatorType.BCR:
            mutated_ast = self._replace_break_continue(mutated_ast, mutation_info)

        elif mutation_type == MutationOperatorType.EHD:
            mutated_ast = self._delete_exception_handler(mutated_ast, mutation_info)

        elif mutation_type == MutationOperatorType.DDL:
            mutated_ast = self._delete_decorator(mutated_ast, mutation_info)

        elif mutation_type == MutationOperatorType.BOD:
            mutated_ast = self._delete_binary_operand(mutated_ast, mutation_info)

        ast.fix_missing_locations(mutated_ast)
        return mutated_ast

    def _sample_mutations(self, mutations: List[Dict]) -> List[Dict]:
        """Sample mutations to control explosion"""
        # Group by location
        by_location = {}
        for m in mutations:
            key = (m['line'], m['col'])
            if key not in by_location:
                by_location[key] = []
            by_location[key].append(m)

        # Sample from each location
        sampled = []
        for location, location_mutations in by_location.items():
            if len(location_mutations) <= self.max_mutants_per_location:
                sampled.extend(location_mutations)
            else:
                sampled.extend(random.sample(
                    location_mutations,
                    self.max_mutants_per_location
                ))

        return sampled

    def _extract_location(self, mutation_info: Dict) -> Dict[str, Any]:
        """Extract detailed location information"""
        return {
            'line': mutation_info['line'],
            'col': mutation_info['col'],
            'function_name': mutation_info.get('function_name', 'module_level'),
            'code_context': self.source_lines[mutation_info['line']-1]
                           if mutation_info['line'] <= len(self.source_lines) else ''
        }

    def _generate_diff(self, original: str, mutated: str) -> str:
        """Generate unified diff"""
        diff = difflib.unified_diff(
            original.splitlines(keepends=True),
            mutated.splitlines(keepends=True),
            fromfile='original.py',
            tofile='mutated.py',
            lineterm=''
        )
        return ''.join(diff)

    def _is_likely_equivalent(self, original: str, mutated: str) -> bool:
        """Heuristic to detect equivalent mutants (experimental)"""
        return original.strip() == mutated.strip()

    # Keep all the _replace_* and _delete_* methods from before
    # (I'm omitting them here for brevity, but they stay the same)
    def _replace_operator(self, tree: ast.Module, mutation_info: Dict) -> ast.Module:
        """Replace operator mutations (AOR, ROR, LOR, etc.)"""
        class OperatorReplacer(ast.NodeTransformer):
            def __init__(self, target_line, target_col, new_op, mut_info):
                self.target_line = target_line
                self.target_col = target_col
                self.new_op = new_op
                self.mut_info = mut_info
                self.replaced = False

            def visit_BinOp(self, node):
                if (node.lineno == self.target_line and
                    node.col_offset == self.target_col and
                    not self.replaced):
                    node.op = self.new_op()
                    self.replaced = True
                return self.generic_visit(node)

            def visit_Compare(self, node):
                if (node.lineno == self.target_line and
                    node.col_offset == self.target_col and
                    not self.replaced):
                    if 'position' in self.mut_info:
                        node.ops[self.mut_info['position']] = self.new_op()
                    self.replaced = True
                return self.generic_visit(node)

            def visit_BoolOp(self, node):
                if (node.lineno == self.target_line and
                    node.col_offset == self.target_col and
                    not self.replaced):
                    node.op = self.new_op()
                    self.replaced = True
                return self.generic_visit(node)

            def visit_AugAssign(self, node):
                if (node.lineno == self.target_line and
                    node.col_offset == self.target_col and
                    not self.replaced):
                    node.op = self.new_op()
                    self.replaced = True
                return self.generic_visit(node)

        replacer = OperatorReplacer(
            mutation_info['line'],
            mutation_info['col'],
            mutation_info['mutated_op'],
            mutation_info
        )
        return replacer.visit(tree)

    def _replace_constant(self, tree: ast.Module, mutation_info: Dict) -> ast.Module:
        """Replace constant values"""
        class ConstantReplacer(ast.NodeTransformer):
            def __init__(self, target_line, target_col, new_value):
                self.target_line = target_line
                self.target_col = target_col
                self.new_value = new_value
                self.replaced = False

            def visit_Constant(self, node):
                if (node.lineno == self.target_line and
                    node.col_offset == self.target_col and
                    not self.replaced):
                    node.value = self.new_value
                    self.replaced = True
                return node

        replacer = ConstantReplacer(
            mutation_info['line'],
            mutation_info['col'],
            mutation_info['mutated_value']
        )
        return replacer.visit(tree)

    def _delete_statement(self, tree: ast.Module, mutation_info: Dict) -> ast.Module:
        """Delete a statement"""
        class StatementDeleter(ast.NodeTransformer):
            def __init__(self, parent_line, stmt_position):
                self.parent_line = parent_line
                self.stmt_position = stmt_position
                self.deleted = False

            def visit_FunctionDef(self, node):
                if node.lineno == self.parent_line and not self.deleted:
                    if self.stmt_position < len(node.body):
                        del node.body[self.stmt_position]
                        if not node.body:
                            node.body = [ast.Pass()]
                        self.deleted = True
                return self.generic_visit(node)

        deleter = StatementDeleter(
            mutation_info['parent_node'].lineno,
            mutation_info['position']
        )
        return deleter.visit(tree)

    def _modify_conditional(self, tree: ast.Module, mutation_info: Dict) -> ast.Module:
        """Modify conditional (COD, COI)"""
        class ConditionalModifier(ast.NodeTransformer):
            def __init__(self, target_line, new_test):
                self.target_line = target_line
                self.new_test = new_test
                self.modified = False

            def visit_If(self, node):
                if node.lineno == self.target_line and not self.modified:
                    node.test = copy.deepcopy(self.new_test)
                    self.modified = True
                return self.generic_visit(node)

        modifier = ConditionalModifier(
            mutation_info['line'],
            mutation_info['mutated_test']
        )
        return modifier.visit(tree)

    def _replace_break_continue(self, tree: ast.Module,
                                mutation_info: Dict) -> ast.Module:
        """Replace break with continue or vice versa"""
        class BreakContinueReplacer(ast.NodeTransformer):
            def __init__(self, target_line, original):
                self.target_line = target_line
                self.original = original
                self.replaced = False

            def visit_Break(self, node):
                if node.lineno == self.target_line and not self.replaced:
                    if self.original == 'break':
                        self.replaced = True
                        return ast.Continue()
                return node

            def visit_Continue(self, node):
                if node.lineno == self.target_line and not self.replaced:
                    if self.original == 'continue':
                        self.replaced = True
                        return ast.Break()
                return node

        replacer = BreakContinueReplacer(
            mutation_info['line'],
            mutation_info['original']
        )
        return replacer.visit(tree)

    def _delete_exception_handler(self, tree: ast.Module,
                                  mutation_info: Dict) -> ast.Module:
        """Delete exception handler"""
        class HandlerDeleter(ast.NodeTransformer):
            def __init__(self, target_line, handler_position):
                self.target_line = target_line
                self.handler_position = handler_position
                self.deleted = False

            def visit_Try(self, node):
                if node.lineno == self.target_line and not self.deleted:
                    if self.handler_position < len(node.handlers):
                        del node.handlers[self.handler_position]
                        self.deleted = True
                return self.generic_visit(node)

        deleter = HandlerDeleter(
            mutation_info['line'],
            mutation_info['position']
        )
        return deleter.visit(tree)

    def _delete_decorator(self, tree: ast.Module, mutation_info: Dict) -> ast.Module:
        """Delete decorator"""
        class DecoratorDeleter(ast.NodeTransformer):
            def __init__(self, target_line, decorator_position):
                self.target_line = target_line
                self.decorator_position = decorator_position
                self.deleted = False

            def visit_FunctionDef(self, node):
                if node.lineno == self.target_line and not self.deleted:
                    if self.decorator_position < len(node.decorator_list):
                        del node.decorator_list[self.decorator_position]
                        self.deleted = True
                return self.generic_visit(node)

        deleter = DecoratorDeleter(
            mutation_info['line'],
            mutation_info['position']
        )
        return deleter.visit(tree)


    def _delete_binary_operand(self, tree: ast.Module, 
                           mutation_info: Dict) -> ast.Module:
        """Delete one operand of binary operation"""
        class OperandDeleter(ast.NodeTransformer):
            def __init__(self, target_line, target_col, kept_operand):
                self.target_line = target_line
                self.target_col = target_col
                self.kept_operand = kept_operand
                self.deleted = False
            
            def visit_BinOp(self, node):
                # Visit children first
                node = self.generic_visit(node)
                
                # Check if this is our target
                if (hasattr(node, 'lineno') and 
                    node.lineno == self.target_line and 
                    node.col_offset == self.target_col and 
                    not self.deleted):
                    
                    self.deleted = True
                    
                    # Return the appropriate operand
                    if self.kept_operand == 'left':
                        result = copy.deepcopy(node.left)
                    else:
                        result = copy.deepcopy(node.right)
                    
                    # CRITICAL: Copy location info
                    if hasattr(node, 'lineno'):
                        ast.copy_location(result, node)
                    
                    return result
                
                return node
        
        deleter = OperandDeleter(
            mutation_info['line'],
            mutation_info['col'],
            mutation_info['kept_operand']
        )
        mutated_tree = deleter.visit(tree)
        
        # CRITICAL: Fix missing locations
        ast.fix_missing_locations(mutated_tree)
        
        return mutated_tree

import re
def generate_mutants_from_json(json_file_path: str):
    """
    Generate ONE mutant per operator type from JSON file

    Expected JSON format:
    {
        "file_id": "src/utils.py",
        "content": "def foo():\n    return 42",
        "metadata": {...}
    }
    """
    #open json file
    with open(json_file_path, 'r') as f:
        json_file = json.load(f)




    all_results = []
    invalid_syntax_files = 0
    num_files = 0
    num_mutants = 0
    print(invalid_syntax_files)

    for item in tqdm(json_file, desc="Processing files"):
        num_files += 1
        file_id = item["instance_id"]
        source_code = item["fixed_code"]

        # pattern_1 = r'^\[[^]]*start of[^]]*]\s*'
        # source_code = re.sub(pattern_1, '', source_code)
        # pattern_2 = r'\s*\[[^]]*end of[^]]*]$'
        # source_code = re.sub(pattern_2, '', source_code)

        pattern = r'^\[[^]]*start of[^]]*]\s*|\s*\[[^]]*end of[^]]*]$'
        source_code = re.sub(pattern, '', source_code)

        print(f"\n{'='*60}")
        print(f"Processing: {file_id}")
        print(f"{'='*60}")
        # print(source_code)

        try:
            ast.parse(source_code)
        except SyntaxError as e:
            invalid_syntax_files += 1
            print(f"  ⚠ Skipping {file_id}: cannot parse fixed_code")
            print(f"     {type(e).__name__}: {e}")
            continue

        # Generate mutants with ONE PER OPERATOR flag enabled
        generator = ResearchGradeMutationGenerator(
            source_code=source_code,
            # operators=['AOR', 'ROR', 'LOR', 'SDL', 'CRP', 'COD'],
            operators=[
                    'AOR', 'ROR', 'LOR', 'LCR', 'ASR',  # Operator-based
                    'SDL', 'EHD', 'DDL',                # Statement-based
                    'CRP',                              # Constant-based
                    'COD', 'COI', 'BCR', 'BOD'          # Decision + Variable
                ],
            one_per_operator=True,  # ← NEW: Only generate 1 mutant per operator
            filter_equivalent=False
        )

        mutants = generator.generate_all_mutants()


        new_item = {}
        # Collect statistics by category
        stats = {
            'file_id': file_id,
            'total_mutants': len(mutants),
            'by_operator': {},
            'by_category': {},
            'by_difficulty': {},
            'mutants': []
        }
        print(f"Invalid syntax files: {invalid_syntax_files}")
        for mutant in mutants:
            # Aggregate statistics
            op_name = mutant.mutation_type.name
            cat_name = mutant.category.value
            diff = mutant.metadata.detection_difficulty
            stats['by_operator'][op_name] = stats['by_operator'].get(op_name, 0) + 1
            stats['by_category'][cat_name] = stats['by_category'].get(cat_name, 0) + 1
            stats['by_difficulty'][diff] = stats['by_difficulty'].get(diff, 0) + 1

            # Store mutant data
            stats['mutants'].append({
                'mutant_id': mutant.mutant_id,
                'mutation_type': op_name,
                'category': cat_name,
                'complexity': mutant.metadata.complexity,
                'difficulty': diff,
                'equivalent_risk': mutant.metadata.equivalent_mutant_risk,
                'location': mutant.location,
                'mutated_code': mutant.mutated_code,
                'diff': mutant.diff
            })



            new_item['instance_id'] =  file_id
            new_item['mutation_type'] = op_name
            new_item['mutation_category'] = cat_name
            new_item['mutation_complexity'] = mutant.metadata.complexity
            new_item['mutation_difficulty'] = diff
            new_item['mutation_equivalent_risk'] = mutant.metadata.equivalent_mutant_risk
            new_item['mutation_location'] = mutant.location
            new_item['mutated_code'] = mutant.mutated_code
            new_item['mutation_diff'] = mutant.diff

            
            num_mutants += 1
            print(f"Processed files: {num_files}")
            print(f"Total mutants generated: {num_mutants}")


            
            all_results.append(new_item)

    print(f"\n{'='*60}")
    print(f"FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Total Mutants Generated: {len(mutants)}")
    print(f"By Operator: {stats['by_operator']}")
    print(f"By Category: {stats['by_category']}")
    print(f"By Difficulty: {stats['by_difficulty']}")

    # Show which mutant was selected for each operator
    print(f"\n{'='*60}")
    print(f"SELECTED MUTANTS (1 per operator)")
    print(f"{'='*60}")
    for i, mutant_data in enumerate(stats['mutants']):
        print(f"\n[{i+1}] {mutant_data['mutation_type']}:")
        print(f"    Location: Line {mutant_data['location']['line']}, Col {mutant_data['location']['col']}")
        print(f"    Context: {mutant_data['location']['code_context'][:60]}...")
        print(f"    Difficulty: {mutant_data['difficulty']}")

    # print(all_results)

    with open(f'mutation_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nMutation results saved to mutation_results.json")
    print(f"Invalid syntax files: {invalid_syntax_files}")
    print(f"Processed files: {num_files}")
    print(f"Total mutants generated: {len(all_results)}")
    return all_results



# Run it
results = generate_mutants_from_json(
    '/scratch/zt1/project/cmsc848n/shared/hsoora/SWE-bench/swebench/inference/make_datasets/null_resolution/null_data.json')
