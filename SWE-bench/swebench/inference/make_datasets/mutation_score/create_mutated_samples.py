import json
import random
from typing import List, Dict, Optional

import ast
import copy
from typing import List, Tuple
from dataclasses import dataclass

@dataclass
class Mutant:
    """Represents a single mutated version of code"""
    mutant_id: int
    original_code: str
    mutated_code: str
    mutation_type: str
    location: Tuple[int, int]  # (line, col)
    description: str

class MutationOperator(ast.NodeTransformer):
    """Base class for all mutation operators"""
    def __init__(self):
        self.mutations = []
        self.mutant_counter = 0

    def get_mutations(self) -> List[Mutant]:
        return self.mutations


class ArithmeticOperatorReplacement(MutationOperator):
    """AOR: Replace +, -, *, /, %, //, ** with each other"""

    OPERATORS = {
        ast.Add: [ast.Sub, ast.Mult, ast.Div],
        ast.Sub: [ast.Add, ast.Mult, ast.Div],
        ast.Mult: [ast.Add, ast.Sub, ast.Div],
        ast.Div: [ast.Add, ast.Sub, ast.Mult],
        ast.Mod: [ast.Add, ast.Sub],
        ast.FloorDiv: [ast.Div, ast.Mult],
        ast.Pow: [ast.Mult, ast.Div]
    }

    def visit_BinOp(self, node):
        """Visit binary operations like a + b"""
        self.generic_visit(node)  # Continue traversal

        op_type = type(node.op)
        if op_type in self.OPERATORS:
            for replacement_op in self.OPERATORS[op_type]:
                # Create mutant by replacing operator
                mutated_node = copy.deepcopy(node)
                mutated_node.op = replacement_op()

                self.mutations.append({
                    'type': 'AOR',
                    'line': node.lineno,
                    'col': node.col_offset,
                    'original_op': op_type.__name__,
                    'mutated_op': replacement_op.__name__,
                    'mutated_ast': mutated_node
                })

        return node

class RelationalOperatorReplacement(MutationOperator):
    """ROR: Replace <, <=, >, >=, ==, != with each other"""

    OPERATORS = {
        ast.Lt: [ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq],
        ast.LtE: [ast.Lt, ast.Gt, ast.GtE, ast.Eq, ast.NotEq],
        ast.Gt: [ast.GtE, ast.Lt, ast.LtE, ast.Eq, ast.NotEq],
        ast.GtE: [ast.Gt, ast.Lt, ast.LtE, ast.Eq, ast.NotEq],
        ast.Eq: [ast.NotEq, ast.Lt, ast.Gt],
        ast.NotEq: [ast.Eq, ast.Lt, ast.Gt]
    }

    def visit_Compare(self, node):
        self.generic_visit(node)

        for i, op in enumerate(node.ops):
            op_type = type(op)
            if op_type in self.OPERATORS:
                for replacement_op in self.OPERATORS[op_type]:
                    mutated_node = copy.deepcopy(node)
                    mutated_node.ops[i] = replacement_op()

                    self.mutations.append({
                        'type': 'ROR',
                        'line': node.lineno,
                        'col': node.col_offset,
                        'original_op': op_type.__name__,
                        'mutated_op': replacement_op.__name__,
                        'mutated_ast': mutated_node
                    })

        return node

class LogicalOperatorReplacement(MutationOperator):
    """LOR: Replace 'and' with 'or' and vice versa"""

    def visit_BoolOp(self, node):
        self.generic_visit(node)

        if isinstance(node.op, ast.And):
            mutated_node = copy.deepcopy(node)
            mutated_node.op = ast.Or()
            self.mutations.append({
                'type': 'LOR',
                'line': node.lineno,
                'col': node.col_offset,
                'original_op': 'And',
                'mutated_op': 'Or',
                'mutated_ast': mutated_node
            })
        elif isinstance(node.op, ast.Or):
            mutated_node = copy.deepcopy(node)
            mutated_node.op = ast.And()
            self.mutations.append({
                'type': 'LOR',
                'line': node.lineno,
                'col': node.col_offset,
                'original_op': 'Or',
                'mutated_op': 'And',
                'mutated_ast': mutated_node
            })

        return node

class ConstantReplacement(MutationOperator):
    """CRP: Replace constants with boundary values"""

    REPLACEMENTS = {
        0: [1, -1],
        1: [0, 2, -1],
        -1: [0, 1],
        True: [False],
        False: [True],
        '': ['mutant'],
        None: [0, False]
    }

    def visit_Constant(self, node):
        self.generic_visit(node)

        value = node.value
        # Handle numeric constants
        if isinstance(value, (int, float)) and value not in self.REPLACEMENTS:
            replacements = [value + 1, value - 1, 0, 1, -1]
        elif value in self.REPLACEMENTS:
            replacements = self.REPLACEMENTS[value]
        else:
            replacements = []

        for replacement in replacements:
            mutated_node = copy.deepcopy(node)
            mutated_node.value = replacement

            self.mutations.append({
                'type': 'CRP',
                'line': node.lineno,
                'col': node.col_offset,
                'original_value': value,
                'mutated_value': replacement,
                'mutated_ast': mutated_node
            })

        return node

class StatementDeletion(MutationOperator):
    """SDL: Delete statements (most impactful operator per paper)"""

    def visit_FunctionDef(self, node):
        self.generic_visit(node)

        # Try deleting each statement in function body
        for i, stmt in enumerate(node.body):
            # Don't delete docstrings or last return
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                continue
            if i == len(node.body) - 1 and isinstance(stmt, ast.Return):
                continue

            mutated_node = copy.deepcopy(node)
            del mutated_node.body[i]

            # If body becomes empty, add 'pass'
            if not mutated_node.body:
                mutated_node.body = [ast.Pass()]

            self.mutations.append({
                'type': 'SDL',
                'line': stmt.lineno,
                'col': stmt.col_offset,
                'deleted_stmt': ast.unparse(stmt),
                'mutated_ast': mutated_node
            })

        return node


class MutationGenerator:
    """Orchestrates mutation generation"""

    def __init__(self, source_code: str):
        self.source_code = source_code
        self.original_ast = ast.parse(source_code)
        self.operators = [
            ArithmeticOperatorReplacement(),
            RelationalOperatorReplacement(),
            LogicalOperatorReplacement(),
            ConstantReplacement(),
            StatementDeletion()
        ]

    def generate_mutants(self) -> List[Mutant]:
        """Generate all possible mutants"""
        all_mutants = []
        mutant_id = 0

        for operator in self.operators:
            # Each operator traverses AST and collects mutations
            tree_copy = copy.deepcopy(self.original_ast)
            operator.visit(tree_copy)

            # Convert each mutation to executable code
            for mutation_info in operator.get_mutations():
                try:
                    # Create new AST with this single mutation
                    mutated_tree = self._apply_mutation(
                        self.original_ast,
                        mutation_info
                    )

                    # Convert back to source code
                    mutated_code = ast.unparse(mutated_tree)

                    mutant = Mutant(
                        mutant_id=mutant_id,
                        original_code=self.source_code,
                        mutated_code=mutated_code,
                        mutation_type=mutation_info['type'],
                        location=(mutation_info['line'], mutation_info['col']),
                        description=self._describe_mutation(mutation_info)
                    )
                    all_mutants.append(mutant)
                    mutant_id += 1

                except Exception as e:
                    # Skip invalid mutants
                    continue

        return all_mutants

    def _apply_mutation(self, original_ast, mutation_info):
        """Apply a single mutation to the AST"""
        # This is simplified - production version needs precise node replacement
        class MutationApplier(ast.NodeTransformer):
            def __init__(self, target_line, target_col, new_node):
                self.target_line = target_line
                self.target_col = target_col
                self.new_node = new_node
                self.applied = False

            def visit(self, node):
                if (hasattr(node, 'lineno') and
                    node.lineno == self.target_line and
                    node.col_offset == self.target_col and
                    not self.applied):
                    self.applied = True
                    return self.new_node
                return self.generic_visit(node)

        mutated_tree = copy.deepcopy(original_ast)
        applier = MutationApplier(
            mutation_info['line'],
            mutation_info['col'],
            mutation_info['mutated_ast']
        )
        return applier.visit(mutated_tree)

    def _describe_mutation(self, mutation_info):
        """Human-readable description"""
        if mutation_info['type'] == 'AOR':
            return f"Replace {mutation_info['original_op']} with {mutation_info['mutated_op']}"
        elif mutation_info['type'] == 'ROR':
            return f"Replace {mutation_info['original_op']} with {mutation_info['mutated_op']}"
        elif mutation_info['type'] == 'CRP':
            return f"Replace constant {mutation_info['original_value']} with {mutation_info['mutated_value']}"
        elif mutation_info['type'] == 'SDL':
            return f"Delete statement: {mutation_info['deleted_stmt']}"
        return mutation_info['type']



import subprocess
import tempfile
import os

class MutationTester:
    """Executes tests against mutants and computes score"""

    def __init__(self, test_command: str = "pytest"):
        self.test_command = test_command

    def run_mutation_testing(self, mutants: List[Mutant],
                            original_file_path: str) -> dict:
        """
        Run tests against all mutants

        Args:
            mutants: List of generated mutants
            original_file_path: Path to original source file

        Returns:
            dict with mutation score and details
        """
        results = {
            'total_mutants': len(mutants),
            'killed_mutants': 0,
            'survived_mutants': 0,
            'timeout_mutants': 0,
            'mutant_details': []
        }

        # Backup original file
        with open(original_file_path, 'r') as f:
            original_content = f.read()

        for mutant in mutants:
            # Write mutated code to file
            with open(original_file_path, 'w') as f:
                f.write(mutant.mutated_code)

            # Run tests
            status = self._run_tests()

            mutant_result = {
                'mutant_id': mutant.mutant_id,
                'mutation_type': mutant.mutation_type,
                'location': mutant.location,
                'description': mutant.description,
                'status': status
            }

            if status == 'killed':
                results['killed_mutants'] += 1
            elif status == 'survived':
                results['survived_mutants'] += 1
            else:
                results['timeout_mutants'] += 1

            results['mutant_details'].append(mutant_result)

        # Restore original file
        with open(original_file_path, 'w') as f:
            f.write(original_content)

        # Calculate mutation score
        testable_mutants = results['total_mutants'] - results['timeout_mutants']
        if testable_mutants > 0:
            results['mutation_score'] = (
                results['killed_mutants'] / testable_mutants
            ) * 100
        else:
            results['mutation_score'] = 0.0

        return results

    def _run_tests(self, timeout: int = 30) -> str:
        """
        Run test suite and determine if mutant is killed

        Returns:
            'killed' if tests fail (mutant detected)
            'survived' if tests pass (mutant not detected)
            'timeout' if tests hang
        """
        try:
            result = subprocess.run(
                self.test_command.split(),
                capture_output=True,
                timeout=timeout,
                text=True
            )

            # If tests fail, mutant is killed
            if result.returncode != 0:
                return 'killed'
            else:
                return 'survived'

        except subprocess.TimeoutExpired:
            return 'timeout'
        except Exception as e:
            # Syntax errors kill the mutant
            return 'killed'


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

            print(f"[{op_name}] Found {len(mutations)} mutation points")

            # NEW: If one_per_operator, randomly sample 1 mutation
            if self.one_per_operator and len(mutations) > 0:
                mutations = [random.choice(mutations)]
                print(f"  → Randomly selected 1 mutation point")
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

    for item in json_file:
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

    print(all_results)

    with open(f'mutation_data_lite.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    return all_results



# Run it
results = generate_mutants_from_json(
    '/scratch/zt1/project/cmsc848n/shared/hsoora/SWE-bench/swebench/inference/make_datasets/null_resolution/gt_fixed_patches_lite.json')
