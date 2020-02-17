import lark
import torch

from convlogic.parser import LogicParser

class Functions:

    @staticmethod
    def not_(x):
        return -x

    @staticmethod
    def and_(a, b):
        return torch.min(a, b)

    @staticmethod
    def or_(a, b):
        return torch.max(a, b)

    @staticmethod
    def finally_(f):
        return torch.max(f)

    @staticmethod
    def globally_(f):
        return torch.min(f)


@lark.v_args(inline=True)
class _CodeBuilder(lark.Transformer):

    def atom(self, *args):
        operand_a, operator, operand_b = args

        if operator == '>=':
            return f'{operand_a} - {operand_b}'
        elif operator == '>':
            raise NotImplementedError

        elif operator == '<=':
            return f'{operand_b} - {operand_a}'
        elif operator == '<':
            raise NotImplementedError

        elif operator == '==' or '!=':
            raise NotImplementedError

    def op_not(self, preposition):
        return 'fn.not_(' + preposition + ')'

    def op_and(self, preposition_a, preposition_b):
        args = [preposition_a, preposition_b]
        return 'fn.and_(' + ', '.join(args) + ')'

    def op_or(self, preposition_a, preposition_b):
        args = [preposition_a, preposition_b]
        return 'fn.or_(' + ', '.join(args) + ')'

    def ltl_op(self, *parameters):
        return list(map(lambda x: str(x.children[0]), parameters))

    def operator(self, params, preposition):
        if len(params) > 1:
            raise NotImplementedError
        else:
            letter = params[0]
            operator_args = [preposition]

        if letter == 'F':
            function = 'fn.finally_'
        elif letter == 'G':
            function = 'fn.globally_'

        return function + '(' + ', '.join(operator_args) + ')'

    def start(self, preposition):
        return str(preposition)


class DiffQuantitativeSemantic:

    def __init__(self, logic_formula):
        """Get the parse-tree and call the method _build on it"""
        if isinstance(logic_formula, str):
            self.logic_parser = LogicParser(logic_formula)
        else:
            self.logic_parser = logic_formula

        self._code = self._build()

    def _build(self):
        """Compute the internal representation for the semantic"""
        tree = self.logic_parser.parse_tree
        code = _CodeBuilder().transform(tree)
        return code

    def compute(self, t, **signals):
        environment = {
            'fn': Functions,
            't_': t,
        }
        environment.update(signals)

        return eval(self._code, environment)

    def __str__(self):
        return self._code
