from itertools import product
import time
import sys

TESTDATA_FILE = "testdata.txt"

all_digits = [str(x) for x in range(0, 10)]
all_hex = [str(x) for x in range(0, 10)] + ['A', 'B', 'C', 'D', 'E', 'F']
binary_ops = ['+', '-', '*', '//', '**', '<<', '>>', '/', '^', '&', '|']
unary_ops = ['~', '-']
brackets = ['(', ')']


class Expression:
    __slots__ = ('result', 'repr', 'len', 'precedence')
    result: int
    repr: str
    len: int
    precedence: int

    def __str__(self):
        return self.repr

    def __repr__(self):
        return self.repr

    def __eq__(self, other):
        return self.result == other.result

    def __hash__(self):
        return hash(self.result)

    @classmethod
    def is_valid(cls, left, right):
        return True


class BinaryOp(Expression):
    op: str = ""

    def __init__(self, left, right):
        self.repr = f"{left}{self.op}{right}"
        self.result = self._compute(left.result, right.result)
        self.len = len(self.repr)

    @classmethod
    def create(cls, left, right):
        if not cls.is_valid(left, right):
            return None
        curr = cls.precedence
        l = left.precedence
        r = right.precedence
        if cls == Pow:
            # Right-associative: parens if left has same OR lower precedence
            if l <= curr or left.repr[0] in '-~':
                left = Parenthesized.create(left)
            if r < curr:
                right = Parenthesized.create(right)
        else:
            # Left-associative: parens if right has same OR lower precedence
            if l < curr:
                left = Parenthesized.create(left)
            if r <= curr:
                right = Parenthesized.create(right)
        return cls(left, right)

    def _compute(self, left, right):
        raise NotImplementedError()


class UnaryExpression(Expression):
    op: str = ""
    precedence: int = 100

    def __init__(self, expression):
        self.repr = self.op + str(expression)
        self.result = self._compute(expression.result)
        self.len = len(self.repr)

    def inv(self):
        return Inv.create(self)

    def negate(self):
        return Negative.create(self)

    @classmethod
    def create(cls, expression):
        if isinstance(expression.result, float):
            return None
        if isinstance(expression, UnaryExpression) or isinstance(expression, Number):
            return cls(expression)
        return cls(Parenthesized.create(expression))


# https://www.w3schools.com/python/python_operators_precedence.asp
class Parenthesized(Expression):
    precedence = 100

    def __init__(self, inner):
        self.repr = "(" + str(inner) + ")"
        self.result = inner.result
        self.len = len(self.repr)

    def create(inner):
        return Parenthesized(inner) if inner is not None else None


class Negative(UnaryExpression):
    op = "-"

    def _compute(self, value):
        return -value


class Inv(UnaryExpression):
    op = "~"

    def _compute(self, value):
        return ~value


class Number(UnaryExpression):
    precedence = 100

    def __init__(self, num):
        self.result = num
        self.len = len(str(num))
        self.repr = str(num)

    @classmethod
    def hex(cls, string):
        r = cls(-1)
        r.result = int(string, 16)
        r.len = len(string)
        r.repr = string
        return r

    @classmethod
    def true(cls):
        r = cls(-1)
        r.result = 1
        r.len = 4
        r.repr = 'True'
        return r

    @classmethod
    def totally_sane_number(cls):
        r = cls(-1)
        r.result = 1
        r.len = 5
        r.repr = 'not[]'
        return r

    @classmethod
    def even_more_sane_number(cls):
        r = cls(-1)
        r.result = 1
        r.len = 6
        r.repr = '()==()'
        return r


class Pow(BinaryOp):
    precedence = 90
    op = "**"

    @classmethod
    def is_valid(cls, left, right):
        l = left.result
        r = right.result
        return isinstance(l, int) and isinstance(r, int) and l > 0 and r > 0 and l < 100 and r < 100

    def _compute(self, left, right): return left ** right


class Multiplication(BinaryOp):
    precedence = 85
    op = "*"

    def _compute(self, left, right): return left * right


class IntDiv(BinaryOp):
    precedence = 85
    op = "//"

    @classmethod
    def is_valid(cls, left, right):
        l = left.result
        r = right.result
        return l > 0 and r > 0

    def _compute(self, left, right): return left // right


class Addition(BinaryOp):
    precedence = 80
    op = "+"

    def _compute(self, left, right): return left + right


class Subtraction(BinaryOp):
    precedence = 80
    op = "-"

    def _compute(self, left, right): return left - right


class Shl(BinaryOp):
    precedence = 75
    op = "<<"

    @classmethod
    def is_valid(cls, left, right):
        return isinstance(right.result, int) and isinstance(left.result, int) and 0 < right.result < 16

    def _compute(self, left, right): return left << right


class Shr(BinaryOp):
    precedence = 75
    op = ">>"

    @classmethod
    def is_valid(cls, left, right):
        return isinstance(right.result, int) and isinstance(left.result, int) and 0 < right.result < 16

    def _compute(self, left, right): return left >> right


class And(BinaryOp):
    precedence = 70
    op = "&"

    @classmethod
    def is_valid(cls, left, right):
        return isinstance(right.result, int) and isinstance(left.result, int)

    def _compute(self, left, right): return left & right


class Xor(BinaryOp):
    precedence = 65
    op = "^"

    @classmethod
    def is_valid(cls, left, right):
        return isinstance(right.result, int) and isinstance(left.result, int)

    def _compute(self, left, right): return left ^ right


class Or(BinaryOp):
    precedence = 60
    op = "|"

    @classmethod
    def is_valid(cls, left, right):
        return isinstance(right.result, int) and isinstance(left.result, int)

    def _compute(self, left, right): return left | right


def generate_numbers(allowed_digits, dot_allowed, max_len):
    result = {}
    max_num_len = max_len - 2  # max((max_len - 1) // 2, 2)
    for len in range(1, max_len):
        for digits in product(allowed_digits, repeat=len):
            if digits[0] == '0':
                continue
            num = Number(int(''.join(digits)))
            result[num] = num

    if dot_allowed:
        for num in result.copy().values():
            if num.len > max_num_len - 2:
                continue
            for d in allowed_digits:
                if d == 0:
                    continue
                fl = Number(float(num.repr + "." + d))
                if fl not in result:
                    result[fl] = fl
    return result


def generate_hex_numbers(allowed_hex, max_len):
    result = {}
    if '0' in allowed_hex:
        zero = Number(0)
        result[zero] = zero

    max_num_len = max_len // 2 + 3
    for len in range(1, max_num_len - 2):
        for digits in product(allowed_hex, repeat=len):
            if digits[0] == '0':
                continue
            num = Number.hex("0x" + ''.join(digits))
            if num.len > max_num_len:
                continue
            result[num] = num

    return result


def generate_numbers_no_digits(prohibited_symbols, max_len):
    result = {}
    if all(c not in prohibited_symbols for c in 'True'):
        one = Number.true()
        result[one] = one
    elif all(c not in prohibited_symbols for c in 'not[]'):
        sane = Parenthesized.create(Number.totally_sane_number())
        result[sane] = sane
    else:
        sane = Parenthesized.create(Number.even_more_sane_number())
        result[sane] = sane
    return result


def add_unaries(all_exprs, allowed_unaries, max_len, recursive=True):
    added = False
    new = []
    for expr in all_exprs.values():
        candidates = []
        if '-' in allowed_unaries:
            candidates.append(Negative.create(expr))
        if '~' in allowed_unaries:
            candidates.append(Inv.create(expr))
        for candidate in candidates:
            if candidate is None or candidate.len > max_len:
                continue
            curr = all_exprs.get(candidate.result)
            if curr is None or candidate.len < curr.len:
                new.append(candidate)
                added = True

    for new_expr in new:
        all_exprs[new_expr.result] = new_expr

    if added and recursive:
        # TODO can be much faster, stop copying everything
        add_unaries(all_exprs, allowed_unaries, max_len, recursive)


def solve(target: int, level: int, prohibited_symbols: set[str], max_len: int):
    allowed_digits = set(filter(lambda x: x not in prohibited_symbols, all_digits))
    allowed_hex = set(filter(lambda x: x not in prohibited_symbols, all_hex))
    allowed_unaries = set(filter(lambda x: x not in prohibited_symbols, unary_ops))
    allowed_b_ops = set(filter(lambda x: all(ch not in prohibited_symbols for ch in x), binary_ops))
    dot_allowed = '.' not in prohibited_symbols

    if level < 20:  # Classic bruteforce
        all_numbers = list(generate_numbers(allowed_digits, dot_allowed, max_len).values())
    elif level < 31:
        t = hex(target)
        if all(c.upper() in allowed_hex for c in t[2:]):
            return Number.hex(t)
        all_numbers = list(generate_hex_numbers(allowed_hex, max_len).values())
    else:
        all_numbers = list(generate_numbers_no_digits(prohibited_symbols, max_len).values())

    all_exprs = {}
    for n in all_numbers:
        all_exprs[n.result] = n
    # Note: recursing eats a lot of throughput. Should be used as a fallback or until I optimize it
    recurse = level > 30
    add_unaries(all_exprs, allowed_unaries, max_len, recursive=recurse)

    for n in all_exprs.values():
        if n.result == target:
            return n

    expr_gens = expression_generators(allowed_b_ops)

    # Fast-path for hexes mostly
    for n in all_numbers:
        e = find_complement(n, target, allowed_b_ops, all_exprs, max_len)
        if e is not None:
            return e

    result = bruteforce_expressions(all_exprs, expr_gens, allowed_b_ops, allowed_unaries, recurse, max_len, target)
    return result


def expression_generators(allowed_b_ops):
    expr_gens = []

    def maybe_add(cls):
        nonlocal expr_gens
        expr_gens.append(cls)

    for op in allowed_b_ops:
        match op:
            case '+':
                maybe_add(Addition)
            case '-':
                maybe_add(Subtraction)
            case '*':
                maybe_add(Multiplication)
            case '//':
                maybe_add(IntDiv)
            case '**':
                maybe_add(Pow)
            case '<<':
                maybe_add(Shl)
            case '>>':
                maybe_add(Shr)
            case '|':
                maybe_add(Or)
            case '&':
                maybe_add(And)
            case '^':
                maybe_add(Xor)
        pass
    return expr_gens


# Basic meet-in-the-middle?
def find_complement(n1, target, allowed_ops, all_exprs, max_len):
    # Add: target = n1 + n2 -> n2 = target - n1
    if '+' in allowed_ops:
        needed = target - n1.result
        if needed in all_exprs:
            expr = Addition.create(n1, all_exprs[needed])
            if expr and expr.len <= max_len and expr.result == target:
                return expr

    # Sub: target = n1 - n2 -> n2 = n1 - target
    if '-' in allowed_ops:
        needed = n1.result - target
        if needed in all_exprs:
            expr = Subtraction.create(n1, all_exprs[needed])
            if expr and expr.len <= max_len and expr.result == target:
                return expr

    # Mul: target = n1 * n2 -> n2 = target / n1
    if '*' in allowed_ops and n1.result != 0:
        if isinstance(n1.result, int) and target % n1.result == 0:
            needed = target // n1.result
            if needed in all_exprs:
                expr = Multiplication.create(n1, all_exprs[needed])
                if expr and expr.len <= max_len and expr.result == target:
                    return expr

    # IntDiv: target = n1 // n2 -> n2 = n1 // target
    if '//' in allowed_ops and target > 0 and isinstance(n1.result, int) and n1.result > 0:
        approx = n1.result // target
        for delta in range(-1, 2):
            needed = approx + delta
            if needed > 0 and needed in all_exprs:
                expr = IntDiv.create(n1, all_exprs[needed])
                if expr and expr.len <= max_len and expr.result == target:
                    return expr

    # Xor: target = n1 ^ n2 -> n2 = target ^ n1
    if '^' in allowed_ops and isinstance(n1.result, int):
        needed = target ^ n1.result
        if needed in all_exprs:
            expr = Xor.create(n1, all_exprs[needed])
            if expr and expr.len <= max_len and expr.result == target:
                return expr

    return None

def create_binary_expression(cls, left, right, max_len):
    if left.len + right.len + 1 > max_len:
        return None
    result = cls.create(left, right)
    if result and result.len <= max_len:
        return result
    return None

def bruteforce_expressions(all_exprs: dict[int|float, Expression],
                           expr_gens,
                           allowed_b_ops,
                           allowed_u_ops,
                           recurse_for_unaries,
                           max_len: int, target: int):
    populated = False

    add_unaries(all_exprs, allowed_u_ops, max_len, recurse_for_unaries)
    exprs_copy = list(all_exprs.values())
    for n1 in exprs_copy:
        for n2 in exprs_copy:
            for cls in expr_gens:
                expr = create_binary_expression(cls, n1, n2, max_len)
                if expr is None:
                    continue
                res = expr.result
                if res == target:
                    return expr
                # Prune probable offender
                if res > target * 100:
                    continue
                if isinstance(res, float) and res * 10 != int(res * 10):
                    continue
                curr = all_exprs.get(expr.result)
                if curr is None or expr.len < curr.len:
                    populated = True
                    all_exprs[expr.result] = expr
                    e = find_complement(expr, target, allowed_b_ops, all_exprs, max_len)
                    if e is not None:
                        return e

    if populated:
        return bruteforce_expressions(all_exprs, expr_gens, allowed_b_ops, allowed_u_ops, recurse_for_unaries, max_len, target)
    return None

def main():
    with open(TESTDATA_FILE, "r") as f:
        lines = f.readlines()

    total_start = time.time()
    batch_start = time.time()
    batch_times = []

    for level, line in enumerate(lines, start=1):
        parts = line.rstrip('\n').split(' ', maxsplit=2)
        target = int(parts[0])
        max_len = int(parts[1])
        prohibited_symbols = set(parts[2]) if len(parts) > 2 else set()

        print(f"Solving level {level}, target: {target}, prohibited: {prohibited_symbols}, max length: {max_len}")
        result = solve(target, level, prohibited_symbols, max_len)

        if result is None:
            print(f"FAILED level {level}: No solution found")
            sys.exit(1)

        expr = result.repr
        print(f"Found a solution: '{expr}'")
        verify(expr, level, max_len, prohibited_symbols, target)

        print(f"Passed level {level}")
        if level % 10 == 0:
            batch_elapsed = time.time() - batch_start
            batch_times.append((level - 9, level, batch_elapsed))
            print(f"--- Levels {level - 9}-{level} completed in {batch_elapsed:.3f}s ---")
            batch_start = time.time()

    if len(lines) % 10 != 0:
        last_batch_start = (len(lines) // 10) * 10 + 1
        batch_elapsed = time.time() - batch_start
        batch_times.append((last_batch_start, len(lines), batch_elapsed))

    total_elapsed = time.time() - total_start

    print("\n===== Timings =====")
    for start, end, elapsed in batch_times:
        print(f"Levels {start:3d} -{end:3d}: {elapsed:.3f}s")
    print(f"{'Total':15s}: {total_elapsed:.3f}s")


def verify(expr: str, level: int, max_len: int, prohibited_symbols, target: int):
    if len(expr) > max_len:
        print(f"FAILED level {level}: Expression length {len(expr)} exceeds max length {max_len}")
        sys.exit(1)

    found_prohibited = [ch for ch in prohibited_symbols if ch in expr]
    if found_prohibited:
        print(f"FAILED level {level}: Expression contains prohibited symbols: {found_prohibited}")
        sys.exit(1)

    ev = eval(expr)
    if ev != target:
        print(f"FAILED level {level}: Expected {target}, got {ev}")
        sys.exit(1)


if __name__ == '__main__':
    main()
