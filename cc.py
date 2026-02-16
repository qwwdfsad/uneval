import time
import sys
from collections import deque

# Precedence levels (low to high)
PREC_NOT = -2
PREC_CMP = -1
PREC_BOR = 0
PREC_BXOR = 1
PREC_BAND = 2
PREC_SHIFT = 3
PREC_ADD = 4
PREC_MUL = 5
PREC_UNARY = 6
PREC_POW = 7
PREC_ATOM = 8

# (symbol, precedence, is_right_associative)
BINARY_OPS = [
    ('|', PREC_BOR, False),
    ('^', PREC_BXOR, False),
    ('&', PREC_BAND, False),
    ('<<', PREC_SHIFT, False),
    ('>>', PREC_SHIFT, False),
    ('+', PREC_ADD, False),
    ('-', PREC_ADD, False),
    ('*', PREC_MUL, False),
    ('//', PREC_MUL, False),
    ('%', PREC_MUL, False),
    ('**', PREC_POW, True),
]

UNARY_OPS = [
    ('-', PREC_UNARY),
    ('~', PREC_UNARY),
]

MAX_VAL = 10 ** 18
MAX_ATOMS = 50000  # cap atom generation


def needs_left_parens(left_prec, left_op, op_str, op_prec, is_ra):
    if left_prec < op_prec:
        return True
    if left_prec > op_prec:
        return False
    if is_ra:
        return True
    return False


def needs_right_parens(right_prec, right_op, op_str, op_prec, is_ra):
    if right_prec < op_prec:
        return True
    if right_prec > op_prec:
        return False
    if is_ra:
        return False
    if op_str == '+':
        return False
    if op_str == '*' and right_op == '*':
        return False
    return True


def safe_binary(v1, op, v2):
    try:
        if op == '+':
            r = v1 + v2
        elif op == '-':
            r = v1 - v2
        elif op == '*':
            r = v1 * v2
        elif op == '//':
            if v2 == 0:
                return None
            r = v1 // v2
        elif op == '%':
            if v2 == 0:
                return None
            r = v1 % v2
        elif op == '**':
            if not isinstance(v2, int) or v2 < 0:
                return None
            if v2 > 100:
                return None
            if v1 == 0 and v2 == 0:
                return None
            if abs(v1) > 1 and v2 > 64:
                return None
            if abs(v1) > 10 and v2 > 18:
                return None
            if abs(v1) > 1000 and v2 > 6:
                return None
            r = v1 ** v2
        elif op == '&':
            if not (isinstance(v1, int) and isinstance(v2, int)):
                return None
            r = v1 & v2
        elif op == '|':
            if not (isinstance(v1, int) and isinstance(v2, int)):
                return None
            r = v1 | v2
        elif op == '^':
            if not (isinstance(v1, int) and isinstance(v2, int)):
                return None
            r = v1 ^ v2
        elif op == '<<':
            if not (isinstance(v1, int) and isinstance(v2, int)):
                return None
            if v2 < 0 or v2 > 64:
                return None
            r = v1 << v2
        elif op == '>>':
            if not (isinstance(v1, int) and isinstance(v2, int)):
                return None
            if v2 < 0:
                return None
            r = v1 >> v2
        else:
            return None

        if isinstance(r, float):
            if r != r:
                return None
            if abs(r) > MAX_VAL:
                return None
            if r == int(r):
                r = int(r)
        elif isinstance(r, int):
            if abs(r) > MAX_VAL:
                return None
        else:
            return None
        return r
    except (OverflowError, ValueError, ZeroDivisionError, TypeError, MemoryError):
        return None


def safe_unary(op, v):
    try:
        if op == '-':
            r = -v
        elif op == '~':
            if not isinstance(v, int):
                return None
            r = ~v
        else:
            return None
        if isinstance(r, int) and abs(r) > MAX_VAL:
            return None
        return r
    except (OverflowError, TypeError):
        return None


def gen_numbers_with_digits(allowed_digits, max_len, max_count):
    """Generate numbers (BFS, shortest first) whose decimal repr uses only allowed_digits."""
    if max_len <= 0 or not allowed_digits:
        return []
    results = []
    if '0' in allowed_digits:
        results.append((0, '0'))
    nonzero = [d for d in allowed_digits if d != '0']
    q = deque((d, 1) for d in nonzero)
    while q and len(results) < max_count:
        num_str, length = q.popleft()
        results.append((int(num_str), num_str))
        if length < max_len:
            for d in allowed_digits:
                q.append((num_str + d, length + 1))
    return results


def gen_hex_numbers(hex_chars, max_len, max_count):
    """Generate hex numbers (BFS, shortest first) using only allowed hex chars."""
    if max_len < 3 or not hex_chars:
        return []
    results = []
    max_hd = max_len - 2
    if '0' in hex_chars:
        results.append((0, '0x0'))
    nonzero_h = [d for d in hex_chars if d != '0']
    q = deque((d, 1) for d in nonzero_h)
    while q and len(results) < max_count:
        hs, length = q.popleft()
        results.append((int('0x' + hs, 16), '0x' + hs))
        if length < max_hd:
            for d in hex_chars:
                q.append((hs + d, length + 1))
    return results


def solve_case(target, max_length, prohibited):
    prohibited_set = set(prohibited)

    def is_allowed(s):
        return all(c not in prohibited_set for c in s)

    allowed_binary = [(s, p, ra) for s, p, ra in BINARY_OPS if is_allowed(s)]
    allowed_unary = [(s, p) for s, p in UNARY_OPS if is_allowed(s)]

    # cache: value -> (expr_string, precedence, top_op_str_or_None)
    cache = {}
    new_set = set()

    def try_add(value, expr, prec, top_op=None):
        le = len(expr)
        if le > max_length:
            return False
        if value in cache:
            if len(cache[value][0]) <= le:
                return False
        cache[value] = (expr, prec, top_op)
        new_set.add(value)
        return True

    allowed_digits = [d for d in '0123456789' if d not in prohibited_set]
    search_max = max(max_length - 2, 0)

    def generate_base_atoms():
        """Generate decimal, float, and special atoms for BFS."""
        # Decimal atoms
        for val, s in gen_numbers_with_digits(allowed_digits, search_max, MAX_ATOMS):
            try_add(val, s, PREC_ATOM)

        # Direct decimal match for target
        if target >= 0:
            ts = str(target)
            if is_allowed(ts) and len(ts) <= max_length:
                try_add(target, ts, PREC_ATOM)
        else:
            abs_s = str(abs(target))
            neg_s = '-' + abs_s
            if is_allowed(neg_s) and len(neg_s) <= max_length:
                try_add(target, neg_s, PREC_UNARY, '-')

        # Float atoms: .d format (e.g. .5 = 0.5)
        if '.' not in prohibited_set:
            for dd in range(10):
                dc = str(dd)
                if dc in prohibited_set:
                    continue
                fs = f".{dc}"
                if len(fs) <= search_max:
                    fv = dd / 10.0
                    if fv == int(fv):
                        try_add(int(fv), fs, PREC_ATOM)
                    else:
                        try_add(fv, fs, PREC_ATOM)

        # Float atoms: n.d format (e.g. 1.5)
        if '.' not in prohibited_set and search_max >= 3:
            float_dec_max = max(search_max - 2, 1)
            for val, int_str in gen_numbers_with_digits(allowed_digits, float_dec_max, MAX_ATOMS // 10):
                for dd in range(10):
                    dc = str(dd)
                    if dc in prohibited_set:
                        continue
                    fs = f"{int_str}.{dc}"
                    if len(fs) <= search_max:
                        fv = val + dd / 10.0
                        if fv == int(fv):
                            try_add(int(fv), fs, PREC_ATOM)
                        else:
                            try_add(fv, fs, PREC_ATOM)

        # Short hex atoms for BFS (up to 6 chars total = 4 hex digits, max 1000 atoms)
        if is_allowed('0x'):
            hex_chars = [d for d in '0123456789abcdef' if d not in prohibited_set]
            if hex_chars:
                for val, s in gen_hex_numbers(hex_chars, min(search_max, 6), 1000):
                    try_add(val, s, PREC_ATOM)

        # Special atoms
        if is_allowed('True'):
            try_add(1, 'True', PREC_ATOM)
        if is_allowed('False'):
            try_add(0, 'False', PREC_ATOM)
        if is_allowed('not[]'):
            try_add(1, 'not[]', PREC_NOT, 'not')
        if is_allowed('not()'):
            try_add(1, 'not()', PREC_NOT, 'not')
        if is_allowed('()==()'):
            try_add(1, '()==()', PREC_CMP, '==')
        if is_allowed('[]==[]'):
            try_add(1, '[]==[]', PREC_CMP, '==')
        if is_allowed('()!=()'):
            try_add(0, '()!=()', PREC_CMP, '!=')

    def generate_hex_atoms(atom_limit):
        """Generate hex atoms (for targeted decomposition)."""
        if is_allowed('0x'):
            hex_chars = [d for d in '0123456789abcdef' if d not in prohibited_set]
            if hex_chars:
                for val, s in gen_hex_numbers(hex_chars, search_max, atom_limit):
                    try_add(val, s, PREC_ATOM)
                if target >= 0:
                    hs = hex(target)
                    if is_allowed(hs) and len(hs) <= max_length:
                        try_add(target, hs, PREC_ATOM)

    # Phase 1: Generate base atoms for BFS (no hex - targeted decomposition handles it)
    generate_base_atoms()

    if target in cache:
        return cache[target][0]

    # === Targeted decomposition: try target = v1 OP v2 using cache ===
    def try_decompose_target():
        """O(n * ops) check: can target be reached by combining two cached values?"""
        for op_str, op_prec, is_ra in allowed_binary:
            op_len = len(op_str)
            for v1, (e1, p1, t1) in list(cache.items()):
                l1 = len(e1)
                # --- v1 OP v2 = target: what v2 is needed? ---
                needed_right = []
                if op_str == '+':
                    needed_right.append(target - v1)
                elif op_str == '-':
                    needed_right.append(v1 - target)
                elif op_str == '*':
                    if v1 != 0:
                        if isinstance(v1, int) and isinstance(target, int) and target % v1 == 0:
                            needed_right.append(target // v1)
                        elif isinstance(v1, float):
                            cand = target / v1
                            if cand == int(cand):
                                needed_right.append(int(cand))
                            else:
                                needed_right.append(cand)
                elif op_str == '//':
                    # v1 // v2 = target → v2 in range [v1/(target+1)+1, v1/target] roughly
                    if isinstance(v1, int) and isinstance(target, int) and target != 0:
                        if target > 0 and v1 > 0:
                            # v2 candidates: v1//target and v1//target if v1//v2==target
                            cand = v1 // target
                            if cand != 0 and v1 // cand == target:
                                needed_right.append(cand)
                            cand2 = cand + 1
                            if cand2 != 0 and v1 // cand2 == target:
                                needed_right.append(cand2)
                elif op_str == '**':
                    if isinstance(v1, int) and abs(v1) > 1 and isinstance(target, int):
                        for exp in range(2, 65):
                            p = v1 ** exp
                            if p == target:
                                needed_right.append(exp)
                                break
                            if abs(p) > abs(target) * 2:
                                break
                elif op_str == '<<':
                    if isinstance(v1, int) and isinstance(target, int) and v1 != 0:
                        for sh in range(1, 65):
                            if v1 << sh == target:
                                needed_right.append(sh)
                                break
                            if abs(v1 << sh) > abs(target) * 2:
                                break
                elif op_str == '>>':
                    if isinstance(v1, int) and isinstance(target, int):
                        for sh in range(1, 65):
                            if v1 >> sh == target:
                                needed_right.append(sh)
                            if v1 >> sh == 0 and v1 > 0:
                                break
                elif op_str == '^':
                    if isinstance(v1, int) and isinstance(target, int):
                        needed_right.append(v1 ^ target)
                elif op_str == '&':
                    if isinstance(v1, int) and isinstance(target, int):
                        if v1 & target == target:
                            needed_right.append(target)
                elif op_str == '|':
                    if isinstance(v1, int) and isinstance(target, int):
                        needed = target & ~v1
                        if (v1 | needed) == target:
                            needed_right.append(needed)

                for v2 in needed_right:
                    if v2 not in cache:
                        continue
                    e2, p2, t2 = cache[v2]
                    l2 = len(e2)
                    nlp = needs_left_parens(p1, t1, op_str, op_prec, is_ra)
                    nrp = needs_right_parens(p2, t2, op_str, op_prec, is_ra)
                    total = l1 + (2 if nlp else 0) + op_len + l2 + (2 if nrp else 0)
                    if total > max_length:
                        continue
                    left = f"({e1})" if nlp else e1
                    right = f"({e2})" if nrp else e2
                    expr = f"{left}{op_str}{right}"
                    if try_add(target, expr, op_prec, op_str):
                        return cache[target][0]

                # --- v2 OP v1 = target (v1 on right side) ---
                needed_left = []
                if op_str == '+':
                    pass  # Commutative, already covered
                elif op_str == '-':
                    needed_left.append(target + v1)
                elif op_str == '*':
                    pass  # Commutative, already covered
                elif op_str == '//':
                    if isinstance(v1, int) and isinstance(target, int) and v1 != 0:
                        # ? // v1 = target → ? in [target*v1, (target+1)*v1 - 1]
                        needed_left.append(target * v1)
                elif op_str == '**':
                    if isinstance(v1, int) and 2 <= v1 <= 100 and isinstance(target, int) and target > 0:
                        root = round(target ** (1.0 / v1))
                        for r in [root - 1, root, root + 1]:
                            if r > 0 and r ** v1 == target:
                                needed_left.append(r)
                elif op_str == '<<':
                    if isinstance(v1, int) and isinstance(target, int) and v1 >= 0 and v1 <= 64:
                        if target >> v1 << v1 == target:
                            needed_left.append(target >> v1)
                elif op_str == '>>':
                    if isinstance(v1, int) and isinstance(target, int) and v1 >= 0 and v1 <= 64:
                        needed_left.append(target << v1)
                elif op_str == '^':
                    pass  # Commutative, already covered
                elif op_str == '&':
                    pass  # Commutative, already covered
                elif op_str == '|':
                    pass  # Commutative, already covered

                for v2 in needed_left:
                    if v2 not in cache:
                        continue
                    e2, p2, t2 = cache[v2]
                    l2 = len(e2)
                    nlp = needs_left_parens(p2, t2, op_str, op_prec, is_ra)
                    nrp = needs_right_parens(p1, t1, op_str, op_prec, is_ra)
                    total = l2 + (2 if nlp else 0) + op_len + l1 + (2 if nrp else 0)
                    if total > max_length:
                        continue
                    left = f"({e2})" if nlp else e2
                    right = f"({e1})" if nrp else e1
                    expr = f"{left}{op_str}{right}"
                    if try_add(target, expr, op_prec, op_str):
                        return cache[target][0]

        # Unary decomposition
        for op_str, op_prec in allowed_unary:
            if op_str == '-':
                needed = -target
            elif op_str == '~':
                if isinstance(target, int):
                    needed = ~target
                else:
                    continue
            else:
                continue
            if needed in cache:
                e, p, t = cache[needed]
                np = p < op_prec
                eff = len(e) + len(op_str) + (2 if np else 0)
                if eff <= max_length:
                    operand = f"({e})" if np else e
                    expr = f"{op_str}{operand}"
                    if try_add(target, expr, op_prec, op_str):
                        return cache[target][0]
        # Special | decomposition: check all pairs of submasks of target
        if isinstance(target, int) and target >= 0:
            for op_str, op_prec, is_ra in allowed_binary:
                if op_str != '|':
                    continue
                op_len = len(op_str)
                submasks = []
                for v, (e, p, t) in cache.items():
                    if isinstance(v, int) and 0 <= v and (v & ~target) == 0 and len(e) + op_len + 1 <= max_length:
                        submasks.append((v, e, p, t))
                for i, (v1, e1, p1, t1) in enumerate(submasks):
                    for v2, e2, p2, t2 in submasks:
                        if safe_binary(v1, op_str, v2) != target:
                            continue
                        nlp = needs_left_parens(p1, t1, op_str, op_prec, is_ra)
                        nrp = needs_right_parens(p2, t2, op_str, op_prec, is_ra)
                        total = len(e1) + (2 if nlp else 0) + op_len + len(e2) + (2 if nrp else 0)
                        if total > max_length:
                            continue
                        left = f"({e1})" if nlp else e1
                        right = f"({e2})" if nrp else e2
                        expr = f"{left}{op_str}{right}"
                        if try_add(target, expr, op_prec, op_str):
                            return cache[target][0]

        return None

    result = try_decompose_target()
    if result is not None:
        return result

    # === Iterative search ===
    iteration = 0
    bfs_start = time.time()
    while new_set:
        iteration += 1
        current_new = new_set
        new_set = set()

        # Snapshot grouped by length
        by_length = {}
        for val, (expr, prec, top_op) in cache.items():
            le = len(expr)
            by_length.setdefault(le, []).append((val, expr, prec, top_op))

        new_by_length = {}
        for val in current_new:
            if val not in cache:
                continue
            expr, prec, top_op = cache[val]
            le = len(expr)
            new_by_length.setdefault(le, []).append((val, expr, prec, top_op))

        # Skip brute-force BFS if cache is too large or taking too long
        total_new = sum(len(v) for v in new_by_length.values())
        total_all = sum(len(v) for v in by_length.values())
        if total_new * total_all > 2_000_000_000:
            break
        if time.time() - bfs_start > 30:
            break

        lengths = sorted(by_length.keys())

        # Binary: new x all + all x new
        for op_str, op_prec, is_ra in allowed_binary:
            op_len = len(op_str)

            # new_left x all_right
            for l1 in sorted(new_by_length.keys()):
                for l2 in lengths:
                    if l1 + op_len + l2 > max_length:
                        break
                    for v1, e1, p1, top1 in new_by_length[l1]:
                        for v2, e2, p2, top2 in by_length[l2]:
                            nlp = needs_left_parens(p1, top1, op_str, op_prec, is_ra)
                            nrp = needs_right_parens(p2, top2, op_str, op_prec, is_ra)
                            total = l1 + (2 if nlp else 0) + op_len + l2 + (2 if nrp else 0)
                            if total > max_length:
                                continue
                            result = safe_binary(v1, op_str, v2)
                            if result is None:
                                continue
                            if result in cache and len(cache[result][0]) <= total:
                                continue
                            left = f"({e1})" if nlp else e1
                            right = f"({e2})" if nrp else e2
                            expr = f"{left}{op_str}{right}"
                            try_add(result, expr, op_prec, op_str)
                            if result == target:
                                return cache[target][0]

            # all_left x new_right (skip left in current_new)
            for l1 in lengths:
                for l2 in sorted(new_by_length.keys()):
                    if l1 + op_len + l2 > max_length:
                        break
                    for v1, e1, p1, top1 in by_length[l1]:
                        if v1 in current_new:
                            continue
                        for v2, e2, p2, top2 in new_by_length[l2]:
                            nlp = needs_left_parens(p1, top1, op_str, op_prec, is_ra)
                            nrp = needs_right_parens(p2, top2, op_str, op_prec, is_ra)
                            total = l1 + (2 if nlp else 0) + op_len + l2 + (2 if nrp else 0)
                            if total > max_length:
                                continue
                            result = safe_binary(v1, op_str, v2)
                            if result is None:
                                continue
                            if result in cache and len(cache[result][0]) <= total:
                                continue
                            left = f"({e1})" if nlp else e1
                            right = f"({e2})" if nrp else e2
                            expr = f"{left}{op_str}{right}"
                            try_add(result, expr, op_prec, op_str)
                            if result == target:
                                return cache[target][0]

        # Unary on new values
        for val in current_new:
            if val not in cache:
                continue
            e, p, top = cache[val]
            for op_str, op_prec in allowed_unary:
                np = p < op_prec
                eff = len(e) + len(op_str) + (2 if np else 0)
                if eff > max_length:
                    continue
                result = safe_unary(op_str, val)
                if result is None:
                    continue
                if result in cache and len(cache[result][0]) <= eff:
                    continue
                operand = f"({e})" if np else e
                expr = f"{op_str}{operand}"
                try_add(result, expr, op_prec, op_str)
                if result == target:
                    return cache[target][0]

        # After each BFS round, try targeted decomposition
        result = try_decompose_target()
        if result is not None:
            return result

    # Phase 2: Expand hex atoms for targeted decomposition (no BFS)
    generate_hex_atoms(MAX_ATOMS)
    if target in cache:
        return cache[target][0]
    result = try_decompose_target()
    if result is not None:
        return result

    return None


def main():
    with open('cases.txt') as f:
        cases = f.read().strip().split('\n')

    start_time = time.time()
    for i, line in enumerate(cases):
        parts = line.split(' ', 2)
        target_num = int(parts[0])
        max_len = int(parts[1])
        prohibited = parts[2] if len(parts) > 2 else ''

        level = i + 1
        print(f"Solving level {level}, target: {target_num}, prohibited symbols: {prohibited}")
        sys.stdout.flush()

        solution = solve_case(target_num, max_len, prohibited)

        if solution is not None:
            print(f"Found a solution: {solution}")
            result = eval(solution)
            assert result == target_num, f"eval({solution}) = {result} != {target_num}"
            assert len(solution) <= max_len, f"len({solution}) = {len(solution)} > {max_len}"
            assert all(c not in prohibited for c in solution), \
                f"Prohibited char in solution: {solution}"
        else:
            print(f"No solution found!")

        sys.stdout.flush()
        if level % 10 == 0:
            elapsed = time.time() - start_time
            print(f"Total time: {elapsed:.2f}s")
            sys.stdout.flush()


if __name__ == '__main__':
    main()
