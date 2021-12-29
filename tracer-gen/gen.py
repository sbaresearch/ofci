#!/usr/bin/env python3
import random
import numpy as np
import sys


VAR_NAMES     = ['f', 'g', 'h', 'i', 'j', 'k', 'l']
BINOP_SYMBOLS = ['+', '*', '|', '&', '-', '^']
UNOP_SYMBOLS  = ['-', '~']


class BinOp:
    def __init__(self, symbol, left, right):
        self.symbol = symbol
        self.left = left
        self.right = right

    def __repr__(self):
        return f'({repr(self.left)}{self.symbol}{repr(self.right)})'

    def eval_string(self):
        return f'({self.left.eval_string()}{self.symbol}{self.right.eval_string()})'


class UnOp:
    def __init__(self, symbol, expr):
        self.symbol = symbol
        self.expr = expr

    def __repr__(self):
        return f'({self.symbol}{repr(self.expr)})'

    def eval_string(self):
        return f'({self.symbol}{self.expr.eval_string()})'


class Var:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name

    def eval_string(self):
        return self.__repr__()


class Const:
    def __init__(self, value):
        self.value = np.uint64(value)

    def __repr__(self):
        return str(self.value) + 'ULL'

    def eval_string(self):
        return f'np.uint64({str(self.value)})'


class FunctionGenerator:
    def __init__(self, index, var_values):
        self.index = index
        self.var_values = var_values

    def generate(self):
        proto = f'\nuint64_t fn_{self.index:08} (uint64_t a, uint64_t b, uint64_t c, uint64_t d, uint64_t e) {{'
        s = StatementGenerator(self.var_values)
        proto += s.generate()
        proto += '\n}'
        return proto


class StatementGenerator:
    def __init__(self, var_values):
        self.var_values = var_values
        self.min_steps = 1
        self.max_steps = 3

    def generate(self, level=0):
        generators = []

        if self.max_steps > 1:
            for _ in range(3):
                generators.append(StatementGenerator.gen_if)
        if self.min_steps <= 0:
            for _ in range(1):
                generators.append(StatementGenerator.gen_leaf)

        self.min_steps -= 1
        self.max_steps -= 1

        return random.choice(generators)(self, level + 1)

    def gen_if(self, level):
        indent = '    ' * level
        proto = '\n' + indent + f'if ('
        e = ExpressionGenerator()
        e = e.generate()
        c = Const(random.randint(0, 2**32))
        #print(e.eval_string() + ' > ' + c.eval_string())
        r = eval('exec("import numpy as np") or ' + e.eval_string() + ' > ' + c.eval_string(), self.var_values)
        if r:
            proto += repr(e) + ' > ' + repr(c)
        else:
            proto += repr(e) + ' <= ' + repr(c)
        proto += ') {'
        proto += self.generate(level)
        proto += '\n' + indent + '}'
        proto += '\n' + indent + f'return {level};'
        return proto

    def gen_leaf(self, level):
        indent = '    ' * level
        e = ExpressionGenerator()
        return '\n' + indent + f'return {e.generate()};'


class ExpressionGenerator:
    def __init__(self):
        self.vars = ['a', 'b', 'c', 'd', 'e']
        self.min_length = 4
        self.max_length = 8

    def generate(self):
        generators = []

        if self.max_length > 2:
            for _ in range(5):
                generators.append(ExpressionGenerator.gen_binop)
        if self.max_length > 1:
            for _ in range(2):
                generators.append(ExpressionGenerator.gen_unop)
        if self.min_length <= 0:
            for _ in range(1):
                generators.append(ExpressionGenerator.gen_leaf)

        self.max_length -= 1
        self.min_length -= 1

        return random.choice(generators)(self)

    def gen_leaf(self):
        if random.random() > 0.2:
            return Var(random.choice(self.vars))
        else:
            return Const(random.randint(0, 2**32))

    def gen_unop(self):
        op = random.choice(UNOP_SYMBOLS)
        expr = self.generate()
        return UnOp(op, expr)

    def gen_binop(self):
        op = random.choice(BINOP_SYMBOLS)
        left = self.generate()
        right = self.generate()
        return BinOp(op, left, right)


if __name__ == '__main__':
    count = 1000

    vals = {
        'a': np.uint64(4097993829),
        'b': np.uint64(371910969),
        'c': np.uint64(3718900527),
        'd': np.uint64(673787262),
        'e': np.uint64(471341500)
    }

    random.seed(int(sys.argv[1]))

    print(f'// a: {vals["a"]}')
    print(f'// b: {vals["b"]}')
    print(f'// c: {vals["c"]}')
    print(f'// d: {vals["d"]}')
    print(f'// e: {vals["e"]}')
    print('#include <stdlib.h>')
    print('#include <stdio.h>')
    print('#include <stdint.h>')

    for i in range(count):
        g = FunctionGenerator(i, vals)
        print(g.generate())

    print('\nuint64_t all_targets (uint64_t a, uint64_t b, uint64_t c, uint64_t d, uint64_t e) {')
    for i in range(count):
        print(f'    fn_{i:08}(a, b, c, d, e);')
    print('}')

    print('\nint main(int argc, char *argv[]) {')
    print('''    if (argc <= 5){
        printf("usage: %s <input1> ... <input5>\\n",argv[0]);
        exit(1);
    }
    uint64_t a = atoi(argv[1]);
    uint64_t b = atoi(argv[2]);
    uint64_t c = atoi(argv[3]);
    uint64_t d = atoi(argv[4]);
    uint64_t e = atoi(argv[5]);
    all_targets(a ,b ,c ,d ,e);
    return 0;
}
''')

