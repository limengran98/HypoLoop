"""Executable error weighting and shared candidate policies."""
import ast
import numpy as np

VARIABLES = ('residual', 'target', 'prediction', 'age', 'condition')

FUNCTIONS = {'abs': np.abs, 'sqrt': lambda x: np.sqrt(np.maximum(x, 0)),
             'log1p': lambda x: np.log1p(np.maximum(x, 0)),
             'exp': lambda x: np.exp(np.clip(x, -20, 20)), 'tanh': np.tanh,
             'minimum': np.minimum, 'maximum': np.maximum, 'clip': np.clip}

KEYS = {'loss', 'ridge', 'preservation', 'step', 'scope', 'condition', 'weight_program'}

def parse_program(expression):
    if not isinstance(expression, str) or not 1 <= len(expression) <= 320:
        raise ValueError('Weight program must be a bounded expression')
    tree = ast.parse(expression, mode='eval')
    nodes = list(ast.walk(tree))
    if len(nodes) > 72:
        raise ValueError('Weight program exceeds 72 AST nodes')
    allowed = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name,
               ast.Constant, ast.Load, ast.Add, ast.Sub, ast.Mult, ast.Div,
               ast.USub, ast.UAdd)
    for node in nodes:
        if not isinstance(node, allowed):
            raise ValueError('Unsupported program operation: ' + type(node).__name__)
        if isinstance(node, ast.Name) and node.id not in (*VARIABLES, *FUNCTIONS):
            raise ValueError('Unknown program variable: ' + node.id)
        if isinstance(node, ast.Name) and node.id in FUNCTIONS and not any(
                isinstance(parent, ast.Call) and parent.func is node for parent in nodes):
            raise ValueError('Numerical function names require a call')
        if isinstance(node, ast.Constant) and (type(node.value) not in (int, float)
                or not np.isfinite(node.value) or abs(node.value) > 1000):
            raise ValueError('Constants must be finite bounded numbers')
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in FUNCTIONS or node.keywords:
                raise ValueError('Only registered numerical functions are callable')
            arity = 3 if node.func.id == 'clip' else 2 if node.func.id in ('minimum', 'maximum') else 1
            if len(node.args) != arity:
                raise ValueError('Incorrect function argument count')
    return tree

def run_program(expression, variables, shape):
    """Interpret the restricted AST without eval, imports, attributes or I/O."""
    def walk(node):
        if isinstance(node, ast.Expression): return walk(node.body)
        if isinstance(node, ast.Constant): return float(node.value)
        if isinstance(node, ast.Name): return variables[node.id]
        if isinstance(node, ast.UnaryOp):
            x = walk(node.operand)
            return -x if isinstance(node.op, ast.USub) else x
        if isinstance(node, ast.Call): return FUNCTIONS[node.func.id](*[walk(a) for a in node.args])
        a, b = walk(node.left), walk(node.right)
        if isinstance(node.op, ast.Add): return a+b
        if isinstance(node.op, ast.Sub): return a-b
        if isinstance(node.op, ast.Mult): return a*b
        # Division is total, with an explicit fixed signed denominator floor.
        b = np.asarray(b)
        return a / np.where(np.abs(b) < 1e-6, np.where(b < 0, -1e-6, 1e-6), b)
    with np.errstate(all='raise'):
        value = np.broadcast_to(walk(parse_program(expression)), shape).copy()
    if not np.isfinite(value).all(): raise ValueError('Nonfinite program output')
    return np.clip(value, .05, 20.)

def validate_spec(spec, scopes, conditions, binary=False):
    if not isinstance(spec, dict) or set(spec) != KEYS:
        raise ValueError('Experiment requires exactly ' + ','.join(sorted(KEYS)))
    choices = ('bce', 'brier') if binary else ('mse', 'smooth_mae', 'huber')
    if spec['loss'] not in choices: raise ValueError('Loss incompatible with native output')
    for key, lo, hi in [('ridge', 1e-7, 1e3), ('preservation', 0., 100.), ('step', .05, 1.5)]:
        if type(spec[key]) not in (int, float) or not np.isfinite(spec[key]) or not lo <= spec[key] <= hi:
            raise ValueError('Invalid bounded ' + key)
    if spec['scope'] not in scopes or spec['condition'] not in conditions:
        raise ValueError('Unregistered scope/condition')
    parse_program(spec['weight_program'])
    return dict(spec)

def anchors(binary=False):
    loss = 'bce' if binary else 'mse'
    return [specification(loss, r) for r in (.0001, .01, 1.)] + [
        specification('bce' if binary else 'smooth_mae', .01,
                      weight_program='1 + 9 * target' if binary else '1')]

def fixed_extra(binary=False):
    loss = 'bce' if binary else 'mse'
    return [specification(loss, r, p) for r in (.00001, .001, .1, 10.) for p in (0., .1)]

def numeric_extra(previous, binary, scopes, conditions, rng, count=4):
    """Generate fitting candidates from previous validation records."""
    valid = [x for x in previous if x.get('cv_loss') is not None]
    best = min(valid, key=lambda x: x['cv_loss'])['spec'] if valid else anchors(binary)[1]
    programs = ['1', '1/(1+abs(residual))', '1+abs(residual)', 'exp(-3*age)',
                '1+3*condition', '1/(1+3*condition)', '1+9*target' if binary else
                '1/(0.1+abs(target))']
    result=[]
    for _ in range(count):
        s=dict(best)
        s.update(ridge=float(np.clip(best['ridge'] * 10**rng.uniform(-2,2),1e-7,1e3)),
                 preservation=float(rng.choice([0.,.001,.01,.1,1.])),
                 step=float(rng.choice([.25,.5,.75,1.,1.25])),
                 loss=str(rng.choice(['bce','brier'] if binary else ['mse','smooth_mae','huber'])),
                 scope=str(rng.choice(list(scopes))), condition=str(rng.choice(list(conditions))),
                 weight_program=str(rng.choice(programs)))
        result.append(s)
    return result

def specification(loss='mse', ridge=.01, preservation=0., step=1., scope='full',
                  condition='all', weight_program='1'):
    return dict(loss=loss, ridge=ridge, preservation=preservation, step=step,
                scope=scope, condition=condition, weight_program=weight_program)
