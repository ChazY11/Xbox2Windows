"""Operand counts of the VOPC compare path and the three interpolation handlers.

usage: startup_gfx2hlsl_compare_interp_arity_check.py

`Stage.translate` records any non-`Unsupported` exception as `translator crash in <op>: <ExcType>`, which the
refusal census reads as a translator defect rather than as text it does not lower. Two paths produced that
marker for a short operand list, with different exception types: the VOPC compare unpacks `dst, (a, b) = 'vcc',
ops` (ValueError) and each interpolation handler reads `ops[2]` (IndexError). Both now refuse. The last checks
load the translator with each guard removed and require the old exception, so the fixture cannot pass without
them.
"""
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
sys.path.insert(0, str(HERE.parent / 'tools'))
import gfx2hlsl as G  # noqa: E402

COMPARE_GUARD = """        if len(ops) < 2:
            raise Unsupported('%s with %d operands, expected at least 2' % (base, len(ops)))
        if len(ops) == 3:"""
INTERP_GUARD = """        if len(ops) != 3:
            raise Unsupported('{0} with %d operands, expected 3' % len(ops))"""
failures = []


def check(label, condition, detail=''):
    print('%s: %s' % (label, 'ok' if condition else 'MISMATCH  ' + detail))
    if not condition:
        failures.append(label)


def probe(module):
    class Probe(module.Stage):
        """A Stage with no pipeline: enough state for one instruction to be translated."""

        def __init__(self):
            self.lines, self.sym = [], {}
            self.used_s, self.used_v, self.uniform_v = set(), set(), set()
            self.masks, self.attrs = set(), {}
            self.exec_known, self.kind = True, 'ps'

    return Probe


def outcome(module, line):
    """'ok', 'refused' or the name of another exception for one instruction text."""
    try:
        probe(module)().one(line.split()[0], line)
    except module.Unsupported:
        return 'refused'
    except Exception as error:  # noqa: BLE001 - the name is the observation
        return type(error).__name__
    return 'ok'


# Accepted shapes: three-operand explicit destination, the two-operand implicit form, and one interpolation.
ACCEPTED = (
    'v_cmp_lt_f32 vcc, v1, v2',
    'v_cmp_lt_i32 vcc, v1, v2',
    'v_cmp_eq_f32 vcc, v1, v2',
    'v_cmp_lt_f32 v1, v2',
    'v_cmp_lt_i32 v1, v2',
    'v_interp_p1_f32 v0, v1, attr0.x',
    'v_interp_p2_f32 v0, v1, attr0.x',
    'v_interp_mov_f32 v0, p0, attr0.x',
)
for line in ACCEPTED:
    got = outcome(G, line)
    check('translates: %s' % line, got == 'ok', got)

# Both short forms of both paths are refusals, and the text names the instruction and both counts.
SHORT = (
    ('v_cmp_lt_f32', 'v_cmp_lt_f32 with 0 operands, expected at least 2'),
    ('v_cmp_lt_f32 v1', 'v_cmp_lt_f32 with 1 operands, expected at least 2'),
    ('v_cmp_lt_i32', 'v_cmp_lt_i32 with 0 operands, expected at least 2'),
    ('v_interp_p1_f32', 'v_interp_p1_f32 with 0 operands, expected 3'),
    ('v_interp_p1_f32 v0', 'v_interp_p1_f32 with 1 operands, expected 3'),
    ('v_interp_p2_f32 v0', 'v_interp_p2_f32 with 1 operands, expected 3'),
    ('v_interp_mov_f32 v0', 'v_interp_mov_f32 with 1 operands, expected 3'),
)
for line, expected in SHORT:
    try:
        probe(G)().one(line.split()[0], line)
    except G.Unsupported as reason:
        check('refuses and names both counts: %s' % line, str(reason) == expected, repr(str(reason)))
    except Exception as error:  # noqa: BLE001
        check('refuses and names both counts: %s' % line, False, type(error).__name__)
    else:
        check('refuses and names both counts: %s' % line, False, 'accepted')

# Negative control: remove each guard and require the old exception back.
source = Path(G.__file__).read_text(encoding='utf-8')
check('control: the compare guard is in the translator once', source.count(COMPARE_GUARD) == 1,
      'count=%d' % source.count(COMPARE_GUARD))
guards = {name: INTERP_GUARD.format(name) for name in
          ('v_interp_p1_f32', 'v_interp_p2_f32', 'v_interp_mov_f32')}
check('control: each interpolation guard is in the translator once',
      all(source.count(text) == 1 for text in guards.values()),
      str({name: source.count(text) for name, text in guards.items()}))

unguarded_source = source.replace(COMPARE_GUARD, '        if len(ops) == 3:')
for text in guards.values():
    unguarded_source = unguarded_source.replace(text, '')
spec = importlib.util.spec_from_loader('gfx2hlsl_unguarded_compare_interp', loader=None)
unguarded = importlib.util.module_from_spec(spec)
unguarded.__dict__['__file__'] = G.__file__
exec(compile(unguarded_source, G.__file__, 'exec'), unguarded.__dict__)

check('control: without its guard the compare with 0 operands raises ValueError',
      outcome(unguarded, 'v_cmp_lt_f32') == 'ValueError', outcome(unguarded, 'v_cmp_lt_f32'))
check('control: without its guard the interpolation with 0 operands raises IndexError',
      outcome(unguarded, 'v_interp_p1_f32') == 'IndexError', outcome(unguarded, 'v_interp_p1_f32'))
check('control: without its guards the accepted shapes still translate',
      all(outcome(unguarded, line) == 'ok' for line in ACCEPTED), 'see the accepted list')

print('compare and interpolation operand count checks: %d failed' % len(failures))
if failures:
    raise SystemExit(1)
print('PASS')
