"""Refuse-everything audit for the gfx2hlsl handlers (startup translator).

Every `Stage.op_*` handler is fed operand shapes it can never receive from the decoder: no operands, one to
three plain VGPRs, a truncated register and an attribute token. A handler must answer each one with
`Unsupported`, because that is the only exception the translator treats as a refusal:

  * `Stage.translate` records `Unsupported` as a refusal, but any other exception as
    `handler_error(...)`, which labels the entry `internal error` -- the project's marker for a translator
    defect rather than a missing lowering (#1723 counts those as gaps);
  * the single-shader path (`refusals is None`) does not catch anything, so the same input exits with a
    traceback instead of a refusal line.

These shapes cannot come from `decode_instruction`, so this is robustness rather than a live decode bug.
It is still a checkable invariant, and this is a ratchet: `PENDING` lists the handlers that do not hold it
today. A handler outside the list that starts escaping fails the check, and an entry that no longer escapes
also fails -- remove it, and the list shrinks.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'tools' / 'startup'))
sys.path.insert(0, str(ROOT.parent / 'tools'))

import gfx2hlsl as G  # noqa: E402

# Operand shapes no decoder output produces: empty, 1..4 plain VGPRs, a truncated register, an attr token.
SHAPES = (
    (),
    ('v0',),
    ('v0', 'v1'),
    ('v0', 'v1', 'v2'),
    ('v[0', 'v1'),
    ('attr0', 'v1'),
    ('s0', 'v1', 'v2', 'v3'),
)

# Handlers that answer at least one shape with something other than Unsupported on 489d5797. Each entry is a
# handler that indexes an operand before checking the count, or formats a variable number of operands into a
# fixed string. Removing an entry here is the goal; adding one is a regression.
PENDING = (
    'buffer_load_dword', 'buffer_load_dwordx2', 'buffer_load_dwordx3', 'buffer_load_dwordx4',
    'image_load', 'image_load_mip', 'image_sample', 'image_sample_l', 'image_sample_lz',
    's_and_saveexec_b32', 's_and_saveexec_b64', 's_andn2_saveexec_b64', 's_andn2_wrexec_b32',
    's_andn2_wrexec_b64', 's_bfm_b32',
    's_cmp_eq_i32', 's_cmp_eq_u32', 's_cmp_ge_i32', 's_cmp_ge_u32', 's_cmp_gt_i32', 's_cmp_gt_u32',
    's_cmp_le_i32', 's_cmp_le_u32', 's_cmp_lg_i32', 's_cmp_lg_u32', 's_cmp_lt_i32', 's_cmp_lt_u32',
    's_cmpk_eq_i32', 's_cmpk_eq_u32', 's_cmpk_ge_i32', 's_cmpk_ge_u32', 's_cmpk_gt_i32', 's_cmpk_gt_u32',
    's_cmpk_le_i32', 's_cmpk_le_u32', 's_cmpk_lg_i32', 's_cmpk_lg_u32', 's_cmpk_lt_i32', 's_cmpk_lt_u32',
    's_cselect_b32', 's_cselect_b64', 's_or_saveexec_b32', 's_or_saveexec_b64',
    'tbuffer_load_format_x', 'tbuffer_load_format_xy', 'tbuffer_load_format_xyz',
    'tbuffer_load_format_xyzw',
    'v_cndmask_b32', 'v_cvt_i32_f32', 'v_cvt_pkrtz_f16_f32', 'v_fma_mixhi_f16', 'v_fma_mixlo_f16',
    'v_interp_mov_f32', 'v_interp_p1_f32', 'v_interp_p2_f32', 'v_mov_b32', 'v_readfirstlane_b32',
)


class Probe(G.Stage):
    """A Stage with the tracking state a real stage has, without a pipeline."""

    def __init__(self):
        self.lines, self.sym = [], {}
        self.used_s, self.used_v, self.uniform_v = set(), set(), set()
        self.exec_known = True
        self.attrs = {}
        self.interp_p1 = None
        self.compr_params = set()
        self.blocks = []
        self.pc = 0
        self.kind = 'ps'


def escapes(name):
    """Return the shapes this handler answers with something other than Unsupported."""
    bad = []
    for ops in SHAPES:
        stage = Probe()
        try:
            stage.one(name, ' '.join((name,) + ops))
        except G.Unsupported:
            continue
        except Exception as error:  # noqa: BLE001 - any escape is the finding
            bad.append((ops, type(error).__name__))
    return bad


def handler_names():
    return sorted(attribute[3:] for attribute in dir(G.Stage) if attribute.startswith('op_'))


def check_every_handler_refuses_malformed_operands(failures):
    names = handler_names()
    assert len(names) > 50, 'expected the handler set, found %d' % len(names)

    offending = {}
    for name in names:
        bad = escapes(name)
        if bad:
            offending[name] = bad

    # A handler outside the pending list must not escape: that is the regression this check exists for.
    new = sorted(set(offending) - set(PENDING))
    for name in new:
        failures.append('handler %s answers %r with %s; it must raise Unsupported'
                        % (name, offending[name][0][0], offending[name][0][1]))

    # A pending entry that now refuses can be deleted; a stale entry hides the fact that the list is not the
    # current state of the file.
    fixed = sorted(set(PENDING) - set(offending))
    for name in fixed:
        failures.append('handler %s now refuses every shape; remove it from PENDING' % name)

    # The list must be a subset of the handlers that exist, so a rename shows up here.
    for name in sorted(set(PENDING) - set(names)):
        failures.append('PENDING names %s, which is not a handler' % name)


def main():
    failures = []
    check_every_handler_refuses_malformed_operands(failures)
    if failures:
        for failure in failures:
            print('  ' + failure)
        print('startup gfx2hlsl handler arity checks: %d failed' % len(failures))
        return 1
    print('startup gfx2hlsl handler arity checks: %d handlers, %d pending'
          % (len(handler_names()), len(PENDING)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
