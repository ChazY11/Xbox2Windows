"""The src_scc scalar operand (source code 253) in tools/startup/gfx2hlsl.py.

Source code 253 reads the SCC bit as the integer 0 or 1. The translator refused it as an unknown
operand, so `s_mul_i32 s28, 1.0, src_scc` (the census form in #1820) never lowered. The generated
scalar code keeps SCC in `scc`, so the operand becomes `(scc ? 1u : 0u)`.

Every instruction here is built from the layout tools/disasm_rdna2.py decodes: SOP2 is
`[31:30] == 0b10`, opcode `w0[29:23]`, destination `w0[22:16]`, src1 `w0[15:8]`, src0 `w0[7:0]`;
inline integer 1.0 is source code 242 and src_scc is 253.
"""
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools'))
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
import disasm_rdna2 as D  # noqa: E402
import gfx2hlsl as G  # noqa: E402

SRC_SCC = 253
INLINE_ONE = 242
S_AND_B32 = 0x0e
S_MUL_I32 = 0x26


def sop2(op, sdst, ssrc0, ssrc1):
    return struct.pack('<I', 0x80000000 | ((op & 0x7f) << 23) | ((sdst & 0x7f) << 16)
                       | ((ssrc1 & 0xff) << 8) | (ssrc0 & 0xff))


def stage():
    value = G.Stage.__new__(G.Stage)
    value.sym, value.used_s, value.used_v, value.uniform_v = {}, set(), set(), set()
    value.lines, value.masks = [], set()
    value.exec_known, value.has_branches = True, False
    value.kind, value.pipe = 'vs', None
    return value


def translate(words):
    _, name, asm, _ = D.decode_instruction(words, 0)
    value = stage()
    value.one(name, asm)
    return asm, value.lines


def main():
    # 1. The census form: the raw bits of the inline float times the SCC bit.
    asm, lines = translate(sop2(S_MUL_I32, 28, INLINE_ONE, SRC_SCC))
    assert asm == 's_mul_i32 s28, 1.0, src_scc', asm
    assert lines == ['s28 = 0x3f800000u * (scc ? 1u : 0u);'], lines

    # 2. src_scc in the first source position as well.
    asm, lines = translate(sop2(S_MUL_I32, 29, SRC_SCC, INLINE_ONE))
    assert lines == ['s29 = (scc ? 1u : 0u) * 0x3f800000u;'], lines

    # 3. A bitwise op still updates SCC, so a definition followed by a use emits both statements.
    asm, lines = translate(sop2(S_AND_B32, 28, SRC_SCC, 5))
    assert lines == ['s28 = (scc ? 1u : 0u) & s5;', 'scc = s28 != 0;'], lines

    # 4. Two instructions in sequence: the definition first, then the use that reads it.
    first = stage()
    for words in (sop2(S_AND_B32, 28, SRC_SCC, 5), sop2(S_MUL_I32, 29, INLINE_ONE, SRC_SCC)):
        _, name, asm, _ = D.decode_instruction(words, 0)
        first.one(name, asm)
    assert first.lines == ['s28 = (scc ? 1u : 0u) & s5;', 'scc = s28 != 0;',
                           's29 = 0x3f800000u * (scc ? 1u : 0u);'], first.lines

    # 5. The adjacent refused operands stay refused, so this branch did not widen the operand set.
    for bad in ('ttmp0', 'vcc_hi', 'exec_hi'):
        try:
            G.Stage.u(stage(), bad)
        except G.Unsupported as error:
            assert "operand" in str(error), error
        else:
            raise AssertionError('%s was accepted by u()' % bad)

    print('src_scc operand: 5 cases passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
