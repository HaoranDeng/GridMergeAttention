"""CPU checks for independent numeric and special-value cases."""
import math
import torch
from analysis.c8_numerical_gate import compare_tensor


def main():
    ref = torch.tensor([3., 4.])
    same = compare_tensor(ref, ref, kind='output', chunk=1)
    assert same['passed'] and same['relative_l2'] == 0.
    scaled = compare_tensor(ref*1.1, ref, kind='output', chunk=1)
    assert not scaled['passed'] and math.isclose(scaled['relative_l2'], .1, rel_tol=2e-6)
    small = compare_tensor(ref+1e-4, ref, kind='output', chunk=1)
    assert small['passed']
    inf = torch.tensor([float('inf'), 3.])
    assert compare_tensor(inf, inf, kind='bias', chunk=1)['passed']
    wrong_inf = compare_tensor(-inf, inf, kind='bias', chunk=1)
    assert not wrong_inf['passed'] and wrong_inf['invalid_or_mismatched_special_values'] == 1
    nan = torch.tensor([float('nan')])
    assert not compare_tensor(nan, nan, kind='bias')['passed']
    assert not compare_tensor(torch.ones(1), torch.zeros(1), kind='output')['passed']
    print('Seven CPU numerical-gate cases passed; no GPU initialized.')


if __name__ == '__main__':
    main()
