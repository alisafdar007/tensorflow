# Copyright 2024 The OpenXLA Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from collections.abc import Callable
import dataclasses
import itertools

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from xla.backends.cpu.testlib import kernel_runner
from xla.codegen.testlib import kernel_runner as kernel_runner_base
from xla.python import xla_extension

HloOpcode = kernel_runner_base.HloOpcode
create_literal = kernel_runner_base.create_literal_from_np


@dataclasses.dataclass(frozen=True)
class ElementalHloOpcodeDef:
  op: HloOpcode
  np_op: Callable[[np.ndarray, ...], np.ndarray]
  decimal_precision: int = 6

  # For simple unpacking
  def __iter__(self):
    return iter((self.op, self.np_op, self.decimal_precision))


@parameterized.product(
    op_def=[
        ElementalHloOpcodeDef(HloOpcode.sine, np.sin),
        ElementalHloOpcodeDef(HloOpcode.cosine, np.cos),
        ElementalHloOpcodeDef(HloOpcode.tan, np.tan),
        ElementalHloOpcodeDef(HloOpcode.exponential, np.exp, 4),
        ElementalHloOpcodeDef(HloOpcode.log, np.log),
        ElementalHloOpcodeDef(HloOpcode.sqrt, np.sqrt),
        ElementalHloOpcodeDef(HloOpcode.power, np.pow),
        ElementalHloOpcodeDef(HloOpcode.add, np.add),
    ],
    dtype=[np.float32, np.float64],
)
class ElementalKernelRunnerTest(absltest.TestCase):

  def test_llvm_ir_kernel_runner(
      self,
      op_def: ElementalHloOpcodeDef,
      dtype: np.dtype,
  ):

    [op, np_op, decimal_precision] = op_def

    num_inputs = kernel_runner_base.opcode_arity(op)
    self.assertIsNotNone(num_inputs)

    shape = xla_extension.Shape.array_shape(np.dtype(dtype), [4])
    emitter = kernel_runner.ElementalKernelEmitter(
        [shape] * num_inputs, shape, op
    )

    runner = kernel_runner.KernelRunner.create(emitter.emit_kernel_spec())

    input_arrays = [np.array([1, 2, 3, 4], dtype=dtype)] * num_inputs
    input_literals = [
        create_literal(input_array) for input_array in input_arrays
    ]
    output_literal = create_literal(np.ndarray(shape.dimensions(), dtype=dtype))
    runner.call(list(itertools.chain(input_literals, [output_literal])))
    np.testing.assert_array_almost_equal(
        np.asarray(output_literal),
        np_op(*input_arrays),
        decimal=decimal_precision,
    )


if __name__ == "__main__":
  absltest.main()
