"""Original local vector-FMA FFN, retained pending an upstream BF16-stage port."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import arith
from flydsl.expr import buffer_ops
from flydsl.expr import math as fxmath
from flydsl.expr import rocdl


def _wave_sum(value, lane):
    for distance in (32, 16, 8, 4, 2, 1):
        bits = arith.bitcast(fx.Int32.ir_type, value.ir_value())
        shuffled = rocdl.ds_bpermute(fx.Int32.ir_type, (lane ^ distance) * 4, bits)
        neighbor = fx.Float32(arith.bitcast(fx.Float32.ir_type, shuffled))
        value = value + neighbor
    return value


def _load_four(resource, offset):
    return fx.Vector(
        buffer_ops.buffer_load(resource, offset, vec_width=4, dtype=fx.BFloat16)
    ).to(fx.Float32)


@flyc.kernel
def ffn_partition_kernel(
    hidden: fx.Tensor,
    gate_up: fx.Tensor,
    down: fx.Tensor,
    partial: fx.Tensor,
    hidden_size: fx.Constexpr[int],
    intermediate_size: fx.Constexpr[int],
    rows: fx.Constexpr[int],
    partition_size: fx.Constexpr[int],
):
    lane = fx.thread_idx.x % 64
    wave = fx.thread_idx.x // 64
    partition = fx.block_idx.x
    partition_count = intermediate_size // partition_size
    hidden_resource = buffer_ops.create_buffer_resource(hidden)
    gate_resource = buffer_ops.create_buffer_resource(gate_up)
    down_resource = buffer_ops.create_buffer_resource(down)
    partial_resource = buffer_ops.create_buffer_resource(partial)
    allocator = fx.SharedAllocator()
    activated = allocator.allocate(fx.Array[fx.BFloat16, rows * partition_size]).peek()
    for batch_row in range(rows):
        for local_row in range(partition_size // 4):
            output_row = partition * partition_size + local_row * 4 + wave
            gate_accumulator = fx.full((4,), 0.0, fx.Float32)
            up_accumulator = fx.full((4,), 0.0, fx.Float32)
            for reduction_tile in range(hidden_size // 256):
                column = reduction_tile * 256 + lane * 4
                input_values = _load_four(hidden_resource, batch_row * hidden_size + column)
                gate_values = _load_four(gate_resource, output_row * hidden_size + column)
                up_values = _load_four(
                    gate_resource, (output_row + intermediate_size) * hidden_size + column
                )
                gate_accumulator = gate_accumulator + input_values * gate_values
                up_accumulator = up_accumulator + input_values * up_values
            gate_value = fx.Float32(fx.BFloat16(_wave_sum(gate_accumulator.reduce("add"), lane)))
            up_value = fx.Float32(fx.BFloat16(_wave_sum(up_accumulator.reduce("add"), lane)))
            activation = gate_value / (fx.Float32(1.0) + fxmath.exp(-gate_value))
            activation = fx.Float32(fx.BFloat16(activation))
            if lane == 0:
                fx.ptr_store(
                    fx.BFloat16(activation * up_value),
                    activated.ptr + batch_row * partition_size + local_row * 4 + wave,
                )
    fx.gpu.barrier()
    for batch_row in range(rows):
        input_values = fx.Vector.from_elements(
            [fx.Float32(fx.ptr_load(activated.ptr + batch_row * partition_size + lane * 4 + element)) for element in range(4)],
            fx.Float32,
        )
        for output_tile in range(hidden_size // 4):
            output_row = output_tile * 4 + wave
            weights = _load_four(
                down_resource, output_row * intermediate_size + partition * partition_size + lane * 4
            )
            contribution = _wave_sum((input_values * weights).reduce("add"), lane)
            if lane == 0:
                buffer_ops.buffer_store(
                    contribution,
                    partial_resource,
                    (batch_row * partition_count + partition) * hidden_size + output_row,
                )


@flyc.kernel
def ffn_reduce_kernel(
    partial: fx.Tensor,
    output: fx.Tensor,
    executions: fx.Tensor,
    hidden_size: fx.Constexpr[int],
    partition_count: fx.Constexpr[int],
    rows: fx.Constexpr[int],
):
    position = fx.block_idx.x * 256 + fx.thread_idx.x
    if position == 0:
        counter_resource = buffer_ops.create_buffer_resource(executions)
        previous = fx.Int32(buffer_ops.buffer_load(counter_resource, 0, vec_width=1, dtype=fx.Int32))
        buffer_ops.buffer_store(previous + 1, counter_resource, 0)
    if position < rows * hidden_size:
        batch_row = position // hidden_size
        column = position % hidden_size
        accumulator = fx.Float32(0.0)
        for partition in range(partition_count):
            accumulator = accumulator + partial[(batch_row * partition_count + partition) * hidden_size + column]
        output[position] = fx.BFloat16(accumulator)


@flyc.jit
def launch_ffn(
    hidden: fx.Tensor,
    gate_up: fx.Tensor,
    down: fx.Tensor,
    partial: fx.Tensor,
    output: fx.Tensor,
    executions: fx.Tensor,
    hidden_size: fx.Constexpr[int],
    intermediate_size: fx.Constexpr[int],
    rows: fx.Constexpr[int],
    stream: fx.Stream,
):
    ffn_partition_kernel(
        hidden, gate_up, down, partial, hidden_size, intermediate_size, rows, 256
    ).launch(grid=(intermediate_size // 256, 1, 1), block=(256, 1, 1), stream=stream)
    ffn_reduce_kernel(partial, output, executions, hidden_size, intermediate_size // 256, rows).launch(
        grid=((rows * hidden_size + 255) // 256, 1, 1), block=(256, 1, 1), stream=stream
    )


