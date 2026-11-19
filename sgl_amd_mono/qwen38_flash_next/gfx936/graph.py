import logging

import torch

from .program import FlashContractError


logger = logging.getLogger(__name__)


class GraphExecution:
    def __init__(self, runtime):
        self.runtime = runtime
        self.captured_batch_sizes = []
        self.replays = {}
        self.checks = []

    def capture(self, original, runner, *args, **kwargs):
        if not runner.disable_padding or not set(runner.capture_bs).issubset({1, 2}):
            raise FlashContractError("Flash-Next Graph requires exact batch sizes 1/2 without padding")
        self.runtime.graph_capture_active = True
        remaining = self.runtime.shadow_remaining
        self.runtime.shadow_remaining = 0
        try:
            result = original(runner, *args, **kwargs)
            self.runtime.peers.check()
            self.captured_batch_sizes = list(runner.capture_bs)
            logger.info("AMD_FLASH_GRAPH_CAPTURE batch_sizes=%s", self.captured_batch_sizes)
            return result
        finally:
            self.runtime.shadow_remaining = remaining
            self.runtime.graph_capture_active = False
            self.runtime.save()

    def execute(self, original, runner, batch, *args, **kwargs):
        runtime = self.runtime
        rows = int(batch.batch_size)
        verify = bool(getattr(batch.forward_mode, "is_target_verify", lambda: False)())
        if not runtime.fusion_enabled or not runtime.supports(batch) or rows not in self.captured_batch_sizes:
            raise FlashContractError("Flash-Next Graph cannot change the captured execution route or pad requests")
        validate = not any(check["rows"] == rows and check["accepted"] for check in self.checks)
        # _decode_states(batch) so the state set includes the per-draft GDN scratch and the
        # PLE side states a TARGET_VERIFY pass writes; on a spec target ``rows`` is the
        # request count and never collides with a plain-decode graph.
        states = runtime._decode_states(batch) if validate else []
        before = [tensor.clone() for _, tensor in states]
        runtime.begin_stage_timing_step()
        result = original(runner, batch, *args, **kwargs)
        runtime.peers.check()
        self.replays[rows] = self.replays.get(rows, 0) + 1
        runtime.completed_decode_steps += 1
        runtime.finish_stage_timing_step()
        if validate:
            try:
                self._validate(batch, result, states, before, native_shadow=verify)
            finally:
                runtime.save()
        return result

    def _validate(self, batch, result, states, before, native_shadow=False):
        from .runner import error_metrics

        runtime = self.runtime
        actual_states = [tensor.clone() for _, tensor in states]
        logits = result.next_token_logits.clone()
        hidden = runtime.model.last_hc_hidden_states
        enabled, remaining = runtime.enabled, runtime.shadow_remaining
        runtime.enabled = True
        # A verify batch replays the graph and never runs the eager path, so without this
        # the per-module native-vs-mono shadow would never run for the verify shape. Run
        # the eager reference here with that shadow enabled -- it is the same fatal
        # GDN/QSA/PLE/post check (incl. the bf16-ULP gate) the first eager verify batch does
        # today, so graphed verify keeps identical coverage. Decode keeps native_shadow=False.
        runtime.shadow_remaining = runtime.shadow_steps if native_shadow else 0
        try:
            for (_, tensor), saved in zip(states, before):
                tensor.copy_(saved)
            expected = runtime.original_forward(batch.input_ids, batch.positions, batch)
            runtime.shadow_remaining = 0
            metrics = error_metrics(logits, expected.next_token_logits)
            state_matches = {name: bool(torch.equal(tensor.contiguous().view(torch.uint8), saved.contiguous().view(torch.uint8))) for (name, tensor), saved in zip(states, actual_states)}
            accepted = metrics["finite"] and metrics["max_absolute_error"] == 0 and all(state_matches.values())
            local_accepted = accepted
            if torch.distributed.is_initialized():
                from sglang.srt.distributed import get_tp_group

                outcome = torch.tensor([int(accepted)], dtype=torch.int32)
                torch.distributed.all_reduce(outcome, op=torch.distributed.ReduceOp.MIN, group=get_tp_group().cpu_group)
                accepted = bool(outcome.item())
            check = {"rows": int(batch.batch_size), "logits": metrics, "state_matches": state_matches, "local_accepted": local_accepted, "accepted": accepted}
            self.checks.append(check)
            logger.info("AMD_FLASH_GRAPH_EAGER_CHECK %s", check)
            if not accepted:
                runtime.serving_shadow_failure = f"Flash-Next Graph differs from eager Mono: {check}"
                raise FlashContractError(runtime.serving_shadow_failure)
        finally:
            for (_, tensor), saved in zip(states, actual_states):
                tensor.copy_(saved)
            result.next_token_logits.copy_(logits)
            runtime.model.last_hc_hidden_states = hidden
            runtime.enabled, runtime.shadow_remaining = enabled, remaining
        del actual_states
        runtime.shadow_remaining = 0
        try:
            runtime._full_model_shadow(runtime.original_forward, (batch.input_ids, batch.positions, batch), {}, result, states, before)
            if runtime.full_model_shadow_checks[-1]["accepted"]:
                runtime.validated_batch_sizes.add(int(batch.batch_size))
        finally:
            runtime.enabled = enabled
            runtime.shadow_remaining = 0
            runtime.shadow_remaining_by_batch[int(batch.batch_size)] = 0

    def snapshot(self):
        return {"captured_batch_sizes": self.captured_batch_sizes, "replays_by_batch_size": dict(self.replays), "graph_eager_checks": self.checks, "padding": False}
