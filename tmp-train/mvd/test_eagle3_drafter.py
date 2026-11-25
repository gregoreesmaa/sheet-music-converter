"""CPU-only unit tests for EAGLE-3 Multi-Layer Speculative Draft Head.

Verifies:
  1. Parameter budget for ``small_45m`` (d=384) and ``edge85m`` (d=512):
     ``Eagle3DraftHead`` is < 4% of target model params and < 2.5M params.
  2. ``extract_multilayer_hidden`` captures 3 decoder layers ``(0, L//2, L-1)``
     without breaking KV cache or leaving lingering forward hooks.
  3. ``compute_eagle3_ttt_loss`` (K=4 unroll) produces finite loss and nonzero
     gradients ONLY on ``Eagle3DraftHead`` (target ``UniOMRv2Net`` strictly frozen).
  4. 100% bit-identical greedy token-for-token equivalence between
     ``eagle3_speculative_generate(..., k_draft=4)`` and vanilla
     ``UniOMRv2Net.generate()`` across 24 tokens — both with a random drafter
     and with a slightly trained drafter (plus optional ``PrefixState`` FSA smoke).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Strictly enforce CPU-only execution (never touch GPU while training runs).
# Note: on Windows, CUDA_VISIBLE_DEVICES="" is ignored/unset; "-1" hides all GPUs.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import torch  # noqa: E402

from eagle3_drafter import (  # noqa: E402
    Eagle3DraftHead,
    compute_eagle3_ttt_loss,
    eagle3_speculative_generate,
    extract_multilayer_hidden,
    get_eagle3_tap_indices,
)
from uni_omr import BOS_ID  # noqa: E402
from uni_omr_v2 import UniOMRv2Config, UniOMRv2Net  # noqa: E402


def _make_tiny_model(seed: int = 42) -> UniOMRv2Net:
    torch.manual_seed(seed)
    cfg = UniOMRv2Config(
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        n_dec_layers=3,
        n_axial_layers=1,
        n_global_layers=1,
        ffn_mult=2.0,
        axial_ffn_mult=1.5,
        backbone_channels=(16, 32, 48, 64),
        backbone_blocks=(1, 1, 1, 1),
        dropout=0.0,
        router_hidden=32,
        max_mem_tokens=256,
        use_sys_y_aux=False,
        use_sys_count_aux=False,
        use_staff_seg_aux=False,
        use_stroke_hog_head=False,
    )
    model = UniOMRv2Net(cfg)
    model.eval()
    return model


def test_param_budget_presets() -> None:
    """1. Verify Eagle3DraftHead param count for small_45m (d=384) and edge85m (d=512)."""
    for name, cfg in [
        ("small_45m", UniOMRv2Config.small_45m()),
        ("edge85m", UniOMRv2Config.edge85m()),
    ]:
        # Instantiate on meta device so we don't allocate hundreds of MB on CPU.
        with torch.device("meta"):
            model = UniOMRv2Net(cfg)
            drafter = Eagle3DraftHead.from_config(cfg)

        summary = model.param_summary()
        total_params = summary["total_params"]
        infer_params = summary["infer_params"]
        draft_params = drafter.param_count()
        pct_total = 100.0 * draft_params / total_params
        pct_infer = 100.0 * draft_params / infer_params

        print(
            f"[PARAM] {name} (d={cfg.d_model}): "
            f"drafter={draft_params:,} ({draft_params / 1e6:.3f}M) | "
            f"target_total={total_params:,} ({pct_total:.2f}%) | "
            f"target_infer={infer_params:,} ({pct_infer:.2f}%)"
        )
        assert draft_params < 2_500_000, f"{name}: drafter {draft_params:,} >= 2.5M"
        assert pct_total < 4.0, f"{name}: drafter is {pct_total:.2f}% >= 4% of total params"
        assert pct_infer < 4.0, f"{name}: drafter is {pct_infer:.2f}% >= 4% of infer params"
        if cfg.d_model == 384:
            assert 1_200_000 <= draft_params <= 1_800_000, (
                f"d=384 drafter {draft_params:,} outside 1.2M-1.8M target range"
            )


def test_multilayer_hook_and_kv_parity() -> None:
    """2. Verify extract_multilayer_hidden taps (0, L//2, L-1) and leaves no hooks."""
    model = _make_tiny_model(seed=7)
    assert get_eagle3_tap_indices(len(model.dec_layers)) == (0, 1, 2)

    images = torch.randn(1, 3, 64, 64)
    tgt_ids = torch.tensor([[BOS_ID, 5, 12, 18, 25]], dtype=torch.long)

    with torch.no_grad():
        mem, _, _ = model.encode_image(images, return_aux=False)
        logits_ref, _, _ = model.decode_tokens(tgt_ids, mem, return_mtp=False, return_aux=False)
        logits_hook, _, h_multi = extract_multilayer_hidden(model, tgt_ids, mem, use_cache=False)

    assert torch.allclose(logits_ref, logits_hook, atol=1e-6), "Hook altered decode_tokens logits!"
    assert h_multi.shape == (1, tgt_ids.shape[1], 3 * model.cfg.d_model)
    for layer in model.dec_layers:
        assert len(layer._forward_hooks) == 0, "Forward hook leaked after extract_multilayer_hidden!"

    # Also verify cached decoding via hooks matches uncached logits.
    with torch.no_grad():
        past_kvs = model.precompute_cross_kv(mem)
        step_logits = []
        for t in range(tgt_ids.shape[1]):
            l_t, past_kvs, h_t = extract_multilayer_hidden(
                model, tgt_ids[:, t : t + 1], memory=None, past_kvs=past_kvs, use_cache=True
            )
            assert h_t.shape == (1, 1, 3 * model.cfg.d_model)
            step_logits.append(l_t)
        cat_logits = torch.cat(step_logits, dim=1)
    assert torch.allclose(logits_ref, cat_logits, atol=1e-5), "Cached hook decode diverged!"
    print("[HOOK] extract_multilayer_hidden parity + zero hook leaks: PASS")


def test_eagle3_ttt_loss_and_grad_isolation() -> None:
    """3. Verify K=4 TTT unroll produces finite loss and grads ONLY on Eagle3DraftHead."""
    model = _make_tiny_model(seed=13)
    for p in model.parameters():
        p.requires_grad_(False)

    torch.manual_seed(14)
    drafter = Eagle3DraftHead.from_model(model)
    drafter.train()

    images = torch.randn(2, 3, 64, 64)
    tgt_ids = torch.randint(1, model.cfg.vocab_size, (2, 16), dtype=torch.long)
    tgt_ids[:, 0] = BOS_ID

    out = compute_eagle3_ttt_loss(model, drafter, tgt_ids, images=images, k_draft=4)
    loss = out["loss"]
    assert out["k_steps_computed"] == 4
    assert torch.isfinite(loss), f"Non-finite TTT loss: {loss}"
    assert float(loss.item()) > 0.0

    loss.backward()

    # Target model must have ZERO gradients everywhere.
    for name, p in model.named_parameters():
        assert p.grad is None, f"Target param {name} unexpectedly received gradient!"

    # Every Eagle3DraftHead parameter must receive finite, nonzero gradients.
    nonzero_grads = 0
    for name, p in drafter.named_parameters():
        assert p.grad is not None, f"Drafter param {name} missing gradient!"
        assert torch.isfinite(p.grad).all(), f"Drafter param {name} has non-finite grad!"
        if p.grad.abs().sum().item() > 0:
            nonzero_grads += 1
    total_draft_tensors = sum(1 for _ in drafter.parameters())
    assert nonzero_grads == total_draft_tensors, (
        f"Expected all {total_draft_tensors} drafter params to have nonzero grad, got {nonzero_grads}"
    )
    step_vals = [round(float(s.item()), 4) for s in out["step_losses"]]
    print(f"[TTT] K=4 loss={loss.item():.4f} step_losses={step_vals} grad_tensors={nonzero_grads}/{total_draft_tensors}: PASS")


def test_exact_greedy_equivalence() -> None:
    """4. Verify 100% bit-identical greedy equivalence across 24 tokens (random + trained drafter)."""
    model = _make_tiny_model(seed=101)
    for p in model.parameters():
        p.requires_grad_(False)

    torch.manual_seed(202)
    drafter = Eagle3DraftHead.from_model(model)
    drafter.eval()

    torch.manual_seed(303)
    images = torch.randn(1, 3, 64, 64)
    max_len = 24

    # Reference vanilla autoregressive greedy generation.
    with torch.no_grad():
        ref_seq = model.generate(images, max_len=max_len)

    # (a) Untrained / random drafter (k_draft=4).
    spec_rand, stats_rand = eagle3_speculative_generate(
        model, drafter, images, max_len=max_len, k_draft=4, return_stats=True
    )
    assert torch.equal(ref_seq, spec_rand), (
        f"Random drafter mismatch!\nref ={ref_seq.tolist()}\nspec={spec_rand.tolist()}"
    )
    print(
        f"[SPEC-RANDOM] exact match ({ref_seq.shape[1]-1} toks) | "
        f"mean_accept={stats_rand['mean_accept_len']:.2f} | "
        f"toks/pass={stats_rand['tokens_per_target_pass']:.2f}: PASS"
    )

    # (b) Slightly train drafter on this sequence via EAGLE-3 TTT so acceptance rises!
    drafter.train()
    opt = torch.optim.AdamW(drafter.parameters(), lr=5e-3)
    with torch.no_grad():
        mem, _, _ = model.encode_image(images, return_aux=False)
    init_loss = None
    final_loss = None
    for step in range(25):
        opt.zero_grad()
        res = compute_eagle3_ttt_loss(model, drafter, ref_seq, memory=mem, k_draft=4)
        res["loss"].backward()
        opt.step()
        if step == 0:
            init_loss = float(res["loss"].item())
        final_loss = float(res["loss"].item())
    assert final_loss < init_loss, f"TTT training did not decrease loss ({init_loss:.4f} -> {final_loss:.4f})"

    drafter.eval()
    spec_trained, stats_trained = eagle3_speculative_generate(
        model, drafter, images, max_len=max_len, k_draft=4, return_stats=True
    )
    assert torch.equal(ref_seq, spec_trained), (
        f"Trained drafter mismatch!\nref ={ref_seq.tolist()}\nspec={spec_trained.tolist()}"
    )
    assert stats_trained["mean_accept_len"] > stats_rand["mean_accept_len"], (
        f"Expected trained drafter to increase mean_accept_len "
        f"({stats_rand['mean_accept_len']:.2f} -> {stats_trained['mean_accept_len']:.2f})"
    )
    print(
        f"[SPEC-TRAINED] exact match ({ref_seq.shape[1]-1} toks) | "
        f"loss {init_loss:.4f}->{final_loss:.4f} | "
        f"mean_accept={stats_trained['mean_accept_len']:.2f} | "
        f"toks/pass={stats_trained['tokens_per_target_pass']:.2f} "
        f"(passes={stats_trained['n_target_passes']} vs {ref_seq.shape[1]-1} vanilla): PASS"
    )

    # (c) Batch B=2 equivalence check.
    torch.manual_seed(404)
    images_b2 = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        ref_b2 = model.generate(images_b2, max_len=16)
    spec_b2 = eagle3_speculative_generate(model, drafter, images_b2, max_len=16, k_draft=4)
    assert torch.equal(ref_b2, spec_b2), "Batch B=2 speculative mismatch!"
    print("[SPEC-BATCH2] exact match (B=2, max_len=16): PASS")

    # (d) Optional PrefixState FSA grammar-masked smoke test (verify k_draft=1 == k_draft=4).
    gram_k1 = eagle3_speculative_generate(
        model, drafter, images, max_len=16, k_draft=1, use_grammar=True
    )
    gram_k4 = eagle3_speculative_generate(
        model, drafter, images, max_len=16, k_draft=4, use_grammar=True
    )
    assert torch.equal(gram_k1, gram_k4), "Grammar-masked speculative mismatch (k=1 vs k=4)!"
    print(f"[SPEC-GRAMMAR] PrefixState FSA exact match (k=1 vs k=4, len={gram_k4.shape[1]-1}): PASS")


def main() -> None:
    assert not torch.cuda.is_available(), "CUDA must be hidden (CPU-only test)!"
    test_param_budget_presets()
    test_multilayer_hook_and_kv_parity()
    test_eagle3_ttt_loss_and_grad_isolation()
    test_exact_greedy_equivalence()
    print("\nALL EAGLE-3 DRAFTER TESTS PASSED (CPU-ONLY).")


if __name__ == "__main__":
    main()
