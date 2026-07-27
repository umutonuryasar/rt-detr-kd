"""Regression tests for the pre-campaign audit fix batch (see AUDIT.md).

Every test here was written to FAIL against the code as it stood before the
corresponding fix, and to pass after it. Grouped by audit finding id.
"""

import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.distillation.cwd import CWDLoss
from src.trainer_kd import KDTrainer, build_lr_scheduler

from tools.train_kd import (  # noqa: E402
    _apply_kd_cfg_overrides,
    apply_capture_attn,
    make_loader_generator,
    resume_if_checkpoint,
    seed_worker,
)


# ---------------------------------------------------------------------------
# P0-1: CWD KL reduction must normalize by channels, not by batch alone.
# ---------------------------------------------------------------------------

def test_cwd_kl_is_normalized_by_batch_and_channels():
    """CWD divides the summed KL by B*D, not by B.

    Before the fix the divisor was B alone, inflating the loss by a factor of
    D (256 at hidden_dim=256) and making kd_lambda incomparable with every
    other KD method. Smoke-run evidence: loss_kd ~1242 for CWD vs ~0.25 for
    logit-KD on the same batch.
    """
    torch.manual_seed(0)
    B, D, N = 2, 4, 6
    loss_fn = CWDLoss(student_channels=D, teacher_channels=D, tau=1.0)
    # Make the alignment projection an exact identity so the expected value
    # can be computed in closed form from the inputs.
    with torch.no_grad():
        loss_fn.align.weight.copy_(torch.eye(D).unsqueeze(-1))

    student = torch.randn(B, N, D)
    teacher = torch.randn(B, N, D)

    got = loss_fn(student, teacher)

    s = student.permute(0, 2, 1)
    t = teacher.permute(0, 2, 1)
    summed = F.kl_div(
        F.log_softmax(s, dim=-1).reshape(B * D, N),
        F.softmax(t, dim=-1).reshape(B * D, N),
        reduction="sum",
    )

    assert got.item() == pytest.approx((summed / (B * D)).item(), rel=1e-5)
    # And explicitly NOT the old, D-times-too-large value.
    assert got.item() != pytest.approx((summed / B).item(), rel=1e-3)


def _cwd_reference(student, teacher, tau):
    """Shu et al.: L = tau^2 / C * sum_c sum_n KL(t_c || s_c)."""
    B, N, D = student.shape
    s = student.permute(0, 2, 1) / tau
    t = teacher.permute(0, 2, 1) / tau
    batchmean = F.kl_div(
        F.log_softmax(s, dim=-1).reshape(B * D, N),
        F.softmax(t, dim=-1).reshape(B * D, N),
        reduction="batchmean",
    )
    return batchmean * (tau ** 2)


def test_cwd_matches_batchmean_reduction():
    """The explicit divisor is exactly torch's batchmean over the [B*D, N] view."""
    torch.manual_seed(1)
    B, D, N = 3, 8, 5
    tau = 2.0
    loss_fn = CWDLoss(student_channels=D, teacher_channels=D, tau=tau)
    with torch.no_grad():
        loss_fn.align.weight.copy_(torch.eye(D).unsqueeze(-1))

    student = torch.randn(B, N, D)
    teacher = torch.randn(B, N, D)
    got = loss_fn(student, teacher)

    assert got.item() == pytest.approx(_cwd_reference(student, teacher, tau).item(), rel=1e-5)


# ---------------------------------------------------------------------------
# P-2 (applied): CWD carries the reference tau^2 gradient-scaling factor.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tau", [0.5, 1.0, 2.0, 4.0])
def test_cwd_includes_tau_squared_factor(tau):
    """Without the tau^2 factor the effective KD weight scales with 1/tau^2.

    Before the fix the returned value was the batchmean alone, so this test
    fails at every tau != 1.0 and passes at tau == 1.0 (where the factor is a
    no-op — which is precisely why no planned ablation run changes).
    """
    torch.manual_seed(2)
    B, D, N = 2, 6, 7
    loss_fn = CWDLoss(student_channels=D, teacher_channels=D, tau=tau)
    with torch.no_grad():
        loss_fn.align.weight.copy_(torch.eye(D).unsqueeze(-1))

    student = torch.randn(B, N, D)
    teacher = torch.randn(B, N, D)

    got = loss_fn(student, teacher)
    assert got.item() == pytest.approx(_cwd_reference(student, teacher, tau).item(), rel=1e-5)


def test_cwd_tau_one_is_unchanged_by_the_tau_squared_fix():
    """Guards the 'no planned run changes' claim: tau=1.0 must be a no-op."""
    torch.manual_seed(3)
    B, D, N = 2, 5, 4
    loss_fn = CWDLoss(student_channels=D, teacher_channels=D, tau=1.0)
    with torch.no_grad():
        loss_fn.align.weight.copy_(torch.eye(D).unsqueeze(-1))

    student = torch.randn(B, N, D)
    teacher = torch.randn(B, N, D)

    s = student.permute(0, 2, 1)
    t = teacher.permute(0, 2, 1)
    without_factor = F.kl_div(
        F.log_softmax(s, dim=-1).reshape(B * D, N),
        F.softmax(t, dim=-1).reshape(B * D, N),
        reduction="batchmean",
    )
    assert loss_fn(student, teacher).item() == pytest.approx(without_factor.item(), rel=1e-9)


# ---------------------------------------------------------------------------
# P1-1: training data order must depend on --seed only, never on how much
#       RNG the model/loss construction consumed.
# ---------------------------------------------------------------------------

class _IndexDataset(Dataset):
    def __len__(self):
        return 64

    def __getitem__(self, idx):
        return idx


def _epoch_order(seed: int, rng_burn: int) -> list[int]:
    """Shuffle order for one epoch after burning `rng_burn` draws globally.

    `rng_burn` stands in for the differing amounts of RNG that each KD method
    consumes at construction time (CWDLoss/MGDLoss build a Xavier-initialised
    Conv1d; FeatureKD/QueryKD build nn.Identity; the baseline builds no
    teacher at all).
    """
    torch.manual_seed(seed)
    if rng_burn:
        torch.randn(rng_burn)
    loader = DataLoader(
        _IndexDataset(),
        batch_size=4,
        shuffle=True,
        num_workers=0,
        generator=make_loader_generator(seed),
        worker_init_fn=seed_worker,
    )
    return [int(i) for batch in loader for i in batch]


def test_data_order_independent_of_prior_rng_consumption():
    """Two runs differing only in KD type must see identical data order."""
    assert _epoch_order(42, rng_burn=0) == _epoch_order(42, rng_burn=100_000)


def test_data_order_still_depends_on_seed():
    """Sanity: the fix pins order to the seed, it does not make it constant."""
    assert _epoch_order(42, rng_burn=0) != _epoch_order(43, rng_burn=0)


def _unseeded_epoch_order(seed: int, rng_burn: int) -> list[int]:
    """The pre-fix behaviour: no explicit generator on the DataLoader."""
    torch.manual_seed(seed)
    if rng_burn:
        torch.randn(rng_burn)
    loader = DataLoader(_IndexDataset(), batch_size=4, shuffle=True, num_workers=0)
    return [int(i) for batch in loader for i in batch]


def test_unseeded_loader_demonstrates_the_bug():
    """Documents WHY the explicit generator is required.

    Without generator=..., the sampler seeds itself from the global RNG at
    iterator-creation time, so prior RNG consumption changes the data order.
    """
    assert _unseeded_epoch_order(42, rng_burn=0) != _unseeded_epoch_order(
        42, rng_burn=100_000
    )


def test_seed_worker_makes_python_and_numpy_streams_deterministic():
    """Augmentation draws from python's `random`; it must be seed-derived."""
    torch.manual_seed(7)
    seed_worker(0)
    first = (random.random(), float(np.random.rand()))
    torch.manual_seed(7)
    seed_worker(0)
    assert (random.random(), float(np.random.rand())) == first


# ---------------------------------------------------------------------------
# P1-2: a KD YAML key must never silently override an explicit CLI flag.
# ---------------------------------------------------------------------------

def test_kd_cfg_conflicting_with_explicit_cli_flag_raises():
    kd_cfg = {"type": "query", "query_matching": "index"}
    with pytest.raises(ValueError, match="--query-matching"):
        _apply_kd_cfg_overrides(
            kd_cfg,
            {"query_matching": "hungarian"},
            explicit_flags={"--query-matching"},
        )


def test_kd_cfg_may_override_a_default_valued_flag():
    kd_cfg = {"type": "cwd", "tau": 1.0}
    _apply_kd_cfg_overrides(kd_cfg, {"tau": 4.0}, explicit_flags=set())
    assert kd_cfg["tau"] == 4.0


def test_kd_cfg_agreeing_with_explicit_flag_is_not_a_conflict():
    """The live ablation configs restate values the script also passes on the
    CLI (e.g. kd_lambda: 1.0 with --kd-lambda 1.0). That must stay legal."""
    kd_cfg = {"type": "cwd", "lambda": 1.0}
    _apply_kd_cfg_overrides(
        kd_cfg,
        {"type": "cwd", "lambda": 1.0},
        explicit_flags={"--kd-type", "--kd-lambda"},
    )
    assert kd_cfg["lambda"] == 1.0


# ---------------------------------------------------------------------------
# P1-3: the cosine LR schedule must not turn back up past total_iters.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# P1-4: a resumed run must continue the same stream it would have followed
#       uninterrupted (RNG state persisted; data order pinned to seed+epoch).
# ---------------------------------------------------------------------------

class _TinyBatchDataset(Dataset):
    """Yields (image, target) pairs whose target carries the sample index."""

    def __len__(self):
        return 16

    def __getitem__(self, idx):
        return torch.zeros(3, 8, 8), {"idx": idx}


def _tiny_collate(batch):
    images, targets = zip(*batch)
    return torch.stack(images, 0), list(targets)


class _RecordingModel(torch.nn.Module):
    """Minimal stand-in for RTDETR: one real parameter, records nothing."""

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))

    def forward(self, images):
        return {"pred": images.mean() + self.w}


class _TinyLoss(torch.nn.Module):
    def forward(self, outputs, targets, epoch: int = 0):
        total = outputs["pred"].abs()
        return {"loss_total": total, "loss_det": total.detach()}


def _make_trainer(tmp_path, generator, seed, order_log):
    dataset = _TinyBatchDataset()
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        collate_fn=_tiny_collate,
        generator=generator,
        worker_init_fn=seed_worker,
    )

    class _LoggingLoss(_TinyLoss):
        def forward(self, outputs, targets, epoch: int = 0):
            order_log.extend(t["idx"] for t in targets)
            return super().forward(outputs, targets, epoch=epoch)

    model = _RecordingModel()
    loss_fn = _LoggingLoss()
    opt = torch.optim.SGD(model.parameters(), lr=0.0)
    return KDTrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=opt,
        scheduler=None,
        train_loader=loader,
        val_loader=loader,
        cfg={"train": {"use_amp": False, "accumulate_steps": 1},
             "checkpoint": {"output_dir": str(tmp_path)}},
        device=torch.device("cpu"),
        train_loader_generator=generator,
        seed=seed,
    )


def test_epoch_data_order_is_resume_identical(tmp_path):
    """Epoch N's data order must not depend on whether epochs 1..N-1 ran.

    Before the fix the loader generator advanced across epochs, so a run
    resumed at epoch N replayed epoch 1's ordering.
    """
    uninterrupted = []
    t1 = _make_trainer(tmp_path / "a", torch.Generator(), 42, uninterrupted)
    for epoch in (1, 2, 3):
        uninterrupted.clear()
        t1.train_epoch(epoch)
        if epoch == 3:
            order_uninterrupted = list(uninterrupted)

    resumed = []
    t2 = _make_trainer(tmp_path / "b", torch.Generator(), 42, resumed)
    t2.train_epoch(3)

    assert resumed == order_uninterrupted


def test_checkpoint_roundtrip_restores_rng_state(tmp_path):
    """save/load must carry the RNG state, or resumed augmentation diverges."""
    log = []
    trainer = _make_trainer(tmp_path, torch.Generator(), 42, log)

    torch.manual_seed(1234)
    random.seed(1234)
    np.random.seed(1234)
    trainer.save_checkpoint(epoch=3, tag="latest")

    expected = (torch.randn(3), random.random(), float(np.random.rand()))

    # Perturb every stream, then resume.
    torch.manual_seed(999)
    random.seed(999)
    np.random.seed(999)
    trainer.load_checkpoint(str(tmp_path / "checkpoint_latest.pth"))

    got = (torch.randn(3), random.random(), float(np.random.rand()))
    assert torch.equal(got[0], expected[0])
    assert got[1] == expected[1]
    assert got[2] == expected[2]



# ---------------------------------------------------------------------------
# P0-2: a failed resume must abort, never silently restart from epoch 1.
# ---------------------------------------------------------------------------

class _ExplodingTrainer:
    def load_checkpoint(self, path, load_optimizer: bool = True) -> int:
        raise RuntimeError("RNG state must be a torch.ByteTensor")


class _OkTrainer:
    def load_checkpoint(self, path, load_optimizer: bool = True) -> int:
        return 17


def test_failed_resume_raises_instead_of_restarting(tmp_path):
    """The exact live failure mode: load_checkpoint raised, the bare
    `except Exception` swallowed it, and 2 completed epochs were retrained
    from scratch while only a WARNING was logged."""
    ckpt = tmp_path / "checkpoint_latest.pth"
    torch.save({"epoch": 12, "model_state_dict": {}}, ckpt)
    with pytest.raises(RuntimeError, match="ByteTensor"):
        resume_if_checkpoint(_ExplodingTrainer(), str(ckpt))


def test_successful_resume_returns_checkpoint_epoch(tmp_path):
    ckpt = tmp_path / "checkpoint_latest.pth"
    torch.save({"epoch": 17, "model_state_dict": {}}, ckpt)
    assert resume_if_checkpoint(_OkTrainer(), str(ckpt)) == 17


def test_bare_state_dict_still_starts_from_epoch_one(tmp_path):
    """Backbone-only weight files must keep the graceful path."""
    weights = tmp_path / "backbone.pth"
    torch.save({"conv.weight": torch.zeros(1)}, weights)
    assert resume_if_checkpoint(_ExplodingTrainer(), str(weights)) == 0


def test_missing_weights_path_is_not_a_resume():
    assert resume_if_checkpoint(_ExplodingTrainer(), None) == 0
    assert resume_if_checkpoint(_ExplodingTrainer(), "/nonexistent/x.pth") == 0


def test_cuda_mapped_rng_state_restores(tmp_path):
    """load_checkpoint must survive RNG tensors that torch.load put on GPU.

    Reproduces the failure on CPU by saving CUDA-style state as a non-uint8
    tensor view; the fix normalises every RNG tensor back to a CPU ByteTensor.
    """
    log = []
    trainer = _make_trainer(tmp_path, torch.Generator(), 42, log)
    trainer.save_checkpoint(epoch=1, tag="latest")

    ckpt_path = tmp_path / "checkpoint_latest.pth"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt["rng_state"]["torch"] = ckpt["rng_state"]["torch"].to(torch.int64)
    torch.save(ckpt, ckpt_path)

    trainer.load_checkpoint(str(ckpt_path))  # must not raise


def test_lr_schedule_never_rises_after_total_iters():
    """total_iters is an estimate; overshooting it must not restart the cosine."""
    params = [torch.nn.Parameter(torch.zeros(1))]
    opt = torch.optim.SGD(params, lr=1.0)
    sched = build_lr_scheduler(opt, warmup_iters=10, total_iters=100)
    lr_at = {}
    for it in range(140):
        lr_at[it] = opt.param_groups[0]["lr"]
        opt.step()
        sched.step()

    assert lr_at[100] == pytest.approx(0.0, abs=1e-9)
    # Past the horizon the LR must stay pinned at the floor, not rise again.
    for it in range(100, 140):
        assert lr_at[it] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# P-1 (applied): the dead C3 fusion branch was removed from HybridEncoder.
#
# The gate for this change is EQUIVALENCE, not plausibility: the pre-change
# encoder is reconstructed from git, its shared weights are copied into the
# post-change encoder, and both are required to produce bit-identical output
# on the same input. If they ever diverge, the branch was not dead and the
# removal must be reverted.
# ---------------------------------------------------------------------------

_OLD_TO_NEW_PREFIX = {
    # C3 projection and the C3 fusion block are gone; C4/C5 shift down by one.
    "input_proj.1.": "input_proj.0.",   # C4
    "input_proj.2.": "input_proj.1.",   # C5
}


_PRE_P1_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "encoder_pre_p1.py"


def _load_pre_change_encoder_module():
    """Import the vendored pre-P-1 encoder (tests/fixtures/encoder_pre_p1.py).

    This used to run `git show HEAD:src/models/encoder.py`, which worked only
    while the P-1 edit was uncommitted. Once P-1 landed, HEAD *was* the new
    code, so these tests compared the post-change encoder against itself and
    happily reported a parameter delta of 0. The baseline is now vendored at
    commit 023947a, so it cannot drift with the branch and works in a shallow
    clone.
    """
    import importlib.util

    assert _PRE_P1_FIXTURE.is_file(), (
        f"missing pre-change encoder fixture at {_PRE_P1_FIXTURE} — it is "
        "committed on purpose; see the file header."
    )
    spec = importlib.util.spec_from_file_location("encoder_pre_p1", _PRE_P1_FIXTURE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pre_p1_fixture_is_actually_the_old_encoder():
    """Guard the guard: the fixture must not have drifted to current code.

    If someone "fixes" a failing equivalence test by editing the fixture, this
    fires. The fixture's whole value is that it still contains the dead branch.
    """
    old_mod = _load_pre_change_encoder_module()
    enc = old_mod.HybridEncoder(in_channels=[128, 256, 512], hidden_dim=32,
                                num_encoder_layers=1, nhead=4,
                                dim_feedforward=64, num_csp_blocks=1)
    names = dict(enc.named_parameters())
    assert any(n.startswith("fusion_c3.") for n in names), (
        "fixture has no fusion_c3 — it is no longer the pre-P-1 encoder"
    )
    assert len(enc.input_proj) == 3, (
        f"fixture projects {len(enc.input_proj)} scales, expected 3 (C3+C4+C5)"
    )
    # And it must still consume the C3 backbone key the current encoder dropped.
    import inspect
    assert 'features["0"]' in inspect.getsource(old_mod.HybridEncoder.forward)


def _remap_old_state_dict(old_sd, new_sd):
    """Map pre-change parameter names onto post-change ones.

    Anything belonging to the removed branch (input_proj.0.*, fusion_c3.*) is
    dropped. Everything the new encoder needs must be found — a KeyError here
    would mean the re-indexing is wrong.
    """
    remapped = {}
    for name, tensor in old_sd.items():
        if name.startswith("input_proj.0.") or name.startswith("fusion_c3."):
            continue
        for old_prefix, new_prefix in _OLD_TO_NEW_PREFIX.items():
            if name.startswith(old_prefix):
                name = new_prefix + name[len(old_prefix):]
                break
        remapped[name] = tensor

    assert set(remapped) == set(new_sd), (
        "post-change encoder state dict does not line up with the remapped "
        f"pre-change one; missing={set(new_sd) - set(remapped)}, "
        f"extra={set(remapped) - set(new_sd)}"
    )
    return remapped


@pytest.mark.parametrize("mode", ["eval", "train"])
def test_p1_encoder_output_is_bit_identical_to_pre_change(mode):
    """THE GATE for P-1: removing the C3 branch must change no output value."""
    from src.models.encoder import HybridEncoder

    old_mod = _load_pre_change_encoder_module()

    in_channels = [128, 256, 512]
    torch.manual_seed(0)
    old_enc = old_mod.HybridEncoder(in_channels=in_channels, hidden_dim=64,
                                    num_encoder_layers=1, nhead=4,
                                    dim_feedforward=128, num_csp_blocks=2)
    torch.manual_seed(0)
    new_enc = HybridEncoder(in_channels=in_channels, hidden_dim=64,
                            num_encoder_layers=1, nhead=4,
                            dim_feedforward=128, num_csp_blocks=2)

    # Independent init streams (the removed modules consumed draws), so copy
    # the shared weights across explicitly rather than trusting the seed.
    new_enc.load_state_dict(
        _remap_old_state_dict(old_enc.state_dict(), new_enc.state_dict())
    )

    getattr(old_enc, mode)()
    getattr(new_enc, mode)()

    torch.manual_seed(7)
    feats = {
        "0": torch.randn(2, 128, 16, 16),   # C3 — consumed only by the dead branch
        "1": torch.randn(2, 256, 8, 8),     # C4
        "2": torch.randn(2, 512, 4, 4),     # C5
    }

    with torch.no_grad():
        old_out = old_enc(feats)            # old encoder still reads "0"
        new_out = new_enc(feats)            # new encoder ignores it

    assert new_out.shape == old_out.shape
    assert torch.equal(new_out, old_out), (
        "C3 branch was NOT dead — outputs differ by "
        f"{(new_out - old_out).abs().max().item():.3e}. Revert the removal."
    )
    assert new_enc.scale_shapes == old_enc.scale_shapes


@pytest.mark.parametrize(
    "backbone, expected_drop",
    [("resnet18", 477_824), ("resnet50", 576_128)],
)
def test_p1_removes_exactly_the_audited_parameter_count(backbone, expected_drop):
    """A different delta means the edit removed something other than the audit target.

    Measured against the real pre-change encoder from git, at the production
    hidden_dim of 256 — not against a reconstruction.
    """
    from src.models.backbone import BACKBONE_OUT_CHANNELS
    from src.models.encoder import HybridEncoder

    old_mod = _load_pre_change_encoder_module()
    channels = BACKBONE_OUT_CHANNELS[backbone]

    old_enc = old_mod.HybridEncoder(in_channels=channels, hidden_dim=256)
    new_enc = HybridEncoder(in_channels=channels, hidden_dim=256)

    old_n = sum(p.numel() for p in old_enc.parameters())
    new_n = sum(p.numel() for p in new_enc.parameters())

    assert old_n - new_n == expected_drop, (
        f"{backbone}: encoder params {old_n} -> {new_n} "
        f"(delta {old_n - new_n}, expected {expected_drop})"
    )

    # And the delta is exactly the two removed modules, nothing else.
    removed = sum(
        p.numel()
        for name, p in old_enc.named_parameters()
        if name.startswith("input_proj.0.") or name.startswith("fusion_c3.")
    )
    assert removed == expected_drop


def test_p1_backbone_no_longer_returns_c3_but_still_computes_it():
    """C3 must stop being returned WITHOUT breaking C4/C5, which depend on it."""
    from src.models.backbone import ResNetBackbone

    backbone = ResNetBackbone(name="resnet18", pretrained=False)
    backbone.eval()
    with torch.no_grad():
        feats = backbone(torch.randn(1, 3, 64, 64))

    assert set(feats) == {"1", "2"}, "C3 ('0') must no longer be returned"
    # Strides preserved: C4 = /16, C5 = /32 of a 64-px input.
    assert feats["1"].shape[-2:] == (4, 4)
    assert feats["2"].shape[-2:] == (2, 2)
    assert feats["1"].shape[1] == 256 and feats["2"].shape[1] == 512


def test_p1_encoder_has_no_untrained_parameters():
    """Every encoder parameter must now receive a gradient.

    This is the property that was violated: input_proj[0] and fusion_c3
    produced a tensor nothing consumed, so they stayed at initialisation for
    the whole run.
    """
    from src.models.encoder import HybridEncoder

    enc = HybridEncoder(in_channels=[128, 256, 512], hidden_dim=32,
                        num_encoder_layers=1, nhead=4, dim_feedforward=64,
                        num_csp_blocks=1)
    out = enc({"1": torch.randn(2, 256, 8, 8), "2": torch.randn(2, 512, 4, 4)})
    out.sum().backward()

    ungrad = [n for n, p in enc.named_parameters()
              if p.requires_grad and (p.grad is None or p.grad.abs().sum() == 0)]
    assert ungrad == [], f"parameters receive no gradient: {ungrad}"


# ---------------------------------------------------------------------------
# P-4 (applied): capture_attn is threaded into the TEACHER config too.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "kd_type, expected",
    [
        # consumes teacher attention maps
        ("feature", True),
        ("query", True),
        ("stage_adaptive", True),
        ("combined", True),
        # does not
        ("none", False),
        ("logit", False),
        ("cwd", False),
        ("mgd", False),
    ],
)
def test_capture_attn_is_threaded_into_teacher_config(kd_type, expected):
    """Teacher and student must agree — it is one setting, not two.

    Before the fix only the student config was set; the teacher YAML never
    carries the key, so build_rtdetr defaulted it to True and the teacher
    stored dense [L, B, H, Q, N] attention maps even for logit-KD and CWD,
    which never read them.
    """
    student_cfg = {"model": {}}
    teacher_cfg = {"model": {}}

    got = apply_capture_attn(kd_type, student_cfg, teacher_cfg)

    assert got is expected
    assert student_cfg["model"]["capture_attn"] is expected
    assert teacher_cfg["model"]["capture_attn"] is expected
    assert (
        teacher_cfg["model"]["capture_attn"]
        == student_cfg["model"]["capture_attn"]
    )


def test_capture_attn_reaches_the_built_teacher_model():
    """End-to-end: the threaded flag actually lands on the teacher's decoder."""
    from src.models.rtdetr import build_rtdetr

    teacher_cfg = {"model": {"backbone": "resnet18", "num_queries": 4,
                             "num_decoder_layers": 1, "hidden_dim": 32,
                             "nhead": 4, "dim_feedforward": 64,
                             "pretrained_backbone": False}}
    student_cfg = {"model": dict(teacher_cfg["model"])}

    apply_capture_attn("cwd", student_cfg, teacher_cfg)
    teacher = build_rtdetr(teacher_cfg)
    assert teacher.decoder.capture_attn is False
    assert all(layer.capture_attn is False for layer in teacher.decoder.layers)

    apply_capture_attn("feature", student_cfg, teacher_cfg)
    teacher = build_rtdetr(teacher_cfg)
    assert teacher.decoder.capture_attn is True
    assert all(layer.capture_attn is True for layer in teacher.decoder.layers)


def test_capture_attn_without_teacher_cfg_still_sets_student():
    """Baseline runs pass no teacher config; that path must not break."""
    student_cfg = {}
    assert apply_capture_attn("none", student_cfg, None) is False
    assert student_cfg["model"]["capture_attn"] is False
