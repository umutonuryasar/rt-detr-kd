# Archived KD configs — unused variants

None of these configs produced a reported number. No run in `scripts/run_ablation.sh`
loads any of them, except the commented-out MGD run, which was never executed.

They are kept because the loss code they configure is live and tested
(`src/distillation/mgd.py`, the schedule shapes in
`src/distillation/stage_adaptive_kd.py`), so these files document what that code
accepts. The configs the campaign actually used are one directory up:
`cwd_kd.yml`, `query_kd.yml`, `stage_adaptive_kd.yml`.
