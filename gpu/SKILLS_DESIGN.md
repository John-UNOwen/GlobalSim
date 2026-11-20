# Skills port: numeric instruction table design

The race kernel must not change when the skill set changes. A skill is
compiled by the host into a flat numeric table (`uint` words, f32/f64 payloads
bit-punned as word pairs). The kernel runs one interpreter for every table, so
new skills need no kernel rebuild. A construct the interpreter has no model
for never reaches the kernel: the compiler refuses the table and the host
flags the whole race for CPU fallback (step 5 semantics: any unsupported
construct in any skill of a race flags that race).

Call graph per tick, mirroring `step_all_retail` in `race_solver_retail.rs`:

```
race.cl per tick
  straight-flag row update
  duel_collect, lane_collect            (pre-skill rows)
  prepare loop per runner               (spot/asitame timers, corner, slope,
                                         last-straight entry latch)
  skills_advance (skills.cl)            (skill timers += dt, expiry sweep,
                                         cooldown start from waiting)
  skills_scan x2 (skills.cl)            (priority 1 then 0, fire gate,
                                         receiver registration)
  sk_columns                            (fold the receiver lists into the
                                         live columns)
  blocking sample, duel, field.update   (post-skill rows)
  race_step per runner                  (mods folded in at the engine's
                                         application points, see below)
```

Before the loop runs, one `skills_scan` x2 plus `sk_columns` pass runs with the
race clock at 0 and the course events not ready. The engine activates the
immediate records in the solver constructor, before its first timer step, so a
record armed there gives up its effect one tick earlier than one armed by the
loop's first scan. The kernel's pre-loop pass reproduces that ordering, which is
what puts a 2.4 s immediate effect's expiry on the engine's tick instead of one
tick later.

Two constructor-pass facts carry an `init_pass` flag from the pre-loop scan into
the fire path:

- A record whose condition or precondition reads a field or retail-loop leaf
  cannot be judged at time zero: the engine's closure reports false with no
  field, so the record first fires on the loop's first tick, not the constructor
  pass. The compiler marks such records `SK_FLAG_INIT_DEFER` (bit 6), set when a
  leaf is `order`, `order_rate`, `near_count`, `overtake_target_time`,
  `distance_diff_top`, `distance_diff_top_float`, `distance_diff_rate`,
  `change_order_up_end_after`, `change_order_up_middle`,
  `change_order_up_finalcorner_after`, `lane_type`, `is_behind_in`,
  `is_move_lane`, `blocked_front_continuetime`, `blocked_all_continuetime`,
  `blocked_front`, `near_infront_count`, `visiblehorse`, `is_surrounded`,
  `temptation_count_behind`, `temptation_count_infront`,
  `temptation_opponent_count_behind`, `temptation_opponent_count_infront`,
  `running_style_temptation_count_*`, `running_style_temptation_opponent_count_*`,
  `order_rate_in*_continue`, `overtake_target_no_order_up_time` or
  `overtake_target_no_order_down_time`. The pre-loop scan evaluates such a
  record's gate and state normally but drops the fire; the loop scans fire
  normally. The activation-history leaves are the exception: the counters, the
  heal flag and the used-skill set are the solver's own state, present in the
  constructor, so they are never deferred.
- A record that does fire in the constructor pass has no retail timer map yet,
  so its cooldown starts at once as an absolute ready time
  `-(f32(cooldown) * f32(distance/1000))` in the Cooldown state, skipping the
  Waiting-to-instance-expiry step a loop fire takes. A cooldown of 0 still marks
  the record dead.

Initialization mirrors `retail_initialization.rs::initialize()`: the trigger
samplers, the wisdom lottery and the start-delay multiplier run inside
`im_init` (initialization.cl) before the existing rushing and section draws, in
the engine's stream order, so the shared `RetailRandom` substream order stays
bit-exact in solo mode. Paired mode takes the same draws from keyed substreams.

## Table encoding (host to kernel)

One `uint[SK_STRIDE]` block per race, kernel argument `sk`. Every f64 payload
occupies two words (low, high) via bit cast, every f32 payload one word.

Header words 0 to 9: `nrec, nprog, nreg, neff, ntag, nstr, ncor, nslope, nterm,
viewer_row` (the `playerHorseIndex` the `remain_distance_viewer_id` leaf reads,
0 when the request omits it).

Sections (word offsets inside the block):

| area | offset | stride | count | fields |
|---|---|---|---|---|
| straights | 16 | 2 | 12 | f64 start, end (course JSON, sorted) |
| corners | 40 | 2 | 12 | f64 start, start+length |
| slopes | 64 | 3 | 16 | f32 start, f32 end, f32 value |
| records | 112 | 26 | 960 | see below |
| effects | 25072 | 10 | 192 | see below |
| programs | 26992 | 4 | 3072 | see below |
| arms | 39280 | 8 | 256 | see below |
| regions | 41328 | 4 | 2048 | f64 start, end ([start,end), compiler statics) |
| terms | 49520 | 8 | 160 | see below |
| tags | 50800 | 1 | 4096 | tag bytes for substream hashing |

Record (26 words): `skill_id, runner, flags, priority, rarity, n_eff,
eff_base, arm0_prog_len, arm0_prog_base, sampled, term_index, arm0_pre_n,
arm0_pre_base, cd_lo/hi (f64), wis_off, wis_len, delay_bits, arm0_post_n,
arm0_post_base, n_arms, arms_base, pre_n, pre_reg_base, pre_prog_len,
pre_prog_base`. The arm-0 fields mirror the first arm entry (see arms below)
for the scan's fast path. `pre_*` hold the alternative's precondition: a
region list the gate's `sk_prereq_check` reads (its verdict latches into the
record scratch after the first passing check, mirroring the engine's once
satisfied) and a program evaluated before the condition. `flags`: bit0
retain-for-random (detail 0 and rarity White, Gold or Evolution), bit1
wisdom roll leader (at init a failed roll marks the record always-false when
retained and drops it otherwise, and the skill's other records inherit the
verdict), bit2 carries a sampled term, bit3 always-false (the condition
folded false: the record stays in the table, spends its sampled draw and the
kernel skips the gate), bit4 any-skill observe (see the matrix), bit5 wisdom
follower, bit6 init-defer (a field or loop leaf defers the fire past the
constructor pass, see the initialization note), bit7 green tag (a tag in
601 to 615 records the id once per runner for the activated-green set that
gates random re-activation, `lib.rs:187-193`), bits 8 to 12 the random
re-activation selection bits: bit8 primary record (alternative 0), bit9
re-activation eligibility (every effect carries a non-negative baseDuration
and a kind above 5, the engine's pool filter), bit10 gold rarity (2 and 6,
the priority-pass pool), bit11 white rarity (1, the priority-0 pass joins
it), bit12 `isGeneralSkill`, bit13 bumped (the kinds 4 and 5 self writes are
folded construction bumps, so the arms skip them) and bit14 init-bump (a
roll-gated bump the kernel bakes itself after the wisdom lottery, see the
application points). `delay_bits` is the f32 start-delay multiplier, the folded product
over the skill's enabled always-conditioned delay effects. The compiler
resolves the level before the kernel runs: the effect stores the
level-resolved modifier `retail_skill_value(raw, multiplier)` and
`baseDuration * 0.0001`, so the kernel does no table work.

Effect (10 words): `effect_type, target (1 self, 4 visible, 9 ahead, 10
behind), value_usage, duration_usage, target_value (max receivers, clamped to
the roster cap), 0, modifier_lo/hi (f64), base_duration_lo/hi (f64)`.

Arm (8 words): `prog_len, prog_base, pre_n, pre_base, post_n, post_base,
sampled_flag, term_index`. A top-level `@` in the condition splits it into
arms, at most two: the gate passes when the position is in one arm's post
windows and that arm's program holds (the Or closure the engine's apply
builds). `pre_*` is the arm's window list before the sampled term's
intersection, `post_*` after it. `sampled_flag` bit0 marks the arm carrying
the sampled term, bit1 an arm whose region chain collapsed (draw-only).

Program node (4 words): `op, a, b_lo, b_hi`, evaluated postfix on a 64-entry
boolean stack. Opcodes: 0 TRUE (`a` is the value), 1 AND, 2 OR, 4 FEAT_I. The
program carries only the dynamic leaves: a static true or false fold emits a
TRUE node (a region or sampled leaf is TRUE here because it acts through the
window gate), and a construct the compiler cannot lower flags the race
instead of emitting a node. FEAT_I packs `a = feat * 8 + cmp` with the
comparison in engine token order (0 `==`, 1 `!=`, 2 `<`, 3 `<=`, 4 `>`, 5
`>=`), and `b` is the f64 argument.

Term (8 words): `kind, arg, tag_off, tag_len, corner_idx, 0, 0, 0`. `kind` is
the sampler id (0 all_corner_random, 1 straight_random, 2 down_slope_random,
3 up_slope_random, 4 phase_random, 5 phase_firsthalf_random,
6 phase_laterhalf_random, 7 is_finalcorner_random, 8 distance_rate_after_random,
9 corner_random), and `corner_idx` selects the qualifying final corner for id 7
and the resolved index-tagged corner row for id 9. The sampler runs at init on
the shared stream (solo) or a keyed substream (paired, tag
`T{sid}:{alt}:condition:{name}`) and produces the sampled windows. The init
kernel intersects them with the record's static region list in engine `rmap_*`
order and writes the final windows, trigger and single flag into scratch. The
runtime checks a sampled record the same way as any other: the window gate
reads the final windows.

## Evaluation model (interpreter)

Per tick, per runner, per priority pass (1 then 0), records in wire order:

```
skip dead or always-false records
if pos in [trigger.start, trigger.end):
    if not single and no window covers pos:  skip
    if not cooldown_ready(state):             skip
    if not program():                          skip
    fire:
      state := dead if cooldown <= 0 else waiting
               (a constructor-pass fire goes straight to Cooldown, below)
      per effect: append the modifier to its receiver list,
                  start the effect timer at -f32(scaled_duration)
      start the (skill,detail) instance timer at -runtime_duration
else:
    if pos >= trigger.end and not retain: state := dead
```

`cooldown_ready`: Ready is ready, Waiting is not, Cooldown is ready when the
timer reaches 0 or above. After firing, the cooldown stays Waiting until the
instance timer reaches 0 or above (checked in `skills_advance`), then the
cooldown timer starts at `-(f32(cooldown) * f32(distance/1000))`. The record's
`cooldown` is the request value divided by 10000 (the same fixed-point scale as
the effect `baseDuration`), so the `distance/1000` factor here is the engine's
distance scaling of a cooldown measured in course-length units. This is the
engine's `waiting_skill_cooldowns` state machine (`fire_pending_skill`,
`start_expired_skill_cooldowns`, `pending_cooldown_ready`). A constructor-pass
fire skips the Waiting step: it enters Cooldown directly with that same absolute
timer, since the engine has no retail timer map to hold the instance yet. The
pre-loop scan additionally drops the fire for `SK_FLAG_INIT_DEFER` records, whose
field or loop leaf cannot be judged before the first tick.

Timers step `(double)(convert_float_rte((float)timer) + dt)` and expire at 0 or
above, matching `new_skill_timer` and `SkillTimer::step`.

Duration: `((f32(base_duration) * f32(distance/1000.0)) * f32(coeff)) * 1.0f`
as f64, with `coeff` from `duration_usage` (1 is 1.0, 2 is the retail gap
tiers, 3 and 7 are the retail HP tiers). Modifier dispatch follows retail
`apply_activation` with `value_usage` 1:

- `TargetSpeed(27)`: append `(token, modifier as f32)` to the runner's ordered
  target-receiver list.
- `Accel(31)`: same for the ordered accel-receiver list.
- `CurrentSpeed(21)`: same for the current-speed list (non-natural).
- `CurrentSpeedWithNaturalDeceleration(22)`: same list, marked natural, so its
  fold sits in the natural group and its expiry is more than a removal (below).
- `Recovery(9)`: adds `f32(max_hp) * f32(modifier)` to the runner's pending-HP
  accumulator (the receiver's max when delivered); the heal count the
  `activate_count_heal` condition reads bumps only for a positive self effect.
- `Vision(8)`: a receiver token in the vision order, whose fold
  `fmax(20 + sum, 0)` in f64 with one `convert_float_rte` feeds the lane
  logic's visible distance.
- `GutsUp(4)` and `ModifyKakariChance(29)`: a construction-time fire is a
  bump the host applies to the init inputs (stats through the
  `add_stat_modifier` f32 chain, kakari chance folded into the init roll).
  A mid-race or cross fire of 4 moves the live accumulator and field, and
  the guts-derived `powf` terms recompute from the field at the points the
  engine reads them (see Mid-race stat fires below).
- `WisdomUp(5)`: a construction fire is a bump like those two, and the
  record carries `SK_FLAG_BUMPED` so the kernel's init-pass arms skip the
  write the host already folded. A self mid-race fire needs a phase gate
  false at time zero on every arm, then moves the accumulator and the live
  field that the pace and downhill draws read each tick. The unclamped f32
  accumulator rides the `PRD_WIS_ACC` word (NaN when no bump folded), so a
  loop fire chains from a construction bump. The spurt plan keeps the
  captured wisdom until a receiver refresh recaptures it, as in the engine.
- Roll-gated constructor bump. When a static-true self fire of kinds `2`,
  `4` or `5` also carries the wisdom lottery (`wisdomCheck` set, cooldown 0),
  the host cannot precompute whether it fires, so it pushes no `bumps` entry
  and the record carries `SK_FLAG_INIT_BUMP` (with `SK_FLAG_BUMPED`, which
  already suppresses the init-pass self write). The kernel replays the bump
  once the roll lands: `sk_init_wis_bump` adds the enabled records' `WisdomUp`
  modifiers to the section and rushing wisdom input inside `im_init` (after
  `sk_init_wisdom`, so the section, rushing and temptation draws see the
  bumped value the engine's post-constructor `horse.wisdom` feeds at
  `lib.rs:760`), and `sk_init_stat_bump` replays the enabled records' `2`, `4`
  and `5` effects onto the stat output columns before the HP chain, so the
  `GameHpState` and every later seed start from the bumped field. A failed roll
  leaves the record `SK_S_DISABLED`, so neither helper moves anything and no
  bump lands, matching the engine not installing the skill.
- `RunAway(6)`: the runtime arm is SkillAbilityNull, so the fire is inert
  beyond its event and instance bookkeeping. The build-time rule that matters
  is elsewhere: a Nige that learned a type 6 effect races as Oonige
  (`race_solver_builder.rs:858-865`), and the host prepass applies the same
  upgrade before anything reads the strategy.
- `SetStartDelay(14)`: writes the solver's `start_delay`, which the retail
  physics never reads (the retail start delay resolved in initialization), so
  the kernel fires it inert.
- `MultiplyStartDelay(10)`: for the self target it folds into the init-time
  start-delay factor. Aimed at another runner the retail receiver arm does
  nothing (initialization already applied the gate), so the record fires inert:
  event, counters and cooldown, no state write. Kinds `1`, `4` and `5` aimed at
  another runner stay refused (the receiver stat bump and its refresh).
- `PreferOutsideLane(11)` and `PreferInsideLane(12)`: arm the owner's lane
  preference presence cell. While present the runner's lane search carries the
  outside (bit 1) or inside (bit 2) preference flag, and both present reads 3,
  which the engine treats as no preference (`lane_preference_flags()`,
  `race_solver.rs:1310-1314`). The flag also moves the skill lane cost factor
  to 1.5 when it points against the gap (`retail_lane.rs`).
- `SetTargetLane(35)`: push the lane onto the owner's ordered goal vector (four
  slots), and the lane search reads the last live entry (`lane_target`'s fixed
  goal, `race_solver_retail.rs:766-771`).
- `ExtendKakari(13)`: on a receiver already rushing, `set_forced_end` replaces
  the rush's forced remaining with `fmax(modifier as f32, 0)` and the forced
  stretch skips the natural clocks and the per-tick end lottery
  (`race_solver.rs:2584-2590`, `retail_states.rs:154-158, 178-181`). The kind
  is a debuff (`is_skill_demerit`), so the same-team drop and immunity gate
  it, and target `21` resolves the strategy group carried in `target_value`
  among rushing runners, uncapped and without self (`race_field.rs:439-445`).
- `DebuffImmunity(38)`: arm the owner's immunity presence cell. While present,
  debuff deliveries skip the runner in the target filter, and the owner's own
  debuff fire skips delivery to self (`race_solver.rs:2526`,
  `race_field.rs:458-461`).

Kinds 11, 12, 35 and 38 write per-runner aux cells that the per-tick advance
steps and retires like the record timers (`race_solver.rs:1990-1996`): the
effect is present while the cell
is at or below zero, armed from the per-effect scaled duration
(`race_solver.rs:2530, 2540-2548`), and a negative duration arms it for the
rest of the race.

The lists mirror `AdditiveReceiver`: registration-order folds, in-place token
update on refire, order-preserving token removal at expiry.

Cross-target delivery (targets 9, 10, 4): the fire resolves receivers from the
rank columns of the previous field update, ahead taking the runners above self
nearest-first, behind taking self and the runners below, and visible
additionally demanding a slot in the lane cache as it stood before this tick's
scan (the engine selects before `field_update`, so effects expiring this tick
still count). Each receiver gets the caster's token in its order plus, for
recovery, the HP amount computed from the receiver's max. A delivery never
registers a receiver-side instance: `apply_external_effect` passes
`follow_up = true` (`race_solver.rs:1325-1328`), and the follow-up path returns
before the `random_active_details` insert (`race_solver.rs:2644-2653`), so the
receiver's own cooldown map and instance timers stay untouched. The constructor
pass resolves no receivers (the engine has no field to deliver into at time
zero) while still counting the fire's runtime duration, event and cooldown. The
engine's `skill_targets` filter drops a debuff receiver who shares the owner's
non-zero team (`race_field.rs:458-461`), applied after the `target_value` cap so
no later candidate backfills the slot. The kernel mirrors it: the owner's team
and each receiver's team reach `skills_fire` through `SI_TEAM`, and a negative
modifier skips the same-team receivers off the capped list. Debuff immunity, the
filter's other arm, reads a per-runner presence cell `SD_IMMUNE` armed by a kind
`38` fire, and an immune receiver skips the drop before the team test.

## Modifier application points (exact engine sites)

- Target chain, `phys_target_speed`: the compiler-fed ordinary array replaces
  the NULL mods and 0 length, while `spot(token1)` and `duel(token4)` stay
  separate arguments (`values_except_tokens(&[1,4])` then `value(1)` then
  `value(4)`).
- Accel chain, `phys_acceleration_from_base`: the ordinary array fills the
  NULL slots, duel then cp/asitame append last
  (`values_except_tokens(&[cp,4])` then `value(4)` then `value(cp)`, and no cp
  skill is in this subset).
- Current speed (`retail_current_speed`, `race_solver_retail.rs:729`):
  `observed = (f32(nat_fold) + f32(ord_fold)) + speed` in f32, then clamped to
  [min, cap]. SD_CS keeps the raw speed, since HP and the accel branch read
  `current_speed`, not the folded value.
- Natural current-speed expiry (`race_solver.rs:1967`): the expired modifier
  also lands in `modifiers_one_frame_accel`, and the speed chain reads
  `current_speed` plus that accumulator as its one input, then the retail step
  clears the accumulator. In the kernel `skills_expire_current` drains the
  current-speed tokens from the tail of the activation order into `SD_OFA` (the
  reverse order matches the engine's reverse list drain, which is the sum order
  when a runner loses two of them in one tick), and `race_step` in race.cl reads
  the sum into `current` and clears it at the end of a step that ran. The
  impulse is what keeps the speed decaying toward the lower target over several
  ticks instead of snapping to it, and a gated rest keeps it for the next tick,
  as the engine's early return does.
- Lane and roster target (`retail_target_speed_with_current_skills`,
  `race_solver_retail.rs:740`): `apply_additive(f32(SD_TS), cur_ord_f32s)`,
  recomputed per tick after prepare, because the list state changes between
  ticks. Consumers: the lane owner context, the lane target search and the
  live-horse roster (blocking sample and duel inputs).
- `has_speed_skill` (`race_solver_retail.rs:912`, position keep):
  `pace_cancel > 0 || apply_additive(0., target_ord) > 0`.
- Recovery(9) feeds `pending_hp` (`race_solver_retail.rs:994`): the bank lands
  at the step head before the start-delay gate, and a gain in the last two
  phases defers the last-spurt plan, as the engine's flag does.

## RNG

No draw touches the race-phase stream: sampled windows and the wisdom roll
happen in initialization only. `r128_substream_hash(seed, runner, tag)` in
`rng_core.cl` hashes arbitrary global tag bytes. Tags are built byte-for-byte
like the engine: `T{skill_id}:{alt}:{pre|cond}:{term}` for sampled terms
(paired mode only, solo mode draws in stream order) and `W{skill_id}` for
wisdom rolls (paired). Chained substreams re-key from the parent state exactly
as `RetailRandom::substream` does (FNV-1a-64 plus avalanche, seed is
`hash >> 33` as i32).

A sampled draw is spent for every random trigger term in roster order, in
`im_init` before any region is judged, exactly as `retail_initialization::initialize`
does. That includes a record the compiler marks dead (its region chain collapses):
it still consumes its sampled draw so the shared solo stream stays aligned for every
later term and runner, then stays inert. Only `is_finalcorner_random` with no
qualifying corner takes no draw, since the engine draws nothing there. A dead record
that skipped its draw shifts the whole stream after it and corrupts start delay.

## Condition-support matrix

Engine vocabulary from the `c.insert` list in `activation_conditions.rs`.
Ported means `classify` lowers the name to a static fold or a dynamic leaf.
Any name the match does not handle, or a registered name with an unregistered
comparison, flags the race with the offending construct named.

The compiler lowers each condition term to a `Leaf`. Fold leaves resolve at
compile time (TRUE, FALSE, or a window list the init kernel intersects), so
the program carries only the dynamic leaves: `phase`, `distance_rate`,
`remain_distance`, `corner`, `slope`, `is_finalcorner`,
`is_finalcorner_laterhalf`, `accumulatetime`, `order`, `order_rate`,
`change_order_onetime`, `hp_per`, `is_overtake`, `is_temptation`,
`temptation_count`, `near_count`, `infront_near_lane_time`,
`behind_near_lane_time`, `behind_near_lane_time_set1`,
`blocked_side_continuetime`, `blocked_front_continuetime`, `bashin_diff_behind`,
`bashin_diff_infront`, `activate_count_start`, `activate_count_middle`,
`activate_count_heal`, `is_lastspurt`, `is_last_straight_onetime` (entry latch),
the out-band cells (`order_rate_out*_continue`) and `is_activate_any_skill`, the
wave-2 live fields `lane_type`, `is_move_lane`, `is_behind_in`,
`distance_diff_top`, `overtake_target_time`, `change_order_up_end_after`,
`compete_fight_count`, and the wave-3 leaves `order_rate_in*_continue` (in-band
cells), `near_infront_count`, `visiblehorse`, `is_used_skill_id`,
`is_activate_heal_skill`, `activate_count_all`, `activate_count_end_after`,
`activate_count_later_half`, `overtake_target_no_order_up_time`,
`overtake_target_no_order_down_time`, `change_order_up_middle`,
`change_order_up_finalcorner_after`, `blocked_all_continuetime`,
`blocked_front`, `is_surrounded`, `temptation_count_behind`,
`temptation_count_infront`, `temptation_opponent_count_behind`,
`temptation_opponent_count_infront`, `running_style_temptation_count_*`,
`running_style_temptation_opponent_count_*`, `distance_diff_rate`,
`distance_diff_top_float`, and the activation-history fold
`is_activate_other_skill_detail`. A condition whose
comparison is not registered for its name falls to the `classify` catch-all
and flags the race.

Fold columns: `six` is `==,!=,<,<=,>,>=`, the engine's six value_filter
comparators (`activation_conditions.rs:229-243`).

| name | fold |
|---|---|
| always | TRUE |
| phase | dynamic FEAT_RPHASE, the unclamped runtime phase (the engine compares `retail.phase`, which keeps 3 while the solver field clamps at 2, `race_solver_retail.rs:1311`) |
| distance_rate | dynamic FEAT_DIST_RATE, `arg` rescaled by `distance*rate/100` |
| remain_distance | dynamic FEAT_REMAIN |
| is_finalcorner | dynamic FEAT_FINALE |
| is_finalcorner_laterhalf | dynamic FEAT_FC_HALF, the last-corner second half |
| is_last_straight | static window over the last straight, `==1` |
| is_last_straight_onetime | static window `[ls.start, ls.start+10)` plus the dynamic FEAT_LS_ENTRY entry latch, `==1` |
| is_lastspurt | static window over phase 3 plus the dynamic FEAT_LASTSPURT, `==1` |
| corner | dynamic FEAT_CORNER, corner id at the position |
| slope | dynamic FEAT_SLOPE, 0 gap, 1 up, 2 down |
| order | dynamic FEAT_ORDER, 1-based rank |
| order_rate | dynamic FEAT_ORDER_RATE, rank versus `round(n*rate/100)` |
| order_rate_out20/40/50/70_continue | dynamic FEAT_OUTBAND, `==1` only, the rate in the name picks the band cell and rides in the comparison argument |
| is_activate_any_skill | dynamic FEAT_ANYACT, `==1` only, the v75 observe flag (see below) |
| accumulatetime | dynamic FEAT_CLOCK, the gated race clock |
| activate_count_start, activate_count_middle | dynamic, `>=` only |
| change_order_onetime | dynamic FEAT_ORDER_DELTA, rank change over the last tick |
| hp_per | dynamic FEAT_HP_PER, `trunc(f64(f32(hp)/f32(max))*100)`, `<=,>=` only |
| is_overtake | dynamic FEAT_IS_OVERTAKE, lane target list non-empty |
| is_temptation | dynamic FEAT_IS_TEMPTATION, kakari flag, `==` only |
| temptation_count | dynamic FEAT_TEMPT, the temptation accumulator, `==` only |
| near_count | dynamic FEAT_NEAR |
| infront_near_lane_time | dynamic FEAT_IFNEAR |
| behind_near_lane_time, behind_near_lane_time_set1 | dynamic FEAT_BHNEAR, FEAT_BHNEAR1 |
| blocked_side_continuetime | dynamic FEAT_BSIDE |
| bashin_diff_behind, bashin_diff_infront | dynamic neighbour gaps |
| distance_type | course `distanceType`, six |
| ground_type | course `surface`, six |
| track_id | course `raceTrackId`, six |
| rotation | course `turn`, six |
| ground_condition | request `groundCondition` enum (`Good=1..Heavy=4`), six, refuses a request without a parsable value |
| weather | request `weather` enum (`Sunny=1..Snowy=4`), six, refuses a request without a parsable value |
| season | request `season` enum (`Spring=1..Sakura=5`), six, refuses a request without a parsable value |
| is_dirtgrade | request `isDirtGrade` flag when present, else `track_id` in `DIRT_GRADE_TRACK_IDS` (`[10101,10103,10104,10105]`), `==,!=` only (`activation_conditions.rs:1572,734-742`) |
| grade | request `grade` enum, the `g1` default when absent (`lib.rs:284`), six, refuses a present but unparsable grade |
| time | request `time` enum, the midday default when absent (`lib.rs:285`), six, refuses a present but unparsable time |
| motivation | horse `mood` clamped to `[-2,2]` plus three (`lib.rs:505`), six |
| base_power | the runner's ranked wire stat: `ranked_base_stat` (overcap, team rank bonus) truncated as `as i32`, six (`race_solver_builder.rs:872-881`) |
| post_number | the client gate block mirrored from the wire `gateIndex` (absent means the roster index), `==,<=,>=` only, the three arms the engine registers (`activation_conditions.rs:1002-1039`) |
| popularity | horse `popularity` (default 0), six |
| running_style | horse strategy group, `==` only |
| all_corner_random, straight_random, down_slope_random, up_slope_random | sampled term (`==1`), init-time windows through the scan gate |
| phase_random, phase_firsthalf_random, phase_laterhalf_random | sampled term (`==0..4` phase index), window from the six-way phase split |
| is_finalcorner_random | sampled term, window over the qualifying final corner (an id >= 4 starting within the last 1000 m), else no draw and the record stays inert |
| distance_rate_after_random | sampled term (`==`, argument >= 0), init windows like the other samplers |
| corner_random | sampled term (`==`, the argument resolves the course corner row: the index-tagged row when the course tags them, else the legacy `rows.len() + arg - 5` slot), the row's start..end as the one window, an untagged non-integer course index refuses (the engine's init error) and no resolved row folds the arm dead (`activation_conditions.rs:549-572`, `retail_initialization.rs:135-160`) |
| lane_type | dynamic FEAT_LANETYPE, `HorseRaceInfo.get_Lane` bands: 0..3 from the live lane crossing 0.2, 0.4, 0.6, six |
| is_move_lane | dynamic FEAT_MOVELANE, 0 idle, 1 while a lane move runs and 2 when it is directional, six |
| blocked_front_continuetime | dynamic FEAT_BFRONT, `trunc` of the continuous blocked-front clock, six |
| overtake_target_time | dynamic FEAT_OTIME, `floor` of the live overtake-target clock, six |
| change_order_up_end_after | dynamic FEAT_CHGUP_END, the live up-order-after-end counter, six |
| compete_fight_count | dynamic FEAT_FIGHT, the duel fight counter, `>` only (the retail filter table) |
| activate_count_heal | dynamic FEAT_HEAL, the heal-type activation count, `>=` only |
| is_behind_in | dynamic FEAT_BHIN, the sign of the lane gap to the behind runner; last place reports `None` (SK_NAN) so every comparison fails |
| distance_diff_top | dynamic FEAT_DDTOP, `floor` of the f32 gap to the leader, using my own position while I lead, six |
| running_style_count_nige/senko/sashi/oikomi_otherself | roster fold: the count of other runners in that style group (compiler `Statics.other_style`), six |
| order_rate_in20/40/50/80_continue | dynamic FEAT_INBAND, `==1` only, the rate in the name picks the in-band cell and rides in the comparison argument |
| near_infront_count | dynamic FEAT_NEARIF, live scan: `0 < gap <= 3.0` and `|lane gap| <= f64(0.16666f)` in f64 gaps |
| visiblehorse | dynamic FEAT_VISN, the lane cache's visible count (`SI_L_VISN`) |
| is_used_skill_id | dynamic FEAT_USED, `==` only: true once the runner has activated that id; an id no roster horse carries folds FALSE at compile time |
| is_activate_heal_skill | `==` only: arg 1 lowers to `FEAT_HEAL >= 1` and any other arg to `FEAT_HEAL == 0`, the engine's boolean filter |
| is_activate_other_skill_detail | FALSE fold: the engine's check reads its own id from `used_skills` and `used_skills` takes only self-fired ids, so the record's first fire is the only writer and the same gate blocks it (dead in the table, battery-proven never fired on the CPU) |
| activate_count_all | dynamic FEAT_ACTALL, the sum of the three phase counters, `<=,>=` only |
| activate_count_end_after | dynamic FEAT_ACT2, the phase-2 counter, `>=` only |
| activate_count_later_half | dynamic FEAT_ACTLH, fires past `distance/2`, `>=` only |
| overtake_target_no_order_up_time | dynamic FEAT_NOTOP, `floor` of the f32-into-f64 clock armed while the runner has lane targets, six |
| overtake_target_no_order_down_time | dynamic FEAT_NOTOD, same clock on the down side, six |
| change_order_up_middle | dynamic FEAT_CHGUP_MID, passes taken in phase 1, six |
| change_order_up_finalcorner_after | dynamic FEAT_CHGUP_FC, passes taken while the tick-start final-corner flag ran, six |
| blocked_all_continuetime | dynamic FEAT_BALL, `trunc(min(front, side))` of the two block clocks, six |
| blocked_front | dynamic FEAT_BFRONTB: the live front-blocker scan at evaluation time (`lane_blocking_sample`), not the field-update stored sample: the engine's getter rebuilds from live solver rows |
| is_surrounded | dynamic FEAT_SURR: out, front and behind membership in one scan (f32-rounded gaps, 0.16666/0.08333 lane tolerances), six |
| temptation_count_behind, temptation_count_infront | dynamic FEAT_TEMPT_B/F, published-order slices behind and ahead, counting kakari rows, six |
| temptation_opponent_count_behind, temptation_opponent_count_infront | same leaves, gated to all-zero-team rosters: `is_opponent` is identity there, a team in the roster refuses by name |
| running_style_temptation_count_nige/senko/sashi/oikomi | dynamic FEAT_RTEMP 58 to 61, the group in the low two id bits: kakari rows of that group including self, six |
| running_style_temptation_opponent_count_nige/senko/sashi/oikomi | dynamic FEAT_RTEMP_OPP 62 to 65, the same rows with self dropped like the engine's `opponents_only` arm, all-zero-team gate as the pair above |
| distance_diff_rate | dynamic FEAT_DDRATE, the ranked endpoints with self substitution, the f32 `((first-own)/spread)*100` floored, an approximately-zero spread returning exactly 0 |
| distance_diff_top_float | dynamic FEAT_DDTOPF, `floor(gap*10)/10` in f32 from the leader, six |
| running_style_count_same | roster fold: the runner's own base-style count including self, six (`roster_condition`, refuses without a parsable strategy) |
| running_style_count_same_rate | roster fold: the same count against the f32 `+0.5`-truncation threshold `arg% * n`, six; an empty field (`n<=0`) folds false |
| running_style_equal_popularity_one | roster fold: true when exactly one horse holds `popularity == 1` and its strategy group matches the runner's; no unique favorite folds false for every comparator |
| same_skill_horse_count | roster fold: horses whose skill list holds the record's id, six |
| remain_distance_viewer_id | dynamic FEAT_VREMAIN, course distance minus `floor` of the viewer row's position, the viewer from the request `playerHorseIndex` (default 0) through the `SK_VIDX` header word, six; an out-of-range or non-integer present value fails the engine's spec parse (`lib.rs:292-300`), so it refuses, and the leaf defers the record out of the constructor pass |

The out-band cells live in race scratch (`SI_BND20/40/50/70` per runner),
updated every tick in `field_update` from the previous-tick rank once the
clock passes 5, one cell per rate: unset becomes set while the rank stays at
or over `round(rate*n)`, and a tick below the threshold latches the cell
false forever (the engine's NaN latch, `race_field.rs:417-448`). The kernel
check mirrors the retail one: cell set means true, and only `==1` passes
(`order_out_filter`, `activation_conditions.rs:1562`).

The in-band cells (`order_rate_in*_continue`) live beside them in race scratch
(`SI_BNDIN20/40/50/80`). `track_order_bands` updates both directions on every
rank pass (`race_field.rs:239-246`), so the in-cells are live, not inert: the
cell tracks the pre-publish rank, unset becomes the armed timer value while the
rank stays at or under `round(rate*n/100)` (thresholds index 0, 1, 2, 4 for
20/40/50/80), a pass over the threshold latches NaN, the first five clock
seconds are skipped, and only the armed state reaches the kernel as true
(`order_in_filter`, `activation_conditions.rs:1604-1607`).

The activation history rides per-runner scratch counters bumped in
`skills_fire` at the engine's `apply_activation` sites (`race_solver.rs:2662-2669`):
`SI_ACT0/1/2` per phase, `SI_ACTLH` for a fire at or past `distance/2`, and a
per-record used bit (`SKR_USED`) set on the record's first fire, mirroring the
unconditional `used_skills` insert. The engine's follow-up deliveries return
before that insert, so the used set holds exactly the self-fired ids.

The no-order pair (`SD_NOUP/SD_NODOWN`) accumulates in the movement loop beside
`SD_OTIME` on the same f32-into-f64 step, armed by the runner's own lane target
count, and `field_update` zeroes the up timer on a rank-gain event and the down
timer on any rank drop, before the frozen-past-line guard skips further
resets (`race_field.rs:258-276`). `FEAT_CHGUP_MID` and `FEAT_CHGUP_FC` count
the same event pass into the tick-start phase and the tick-start final-corner
flag (`race_solver.rs:1755-1774`).

`is_activate_any_skill` models the engine's filter
(`activation_conditions.rs:663-703`) and its `observe_anyskill`
(`race_solver.rs:1883-1899`). The runner carries one cumulative activation
counter (`SI_ACTALL`), bumped by every skill fire. A record flagged
`SK_FLAG_ANYACT` observes it at the top of every scan visit, before the
removal and gate passes, exactly where the engine observes: when the record
is checkable (cooldown ready, precondition satisfied), the value is the
count minus the activations already counted, true on a visit that saw the
count rise. A non-checkable visit reports false without touching the
per-record state. The per-record slots are `SKD_ANY_LAST/TRUES/RESULT`.

One sampled term per record is required. A second random term in one condition flags
the race (`skill {sid} has more than one random term`), and a random name outside the
samplers above refuses by name.

Not yet ported (fallback-flagged, plan order in the README):

- no engine name behind them, so the registry itself cannot answer:
  `behind_time`, `near_time` and `infront_time` (the nearest entries are the
  ported `*_near_lane_time` timers)
- `is_other_character_activate_advantage_skill` refuses on equivalence: the
  engine reads a field-wide previous-tick `(runner, kind)` advantage pair set
  with same-team exclusion, swapped at every field update
  (`race_field.rs:373-398`), and no per-runner kernel flag mirrors it
- precondition `is_activate_any_skill`: the engine drops precondition nodes
  before the observe map is built (`race_solver.rs:551`), so the flag never
  becomes readable and the compiler refuses it by name
- `base_speed`, `base_stamina`, `base_guts` and `base_wiz` (the sibling base
  stat reads, no consumer yet), `course_distance`, `furlong`
- effects outside the accepted set 1, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 21,
  22, 27, 28, 29, 31, 35, 36, 37 and 38 (`skills_compile.rs` refuses the rest
  by kind): `32` AllStatusUp (a mid-race all-stat bump with no kernel consumer)
  and `33` and `34` (no `SkillType` variant: the retail builder errors, the
  no-ffi path degrades to Noop). `36` and `37` (random skill activation,
  `race_solver.rs:2677-2737`) are accepted at self target only.
- effect targets outside 1, 4, 9, 10 and 21, and target `21` only with kind
  `13` (the rushing strategy-group receiver set). `10`
  `MultiplyStartDelay` at another runner fires inert in retail (the receiver
  keeps its init-applied gate) and is accepted. The same-team debuff drop is a
  real delivery gate now, not a roster flag: each runner's fixed team id reaches
  the kernel (`RR_TEAM` into `SI_TEAM`) and `skills_fire` drops same-team
  receivers off the capped receiver list, matching `race_field.rs:458-461`.
  The immunity arm of that filter reads the `SD_IMMUNE` presence cell.
- a self `GutsUp(4)` or `WisdomUp(5)` whose arms are not all gated false at
  time zero (an ungated arm short-circuits the field leaves and can fire in
  the constructor pass, ahead of the init draws the kernel bakes), and more
  than 8 effects on one record (the per-record token cap the receiver orders
  index)
- structural: `value_usage` with no arm in the engine's tier fold (0, 15 and up,
  and negatives: the builder and the runtime both pass the raw modifier through,
  so the kernel does the same. Usages 3 to 7, 10, 12 and 13 fold at compile from
  request-static data, `race_solver_builder.rs:908-939`. Usages 8 and 9 draw
  `RetailRandom` at each application and usages 11 and 14 read runtime counters,
  `race_solver.rs:2485-2503`),
  `duration_usage` 4 on kinds 11, 12, 35 or 38 (the overtake extension walks the
  aux entry those kinds register per fire, and the kernel keeps one presence
  cell per runner for them),
  `additional_activate` on a kind outside 21, 22, 27, 28 and 31, or on those
  kinds at a target outside 1, 4, 9 and 10 (a follow-up registers one token
  holding one value, and the other kinds and targets write per-receiver state),
  a `36` or `37` effect at another runner,
  rarity outside 1..6 (3, 4 and 5 all fold to 3), priority outside {0,1}, the
  legacy non-retail wisdom path

Multi-alternative skills compile one record per alternative. An alternative
whose region chain folds empty (a false precondition or condition) stays as a
draw-only dead record: the engine samples every alternative's random terms and
rolls wisdom once per runner and skill id, so the first emitted record of the
skill carries the roll (`SK_FLAG_WISDOM`) and the rest inherit it
(`SK_FLAG_WISDOM_FOLLOW`). A later live alternative with the same condition
text, a matching precondition signature and no sampled term is shadowed by
the first live one (shared cooldown key, scanned first, identical gate), gets
no record, and only its start-delay factor folds into the first record's
multiplier. A second live alternative with a different gate lowers as its own
record too: every alternative shares the skill cooldown key
(`race_solver_builder.rs:589`), so the fire of one parks the key for the rest.
The kernel models the key at fire time: the fire fans its parked state and
both timers over the ready siblings of the (runner, skill id) group, where a
READY record and a COOLDOWN record whose timer already ran up both read as
ready, the same pair the engine's single key map reports
(`race_solver.rs:2371-2435`). WAITING siblings hold their own running
instance, and DEAD or DISABLED siblings left the engine's pending list before
the key was consulted; none of them inherits the park.
`accumulatetime`-style precondition terms lower to a precondition program and
region list the gate checks before the condition, mirroring the engine's
`pre_condition` pass.

Activation-time values and turns. `sk_vu_factor` (`retail_core.cl`) maps each
effect's `value_usage` at the moment the engine reads it, inside the effect loop
of `skills_fire` and again for every additionalActivate application: 8 and 9
draw the paired-run or shared bank the fire already uses, 11 reads `SI_LCO`
(late-corner overtakes, bumped in `field_update` off the rank-neighbor flip),
14 reads `SI_GREEN` (distinct activated green-tag ids, bumped once per id at
fire). The applied value is staged per token (`sk_tok_stage` into `SK_VVAL`)
because the pace gate, the target column and the physics chains all read the
engine's active-skill entry, never the wire modifier.

additionalActivate defers: a counted slot contributes to the instance duration
but not to the fire, the fire registers the record in the runner's running list
(`SK_RUN`, 8 slots), and every later activation or overtake files pending turns
(`SI_AANU` counts the holders). At each record-id boundary of the priority-0
scan, and once more per runner at the pass end, `aa_turns` delivers up to
`min(count, cap - delivered)` applications, cap 3 for kind 2 slots and 2 for
kind 3, and each application allocates a pool token (`SK_AAP`, 128 entries)
whose value and timer ride the same per-kind orders. Each application re-runs the
target filter, the immunity skip and the same-team drop, so a fan at target 4, 9
or 10 lands on the runners the engine picks at that moment. `aa_retire`
reproduces the
engine's deferral: an entry past its timer and past the id floor retires unless
a later unretired entry owns a kind-2 or kind-3 slot.

The running instance keeps its own timer. A `duration_usage` 4 fire extends
`SKD_RUN_T` by `course.distance / 1000`, at most three times, once per overtake
(`race_solver.rs:1824-1839`), and the retire sweep, the kind-1 event, the kind-2
and kind-3 turns and the kind-2 queue gate read that extended value, matching
the engine's `RunningSkill.timer`. `SKD_INSTANCE` stays the
`random_active_details` countdown, which the extension never reaches, so the
flip gate reads it. The entry walk runs in the orders of the runner whose
overtake it is and skips an effect whose target is not self, since a copy
delivered to another runner carries the Other perspective. A sibling the shared
cooldown key parked copies the firing record's instance, and the walk keeps that
copy in step with the extension: the flip for a parked sibling waits for the
firing record's running entry, not only its `random_active_details` entry
(`race_solver.rs:2404-2444`).

Kinds 36 and 37 only file a request (`SD_RANY`/`SD_RGOLD`). `aa_select` at the
end of each priority-0 pass rebuilds the ordered candidate list, draws one index
per requested pick from the same bank, and marks the winners' records; the next
visit of a marked record fires with the arms, the windows and the extra
condition bypassed, and the run-skip fast path stays off while any reservation
is live. A wisdom-failed retained primary is a candidate and can fire exactly
this way (`race_solver_builder.rs:1002-1005`), and `SKR_WISFAIL` sends it back
to DISABLED rather than READY when its cooldown expires.

Mid-race stat fires. A self fire writes the accumulator and the clamped
field and nothing else: the engine's self path registers its timers and
returns without a refresh (`race_solver.rs:2644-2674`). A cross delivery is
one `apply_external_effect` per (effect, receiver) with the effect rewritten
to the self target (`race_field.rs:465-484`), and the follow-up path runs
`refresh_stat_effects` once per delivery (`race_solver.rs:2644-2648`), so
the kernel does both per receiver inside the delivery loop. Of the refresh's
five writes (`race_solver.rs:1330-1340`) three are live in the retail loop:
the target base [0..2] flips to the legacy f64 formula (`race_solver.rs:43-60`)
reading the clamped speed field, which replaces the retail-form values the
enable baked (`race_solver_retail.rs:647-655`) for the rest of the race, the
HP goes through the init-plus-spend re-round (one f32 round trip of
`max_hp - spent`, so each delivery re-rounds exactly as the engine does, and
an already round value re-rounds to itself), and the hp context recaptures
`retail.wisdom`, the spurt plan's capture, from the live field. The
`min_speed` and legacy `last_spurt_speed` rewrites are dead: the step head
recomputes the retail minimum and the phase-2 spurt before any consumer can
read them (`race_solver_retail.rs:961,1007-1034`), and the only pre-phase-2
reader of the spurt speed is the staminasyoubu hook's region-gated extra
condition (`race_solver_builder.rs:819-843`), which is not in the ported
set. `base_accel` is dead too: the retail acceleration chain recomputes from
power, untouched by kinds 1 and 5. `sk_stat_refresh` (`retail_core.cl`)
mirrors the three live writes. Cross SpeedUp additionally repatches the
cached spurt maximum from the bumped speed, the value the engine's plan
closure recomputes from the live field at the next spurt decision.
GutsUp, self (behind the same phase gate as WisdomUp) or cross, moves the
receiver's `SD_HPG_A` accumulator and `SD_HPG` field, seeded by the
constructor bump through `PRD_GUTS_ACC` exactly like the wisdom pair. Any
such fire sets the plan's `live_guts` flag, which prepare bakes into the
race word `PD_LIVE_GUTS`, and the guts-derived `powf` terms then recompute
from the field at the points the engine reads them: the spurt plan closure
`powf(bounded_stat(guts) * spurt[2], spurt[3])` (`retail_physics.rs:362`)
repatches `SF_SPURT_MAX` at each bump, the duel entry (`race_solver.rs:3194-3207`)
snapshots both modifiers into `SD_DUEL_T` and `SD_DUEL_A`, the spot entry
(`race_solver.rs:2928-2939`) snapshots its modifier into `SD_SPOT_MOD` and
its duration into the timer, and the step-head minimum recomputes
`SF_BMIN_VAL`. Each recomputation uses the bit-exact ucrtbase ports `gs_pow`
(f64) and `gs_powf` (f32) from `powf_core.cl`, gated against `f64::powf` and
`f32::powf` at 419.4M pairs with zero bit differences (`wave8_powf_gate.rs`).
Without any live guts fire the flag stays zero and every term keeps the
prepare-baked word, so a guts-static race is unchanged. The self WisdomUp
phase gate (`skills_compile.rs`, `phase_gated`): phase zero is below every
accepted gate, so the record cannot fire in the engine's constructor pass,
where the field move would land ahead of the kernel's baked init draws and
section rolls. The live field reads are the
pace context and the downhill draw (`race_solver_retail.rs:899` and `1118`
read `horse.wisdom as f32` per tick, and the kernel reads the `SD_HPW`
column there), while the plan capture stays on `SF_HPW` until a refresh.

Rule: an unsupported construct is refused by the compiler before upload, the
host flags the whole race for CPU fallback, and the flag reason names the
construct (for example `'lane_type' is not ported`, or `skill 200333
duration_usage 4 extends type 12's per-fire aux entry, which the kernel keeps as
one presence cell per runner`). The kernel never sees a table it cannot run.

One limit is measured rather than refused: a cross-target `additionalActivate`
fan that several hosts refan at once drifts off the engine, on two of the three
seeds the scene runs and from tick 525 on, while the single-fire fan shapes stay
bit-exact. See `fan_churn_tracks_the_engine` in `tests/wave9_gate.rs`, kept
ignored as a standing reproduction until the drift is pinned.

## Bounds (enforced host-side before launch)

`nrec<=960` (8 record slots per runner at the 120-horse grid cap), `neff<=192`, 8 effects per record, `nprog<=3072`, `narm<=256` and two arms per record,
`ntag<=4096`, `nstr<=12`, `ncor<=12`, `nslope<=16`, at most 2048 regions per
race and 16 per arm chain, terms at most 160 per race, one sampled term per
record, 4 sampled windows per term, 12 receiver entries per runner across the
three lists, 64 active effect entries per race, 40 receiver tokens per
runner and kind, 8 running additionalActivate entries per runner, 128 pool
tokens per race, and 16 candidates per reservation pass (an overflow raises the
`FB_SKILL_SEQ` fallback, not a wrong
fold). Overflow
is a host `flagged("skill table overflow ...")`, never a kernel fault.
