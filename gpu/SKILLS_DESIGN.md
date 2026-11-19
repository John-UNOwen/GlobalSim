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
  leaf is `order_rate`, `overtake_target_time`, `distance_diff_top`,
  `change_order_up_end_after`, `lane_type`, `is_behind_in`, `is_move_lane` or
  `blocked_front_continuetime`. The pre-loop scan evaluates such a record's
  gate and state normally but drops the fire; the loop scans fire normally.
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

Header words 0 to 8: `nrec, nprog, nreg, neff, ntag, nstr, ncor, nslope, nterm`.

Sections (word offsets inside the block):

| area | offset | stride | count | fields |
|---|---|---|---|---|
| straights | 16 | 2 | 12 | f64 start, end (course JSON, sorted) |
| corners | 40 | 2 | 12 | f64 start, start+length |
| slopes | 64 | 3 | 16 | f32 start, f32 end, f32 value |
| records | 112 | 26 | 160 | see below |
| effects | 4272 | 10 | 192 | see below |
| programs | 6192 | 4 | 3072 | see below |
| arms | 18480 | 8 | 256 | see below |
| regions | 20528 | 4 | 2048 | f64 start, end ([start,end), compiler statics) |
| terms | 28720 | 8 | 160 | see below |
| tags | 30000 | 1 | 4096 | tag bytes for substream hashing |

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
constructor pass, see the initialization note). `delay_bits` is the f32 start-delay multiplier, the folded product
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
6 phase_laterhalf_random, 7 is_finalcorner_random), and `corner_idx` selects
the qualifying final corner for id 7. The sampler runs at init on the shared
stream (solo) or a keyed substream (paired, tag `T{sid}:{alt}:condition:{name}`)
and produces the sampled windows. The init kernel intersects them with the
record's static region list in engine `rmap_*` order and writes the final
windows, trigger and single flag into scratch. The runtime checks a sampled
record the same way as any other: the window gate reads the final windows.

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
- `GutsUp(4)`, `WisdomUp(5)` and `ModifyKakariChance(29)`: portable only as
  construction-time bumps, the host applies them to the init inputs (stats
  through the `add_stat_modifier` f32 chain, kakari chance folded into the
  init roll). A mid-race fire of 4 or 5 refuses.
- `RunAway(6)`: the runtime arm is SkillAbilityNull, so the fire is inert
  beyond its event and instance bookkeeping. The build-time rule that matters
  is elsewhere: a Nige that learned a type 6 effect races as Oonige
  (`race_solver_builder.rs:858-865`), and the host prepass applies the same
  upgrade before anything reads the strategy.
- `SetStartDelay(14)`: writes the solver's `start_delay`, which the retail
  physics never reads (the retail start delay resolved in initialization), so
  the kernel fires it inert.

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
zero) while still counting the fire's runtime duration, event and cooldown.

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
the out-band cells (`order_rate_out*_continue`) and `is_activate_any_skill`, and
the wave-2 live fields `lane_type`, `is_move_lane`, `is_behind_in`,
`distance_diff_top`, `overtake_target_time`, `change_order_up_end_after`,
`compete_fight_count`. A condition whose
comparison is not registered for its name falls to the `classify` catch-all
and flags the race.

Fold columns: `six` is `==,!=,<,<=,>,>=`, the engine's six value_filter
comparators (`activation_conditions.rs:229-243`).

| name | fold |
|---|---|
| always | TRUE |
| phase | dynamic FEAT_PHASE |
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
| popularity | horse `popularity` (default 0), six |
| running_style | horse strategy group, `==` only |
| all_corner_random, straight_random, down_slope_random, up_slope_random | sampled term (`==1`), init-time windows through the scan gate |
| phase_random, phase_firsthalf_random, phase_laterhalf_random | sampled term (`==0..4` phase index), window from the six-way phase split |
| is_finalcorner_random | sampled term, window over the qualifying final corner (an id >= 4 starting within the last 1000 m), else no draw and the record stays inert |
| distance_rate_after_random | sampled term (`==`, argument >= 0), init windows like the other samplers |
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

The out-band cells live in race scratch (`SI_BND20/40/50/70` per runner),
updated every tick in `field_update` from the previous-tick rank once the
clock passes 5, one cell per rate: unset becomes set while the rank stays at
or over `round(rate*n)`, and a tick below the threshold latches the cell
false forever (the engine's NaN latch, `race_field.rs:417-448`). The kernel
check mirrors the retail one: cell set means true, and only `==1` passes
(`order_out_filter`, `activation_conditions.rs:1562`).

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
eight samplers above refuses by name.

Not yet ported (fallback-flagged, plan order in the README):

- stateful machines beyond the set above: `behind_time`, `near_time`,
  `infront_time`, `blocked_*` beyond `blocked_side_continuetime` and
  `blocked_front_continuetime`,
  `near_infront_count`, `visiblehorse`,
  `change_order_*` beyond `change_order_onetime` and `change_order_up_end_after`
- `is_used_skill_id`, `is_activate_*` beyond `is_activate_any_skill`
  (each registers its own skill-id argument)
- precondition `is_activate_any_skill`: the engine drops precondition nodes
  before the observe map is built (`race_solver.rs:551`), so the flag never
  becomes readable and the compiler refuses it by name
- in-band cells: `order_rate_in*_continue` (never inserted into the retail
  `order_band` map, so the engine's filter always errors)
- roster reads beyond rank, rank rate and the two bashin gaps:
  `same_style_rate`, `distance_diff_*` beyond `distance_diff_top`
- `base_*` (base stat reads), `motivation`, `grade`, `time`, `post_number`,
  `course_distance`, `furlong`
- effects outside the accepted set 1, 4, 5, 6, 8, 9, 10, 14, 21, 22, 27, 28,
  29 and 31 (`skills_compile.rs` refuses the rest by kind): `13` ExtendKakari,
  `35`, `11`, `12` lane goals, `32` AllStatusUp, `36` and `37` random skill
  activation, `38` debuff immunity
- effect targets outside 1, 4, 9 and 10, kinds 1, 4, 5 and 10 aimed at another
  runner, and a cross debuff (negative modifier on a targeted queued kind) on
  a roster with any team id set: the engine drops same-team receivers, which
  only folds to identity on all-zero teams
- a mid-race (non-constructor) fire of `GutsUp(4)` or `WisdomUp(5)`, and more
  than 8 effects on one record (the per-record token cap the receiver orders
  index)
- structural: two live alternatives whose condition or precondition differs
  (refused as `skill {sid} has other than one alternative`, see below),
  `value_usage` 3 to 14 (stat, win and fan tiers, the activation rolls, the
  late-corner count and the green-tag count, the modifier rescaled from state
  the numeric table cannot carry),
  `duration_usage` 4 (the delayed running-instance path),
  `additional_activate` not 0,
  rarity outside 1..6 (3, 4 and 5 all fold to 3), priority outside {0,1}, debuff
  immunity, the legacy non-retail wisdom path, `isGeneralSkill` gates

Multi-alternative skills compile one record per alternative. An alternative
whose region chain folds empty (a false precondition or condition) stays as a
draw-only dead record: the engine samples every alternative's random terms and
rolls wisdom once per runner and skill id, so the first emitted record of the
skill carries the roll (`SK_FLAG_WISDOM`) and the rest inherit it
(`SK_FLAG_WISDOM_FOLLOW`). A later live alternative with the same condition
text, a matching precondition signature and no sampled term is shadowed by
the first live one (shared cooldown key, scanned first, identical gate), gets
no record, and only its start-delay factor folds into the first record's
multiplier. Any other second live alternative is refused by name.
`accumulatetime`-style precondition terms lower to a precondition program and
region list the gate checks before the condition, mirroring the engine's
`pre_condition` pass.

Rule: an unsupported construct is refused by the compiler before upload, the
host flags the whole race for CPU fallback, and the flag reason names the
construct (for example `'lane_type' is not ported`, or `skill 200333 effect
duration_usage 4 (delayed running instance) is not ported`). The kernel never
sees a table it cannot run.

## Bounds (enforced host-side before launch)

`nrec<=160`, `neff<=192`, 8 effects per record, `nprog<=3072`, `narm<=256` and two arms per record,
`ntag<=4096`, `nstr<=12`, `ncor<=12`, `nslope<=16`, at most 2048 regions per
race and 16 per arm chain, terms at most 160 per race, one sampled term per
record, 4 sampled windows per term, 12 receiver entries per runner across the
three lists, 64 active effect entries per race, and 40 receiver tokens per
runner and kind (an overflow raises the `FB_SKILL_SEQ` fallback, not a wrong
fold). Overflow
is a host `flagged("skill table overflow ...")`, never a kernel fault.
