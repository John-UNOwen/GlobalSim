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

Initialization mirrors `retail_initialization.rs::initialize()`: the trigger
samplers, the wisdom lottery and the start-delay multiplier run inside
`im_init` (initialization.cl) before the existing rushing and section draws, in
the engine's stream order, so the shared `RetailRandom` substream order stays
bit-exact in solo mode. Paired mode takes the same draws from keyed substreams.

## Table encoding (host to kernel)

One `uint[SK_STRIDE]` block per race, kernel argument `sk`. Every f64 payload
occupies two words (low, high) via bit cast, every f32 payload one word.

Header words 0 to 7: `nrec, nprog, nreg, neff, ntag, nstr, ncor, nslope`.

Sections (word offsets inside the block):

| area | offset | stride | count | fields |
|---|---|---|---|---|
| straights | 16 | 2 | 12 | f64 start, end (course JSON, sorted) |
| corners | 40 | 2 | 12 | f64 start, start+length |
| slopes | 64 | 2 | 16 | f32 start, f32 length (sign carries up or down) |
| records | 112 | 20 | 32 | see below |
| effects | 752 | 10 | 48 | see below |
| programs | 1232 | 4 | 320 | see below |
| regions | 2512 | 4 | 256 | f64 start, end ([start,end), compiler statics) |
| tags | 3536 | 1 | 512 | tag bytes for substream hashing |
| terms | 4048 | 8 | 48 | see below |

Record (20 words): `skill_id, runner, flags, priority, rarity, n_eff,
eff_base, prog_len, prog_base, sampled, term_base, n_static_a, reg_base_a,
cd_lo, cd_hi, wis_off, wis_len, delay_mult, n_static_b, reg_base_b`. The two
static lists hold the engine's two region-fold stages: list A before the
sampled intersection, list B after it. `flags`: bit0 retain-for-random
(detail 0 and rarity White, Gold or Evolution), bit1 wisdom check (at init a
failed roll marks the record always-false when retained and drops it
otherwise), bit2 carries a sampled term, bit3 always-false (the condition
folded false: the record stays in the table and the kernel skips it). The
compiler resolves the level before the kernel runs: the effect stores the
level-resolved modifier `retail_skill_value(raw, multiplier)` and
`baseDuration * 0.0001`, so the kernel does no table work.

Effect (10 words): `effect_type, target(1 only), value_usage, duration_usage,
0, 0, modifier_lo/hi (f64), base_duration_lo/hi (f64)`.

Program node (4 words): `op, a, b_lo, b_hi`, evaluated postfix on a 64-entry
boolean stack. Opcodes: 0 TRUE (`a` is the value), 1 AND, 2 OR, 4 FEAT_I. The
program carries only the dynamic leaves: a static true or false fold emits a
TRUE node (a region or sampled leaf is TRUE here because it acts through the
window gate), and a construct the compiler cannot lower flags the race
instead of emitting a node. FEAT_I packs `a = feat * 8 + cmp` with the
comparison in engine token order (0 `==`, 1 `!=`, 2 `<`, 3 `<=`, 4 `>`, 5
`>=`), and `b` is the f64 argument.

Term (8 words): `kind, arg, tag_off, tag_len, 0, 0, 0, 0`. `kind` is the
sampler id (0 all_corner_random, 1 straight_random, 2 down_slope_random,
3 up_slope_random). The sampler runs at init on the shared stream (solo) or a
keyed substream (paired, tag `T{sid}:{alt}:{col}:{name}`) and produces the
sampled windows. The init kernel intersects them with the record's static
region list in engine `rmap_*` order and writes the final windows, trigger and
single flag into scratch. The runtime checks a sampled record the same way as
any other: the window gate reads the final windows.

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
`start_expired_skill_cooldowns`, `pending_cooldown_ready`).

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

The lists mirror `AdditiveReceiver`: registration-order folds, in-place token
update on refire, order-preserving token removal at expiry.

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
- Lane and roster target (`retail_target_speed_with_current_skills`,
  `race_solver_retail.rs:740`): `apply_additive(f32(SD_TS), cur_ord_f32s)`,
  recomputed per tick after prepare, because the list state changes between
  ticks. Consumers: the lane owner context, the lane target search and the
  live-horse roster (blocking sample and duel inputs).
- `has_speed_skill` (`race_solver_retail.rs:912`, position keep):
  `pace_cancel > 0 || apply_additive(0., target_ord) > 0`.
- Recovery(9) would feed `pending_hp` (`race_solver_retail.rs:994`), not yet
  ported, see the matrix.

## RNG

No draw touches the race-phase stream: sampled windows and the wisdom roll
happen in initialization only. `r128_substream_hash(seed, runner, tag)` in
`rng_core.cl` hashes arbitrary global tag bytes. Tags are built byte-for-byte
like the engine: `T{skill_id}:{alt}:{pre|cond}:{term}` for sampled terms
(paired mode only, solo mode draws in stream order) and `W{skill_id}` for
wisdom rolls (paired). Chained substreams re-key from the parent state exactly
as `RetailRandom::substream` does (FNV-1a-64 plus avalanche, seed is
`hash >> 33` as i32).

## Condition-support matrix

Engine vocabulary from the `c.insert` list in `activation_conditions.rs`.
Ported means `classify` lowers the name to a static fold or a dynamic leaf.
Any name the match does not handle, or a registered name with an unregistered
comparison, flags the race with the offending construct named.

The compiler lowers each condition term to a `Leaf`. Most leaves are static:
they fold to TRUE/FALSE or to a window list the init kernel intersects, so the
program carries only the six dynamic leaves (`order`, `order_rate`, `hp_per`,
`is_overtake`, `is_temptation`, and the last-straight entry latch). A
condition whose comparison is not registered for its operator falls to the
`classify` catch-all and flags the race.

| name | fold |
|---|---|
| always | TRUE/FALSE (`always==1` true, any other comparison false) |
| phase | static window `[phase_start, phase_end)`, +10 m end for the fudge list; `==,<,<=,>,>=` |
| distance_rate | static window from `distance*arg/100`; `<=,>=` |
| remain_distance | static window from `distance-arg`; `==,<=,>=` |
| furlong | static window `[(arg-1)*201.168, arg*201.168)`; `==` |
| is_finalcorner | static window split at the last corner start; `==0`/`==1` |
| is_last_straight | static window over the last straight, course must have straights; `==1` |
| is_last_straight_onetime | static window `[ls.start, ls.start+10)` plus the dynamic FEAT_LS_ENTRY entry latch; `==1` |
| corner | static windows over the sorted corner list: `!=0` the corner runs, `==0` the complement gaps, `==num` every 4th corner from index `len+num-5` |
| slope | static windows over the sorted slope list: `==0` the gaps, `==1` up slopes, `==2` down slopes |
| order | dynamic FEAT_ORDER, 1-based rank |
| order_rate | dynamic FEAT_ORDER_RATE, rank versus `round(n*rate/100)` |
| hp_per | dynamic FEAT_HP_PER, `trunc(f64(f32(hp)/f32(max))*100)`; `<=,>=` only |
| is_overtake | dynamic FEAT_IS_OVERTAKE, lane target list non-empty |
| is_temptation | dynamic FEAT_IS_TEMPTATION, kakari flag; `==` only |
| all_corner_random, straight_random, down_slope_random, up_slope_random | sampled term (`==1`), init-time windows through the scan gate |

Not yet ported (fallback-flagged, plan order in the README):

- stateful time machines: `accumulatetime`, `behind_time`, `near_time`,
  `infront_time`, `blocked_*`, `temptation_count`, `compete_fight_count`,
  `overtake_target*`, `near_count`, `near_infront_count`,
  `infront_near_lane_time`, `change_order_*`, `visiblehorse`, `lane_type`,
  `is_move_lane`, `lastspurt`
- activation counters and history: `activate_count*`, `is_used_skill_id`,
  `is_activate_*`
- `distance_rate_after_random` (random-term state),
  `order_rate_in*_continue` and `order_rate_out*_continue` (the retail band
  cells need their update point verified)
- roster reads beyond rank and rank rate: `same_style_rate`, `bashin_diff_*`,
  `distance_diff*` (the ahead and behind neighbour gaps are the next step)
- `base_*` (base stat reads), `motivation`
- conditions the `classify` match does not name, so they hit the catch-all and
  flag the race: `is_finalcorner_laterhalf`, the static context values
  (`course_distance`, `distance_type`, `ground_condition`, `ground_type`,
  `grade`, `is_dirtgrade`, `track_id`, `weather`, `season`, `time`,
  `running_style`, `post_number`), and the wider sampled family
  (`corner_random`, `is_finalcorner_random`, `last_straight_random`,
  `phase_straight_random`, `random_lot`)
- effects outside 27, 31 and 21: `Recovery(9)` (application point known),
  `22` natural deceleration (expires into `modifiers_one_frame_accel` first),
  `13` ExtendKakari, `10` and `14` start delay (init already applies them),
  `28` lane move speed, `35`, `11`, `12` lane goals, `32` AllStatusUp, `36`
  and `37` random skill activation
- structural: multi-alternative skills (`alternatives` longer than 1), a
  non-empty `precondition`, `value_usage` in {0,8,9,11,14} (rng draws and
  green-tag counts), `duration_usage` 4 (the delayed running-instance path),
  `target` not 1 (application to other runners), `additional_activate` not 0,
  a repeated effect kind on one alternative (only one duration per kind),
  rarity outside 1..6 (3, 4 and 5 all fold to 3), priority outside {0,1}, debuff
  immunity, the legacy non-retail wisdom path, `isGeneralSkill` gates

Rule: an unsupported construct is refused by the compiler before upload, the
host flags the whole race for CPU fallback, and the flag reason names the
construct (for example `'blocked_time' is not ported`, or `skill 200333 effect
duration_usage 4 (delayed running instance) is not ported`). The kernel never
sees a table it cannot run.

## Bounds (enforced host-side before launch)

`nrec<=32`, `neff<=48`, `nprog<=320`, `ntag<=512`, `nstr<=12`, `ncor<=12`,
`nslope<=16`, at most 256 regions per race and 16 per record or per sampled
split, terms at most 48 per race, one sampled term per record, an
or-condition only when every group is dynamic (mixing region or sampled terms
with an `or` flags the race), 4 sampled windows per term, 12 receiver entries
per runner across the three lists, 64 active effect entries per race. Overflow
is a host `flagged("skill table overflow ...")`, never a kernel fault.
