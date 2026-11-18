# Field port scoping

Function-by-function call graph of what a multi-runner retail race adds over the
solo loop, with the kernel decision per function. The engine tick driver is
`RaceField::step_all`, which calls `step_all_retail(dt, visited = all 0..n)`
once per tick (the retail branch in `src/lib.rs`), then finish detection and the
frame capture.

## Per-tick call graph

1. `rows` and `roster` collection (`race_field_retail.rs:182-220`).
   rows per runner: `(pos f32, lane f32, on the final straight)`, pre-movement.
   roster per runner: `(pos, lane, observed_speed, target_speed)`. The
   `resort_by_position` result feeds only the sorted blocking and sorted lane
   paths, which the kernel replaced with the indexed `lane_blocking_sample` and
   `lane_collect` variants (the engine test build asserts the equality). The SI,
   SF and SD columns already carry every row field. New: the straight flag
   (from PC_LAST_STRAIGHT_START/END and the corners-present gate) and
   `any_straight` for the duel pass.

2. `collect_retail_duel_proximity` (`race_solver_retail.rs:576-605`), per runner.
   It snapshots `duel_preupdate_speed`, `duel_preupdate_finished` and
   `duel_preupdate_final_straight`, then maintains the per-runner proximity
   timer map. A pair qualifies while `|dpos| <= duel_distance && |dlane| <=
   duel_lane && (straight || other_straight)`. An existing timer advances as
   `(old as f32 + dt) as f64`, a new entry starts at 0.0, and a tick with no
   runner on the final straight clears every map. No RNG. New in the kernel: a
   per-runner per-peer timer column with a -1.0 absent sentinel, three snapshot
   flags and a goal-crossing flag.

3. `lane.collect_roster` (`retail_lane.rs`) is `lane_collect`, already ported.
   It works for n > 1 unchanged.

4. `prepare_retail_tick` (`race_solver_retail.rs:807-857`).
   New for the field: the timer loop `timer.set((timer.get() as f32 + dt) as
   f64)` advances `spot_struggle_timer`, observable through the spot state
   machine and the token removal in step 6. The other timers (accumulatetime,
   compete_spurt, secure_lead, duel_timer, skills) feed only skill state. The
   corner state read (`geo_corner_state`) is needed now, because the duel entry
   test reads `duel_preupdate_finished && final_corner && corner == 0`. Skill
   expiry and activation stay the host-flagged fallback domain.

5. `blocking_sample` and `update_retail_blocking`. The front index, gap and
   side flag are ported (`lane_blocking_sample`). The blocked-time counters
   feed only skill conditions. Unchanged for multi.

6. `step_forward_retail_with_roster` (`race_solver_retail.rs:976-1338`), per
   runner in roster order against the live roster, where earlier runners have
   already moved. The multi-sensitive parts:
   - `update_retail_pace` to `pace_update` (ported): real rank, real pacemaker
     (`field.pacemaker()`, the section pacemaker state machine),
     `pacemaker_rank`, signed `pacemaker_gap` (f64 difference rounded to f32),
     `later_style_ahead`, `distance_to_behind` over the rank order, and
     `has_speed_skill`, which is true while a spot or duel token is
     registered. It draws the pace lotteries, so every input must match.
   - lane `update_targets` and `target_sorted_unchecked` are ported
     (`lane_update_targets`, `lane_passing_target`, `lane_target`). They own
     the outward lottery draw and work for n > 1 unchanged.
   - `rushing.step`, spurt and downhill read no field state. The temptation
     counts belong to the non-retail path. The bank order already matches.
   - `spot_struggle_active && timer >= 0` removes target token 1 before the
     target chain. New: the spot token bit.
   - The target chain `target_speed_with_late_terms_iter` gets
     `mods = values_except_tokens([1, 4])`, empty without skills, plus the late
     terms `speed += spot_token; speed += duel_token;` after the slope
     correction (`retail_physics.rs:396-397`). New: two f32 arguments in
     `phys_target_speed`.
   - The blocking cap reads the blocker's live observed speed, already ported.
   - `MoveState::step` takes `rank = field.order_of(i)`, a real rank now.
   - The accel chain `values_except_tokens([cp, 4]).chain(value(4))` is
     `[duel_accel]` while the duel token is up, else empty. The cp token
     exists only for the asitame skill, and `release_conserve_power_now`
     returns early without the pending asitame skill, so conserve power is
     inert and unobserved without skills.
   - The hp context `is_compete_top: spot_struggle_active && timer < 0`
     switches the hp drain rate index, so the spot state must exist.
   - First goal crossing (`pos > distance`) also sets
     `duel_goal_crossing_snapshot`. New: one flag.
   - elapsed is dt unless the start-delay gate rests the runner. Gated runners
     skip the passes below.

7. `resolve_overlaps_with_order` (`retail_lane.rs:913`).
   comparison_sort (introsort) on the `trunc(lane)` key, whose equal-key
   permutation matters, then the outward `0.4 / 18` shifts. The kernel uses a
   stable insertion sort today, so it gets a faithful introsort port
   (`ro_sort`, mirroring the `sort_candidates` structure).

8. `prepare_retail_duel_start`, `latch_active_spot_cutoff` and
   `step_field_conditions(elapsed)` run for every advanced runner
   (`race_field_retail.rs:325-341`), reading the post-movement live field with
   the pre-tick snapshots:
   - `update_dueling` entry (`race_solver.rs:3061-3192`): the final-straight
     gate, the proximity map walk in ascending index, and `qualifies`, which
     needs both hp ratios at least 0.15, a preupdate speed gap at most the f32
     0.6 compared at f64, and the rank half gate. ready picks the longest timer
     with a strict `>`, target is ready, else eligible, else the first map
     key. Activation sets both duel flags, takes the group as the peer group,
     else the own group, else `min(i, j)`, and registers target token 4 and
     accel token 4 with the guts-derived modifiers. The duel modifiers are race
     constants (powf over guts and the CompeteFight coefficients), so the host
     precomputes them, following the PT_POWF pattern. Neither exit nor
     activation draws RNG.
   - Duel exit (`update_dueling` active branch): any peer with
     `fight_count > 0`, the same group and `|dpos| <= 5.0` (live f64 position,
     finishers included in retail) keeps the duel alive. hp ratio under 0.05
     or no peer ends it and removes token 4 from both chains.
   - `latch_active_spot_cutoff` and `update_spot_struggle`
     (`race_solver.rs:2788-2926`): the Nige/Oonige state machine, with the 5.0
     m and 0.416 lane distance exit, the cutoff latch at section 8, and group
     entry that needs two same-strategy runners whose cached-order group front
     leads the eligible field, with one participant before section 6 and a
     3.75 m and 0.165 lane invitation window. The timer is set to
     `-((700 * guts)^0.5 * 0.012 * strat_prof[stapt])` and token 1 registers.
     Ported in full, no RNG, unreachable with the all-Senkou gate rosters.
   - `update_near_lane_timers` feeds only skill conditions, skipped.

9. `field.update()` (`race_field.rs:205-287`), once per tick after the duel
   and spot passes.
   `prev_rank` copies `rank`. A stable insertion sort descending by position
   over the carried order (ties keep the carried order) yields `rank_of` and
   `order`. `style_ahead` collects prefix maxima and `update_pacemaker` runs the
   full section state machine (`race_field.rs:753-843`) on the previous order
   with live positions: section from the overall leader, the forward-style
   group, the handover count and range rules, and the restoration branch. Pace
   and the duel and spot order queries consume the result. The constructor runs
   the same update once before tick 0 (all positions 0, so rank = index + 1,
   order = 0..n, and the pacemaker is the first forward-style runner), and the
   kernel reproduces that state instead of the solo rank-1 hack. The overtake
   cooldown block (`race_field.rs:258-277`) only mutates overtake timers and
   the response overtake count, invisible in frames and finish times, skipped
   with a note. `track_order_bands` and the blocked-time counters likewise.

## RNG bank

The duel and spot paths draw nothing. The bank keeps the existing per-race
shared stream consumed in roster order. The pace, rush, spurt, downhill and
outward-lane draws gain real field inputs to their branch conditions, which is
exactly what must match to stay bit-exact.

## Host-side additions

`prepare()` parses the CompeteFight group (DistanceGap, LaneGap,
AddParam1/2Coef1-3) into new pd slots. The stats pre-pass already runs one case
per runner, so each runner's spurt powf bits, duel speed and accel modifiers
(`((guts as f32 * c0) as f64).powf(c1 as f64) as f32 * c2`) and spot modifier
and duration (`(guts as f32 * 500.0)^0.6_f32 as f32 * 0.0001` and `(700 *
guts)^0.5 * 0.012 * strat_prof[stapt]`) move from the single PT_POWF slot to
per-runner slots. A missing CompeteFight or CompeteTop group keeps the host
fallback, since the engine also fails to build the retail config. Roster size
stays capped at RACE_MAXR = 24, asserted on the host and in the kernel. The
gate races carry 9 runners.

## Fallback domains after the port

Skills (host flag), paired random streams, startup mode, init and geometry
failures, and roster size above 24. Spot struggle needs two same-strategy
Nige/Oonige runners, unreachable with the gate rosters, but ported so such
rosters run unflagged.
