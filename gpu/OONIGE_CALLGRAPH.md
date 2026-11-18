# Oonige (strategy 5) call graph

What the engine changes when one runner's strategy is Oonige, and where the
race kernel meets it. Engine line refs are the read-only `src/` in this repo.

## At init

- Retail init sees the base style, not 5. `lib.rs:1533` builds the
  initialization horse Value with `running_style = base_running_style()`, so
  an Oonige enters `retail_initialization::initialize` as style 1. The only
  style consumer there is the force-in-move pass
  (`retail_initialization.rs:239-243`, repeated in `initialize_cached` at
  353-358 and 445-450): style 1 takes
  `forceInMoveTargetSpeedAddRunningOut` (0.02, positive) and consumes one
  shared-stream `real(0, forceInMoveTargetSpeedAddRandomRange)` draw before
  the start-delay pass. That draw shifts every later shared-stream draw:
  all runners' delays, all runners' 23 section modifiers, and the rushing
  lotteries. Kernel path: `race.cl` packs the IM_INIT triple from `RR_STYLE`
  (`race.cl:467`), and `initialization.cl:477` skips the draw when the style
  is outside 1..4. `RR_STYLE` must carry the base style.
- The Oonige's own start parameters are otherwise the common ones: same
  delay draw, same rank-driven starting lane, same section modifier draws.
- Initial HP uses strategy 5 raw: `initial_hp` indexes `hp_style[strategy-1]`
  = `HpMaxCoefOoNige` (`retail_physics.rs:404-407`). Kernel `im_hp` already
  takes the raw strategy.

## Per tick

- Target speed and acceleration use the Runaway rows: `phase_speed[4]` =
  `PhaseBaseTargetSpeedPerExArray`, `phase_accel[4]` = `PhaseAccelCoefExArray`
  (`retail_physics.rs:313`, index `strategy-1`). The pd block packs 4 regular
  rows plus the Ex row, so the raw 5 indexes them.
- Start dash (`race_solver_retail.rs:1256`, `:1291`, `:1331-1334`): dash cap
  `start_dash_speed` (MinSpeedRate x course speed), minimum 3 while dashing,
  `+StartAccelAdd` (`accel[3]`) in `acceleration_from_base_iter`
  (`retail_physics.rs:480`), release when the clamped observed speed reaches
  the cap. Strategy-independent, already ported and gated in `race.cl`.
- HP during the dash: `hp.tick_start_dash` (`hp_policy.rs:279-285`) with
  `is_runaway: strategy == Oonige` (`race_solver_retail.rs:1295`).
  `hp_state_rate` adds the runaway bit to the competing indices
  (`(rushing?5:3)+runaway`, `retail_physics.rs:413`). Kernel
  `hp_tick_start_dash` and `hp_state_rate` take the runaway bit.
- Rushing: `choose_mode` returns mode 4 for style 1 or 5 with no RNG draw
  (`retail_states.rs:116-118`). Kernel `rushing_choose_mode` matches.
- Position keep (`retail_position_keep.rs`): `runaway = (strategy == 5)`
  selects `speed_end[2]` (`:262-264`), `overtake_end[runaway]`,
  `wisdom[index][runaway]`, `continue_sections[runaway]` (the single
  `PositionKeepContinueSectionExArray` entry), and the coefficient alternate
  column. The controller `style` is `base_style(strategy)`, so an Oonige
  takes the style-1 branch of `choose()` (the SpeedUp/Extra/rank branch).
  Kernel `pace_update` carries all of these.

## What the field reads from an Oonige

- Pacemaker (`race_field.rs:757`): `base_running_style` puts Oonige in the
  Front group, it does not outrank it. Kernel `field_pacemaker` and
  `field_style` map 5 to 1.
- `style_ahead` / `later_style_ahead` (`race_field.rs:251-256`, `:710-711`):
  `base_style` again, Oonige counts as 1. Kernel `race.cl:137` compares with
  `field_style`.
- `ordinary_front_runner_count` (`race_field.rs:706-708`): the exact style
  count, an Oonige does not count. Kernel counts raw `== 1` (`race.cl:510`).
- Spot struggle (`race_solver.rs:2792-2926`): only strategies 1 and 5, and
  within the exact strategy (a Nige group excludes Oonige). Ordinary Front
  runners ignore Oonige rivals for the lead check, an Oonige must lead the
  whole field (`:2866-2868`, `:2891-2896`). Kernel `spot_update` mirrors it
  including the `strategy == 5` branches.
- `strategy_matches` (Nige equals Oonige, `horse_types.rs:101-105`) is used
  by skill target selection and by `is_solo_front_runner`
  (`race_field.rs:694-704`), neither of which the no-skills race path reads.
- Competition recording (`race_field.rs:181-199`, the `!= Oonige` sort at
  `:192`) orders `competition_starts`, which feeds only the full reply's
  `competitionEvents`. The compact contract does not read it, so the kernel
  needs nothing.
- Duel (CompeteFight) proximity, invite and exit carry no strategy condition.

## The port gap found

The kernel packed the raw strategy into `RR_STYLE`, so the IM_INIT force pass
saw style 5, took coefficient 0, and skipped the shared-stream draw. Every
runner's start delay, section modifiers and rushing lotteries then came from
a shifted stream, and the Oonige's own force-in coefficient was 0 instead of
the drawn RunningOut value. Divergence at tick 1 across the whole field.
