//! Input-only shared-stream initialization. Unknown random expressions fail.
//! The normal native race backend is not switched until runtime ordering agrees.
use std::collections::BTreeMap;
use crate::json::Value;
use crate::retail_random::{substream, RetailRandom};

// RaceParameter ctor/GetMotivationCoef (3431530/3431830): integer master
// rate scaled by float32 0.0001, then raw Wit multiplied and clamped.
pub fn initial_base_wisdom(raw: f64, mood: i32) -> f32 {
    let coefficient = (10000 + 200 * mood.clamp(-2, 2)) as f32 * 0.0001_f32;
    ((raw as f32) * coefficient).clamp(1.0, 2000.0)
}

#[cfg(test)]
mod initial_wisdom_tests {
    use super::initial_base_wisdom;

    #[test]
    fn official_export_values_and_constructor_bounds() {
        assert_eq!(initial_base_wisdom(1000., -1) as f64, 979.9999389648438);
        assert_eq!(initial_base_wisdom(1014., -1) as f64, 993.719970703125);
        let coefficients = [0.9599999785423279, 0.9799999594688416, 1.0,
            1.0199999809265137, 1.0399999618530273];
        for (i, coefficient) in coefficients.iter().enumerate() {
            assert_eq!(initial_base_wisdom(1024., i as i32 - 2) as f64, coefficient * 1024.);
        }
        assert_eq!(initial_base_wisdom(1600., 0), 1600.);
        assert_eq!(initial_base_wisdom(2000., 2), 2000.);
        assert_eq!(initial_base_wisdom(0., 0), 1.);
    }
}

#[derive(Debug, Clone)]
pub enum Windows { Distances(Vec<(f32,f32)>), Lottery(bool) }
#[derive(Debug, Clone)]
pub struct Trigger {
    pub skill_id: String,
    pub alternative: usize,
    pub field: String,
    pub name: String,
    pub argument: i32,
    pub windows: Windows,
    /// Shared-stream positions consumed by this trigger. In paired-draws mode the
    /// windows come from a substream instead, so both counts stay equal.
    pub draw_start: usize,
    pub draw_end: usize,
}
#[derive(Default,Clone)]
pub struct HorseInitialization {
    pub runtime_phase: bool,
    pub triggers: Vec<Trigger>,
    pub skill_enabled: BTreeMap<String,bool>,
    pub force_in_move_coefficient: f32,
    pub start_delay: f32,
    pub section_modifiers: Vec<f32>,
    pub rushing_chance: f32,
    pub rushing_section: Option<i32>,
}
#[derive(Clone)]
pub struct Initialization {
    pub horses: Vec<HorseInitialization>,
    pub stages: BTreeMap<String,usize>,
    pub rng: RetailRandom,
}

fn number(value:&Value,key:&str)->Result<f32,String> { Ok(value.get_f64(key)? as f32) }
fn skill_id(entry:&Value)->Result<String,String> { Ok(entry.get_i64("skill_id")?.to_string()) }
fn field<'a>(value:&'a Value,key:&str)->Result<&'a Value,String> {
    value.get(key).ok_or_else(||format!("Missing retail initialization field {key}"))
}

fn terms(text:&str)->Vec<(String,String,i32)> {
    let bytes=text.as_bytes();let mut at=0;let mut found=Vec::new();
    while at<bytes.len() {
        if !(bytes[at].is_ascii_lowercase() || bytes[at]==b'_') {at+=1;continue;}
        let start=at;
        while at<bytes.len() && (bytes[at].is_ascii_lowercase() || bytes[at]==b'_') {at+=1;}
        let name=&text[start..at];let op=at;
        while at<bytes.len() && b"<>=!".contains(&bytes[at]) {at+=1;}
        let operator=&text[op..at];let arg=at;
        while at<bytes.len() && bytes[at].is_ascii_digit() {at+=1;}
        if !operator.is_empty() && arg<at {
            if let Ok(n)=text[arg..at].parse() {found.push((name.to_string(),operator.to_string(),n));}
        }
    }
    found
}

fn distance_window(rng:&mut RetailRandom,start:f32,end:f32)->(f32,f32) {
    let value=rng.real_distance(start,start.max(end-10.0));(value,value+10.0)
}

fn sample_trigger(name:&str,arg:i32,rng:&mut RetailRandom,course:&Value)->Result<Windows,String> {
    let distance=number(course,"distance")?;let section=distance/6.0;
    let phases=[0.0,section,section*4.0,section*5.0,distance,f32::MAX];
    let mut windows=Vec::new();
    match name {
        "all_corner_random" => {
            if arg!=1 {return Err("Unsupported all_corner_random argument".into());}
            let mut candidates=Vec::new();
            for c in course.get_array("corners")? {
                candidates.push((number(c,"start")?,(c.get_f64("start")?+c.get_f64("length")?) as f32));
            }
            for _ in 0..4 {
                if candidates.is_empty() {break;}
                let index=rng.integer(0,candidates.len() as i32) as usize;
                let (mut start,end)=candidates[index];
                let window=distance_window(rng,start,end);windows.push(window);
                if end-window.1<10.0 {candidates.remove(index);}
                else {start=window.1;candidates[index].0=start;}
                candidates.retain(|c|c.0>=start);
            }
            windows.sort_by(|a,b|a.0.total_cmp(&b.0));
        }
        "straight_random" => {
            let rows=course.get_array("straights")?;
            if !rows.is_empty() {
                let row=&rows[rng.integer(0,rows.len() as i32) as usize];
                windows.push(distance_window(rng,number(row,"start")?,number(row,"end")?));
            }
        }
        "up_slope_random" | "down_slope_random" => {
            if arg!=1 {return Err("Unsupported random slope argument".into());}
            let uphill=name=="up_slope_random";
            let mut slopes=Vec::new();
            for row in course.get_array("slopes")? {
                let slope=number(row,"slope")?;
                if slope!=0.0 && (slope>0.0)==uphill {slopes.push(row);}
            }
            if !slopes.is_empty() {
                let row=slopes[rng.integer(0,slopes.len() as i32) as usize];
                windows.push(distance_window(rng,number(row,"start")?,(row.get_f64("start")?+row.get_f64("length")?) as f32));
            }
        }
        "corner_random" | "is_finalcorner_random" => {
            let rows=course.get_array("corners")?;
            // Global SkillTriggerCornerRandom._InitDistance (0x35c3200)
            // reverse-scans explicit corner IDs; absent matches draw nothing.
            let row = if arg <= 0 { None }
            else if name == "is_finalcorner_random" {
                // SkillTriggerFinalCornerRandom uses CalcFinalCornerDistance,
                // which requires a corner numbered >= 4 starting within the
                // final 1000 m. No qualifying corner means no random draw.
                let mut found = None;
                for (i, row) in rows.iter().enumerate().rev() {
                    let id = if row.get("index").is_some() { row.get_i64("index")? }
                        else { i as i64 + 5 - rows.len() as i64 };
                    if id >= 4 && distance - number(row, "start")? <= 1000.0 {
                        found = Some(row);
                        break;
                    }
                }
                found
            }
            else if name == "corner_random" && rows.iter().any(|r| r.get("index").is_some()) {
                for r in rows { r.get_i64("index")?; }
                rows.iter().rev().find(|r| r.get_i64("index").ok() == Some(arg as i64))
            } else {
                // Retain legacy metadata support for unnumbered corners.
                let index=rows.len() as i32+arg-5;
                if index >= 0 { rows.get(index as usize) } else { None }
            };
            if let Some(row) = row {
                windows.push(distance_window(rng,number(row,"start")?,(row.get_f64("start")?+row.get_f64("length")?) as f32));
            }
        }
        "last_straight_random" => {
            if arg!=1 {return Err("Unsupported last straight argument".into());}
            if let Some(corner)=course.get_array("corners")?.last() {
                let start=number(corner,"start")?;
                if distance-start<=1000.0 {
                    for row in course.get_array("straights")?.iter().rev() {
                        if number(row,"start")?>=start {
                            windows.push(distance_window(rng,number(row,"start")?,number(row,"end")?));
                            break;
                        }
                    }
                    // Client 0x35c2dc5 completes initialization without a draw
                    // if no qualifying following straight exists.
                }
            }
        }
        "phase_random" | "phase_firsthalf_random" | "phase_laterhalf_random" => {
            if !(0..=4).contains(&arg) {return Err("Unsupported random phase".into());}
            let (mut start,mut end)=(phases[arg as usize],phases[arg as usize+1]);
            let middle=(end-start)*0.5+start;
            if name=="phase_firsthalf_random" {end=middle;}
            if name=="phase_laterhalf_random" {start=middle;}
            windows.push(distance_window(rng,start,end));
        }
        "distance_rate_after_random" => windows.push(distance_window(rng,distance*arg as f32/100.0,distance)),
        "random_lot" => return Ok(Windows::Lottery(rng.integer(0,100)<arg)),
        _ => return Err(format!("Untraced retail random trigger: {name}")),
    }
    Ok(Windows::Distances(windows))
}

/// One skill term's activation windows. Paired-draws mode samples them from the
/// (runner, skill, term) substream, so the windows do not move when the rest of
/// the field changes.
fn trigger_windows(paired: bool, seed: i32, runner: usize, key: &str, name: &str, argument: i32,
                   rng: &mut RetailRandom, course: &Value) -> Result<Windows, String> {
    if paired {
        let mut stream = substream(seed, runner, key);
        sample_trigger(name, argument, &mut stream, course)
    } else {
        sample_trigger(name, argument, rng, course)
    }
}

/// One skill's wisdom lottery. Paired-draws mode uses that skill's own substream.
fn wisdom_roll(paired: bool, seed: i32, runner: usize, sid: &str, rng: &mut RetailRandom) -> f32 {
    if paired { substream(seed, runner, &format!("W{sid}")).real(0., 100.) } else { rng.real(0., 100.) }
}

pub fn initialize(seed:i32,horses:&[Value],course:&Value,skills:&Value,parameters:&Value,paired:bool)->Result<Initialization,String> {
    let mut rng=RetailRandom::new(seed);
    let mut rows:Vec<HorseInitialization>=(0..horses.len()).map(|_|HorseInitialization::default()).collect();
    for (runner,(horse,row)) in horses.iter().zip(rows.iter_mut()).enumerate() {
        for entry in horse.get_array("skills")? {
            let sid=skill_id(entry)?;
            for (alternative,alt) in field(skills,&sid)?.get_array("alternatives")?.iter().enumerate() {
                for column in ["precondition","condition"] {
                    for (name,operator,argument) in terms(alt.get(column).and_then(Value::as_str).unwrap_or("")) {
                        if !name.contains("random") {continue;}
                        if operator!="==" {return Err(format!("Untraced retail random operator: {name}{operator}"));}
                        let draw_start=rng.draw_count;
                        let windows=trigger_windows(paired,seed,runner,&format!("T{sid}:{alternative}:{column}:{name}"),&name,argument,&mut rng,course)?;
                        row.triggers.push(Trigger {skill_id:sid.clone(),alternative,field:column.into(),name,argument,
                            windows,draw_start,draw_end:rng.draw_count});
                    }
                }
            }
        }
    }
    let mut stages=BTreeMap::from([("after_triggers".into(),rng.draw_count)]);
    let styles=["","RunningOut","Proceding","Insertion","Chasing"];
    for (horse,row) in horses.iter().zip(rows.iter_mut()) {
        let style=horse.get_i64("running_style")?;
        let coefficient=if (1..=4).contains(&style) {number(parameters,&format!("forceInMoveTargetSpeedAdd{}",styles[style as usize]))?} else {0.0};
        let span=number(parameters,"forceInMoveTargetSpeedAddRandomRange")?;
        let offset=if coefficient>0.0 {rng.real(0.0,span)} else if coefficient<0.0 {rng.real(-span,0.0)} else {0.0};
        row.force_in_move_coefficient=coefficient+offset;
    }
    stages.insert("after_ai".into(),rng.draw_count);
    for (runner,(horse,row)) in horses.iter().zip(rows.iter_mut()).enumerate() {
        let chance=(100.0-9000.0/number(horse,"base_wisdom")?).max(20.0);
        for entry in horse.get_array("skills")? {
            let sid=skill_id(entry)?;
            let enabled=field(skills,&sid)?.get_i64("wisdomCheck")?!=1 || chance>wisdom_roll(paired,seed,runner,&sid,&mut rng);
            row.skill_enabled.insert(sid,enabled);
        }
    }
    stages.insert("after_wisdom".into(),rng.draw_count);
    for (horse,row) in horses.iter().zip(rows.iter_mut()) {
        let mut multiplier=1.0f32;
        for entry in horse.get_array("skills")? {
            let sid=skill_id(entry)?;
            if !row.skill_enabled[&sid] {continue;}
            for alt in field(skills,&sid)?.get_array("alternatives")? {
                if alt.get_str("condition")?!="always==1" || !alt.get("precondition").and_then(Value::as_str).unwrap_or("").is_empty() {continue;}
                for effect in alt.get_array("effects")? {
                    if effect.get_i64("type")?==10 && effect.get_i64("target")?==1 {
                        let removal=(10000.0-effect.get_f64("modifier")?) as f32/10000.0;
                        multiplier*=1.0-removal;
                    }
                }
            }
        }
        row.start_delay=rng.real(0.0,number(parameters,"startDelayMax")?)*multiplier;
    }
    stages.insert("after_delays".into(),rng.draw_count);
    Ok(Initialization {horses:rows,stages,rng})
}

/// One skill definition's seed-independent facts, read as `initialize` reads them.
struct SkillFacts {
    /// Random condition terms in visit order: alternative, column, name, argument.
    random_terms: Vec<(usize, &'static str, String, i32)>,
    wisdom_checked: bool,
    /// Start-delay factors, `1 - removal`, of always-active alternatives.
    delay_factors: Vec<f32>,
}

/// Decoded skill facts for one skill table. Clear it whenever the table changes.
#[derive(Default)]
pub struct SkillFactsCache(std::collections::HashMap<String, std::rc::Rc<SkillFacts>>);

impl SkillFactsCache {
    pub fn clear(&mut self) { self.0.clear(); }

    fn facts(&mut self, skills: &Value, sid: &str) -> Result<std::rc::Rc<SkillFacts>, String> {
        if let Some(facts) = self.0.get(sid) { return Ok(facts.clone()); }
        let skill = field(skills, sid)?;
        let alternatives = skill.get_array("alternatives")?;
        let mut random_terms = Vec::new();
        for (alternative, alt) in alternatives.iter().enumerate() {
            for column in ["precondition", "condition"] {
                for (name, operator, argument) in terms(alt.get(column).and_then(Value::as_str).unwrap_or("")) {
                    if !name.contains("random") { continue; }
                    if operator != "==" { return Err(format!("Untraced retail random operator: {name}{operator}")); }
                    random_terms.push((alternative, column, name, argument));
                }
            }
        }
        let wisdom_checked = skill.get_i64("wisdomCheck")? == 1;
        let mut delay_factors = Vec::new();
        for alt in alternatives {
            if alt.get_str("condition")? != "always==1"
                || !alt.get("precondition").and_then(Value::as_str).unwrap_or("").is_empty() { continue; }
            for effect in alt.get_array("effects")? {
                if effect.get_i64("type")? == 10 && effect.get_i64("target")? == 1 {
                    let removal = (10000.0 - effect.get_f64("modifier")?) as f32 / 10000.0;
                    delay_factors.push(1.0 - removal);
                }
            }
        }
        let facts = std::rc::Rc::new(SkillFacts { random_terms, wisdom_checked, delay_factors });
        self.0.insert(sid.to_string(), facts.clone());
        Ok(facts)
    }
}

/// `initialize` with each skill's definition decoded once per table. Draws and
/// arithmetic keep the same order. If any skill fails to decode, the original
/// path runs instead so that its error, or its result, is exactly reproduced.
pub fn initialize_cached(seed:i32,horses:&[Value],course:&Value,skills:&Value,parameters:&Value,
    cache:&mut SkillFactsCache,paired:bool)->Result<Initialization,String> {
    let mut decoded = Vec::with_capacity(horses.len());
    for horse in horses {
        let Ok(entries) = horse.get_array("skills") else { return initialize(seed,horses,course,skills,parameters,paired) };
        let mut row = Vec::with_capacity(entries.len());
        for entry in entries {
            let Ok(sid) = skill_id(entry) else { return initialize(seed,horses,course,skills,parameters,paired) };
            let Ok(facts) = cache.facts(skills, &sid) else { return initialize(seed,horses,course,skills,parameters,paired) };
            row.push((sid, facts));
        }
        decoded.push(row);
    }
    let mut rng=RetailRandom::new(seed);
    let mut rows:Vec<HorseInitialization>=(0..horses.len()).map(|_|HorseInitialization::default()).collect();
    for (runner,(skills,row)) in decoded.iter().zip(rows.iter_mut()).enumerate() {
        for (sid,facts) in skills {
            for (alternative,column,name,argument) in &facts.random_terms {
                let draw_start=rng.draw_count;
                let windows=trigger_windows(paired,seed,runner,&format!("T{sid}:{alternative}:{column}:{name}"),name,*argument,&mut rng,course)?;
                row.triggers.push(Trigger {skill_id:sid.clone(),alternative:*alternative,field:(*column).into(),
                    name:name.clone(),argument:*argument,windows,draw_start,draw_end:rng.draw_count});
            }
        }
    }
    let mut stages=BTreeMap::from([("after_triggers".into(),rng.draw_count)]);
    let styles=["","RunningOut","Proceding","Insertion","Chasing"];
    for (horse,row) in horses.iter().zip(rows.iter_mut()) {
        let style=horse.get_i64("running_style")?;
        let coefficient=if (1..=4).contains(&style) {number(parameters,&format!("forceInMoveTargetSpeedAdd{}",styles[style as usize]))?} else {0.0};
        let span=number(parameters,"forceInMoveTargetSpeedAddRandomRange")?;
        let offset=if coefficient>0.0 {rng.real(0.0,span)} else if coefficient<0.0 {rng.real(-span,0.0)} else {0.0};
        row.force_in_move_coefficient=coefficient+offset;
    }
    stages.insert("after_ai".into(),rng.draw_count);
    for (runner,((horse,skills),row)) in horses.iter().zip(&decoded).zip(rows.iter_mut()).enumerate() {
        let chance=(100.0-9000.0/number(horse,"base_wisdom")?).max(20.0);
        for (sid,facts) in skills {
            let enabled=!facts.wisdom_checked || chance>wisdom_roll(paired,seed,runner,sid,&mut rng);
            row.skill_enabled.insert(sid.clone(),enabled);
        }
    }
    stages.insert("after_wisdom".into(),rng.draw_count);
    for (skills,row) in decoded.iter().zip(rows.iter_mut()) {
        let mut multiplier=1.0f32;
        for (sid,facts) in skills {
            if !row.skill_enabled[sid] {continue;}
            for factor in &facts.delay_factors { multiplier*=factor; }
        }
        row.start_delay=rng.real(0.0,number(parameters,"startDelayMax")?)*multiplier;
    }
    stages.insert("after_delays".into(),rng.draw_count);
    Ok(Initialization {horses:rows,stages,rng})
}

struct PreparedTrigger {
    skill_id: String,
    alternative: usize,
    field: String,
    name: String,
    argument: i32,
}

struct PreparedHorse {
    triggers: Vec<PreparedTrigger>,
    skills: Vec<(String, bool)>,
    delay_modifiers: Vec<(String, Vec<f32>)>,
    wisdom_chance: f32,
    force_coefficient: f32,
}

/// Seed-independent decoding only. Trigger sampling and all random draws remain
/// in their original field-wide pass order for each call to initialize.
pub(crate) struct PreparedInitialization {
    horses: Vec<PreparedHorse>,
    course: Value,
    force_span: f32,
    start_delay_max: f32,
    paired: bool,
}

impl PreparedInitialization {
    pub(crate) fn new(horses: &[Value], course: &Value, skills: &Value, parameters: &Value,
        paired: bool) -> Result<Self, String> {
        let styles = ["", "RunningOut", "Proceding", "Insertion", "Chasing"];
        let mut prepared = Vec::with_capacity(horses.len());
        for horse in horses {
            let mut triggers = Vec::new();
            let mut checks = Vec::new();
            let mut delay_modifiers = Vec::new();
            for entry in horse.get_array("skills")? {
                let sid = skill_id(entry)?;
                let skill = field(skills, &sid)?;
                let alternatives = skill.get_array("alternatives")?;
                for (alternative, alt) in alternatives.iter().enumerate() {
                    for column in ["precondition", "condition"] {
                        for (name, operator, argument) in terms(alt.get(column).and_then(Value::as_str).unwrap_or("")) {
                            if !name.contains("random") { continue; }
                            if operator != "==" { return Err(format!("Untraced retail random operator: {name}{operator}")); }
                            triggers.push(PreparedTrigger { skill_id: sid.clone(), alternative,
                                field: column.into(), name, argument });
                        }
                    }
                }
                checks.push((sid.clone(), skill.get_i64("wisdomCheck")? == 1));
                let mut modifiers = Vec::new();
                for alt in alternatives {
                    if alt.get_str("condition")? != "always==1"
                        || !alt.get("precondition").and_then(Value::as_str).unwrap_or("").is_empty() { continue; }
                    for effect in alt.get_array("effects")? {
                        if effect.get_i64("type")? == 10 && effect.get_i64("target")? == 1 {
                            let removal = (10000.0 - effect.get_f64("modifier")?) as f32 / 10000.0;
                            modifiers.push(1.0 - removal);
                        }
                    }
                }
                delay_modifiers.push((sid, modifiers));
            }
            let style = horse.get_i64("running_style")?;
            prepared.push(PreparedHorse { triggers, skills: checks, delay_modifiers,
                wisdom_chance: (100.0 - 9000.0 / number(horse, "base_wisdom")?).max(20.0),
                force_coefficient: if (1..=4).contains(&style) {
                    number(parameters, &format!("forceInMoveTargetSpeedAdd{}", styles[style as usize]))?
                } else { 0.0 },
            });
        }
        Ok(Self { horses: prepared, course: course.clone(),
            force_span: number(parameters, "forceInMoveTargetSpeedAddRandomRange")?,
            start_delay_max: number(parameters, "startDelayMax")?, paired })
    }

    pub(crate) fn initialize(&self, seed: i32) -> Result<Initialization, String> {
        self.initialize_order(seed, None)
    }

    /// `initialize` for a different field order: `order[position]` names the
    /// prepared horse that runs at that position. The draw sequence stays the
    /// original field-wide pass order, so a gated field matches a preparation
    /// built in that gate's order exactly.
    pub(crate) fn initialize_permuted(&self, seed: i32, order: &[usize]) -> Result<Initialization, String> {
        if order.len() != self.horses.len() {
            return Err(format!("initialization order names {} of {} runners", order.len(), self.horses.len()));
        }
        self.initialize_order(seed, Some(order))
    }

    fn initialize_order(&self, seed: i32, order: Option<&[usize]>) -> Result<Initialization, String> {
        let count = self.horses.len();
        let at = |position: usize| -> Result<usize, String> {
            match order {
                Some(order) => order.get(position).copied()
                    .filter(|&index| index < count)
                    .ok_or_else(|| format!("initialization order index out of range at {position}")),
                None => Ok(position),
            }
        };
        let mut rng = RetailRandom::new(seed);
        let mut rows: Vec<_> = (0..count).map(|_| HorseInitialization::default()).collect();
        for (runner, row) in rows.iter_mut().enumerate() {
            let horse = &self.horses[at(runner)?];
            for trigger in &horse.triggers {
                let draw_start = rng.draw_count;
                let windows = trigger_windows(self.paired, seed, runner,
                    &format!("T{}:{}:{}:{}", trigger.skill_id, trigger.alternative, trigger.field, trigger.name),
                    &trigger.name, trigger.argument, &mut rng, &self.course)?;
                row.triggers.push(Trigger { skill_id: trigger.skill_id.clone(), alternative: trigger.alternative,
                    field: trigger.field.clone(), name: trigger.name.clone(), argument: trigger.argument,
                    windows, draw_start, draw_end: rng.draw_count });
            }
        }
        let mut stages = BTreeMap::from([("after_triggers".into(), rng.draw_count)]);
        for (position, row) in rows.iter_mut().enumerate() {
            let coefficient = self.horses[at(position)?].force_coefficient;
            let offset = if coefficient > 0. { rng.real(0., self.force_span) }
                else if coefficient < 0. { rng.real(-self.force_span, 0.) } else { 0. };
            row.force_in_move_coefficient = coefficient + offset;
        }
        stages.insert("after_ai".into(), rng.draw_count);
        for (runner, row) in rows.iter_mut().enumerate() {
            let horse = &self.horses[at(runner)?];
            for (sid, checked) in &horse.skills {
                let enabled = !checked || horse.wisdom_chance > wisdom_roll(self.paired, seed, runner, sid, &mut rng);
                row.skill_enabled.insert(sid.clone(), enabled);
            }
        }
        stages.insert("after_wisdom".into(), rng.draw_count);
        for (position, row) in rows.iter_mut().enumerate() {
            let horse = &self.horses[at(position)?];
            let mut multiplier = 1.0_f32;
            for (sid, modifiers) in &horse.delay_modifiers {
                if row.skill_enabled[sid] {
                    for modifier in modifiers { multiplier *= modifier; }
                }
            }
            row.start_delay = rng.real(0., self.start_delay_max) * multiplier;
        }
        stages.insert("after_delays".into(), rng.draw_count);
        Ok(Initialization { horses: rows, stages, rng })
    }
}

/// One runner's rushing lottery: the eligibility draw, then (only when it
/// hits) the section draw.
fn rushing_draw(chance:f32,rng:&mut RetailRandom,temptation:&Value)->Result<Option<i32>,String> {
    Ok(if chance>rng.integer(0,100) as f32 {
        Some(rng.integer(temptation.get_i64("TemptationLotSectionMin")? as i32,
                         temptation.get_i64("TemptationLotSectionMax")? as i32+1))
    } else {None})
}

/// Continue the manager's shared stream: all section values, then rushing.
/// Wisdom is the adjusted value after starting green effects; the bonus is
/// in percentage points. These are predicted inputs, not recorded outcomes.
///
/// `paired_seed` (paired-draws mode) gives each runner's rushing lottery its
/// own (seed, runner, "RUSH") substream. On the shared stream the section
/// draw happens only when the eligibility draw hits, and that hit depends on
/// the runner's rushing chance, so a skill that changes the chance shifted
/// every later runner's draws. `None` keeps the retail shared stream.
pub fn initialize_sections_and_rushing(init:&mut Initialization, horses:&[(f32,f32)],
                                      course:&Value,parameters:&Value,
                                      paired_seed:Option<i32>)->Result<(),String> {
    if horses.len()!=init.horses.len() {return Err("Roster changed during initialization".into());}
    let base=20.0-(number(course,"distance")?-2000.0)/1000.0;
    let speeds=field(parameters,"BasetTargetSpeed")?;
    for ((wisdom,_),row) in horses.iter().zip(init.horses.iter_mut()) {
        let logarithm=((*wisdom*0.1) as f64).log10() as f32;
        // Isolated section-only expression-store hypothesis. The logarithm is
        // still the baseline rounded result. Range results are float32 returns.
        let lower=((*wisdom as f64/number(speeds,"BaseTargetSpeedRandomMinusVal2")? as f64)*logarithm as f64
            +number(speeds,"BaseTargetSpeedRandomMinusVal1")? as f64) as f32;
        let upper=((*wisdom as f64/number(speeds,"BaseTargetSpeedRandomPlusVal1")? as f64)*logarithm as f64) as f32;
        row.section_modifiers=(0..24).map(|_| {
            let sample=init.rng.next_double() as f32;
            let width=(upper-lower).abs();
            let random=(sample as f64*width as f64+lower as f64) as f32;
            let random=if random<lower {lower} else if random>upper {upper} else {random};
            ((random as f64 * 0.01_f32 as f64) * base as f64) as f32
        }).collect();
    }
    init.stages.insert("after_sections".into(),init.rng.draw_count);
    let temptation=field(parameters,"Temptation")?;
    for (runner,((wisdom,bonus),row)) in horses.iter().zip(init.horses.iter_mut()).enumerate() {
        let term=*wisdom*0.1+1.0;
        let logarithm=((term*term) as f64).log10() as f32;
        let fraction=number(temptation,"TemptationStartPerVal1")?/logarithm;
        row.rushing_chance=fraction*fraction+bonus;
        row.rushing_section=match paired_seed {
            Some(seed)=>rushing_draw(row.rushing_chance,&mut substream(seed,runner,"RUSH"),temptation)?,
            None=>rushing_draw(row.rushing_chance,&mut init.rng,temptation)?,
        };
    }
    init.stages.insert("after_rushing".into(),init.rng.draw_count);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn prepared_initialization_preserves_draws_and_float_bits_across_seeds() {
        let data = crate::json::parse(include_str!("../tests/fixtures/retail_initialization_official.json")).unwrap();
        // One cache serves every case, as the worker reuses it across requests.
        let mut cache = SkillFactsCache::default();
        for case in data.get_array("cases").unwrap() {
            let course = field(field(&data, "courses").unwrap(), case.get_str("course_id").unwrap()).unwrap();
            let horses = case.get_array("horses").unwrap();
            let skills = field(&data, "skills").unwrap();
            let parameters = field(&data, "parameters").unwrap();
            let prepared = PreparedInitialization::new(horses, course, skills, parameters, false).unwrap();
            for seed in [i32::MIN, -71, 0, 71, i32::MAX, case.get_i64("seed").unwrap() as i32] {
              for cached in [false, true] {
                let mut expected = initialize(seed, horses, course, skills, parameters, false).unwrap();
                let mut actual = if cached {
                    initialize_cached(seed, horses, course, skills, parameters, &mut cache, false).unwrap()
                } else { prepared.initialize(seed).unwrap() };
                assert_eq!(actual.stages, expected.stages);
                for (a, b) in actual.horses.iter().zip(&expected.horses) {
                    assert_eq!(a.skill_enabled, b.skill_enabled);
                    assert_eq!(a.force_in_move_coefficient.to_bits(), b.force_in_move_coefficient.to_bits());
                    assert_eq!(a.start_delay.to_bits(), b.start_delay.to_bits());
                    assert_eq!(a.triggers.len(), b.triggers.len());
                    for (a, b) in a.triggers.iter().zip(&b.triggers) {
                        assert_eq!((&a.skill_id, a.alternative, &a.field, &a.name, a.argument, a.draw_start, a.draw_end),
                                   (&b.skill_id, b.alternative, &b.field, &b.name, b.argument, b.draw_start, b.draw_end));
                        match (&a.windows, &b.windows) {
                            (Windows::Lottery(a), Windows::Lottery(b)) => assert_eq!(a, b),
                            (Windows::Distances(a), Windows::Distances(b)) => assert_eq!(
                                a.iter().map(|(lo,hi)| (lo.to_bits(),hi.to_bits())).collect::<Vec<_>>(),
                                b.iter().map(|(lo,hi)| (lo.to_bits(),hi.to_bits())).collect::<Vec<_>>()),
                            _ => panic!("Trigger kind changed"),
                        }
                    }
                }
                for _ in 0..64 { assert_eq!(actual.rng.next_int(), expected.rng.next_int()); }
              }
            }
            // An undecodable skill defers to the uncached path and its error.
            let mut broken = horses.to_vec();
            if let Some(crate::json::Value::Object(horse)) = broken.first_mut() {
                horse.insert("skills".into(), crate::json::parse(r#"[{"skill_id":999999999,"level":1}]"#).unwrap());
            }
            let error = initialize(7, &broken, course, skills, parameters, false).err();
            assert!(error.is_some());
            assert_eq!(initialize_cached(7, &broken, course, skills, parameters, &mut cache, false).err(), error);
        }
    }
    #[test]
    fn absent_following_straight_keeps_empty_trigger_and_rng_state() {
        let course=crate::json::parse(r#"{"distance":2400,"corners":[{"start":2348,"length":500}],"straights":[{"start":1900,"end":2400}]}"#).unwrap();
        let mut rng=RetailRandom::new(123);
        assert!(matches!(sample_trigger("last_straight_random",1,&mut rng,&course).unwrap(),Windows::Distances(windows) if windows.is_empty()));
        assert_eq!(rng.draw_count,0);
    }
    #[test]
    fn final_corner_random_requires_an_eligible_start() {
        for (start, id, draws) in [(1399.0,4,0), (1400.0,4,1), (1500.0,3,0), (1500.0,90,1)] {
            let course = crate::json::parse(&format!(
                r#"{{"distance":2400,"corners":[{{"start":{start},"length":400,"index":{id}}}]}}"#)).unwrap();
            let mut rng = RetailRandom::new(123);
            let windows = sample_trigger("is_finalcorner_random",1,&mut rng,&course).unwrap();
            assert_eq!(rng.draw_count, draws);
            assert!(matches!(windows, Windows::Distances(rows) if rows.len() == draws));
        }
    }
    #[test]
    fn all_ninety_official_delays_are_bit_exact_from_inputs() {
        let data=crate::json::parse(include_str!("../tests/fixtures/retail_initialization_official.json")).unwrap();
        let mut count=0;
        for case in data.get_array("cases").unwrap() {
            let course=field(field(&data,"courses").unwrap(),case.get_str("course_id").unwrap()).unwrap();
            let result=initialize(case.get_i64("seed").unwrap() as i32,case.get_array("horses").unwrap(),course,
                field(&data,"skills").unwrap(),field(&data,"parameters").unwrap(),false).unwrap();
            for (row,expected) in result.horses.iter().zip(case.get_array("expected_start_delays").unwrap()) {
                assert_eq!(row.start_delay as f64,expected.as_f64().unwrap());count+=1;
            }
        }
        assert_eq!(count,90);
    }

    #[test]
    fn all_official_rushing_eligibility_and_sections_follow_shared_stream() {
        let data=crate::json::parse(include_str!("../tests/fixtures/retail_initialization_official.json")).unwrap();
        let mut runners=0;let mut enabled=0;
        for case in data.get_array("cases").unwrap() {
            let course=field(field(&data,"courses").unwrap(),case.get_str("course_id").unwrap()).unwrap();
            let horses=case.get_array("horses").unwrap();
            let params=field(&data,"parameters").unwrap();
            let mut result=initialize(case.get_i64("seed").unwrap() as i32,horses,course,
                field(&data,"skills").unwrap(),params,false).unwrap();
            let wisdom:Vec<_>=horses.iter().map(|h|(number(h,"base_wisdom").unwrap(),0.0)).collect();
            initialize_sections_and_rushing(&mut result,&wisdom,course,params,None).unwrap();
            for (row,expected) in result.horses.iter().zip(case.get_array("expected_rushing_sections").unwrap()) {
                assert_eq!(row.rushing_section,expected.as_f64().map(|v|v as i32));
                assert_eq!(row.section_modifiers.len(),24);
                runners+=1;enabled+=usize::from(row.rushing_section.is_some());
            }
            assert_eq!(result.stages["after_sections"]-result.stages["after_delays"],24*horses.len());
        }
        assert_eq!((runners,enabled),(90,5));
    }

    #[test]
    fn paired_mode_holds_a_skills_draws_when_the_field_grows() {
        let data=crate::json::parse(include_str!("../tests/fixtures/retail_initialization_official.json")).unwrap();
        let (skills,parameters)=(field(&data,"skills").unwrap(),field(&data,"parameters").unwrap());
        let case=data.get_array("cases").unwrap().first().unwrap();
        let course=field(field(&data,"courses").unwrap(),case.get_str("course_id").unwrap()).unwrap();
        let seed=case.get_i64("seed").unwrap() as i32;
        let plain=case.get_array("horses").unwrap();
        assert!(plain.len()>=2);
        // Duplicate a skill that really draws, 200362, on runner 0: it carries a
        // random condition term and a wisdom check, so the growth consumes extra
        // shared-stream draws (proven below) and must move every runner behind it
        // in shared-stream mode -- and nothing in paired mode. A skill that draws
        // nothing would make the growth invisible and the test vacuous.
        let mut grown=plain.to_vec();
        let mut first=match &grown[0] { crate::json::Value::Object(map)=>map.clone(), _=>panic!("a horse must be an object") };
        let entries=field(&grown[0],"skills").unwrap().as_array().unwrap().clone();
        let drawn=entries.iter().find(|e| e.get_i64("skill_id").ok()==Some(200362))
            .expect("fixture runner 0 owns skill 200362").clone();
        let mut more=entries.clone();
        more.push(drawn);
        first.insert("skills".into(),crate::json::Value::Array(more));
        grown[0]=crate::json::Value::Object(first);
        let fingerprint=|row:&HorseInitialization| {
            let mut keys:Vec<(String,Vec<u32>)>=row.triggers.iter().map(|trigger|(
                format!("{}:{}",trigger.skill_id,trigger.name),
                match &trigger.windows {
                    Windows::Lottery(hit)=>vec![u32::from(*hit)],
                    Windows::Distances(rows)=>rows.iter().flat_map(|(lo,hi)|[lo.to_bits(),hi.to_bits()]).collect(),
                })).collect();
            keys.extend(row.skill_enabled.iter().map(|(sid,on)|(format!("W{sid}"),vec![u32::from(*on)])));
            // The initialization produced at a position, not just its substream
            // draws: the shared-stream values (force-in-move offset, start delay)
            // must stay put too, or a skill change elsewhere still moves them.
            keys.push(("force_in_move".into(),vec![row.force_in_move_coefficient.to_bits()]));
            keys.push(("start_delay".into(),vec![row.start_delay.to_bits()]));
            keys
        };
        let mut before=initialize(seed,plain,course,skills,parameters,true).unwrap();
        let mut paired=initialize(seed,&grown,course,skills,parameters,true).unwrap();
        let mut retail_plain=initialize(seed,plain,course,skills,parameters,false).unwrap();
        let mut retail=initialize(seed,&grown,course,skills,parameters,false).unwrap();
        assert_eq!(fingerprint(&before.horses[1]),fingerprint(&paired.horses[1]),
            "paired mode must not move a later runner's draws");
        assert_ne!(fingerprint(&retail_plain.horses[1]),fingerprint(&retail.horses[1]),
            "the growth does shift the shared stream, so the pairing holds are not vacuous");
        // Sections and rushing continue on the same stream after initialize.
        // Paired mode must leave the whole rest of a position's initialization --
        // section modifiers, rushing chance and section pick -- bit for bit where it was.
        let wisdom:Vec<_>=plain.iter().map(|h|(number(h,"base_wisdom").unwrap(),0.0)).collect();
        for init in [&mut before,&mut paired] {
            initialize_sections_and_rushing(init,&wisdom,course,parameters,Some(seed)).unwrap();
        }
        for init in [&mut retail_plain,&mut retail] {
            initialize_sections_and_rushing(init,&wisdom,course,parameters,None).unwrap();
        }
        assert_eq!(before.stages,paired.stages,
            "paired growth must not move the shared stream the sections draw from");
        assert_ne!(retail_plain.stages,retail.stages,
            "retail growth does move the shared stream");
        let untouched=|a:&HorseInitialization,b:&HorseInitialization| {
            assert_eq!(a.section_modifiers,b.section_modifiers);
            assert_eq!(a.rushing_chance.to_bits(),b.rushing_chance.to_bits());
            assert_eq!(a.rushing_section,b.rushing_section);
        };
        untouched(&before.horses[1],&paired.horses[1]);
    }

    #[test]
    fn paired_mode_holds_other_runners_rushing_when_one_chance_changes() {
        // A skill that changes runner 0's rushing chance (bonus) flips whether
        // its section draw happens. On the shared stream that shifts every later
        // runner's rushing lottery; paired mode must not move any of them.
        let data=crate::json::parse(include_str!("../tests/fixtures/retail_initialization_official.json")).unwrap();
        let (skills,parameters)=(field(&data,"skills").unwrap(),field(&data,"parameters").unwrap());
        let mut retail_moved=false;
        for case in data.get_array("cases").unwrap() {
            let course=field(field(&data,"courses").unwrap(),case.get_str("course_id").unwrap()).unwrap();
            let seed=case.get_i64("seed").unwrap() as i32;
            let horses=case.get_array("horses").unwrap();
            if horses.len()<2 {continue;}
            let wisdom:Vec<(f32,f32)>=horses.iter().map(|h|(number(h,"base_wisdom").unwrap(),0.0)).collect();
            // Runner 0 certain to rush (bonus 100) vs certain not to (bonus -100):
            // its section draw happens in exactly one of the two.
            let mut hot=wisdom.clone(); hot[0].1=100.0;
            let mut cold=wisdom.clone(); cold[0].1=-100.0;
            let run=|w:&[(f32,f32)],paired:Option<i32>| {
                let mut init=initialize(seed,horses,course,skills,parameters,paired.is_some()).unwrap();
                initialize_sections_and_rushing(&mut init,w,course,parameters,paired).unwrap();
                init
            };
            let (ph,pc)=(run(&hot,Some(seed)),run(&cold,Some(seed)));
            assert!(ph.horses[0].rushing_section.is_some() && pc.horses[0].rushing_section.is_none(),
                "the bonus must flip runner 0's own lottery, or the test is vacuous");
            for i in 1..horses.len() {
                assert_eq!(ph.horses[i].rushing_section,pc.horses[i].rushing_section,
                    "paired mode moved runner {i}'s rushing when runner 0's chance changed");
            }
            let (rh,rc)=(run(&hot,None),run(&cold,None));
            if (1..horses.len()).any(|i| rh.horses[i].rushing_section!=rc.horses[i].rushing_section) {retail_moved=true;}
        }
        assert!(retail_moved,"on the shared stream the flip must move some later runner, or the fixture proves nothing");
    }
}
