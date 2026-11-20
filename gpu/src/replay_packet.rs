//! GSIMRP01 packet builder for the batch facade. It hands the reply plus the
//! captured frames to the vendored `binary_replay::encode`, the same function
//! the engine's `into_binary` calls, so packet framing is byte-identical and
//! the meta string returned to the caller is the exact meta bytes embedded
//! in the packet.

use crate::binary_replay;
use crate::json::Value;
use crate::race_exec::loop_end;
use crate::replay_frames::ReplayFrames;

/// The packet magic, the engine's `binary_replay::MAGIC`.
pub use crate::binary_replay::MAGIC;

/// (meta_len, frame_count, horse_count) from a packet header. None for a
/// buffer too short to carry one.
pub fn decode_header(packet: &[u8]) -> Option<(usize, usize, usize)> {
    if packet.len() < 32 {
        return None;
    }
    let meta_len = u64::from_le_bytes(packet[8..16].try_into().unwrap()) as usize;
    let frame_count = u32::from_le_bytes(packet[16..20].try_into().unwrap()) as usize;
    let horse_count = u32::from_le_bytes(packet[20..24].try_into().unwrap()) as usize;
    Some((meta_len, frame_count, horse_count))
}

/// The packet's frame payload, which starts after the 32-byte header and the
/// embedded meta bytes. None for a packet that does not carry them.
pub fn frame_bytes(packet: &[u8]) -> Option<&[u8]> {
    let (meta_len, frame_count, horse_count) = decode_header(packet)?;
    let start = 32 + meta_len;
    let end = start.checked_add(frame_count * (8 + horse_count * 48))?;
    packet.get(start..end)
}

/// Encodes `meta` plus the kernel's frames (frame `k` row `r` at
/// `frames[k][r]`) into (meta string, packet). Frame `k` carries time
/// `loop_end(k)`, the engine's f32 time sum after `k` ticks.
pub fn build(meta: &Value, frames: &[Vec<[f64; 6]>]) -> (String, Vec<u8>) {
    let mut reply = meta.clone();
    let mut stored = ReplayFrames::new(frames.first().map_or(0, |row| row.len()));
    for (tick, rows) in frames.iter().enumerate() {
        let t = loop_end(tick);
        stored.capture(t, |r| rows[r]);
    }
    if let Value::Object(map) = &mut reply {
        map.insert("frames".into(), Value::ReplayFrames(Box::new(stored)));
    }
    let mut metadata = String::new();
    let mut packet = Vec::new();
    binary_replay::encode(&mut reply, &mut metadata, &mut packet);
    (metadata, packet)
}

/// The classic lane's packet. Its capture has no pre-loop frame, so frame `k`
/// carries time `classic_loop_end(k + 1)`, the engine's f64 sum of `1 / 15`
/// (src/lib.rs:581, :939).
pub fn build_classic(meta: &Value, frames: &[Vec<[f64; 6]>]) -> (String, Vec<u8>) {
    let mut reply = meta.clone();
    let mut stored = ReplayFrames::new(frames.first().map_or(0, |row| row.len()));
    for (tick, rows) in frames.iter().enumerate() {
        let t = crate::classic_reply::classic_loop_end(tick + 1);
        stored.capture(t, |r| rows[r]);
    }
    if let Value::Object(map) = &mut reply {
        map.insert("frames".into(), Value::ReplayFrames(Box::new(stored)));
    }
    let mut metadata = String::new();
    let mut packet = Vec::new();
    binary_replay::encode(&mut reply, &mut metadata, &mut packet);
    (metadata, packet)
}
