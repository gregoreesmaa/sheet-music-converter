#!/usr/bin/env python3
"""
Stage 2: 100k Quota-Filling Corpus Puller for sheet-music-converter.

Assembles, classifies across all 14 orthogonal dimensions (101 buckets),
deduplicates (SHA-256), license-verifies (strictly CC0 / PD / CC-BY / MIT / BSD),
and eval-leak-filters 100,000 quota-balanced score-page units into:
  - D:/tmp/corpus100k/corpus100k_manifest.jsonl
  - D:/tmp/corpus100k/quota_report.json
  - D:/tmp/corpus100k/rejected_log.jsonl

Strictly CPU-only (CUDA_VISIBLE_DEVICES=""), max_workers<=4, streaming tar I/O.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import csv
import hashlib
import io
import json
import math
import os
import queue
import re
import struct
import sys
import tarfile
import time
import unicodedata
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

# Enforce strictly CPU-only execution
os.environ["CUDA_VISIBLE_DEVICES"] = ""

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = Path(r"D:\tmp\corpus100k")
CACHE_DIR = OUT_ROOT / "cache"
BLUEPRINT_PATH = OUT_ROOT / "quota_blueprint.json"
MANIFEST_OUT = OUT_ROOT / "corpus100k_manifest.jsonl"
REPORT_OUT = OUT_ROOT / "quota_report.json"
REJECTED_OUT = OUT_ROOT / "rejected_log.jsonl"

SUITE31_MANIFEST = REPO_ROOT / "data-gen" / "score-corpus" / "suite31" / "suite31_hw_manifest.json"
GATE20_MANIFEST = REPO_ROOT / "data-gen" / "score-corpus" / "gate20" / "gate_manifest.json"
DEBUSSY_MANIFEST = Path(r"D:\tmp\debussy-omr\train_pages\manifest.json")

SEED = 20261125
MAX_WORKERS = 4

# =============================================================================
# Deterministic SplitMix64 RNG (matching Rust orchestrator convention)
# =============================================================================

MASK64 = (1 << 64) - 1


def splitmix64_state(seed: int, stage: str, key: str) -> int:
    h = hashlib.sha256(f"{seed}/{stage}/{key}".encode("utf-8")).digest()
    return struct.unpack("<Q", h[:8])[0]


def splitmix64_next(state: int) -> Tuple[int, int]:
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    z = z ^ (z >> 31)
    return state, z


def det_float(seed: int, stage: str, key: str) -> float:
    _, z = splitmix64_next(splitmix64_state(seed, stage, key))
    return (z >> 11) / float(1 << 53)


# =============================================================================
# Normalization & Strict Eval Blocklist
# =============================================================================

def norm_text(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    return s


# Explicit (composer_tokens, work_tokens) pairs covering EVERY piece in
# suite31_hw (45 pieces) + gate20 (16 unique works) + mozart-k458
EVAL_WORK_RULES: List[Tuple[List[str], List[str]]] = [
    # suite31 printed
    (["bach"], ["bwv 66", "bwv66", "bwv 066"]),
    (["joplin"], ["maple leaf"]),
    (["schoenberg", "schonberg"], ["op 19", "op19"]),
    (["beethoven"], ["grosse fuge", "op 133", "op133"]),
    (["mozart"], ["k 155", "k155", "kv 155", "kv155"]),
    (["mozart"], ["k 458", "k458", "kv 458", "kv458", "hunt quartet", "string quartet no 17"]),
    (["beethoven"], ["op 18 no 1", "op 18 n 1", "op18n1", "op 18 1"]),
    (["haydn"], ["op 74 no 1", "op 74 n 1", "op74n1", "op 74 1", "hob iii 72"]),
    (["chopin"], ["op 6 no 2", "op 6 2", "mazurka 6 2", "mazurka06 2", "mazurka op 6"]),
    (["schubert"], ["lindenbaum", "d 911 5", "d911 5"]),
    (["schumann"], ["dichterliebe", "op 48 no 2", "op 48 2", "aus meinen tranen"]),
    (["schumann"], ["op 41 no 1", "op 41 n 1", "op41n1", "op 41 1"]),
    (["clara", "wieck"], ["polonaise op 1", "op 1 no 1", "op1n1"]),
    (["clara", "wieck"], ["op 17", "op17"]),
    (["weber"], ["concertino", "op 26"]),
    (["verdi"], ["la donna e mobile", "la donna"]),
    (["cpe", "c p e", "carl philipp"], ["h 186", "h186", "wq 55"]),
    (["beach"], ["prayer"]),
    (["handel", "haendel"], ["lascia"]),
    (["palestrina"], ["agnus"]),
    (["liliuokalani"], ["aloha oe"]),
    ([], ["aloha oe"]),
    # suite31 real HW
    (["beethoven"], ["op 135", "op135"]),
    (["mozart"], ["k 464", "k464", "kv 464", "kv464", "string quartet no 18"]),
    (["mozart"], ["k 387", "k387", "kv 387", "kv387", "string quartet no 14"]),
    (["mozart"], ["k 421", "k421", "kv 421", "kv421", "string quartet no 15"]),
    (["schubert"], ["d 810", "d810", "death and the maiden", "tod und das madchen"]),
    (["debussy"], ["vetement du blesse", "berceuse heroique", "la plus que lente", "lindaraja"]),
    (["hensel", "mendelssohn"], ["morgengruss", "im herbste", "herbstlied"]),
    (["beethoven"], ["blumchen wunderhold"]),
    (["cornelius"], ["botschaft"]),
    # gate20 works
    (["brahms"], ["bitteres zu sagen", "op 32 7", "op 32 no 7"]),
    (["chabrier"], ["gros dindons"]),
    (["greger"], ["keiner darw dat weiten", "op 34"]),
    (["mahler"], ["gluhend messer", "fahrenden gesellen"]),
    (["paradis"], ["sophia an siegwart"]),
    (["parry"], ["poet s song", "poets song", "op 12"]),
    (["reichardt"], ["traurige wanderer", "kaeuzlein", "kauzlein", "erinnrung zum bach",
                     "vaters klage", "volkslied", "die veilchen", "bergmannslied"]),
    (["schroter", "schroeter"], ["die liebe"]),
    (["mayer"], ["op 14"]),
    (["saint georges", "bologne"], ["op 1 no 3", "op 1 3"]),
]


class EvalBlocklist:
    def __init__(self) -> None:
        self.blocked_sha256: Set[str] = set()
        self.blocked_stems: Set[str] = {
            "hw-mozart-k458", "mozart-k458", "k458", "k.458",
        }
        self._load()

    def _load(self) -> None:
        if SUITE31_MANIFEST.exists():
            d = json.loads(SUITE31_MANIFEST.read_text(encoding="utf-8"))
            for p in d.get("pieces", []):
                slug = p.get("slug", "").lower()
                if slug:
                    self.blocked_stems.add(slug)
                for k in ("clean", "photo", "gt"):
                    sub = p.get(k) or {}
                    if sub.get("sha256"):
                        self.blocked_sha256.add(sub["sha256"].lower())
        if GATE20_MANIFEST.exists():
            d = json.loads(GATE20_MANIFEST.read_text(encoding="utf-8"))
            for p in d.get("pages", []):
                for k in ("clean_sha256", "degraded_sha256", "ground_truth_musicxml_sha256"):
                    if p.get(k):
                        self.blocked_sha256.add(p[k].lower())
                src = p.get("source", "")
                for m in re.findall(r"(?:lc|sq)\d+", src):
                    self.blocked_stems.add(m.lower())
                if p.get("score"):
                    self.blocked_stems.add(p["score"].lower())
        if DEBUSSY_MANIFEST.exists():
            d = json.loads(DEBUSSY_MANIFEST.read_text(encoding="utf-8"))
            for wid in d.get("eval_works_excluded", []):
                self.blocked_stems.add(wid.lower())

    def check(self, text_blob: str, stem: str = "", sha256: str = "") -> Optional[str]:
        if sha256 and sha256.lower() in self.blocked_sha256:
            return f"sha256:{sha256[:12]}"
        stem_l = stem.lower()
        for bs in self.blocked_stems:
            if bs and len(bs) >= 5 and (bs == stem_l or bs in stem_l):
                return f"stem:{bs}"
        nt = norm_text(text_blob)
        if not nt:
            return None
        # Direct k458 check
        if re.search(r"\bk\s*458\b", nt) or "mozart k458" in nt:
            return "rule:mozart_k458"
        for comps, works in EVAL_WORK_RULES:
            if comps:
                if not any(c in nt for c in comps):
                    continue
            for w in works:
                if w in nt:
                    return f"rule:{comps[0] if comps else 'any'}:{w}"
        return None


# =============================================================================
# Strict Permissive License Gate
# =============================================================================

ALLOWED_LICENSES = {
    "cc0", "cc0-1.0", "cc-zero", "publicdomain", "public domain",
    "pdm 1.0", "pdm-1.0", "cc-by-4.0", "cc-by-3.0", "cc-by",
    "mit", "bsd", "bsd-3-clause", "bsd-2-clause",
}


def is_permissive_license(lic: str) -> bool:
    if not lic:
        return False
    l = lic.strip().lower()
    if any(bad in l for bad in ("nc", "nd", "odbl", "sharealike", "by-sa", "gpl", "agpl", "all rights")):
        return False
    if l in ALLOWED_LICENSES:
        return True
    if l.startswith("public domain") or l.startswith("cc0") or l.startswith("pdm") or l.startswith("cc-by"):
        return True
    return False


# =============================================================================
# Composer Era & Genre Heuristics
# =============================================================================

BAROQUE_COMPOSERS = (
    "bach", "handel", "haendel", "vivaldi", "corelli", "scarlatti", "telemann",
    "purcell", "rameau", "couperin", "monteverdi", "buxtehude", "pachelbel",
    "albinoni", "pergolesi", "lully", "marcello", "tartini", "zipoli", "frescobaldi",
)
CLASSICAL_COMPOSERS = (
    "mozart", "haydn", "clementi", "boccherini", "gluck", "salieri", "stamitz",
    "kuhlau", "diabelli", "hummel", "dussek", "dittersdorf", "arriaga", "sammartini",
    "c p e bach", "j c bach", "cpe bach", "czerny", "sor", "carulli", "giuliani",
)
ROMANTIC_COMPOSERS = (
    "chopin", "schumann", "liszt", "brahms", "schubert", "mendelssohn", "debussy",
    "ravel", "rachmaninoff", "rachmaninov", "tchaikovsky", "dvorak", "grieg",
    "wagner", "verdi", "puccini", "mahler", "bruckner", "saint saens", "faure",
    "satie", "scriabin", "elgar", "sibelius", "franck", "berlioz", "bizet",
    "rossini", "donizetti", "bellini", "paganini", "albeniz", "granados", "wolf",
    "strauss", "beethoven", "weber", "moniuszko", "paderewski", "wieniawski",
    "szymanowski", "karlowicz", "noskowski", "elsner", "kurpinski", "dobrzynski",
    "oginski", "mayer", "hensel", "beach", "chabrier", "tosti", "parry", "reichardt",
)
MODERN_COMPOSERS = (
    "schoenberg", "berg", "webern", "stravinsky", "bartok", "prokofiev",
    "shostakovich", "hindemith", "messiaen", "copland", "gershwin", "ives",
    "milhaud", "poulenc", "honegger", "holst", "vaughan williams", "britten",
    "barber", "ligeti", "boulez", "stockhausen", "cage", "penderecki", "lutoslawski",
)


def classify_composer_era(text: str) -> Optional[str]:
    nt = norm_text(text)
    if not nt:
        return None
    for c in BAROQUE_COMPOSERS:
        if c in nt:
            return "classical_baroque"
    for c in CLASSICAL_COMPOSERS:
        if c in nt:
            return "classical_era"
    for c in MODERN_COMPOSERS:
        if c in nt:
            return "classical_modern_atonal"
    for c in ROMANTIC_COMPOSERS:
        if c in nt:
            return "classical_romantic"
    return None


# =============================================================================
# Fast MusicXML / MXL Metadata Extractor (streaming-safe, stdlib ET)
# =============================================================================

def _strip_ns(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def extract_xml_from_mxl(raw_bytes: bytes) -> Optional[bytes]:
    try:
        with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
            # Check META-INF/container.xml first
            rootfile = None
            if "META-INF/container.xml" in zf.namelist():
                try:
                    croot = ET.fromstring(zf.read("META-INF/container.xml"))
                    for elem in croot.iter():
                        if _strip_ns(elem.tag) == "rootfile" and elem.attrib.get("full-path"):
                            rootfile = elem.attrib["full-path"]
                            break
                except Exception:
                    pass
            if rootfile and rootfile in zf.namelist():
                return zf.read(rootfile)
            for name in zf.namelist():
                nl = name.lower()
                if (nl.endswith(".xml") or nl.endswith(".musicxml")) and not nl.startswith("meta-inf"):
                    return zf.read(name)
    except Exception:
        return None
    return None


_RE_WORK_TITLE = re.compile(r"<(?:work-title|movement-title)[^>]*>([^<]+)</")
_RE_COMPOSER = re.compile(r'<creator[^>]*type="composer"[^>]*>([^<]+)</creator>')
_RE_PART_NAME = re.compile(r"<part-name[^>]*>([^<]+)</part-name>")
_RE_STAVES = re.compile(r"<staves>(\d+)</staves>")
_RE_CLEF = re.compile(r"<clef(?:\s+number=\"(\d+)\")?(?:\s+[^>]*)?>(.{1,250}?)</clef>", re.DOTALL)
_RE_SIGN = re.compile(r"<sign>([^<]+)</sign>")
_RE_LINE = re.compile(r"<line>([^<]+)</line>")
_RE_OCT_CHG = re.compile(r"<clef-octave-change>(-?\d+)</clef-octave-change>")
_RE_FIFTHS = re.compile(r"<fifths>(-?\d+)</fifths>")
_RE_TIME = re.compile(r"<time(\s+[^>]*)?>(.{1,250}?)</time>", re.DOTALL)
_RE_BEATS = re.compile(r"<beats>([^<]+)</beats>")
_RE_BEAT_TYPE = re.compile(r"<beat-type>([^<]+)</beat-type>")
_RE_ALTER = re.compile(r"<alter>([^<]+)</alter>")
_RE_ACCID = re.compile(r"<accidental[^>]*>([^<]+)</accidental>")
_RE_TIME_MOD = re.compile(r"<time-modification>(.{1,250}?)</time-modification>", re.DOTALL)
_RE_VOICE = re.compile(r"<voice>([^<]+)</voice>")


def parse_musicxml_bytes(xml_bytes: bytes) -> Optional[Dict[str, Any]]:
    """Ultra-fast C-regex MusicXML feature extractor (~30x faster than ET.iterparse)."""
    try:
        text = xml_bytes.decode("utf-8", errors="ignore")
        if "<note" not in text:
            return None

        m_title = _RE_WORK_TITLE.search(text)
        title = m_title.group(1).strip() if m_title else ""
        m_comp = _RE_COMPOSER.search(text)
        composer = m_comp.group(1).strip() if m_comp else ""
        part_names = [x.strip() for x in _RE_PART_NAME.findall(text)[:24] if x.strip()]

        n_parts = max(1, text.count("<score-part "))
        if n_parts == 1 and "<part " in text:
            n_parts = max(1, text.count("<part "))

        # Staves per system: sum of max <staves> per part (approximated cleanly by splitting on <part )
        part_chunks = text.split("<part ")
        staves_per_system = 0
        distinct_keys_set: Set[int] = set()
        key_changes = 0
        clef_changes = 0
        meter_changes = 0
        max_part_measures = 0

        clefs: Set[str] = set()
        fifths_all: Set[int] = set()
        time_sigs: Set[str] = set()
        time_symbols: Set[str] = set()

        for pchunk in part_chunks[1:]:
            st_vals = [int(x) for x in _RE_STAVES.findall(pchunk)]
            staves_per_system += max(st_vals) if st_vals else 1

            pm = pchunk.count("<measure ")
            if pm > max_part_measures:
                max_part_measures = pm

            # Clefs in this part
            last_clef: Dict[str, str] = {}
            for cnum, cbody in _RE_CLEF.findall(pchunk):
                ms = _RE_SIGN.search(cbody)
                if not ms:
                    continue
                ml = _RE_LINE.search(cbody)
                mo = _RE_OCT_CHG.search(cbody)
                cname = _map_clef(ms.group(1), ml.group(1) if ml else "", int(mo.group(1)) if mo else 0)
                if cname:
                    clefs.add(cname)
                    ck = cnum or "1"
                    if ck in last_clef and last_clef[ck] != cname:
                        clef_changes += 1
                    last_clef[ck] = cname

            # Keys in this part
            p_fifths = [int(x) for x in _RE_FIFTHS.findall(pchunk)]
            if p_fifths:
                distinct_keys_set.add(p_fifths[0])
                fifths_all.update(p_fifths)
                for a, b in zip(p_fifths, p_fifths[1:]):
                    if a != b:
                        key_changes += 1

            # Time signatures in this part
            p_times: List[str] = []
            for tattr, tbody in _RE_TIME.findall(pchunk):
                if 'symbol="common"' in tattr:
                    time_symbols.add("common")
                elif 'symbol="cut"' in tattr:
                    time_symbols.add("cut")
                mb = _RE_BEATS.search(tbody)
                mt = _RE_BEAT_TYPE.search(tbody)
                if mb and mt:
                    ts = f"{mb.group(1).strip()}/{mt.group(1).strip()}"
                    time_sigs.add(ts)
                    p_times.append(ts)
            for a, b in zip(p_times, p_times[1:]):
                if a != b:
                    meter_changes += 1

        if staves_per_system == 0:
            staves_per_system = n_parts
        distinct_keys_in_system = max(1, len(distinct_keys_set))

        note_total = text.count("<note")
        rest_count = text.count("<rest")
        note_count = max(1, note_total - rest_count)
        chord_count = text.count("<chord")
        max_chord_size = 2 if chord_count > 0 else 1
        if chord_count > note_count * 0.45:
            max_chord_size = 4

        voices_found = set(_RE_VOICE.findall(text[:80000]))
        # Normalize voices per staff roughly
        max_voices_per_staff = max(1, min(4, int(math.ceil(len(voices_found) / float(max(1, staves_per_system))))))
        if len(voices_found) >= 2 and staves_per_system == 1:
            max_voices_per_staff = min(4, len(voices_found))

        has_anacrusis = 'implicit="yes"' in text[:8000]
        has_32nd_or_faster = ("<type>32nd</type>" in text or "<type>64th</type>" in text or "<type>128th</type>" in text)
        has_double_dot = "</dot><dot" in text or "<dot/><dot" in text or "<dot /><dot" in text
        grace_count = text.count("<grace")

        tm_matches = _RE_TIME_MOD.findall(text)
        tuplet_count = len(tm_matches)
        non_triplet_tuplet = any(("<actual-notes>3</actual-notes>" not in tm or "<normal-notes>2</normal-notes>" not in tm) for tm in tm_matches[:40])

        has_cross_staff_beam = ("<staff>2</staff>" in text and "<beam " in text and staves_per_system >= 2 and "<staff>1</staff>" in text)

        has_double_accidental = False
        has_microtonal = False
        for av_s in _RE_ALTER.findall(text):
            try:
                av = float(av_s)
                if abs(av) >= 1.9:
                    has_double_accidental = True
                elif av != int(av):
                    has_microtonal = True
            except ValueError:
                pass
        acc_list = _RE_ACCID.findall(text)
        accidental_count = len(acc_list)
        if not has_double_accidental or not has_microtonal:
            for a in acc_list:
                if a in ("double-sharp", "flat-flat", "sharp-sharp", "double-flat"):
                    has_double_accidental = True
                elif "quarter" in a or "three-quarters" in a or "slash" in a or "sori" in a or "koron" in a:
                    has_microtonal = True

        multibar_rest_count = text.count("<multiple-rest")
        slur_count = text.count("<slur") + text.count("<tied")
        dyn_count = text.count("<dynamics")
        hairpin_count = text.count("<wedge")
        artic_count = (
            text.count("<staccato") + text.count("<accent") + text.count("<tenuto") + text.count("<fermata")
        )
        ornament_count = (
            text.count("<trill-mark") + text.count("<mordent") + text.count("<turn")
            + text.count("<arpeggiate") + text.count("<glissando") + text.count("<tremolo")
        )
        lyric_syllables = text.count("<lyric")
        has_harmony = "<harmony" in text
        has_slash = "<beat-repeat" in text or "<slash" in text
        has_fingering_etc = (
            "<fingering" in text or "<up-bow" in text or "<down-bow" in text
            or "<pedal" in text or "<rehearsal" in text or "<fret" in text
        )
        repeat_barlines = text.count("<repeat")
        has_volta_repeat = repeat_barlines > 0 or "<ending" in text or "<coda" in text or "<segno" in text
        page_breaks = text.count('new-page="yes"')

        if page_breaks > 0:
            est_pages = page_breaks + 1
        else:
            if staves_per_system >= 8:
                mpp = 10
            elif staves_per_system >= 4:
                mpp = 18
            elif staves_per_system >= 2:
                mpp = 28
            else:
                mpp = 36
            est_pages = max(1, int(math.ceil(max(1, max_part_measures) / float(mpp))))

        return {
            "title": title[:200],
            "composer": composer[:120],
            "part_names": part_names,
            "n_parts": n_parts,
            "staves_per_system": staves_per_system,
            "clefs": sorted(clefs) if clefs else ["G-2"],
            "clef_changes": clef_changes,
            "distinct_keys_in_system": distinct_keys_in_system,
            "fifths": sorted(fifths_all) if fifths_all else [0],
            "key_changes": key_changes,
            "time_sigs": sorted(time_sigs) if time_sigs else ["4/4"],
            "time_symbols": sorted(time_symbols),
            "meter_changes": meter_changes,
            "has_anacrusis": has_anacrusis,
            "has_32nd_or_faster": has_32nd_or_faster,
            "has_double_dot": has_double_dot,
            "grace_count": grace_count,
            "tuplet_count": tuplet_count,
            "non_triplet_tuplet": non_triplet_tuplet,
            "has_cross_staff_beam": has_cross_staff_beam,
            "has_double_accidental": has_double_accidental,
            "has_microtonal": has_microtonal,
            "accidental_count": accidental_count,
            "note_count": note_count,
            "rest_count": rest_count,
            "measures": max_part_measures,
            "multibar_rest_count": multibar_rest_count,
            "max_voices_per_staff": max_voices_per_staff,
            "max_chord_size": max_chord_size,
            "slur_count": slur_count,
            "dyn_count": dyn_count,
            "hairpin_count": hairpin_count,
            "artic_count": artic_count,
            "ornament_count": ornament_count,
            "has_lyrics": lyric_syllables > 0,
            "lyric_syllables": lyric_syllables,
            "has_harmony": has_harmony,
            "has_slash": has_slash,
            "has_fingering_etc": has_fingering_etc,
            "has_volta_repeat": has_volta_repeat,
            "repeat_barlines": repeat_barlines,
            "est_pages": min(est_pages, 60),
        }
    except Exception:
        return None


def _map_clef(sign: str, line: str, oct_chg: int) -> Optional[str]:
    s = sign.strip().upper()
    if not s:
        return None
    if s == "PERCUSSION":
        return "clef-p-2"
    if s == "TAB":
        return "clef-T-2"
    if s == "G":
        ln = line or "2"
        if oct_chg == -1:
            return "G-2-8vb"
        if oct_chg == 1:
            return "G-2-8va"
        return f"G-{ln}"
    if s == "F":
        ln = line or "4"
        if oct_chg == -1:
            return "F-4-8vb"
        return f"F-{ln}"
    if s == "C":
        ln = line or "3"
        return f"C-{ln}"
    return None


# =============================================================================
# Fast Humdrum .krn Metadata Extractor (NIFC Polish & Chopin)
# =============================================================================

_KRN_CLEF_RE = re.compile(r"\*clef([A-Za-z0-9\-]+)")
_KRN_KEY_RE = re.compile(r"\*k\[([a-g#\-]*)\]")
_KRN_TIME_RE = re.compile(r"\*M(\d+/\d+)")


def parse_krn_file(path: Path) -> Optional[Dict[str, Any]]:
    try:
        raw = path.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        text = raw.decode("utf-8", errors="replace")
    except Exception:
        return None

    title = ""
    composer = ""
    genre_txt = ""
    ain_txt = ""
    part_names: List[str] = []
    kern_spines = 0
    clefs: Set[str] = set()
    clef_hits = 0
    fifths_all: Set[int] = set()
    key_hits = 0
    time_sigs: Set[str] = set()
    time_symbols: Set[str] = set()
    meter_hits = 0
    has_transposing = False
    has_lyrics = False
    lyric_lines = 0
    has_32nd = False
    has_double_dot = False
    grace_count = 0
    has_double_acc = False
    note_count = 0
    rest_count = 0
    measures = 0
    slur_count = 0
    dyn_count = 0
    artic_count = 0
    ornament_count = 0
    has_fingering = False
    has_volta = False
    max_chord_size = 1
    has_multivoice = False

    lines = text.splitlines()
    for line in lines:
        if not line:
            continue
        if line.startswith("!!!"):
            if line.startswith("!!!COM:"):
                composer = line[7:].strip()
            elif line.startswith("!!!OTL:") or line.startswith("!!!OPR:"):
                if not title:
                    title = line[7:].strip()
            elif line.startswith("!!!AGN:"):
                genre_txt = line[7:].strip()
            elif line.startswith("!!!AIN:"):
                ain_txt = line[7:].strip()
            continue
        if line.startswith("**"):
            cols = line.split("\t")
            kern_spines = sum(1 for c in cols if c == "**kern")
            if any(c in ("**text", "**silbe") for c in cols):
                has_lyrics = True
            if any(c == "**fing" for c in cols):
                has_fingering = True
            continue
        if line.startswith("*"):
            if "*clef" in line:
                for m in _KRN_CLEF_RE.findall(line):
                    clef_hits += 1
                    cm = _map_krn_clef(m)
                    if cm:
                        clefs.add(cm)
            if "*k[" in line:
                ks_in_line = set()
                for m in _KRN_KEY_RE.findall(line):
                    f = len(m) // 2
                    if "-" in m:
                        f = -f
                    fifths_all.add(f)
                    ks_in_line.add(f)
                    key_hits += 1
                if len(ks_in_line) >= 2:
                    has_transposing = True
            if "*M" in line:
                for m in _KRN_TIME_RE.findall(line):
                    time_sigs.add(m)
                    meter_hits += 1
            if "*met(c|)" in line:
                time_symbols.add("cut")
            elif "*met(c)" in line:
                time_symbols.add("common")
            if "*ITrd" in line:
                has_transposing = True
            if "*I\"" in line:
                for tok in line.split("\t"):
                    if tok.startswith("*I\""):
                        part_names.append(tok[3:].strip())
            if "*^" in line:
                has_multivoice = True
            continue
        if line.startswith("="):
            measures += 1
            if ":|" in line or "|:" in line:
                has_volta = True
            continue
        if line.startswith("!"):
            continue
        # Data line (sample fast)
        if "##" in line or "--" in line:
            has_double_acc = True
        if "32" in line or "64" in line:
            has_32nd = True
        if ".." in line:
            has_double_dot = True
        if "q" in line:
            grace_count += 1
        if "(" in line:
            slur_count += 1
        if "t" in line or "T" in line or "M" in line:
            ornament_count += 1
        if "'" in line or "^" in line or ";" in line:
            artic_count += 1
        if "\t" in line:
            toks = line.split("\t")
            for t in toks[:kern_spines]:
                if t == ".":
                    continue
                if "r" in t:
                    rest_count += 1
                else:
                    note_count += 1
                    if " " in t:
                        cs = t.count(" ") + 1
                        if cs > max_chord_size:
                            max_chord_size = cs
            if has_lyrics and len(toks) > kern_spines:
                lyric_lines += 1
            if any(t in ("p", "pp", "mp", "mf", "f", "ff", "sfz", "<", ">") for t in toks[kern_spines:]):
                dyn_count += 1

    staves = max(1, kern_spines)
    if staves >= 8:
        mpp = 10
    elif staves >= 4:
        mpp = 18
    elif staves >= 2:
        mpp = 28
    else:
        mpp = 36
    est_pages = max(1, int(math.ceil(max(1, measures) / float(mpp))))

    return {
        "sha256": sha,
        "title": title or path.stem,
        "composer": composer,
        "genre_hint": genre_txt,
        "part_names": part_names[:24] if part_names else ([ain_txt] if ain_txt else []),
        "n_parts": staves,
        "staves_per_system": staves,
        "clefs": sorted(clefs) if clefs else ["G-2"],
        "clef_changes": max(0, clef_hits - staves),
        "distinct_keys_in_system": 2 if has_transposing else 1,
        "fifths": sorted(fifths_all) if fifths_all else [0],
        "key_changes": max(0, (key_hits // max(1, staves)) - 1),
        "time_sigs": sorted(time_sigs) if time_sigs else ["4/4"],
        "time_symbols": sorted(time_symbols),
        "meter_changes": max(0, (meter_hits // max(1, staves)) - 1),
        "has_anacrusis": False,
        "has_32nd_or_faster": has_32nd,
        "has_double_dot": has_double_dot,
        "grace_count": grace_count,
        "tuplet_count": 0,
        "non_triplet_tuplet": False,
        "has_cross_staff_beam": False,
        "has_double_accidental": has_double_acc,
        "has_microtonal": False,
        "accidental_count": int(note_count * 0.1),
        "note_count": max(1, note_count),
        "rest_count": rest_count,
        "measures": measures,
        "multibar_rest_count": 0,
        "max_voices_per_staff": 2 if has_multivoice else 1,
        "max_chord_size": max_chord_size,
        "slur_count": slur_count,
        "dyn_count": dyn_count,
        "hairpin_count": dyn_count // 3,
        "artic_count": artic_count,
        "ornament_count": ornament_count,
        "has_lyrics": has_lyrics,
        "lyric_syllables": lyric_lines,
        "has_harmony": False,
        "has_slash": False,
        "has_fingering_etc": has_fingering,
        "has_volta_repeat": has_volta,
        "repeat_barlines": 2 if has_volta else 0,
        "est_pages": min(est_pages, 60),
    }


def _map_krn_clef(m: str) -> Optional[str]:
    if m.startswith("Gv"):
        return "G-2-8vb"
    if m.startswith("G^"):
        return "G-2-8va"
    if m.startswith("Fv"):
        return "F-4-8vb"
    if len(m) >= 2 and m[0] in ("G", "F", "C") and m[1].isdigit():
        return f"{m[0]}-{m[1]}"
    if m.startswith("X"):
        return "clef-p-2"
    return None


# =============================================================================
# Fast MuseScore .mscx Metadata Extractor (DCML Bach Chorales)
# =============================================================================

def parse_mscx_file(path: Path) -> Optional[Dict[str, Any]]:
    try:
        raw = path.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        text = raw.decode("utf-8", errors="replace")
    except Exception:
        return None
    staves = max(1, len(re.findall(r"<Staff id=\"\d+\">", text)))
    measures = max(1, text.count("<Measure") // max(1, staves))
    fifths = [int(x) for x in re.findall(r"<accidental>(-?\d+)</accidental>", text)[:4]]
    has_lyrics = "<Lyrics>" in text
    has_fermata = "fermata" in text.lower()
    has_repeat = "<endRepeat>" in text or "<startRepeat>" in text
    return {
        "sha256": sha,
        "title": path.stem,
        "composer": "Johann Sebastian Bach",
        "part_names": ["Soprano", "Alto", "Tenor", "Bass"][:staves],
        "n_parts": staves,
        "staves_per_system": staves,
        "clefs": ["G-2", "F-4"],
        "clef_changes": 0,
        "distinct_keys_in_system": 1,
        "fifths": sorted(set(fifths)) if fifths else [0],
        "key_changes": 0,
        "time_sigs": ["4/4"],
        "time_symbols": ["common"],
        "meter_changes": 0,
        "has_anacrusis": True,
        "has_32nd_or_faster": False,
        "has_double_dot": False,
        "grace_count": 0,
        "tuplet_count": 0,
        "non_triplet_tuplet": False,
        "has_cross_staff_beam": False,
        "has_double_accidental": False,
        "has_microtonal": False,
        "accidental_count": text.count("<Accidental>"),
        "note_count": max(1, text.count("<Note>")),
        "rest_count": text.count("<Rest>"),
        "measures": measures,
        "multibar_rest_count": 0,
        "max_voices_per_staff": 2,
        "max_chord_size": 2,
        "slur_count": text.count("<Slur>"),
        "dyn_count": text.count("<Dynamic>"),
        "hairpin_count": text.count("<HairPin>"),
        "artic_count": 10 if has_fermata else 2,
        "ornament_count": 0,
        "has_lyrics": has_lyrics,
        "lyric_syllables": 40 if has_lyrics else 0,
        "has_harmony": "<Harmony>" in text,
        "has_slash": False,
        "has_fingering_etc": False,
        "has_volta_repeat": has_repeat,
        "repeat_barlines": 2 if has_repeat else 0,
        "est_pages": max(1, int(math.ceil(measures / 20.0))),
    }


# =============================================================================
# 14-Dimension Candidate Bucket Classifier
# =============================================================================

def classify_inherent_dims(
    meta: Dict[str, Any],
    source_id: str,
    page_num: int,
    total_pages: int,
    is_tier_c: bool = False,
) -> Tuple[Dict[str, List[str]], str, List[str]]:
    """
    Returns:
      - valid_buckets: dict[dim_name -> list of ALL valid bucket IDs for this unit]
      - supervision_tier: tier_a_v1_exact_gt | tier_b_v2_tagged_gt | tier_c_unlabeled_real_ssl
      - v2_tags: list of v2 extension tags
    """
    vb: Dict[str, List[str]] = {}
    v2_tags: List[str] = []

    title_blob = " ".join([
        str(meta.get("title", "")),
        str(meta.get("composer", "")),
        str(meta.get("genre_hint", "")),
        " ".join(meta.get("part_names", [])),
    ]).lower()

    staves = int(meta.get("staves_per_system", 1))
    n_parts = int(meta.get("n_parts", 1))
    clefs = set(meta.get("clefs", ["G-2"]))
    fifths = meta.get("fifths", [0])
    max_abs_fifth = max((abs(int(f)) for f in fifths), default=0)
    time_sigs = set(meta.get("time_sigs", ["4/4"]))
    time_syms = set(meta.get("time_symbols", []))
    note_cnt = max(1, int(meta.get("note_count", 1)))
    rest_cnt = int(meta.get("rest_count", 0))
    acc_cnt = int(meta.get("accidental_count", 0))
    acc_density = acc_cnt / float(note_cnt)

    # --- V2 tags ---
    if meta.get("has_microtonal"):
        v2_tags.append("v2_microtonal")
    if "clef-T-2" in clefs:
        v2_tags.append("v2_tab_frets")
    if meta.get("has_harmony") or meta.get("has_slash") or source_id == "openewld_jazz":
        v2_tags.append("v2_chord_symbols")
    if meta.get("has_lyrics") and int(meta.get("lyric_syllables", 0)) >= 10:
        v2_tags.append("v2_lyrics")
    if "clef-p-2" in clefs:
        v2_tags.append("v2_percussion_kit")
    if int(meta.get("ornament_count", 0)) >= 2 or meta.get("has_volta_repeat"):
        v2_tags.append("v2_ornaments_expr")

    # --- Dim 13: Supervision Tier ---
    if is_tier_c:
        sup_tier = "tier_c_unlabeled_real_ssl"
    elif "v2_microtonal" in v2_tags or "v2_tab_frets" in v2_tags or "v2_chord_symbols" in v2_tags or "v2_percussion_kit" in v2_tags:
        sup_tier = "tier_b_v2_tagged_gt"
    else:
        sup_tier = "tier_a_v1_exact_gt"
    vb["13_supervision_tier"] = [sup_tier]

    # --- Dim 01: Genre / Tradition ---
    d01: List[str] = []
    if meta.get("has_microtonal") or "experimental" in title_blob or "microtonal" in title_blob:
        d01.append("contemporary_microtonal_exp")
    if source_id == "openewld_jazz" or meta.get("has_harmony") or any(w in title_blob for w in ("jazz", "swing", "bebop", "blues", "bossa")):
        d01.append("jazz_lead_sheets")
    if any(w in title_blob for w in ("pop", "rock", "soundtrack", "film", "musical", "ragtime", "rag ", "metal", "hiphop", "anime", "video game", "electronic", "disney")):
        d01.append("pop_rock_film_theatre")
    if any(w in title_blob for w in ("folk", "jig", "reel", "hornpipe", "klezmer", "balkan", "polska", "traditional", "irish", "scottish", "celtic", "mazurka", "polonaise", "waltz", "worldmusic")):
        d01.append("folk_traditional")
    if source_id == "dcml_bach_chorales" or any(w in title_blob for w in ("choral", "chorale", "hymn", "sacred", "mass", "messe", "missa", "motet", "requiem", "psalm", "salve", "ave maria", "religious", "madrigal", "cantata", "satb", "choir", "chorus")):
        d01.append("choral_sacred_early")
    if any(w in title_blob for w in ("marching", "concert band", "wind ensemble", "big band", "brass band", "brass quintet", "wind quintet", "fanfare", "march", "saxophone", "trumpet", "trombone", "tuba", "euphonium", "snare", "drumline")):
        d01.append("band_bigband_marching")
    if any(w in title_blob for w in ("etude", "étude", "study", "studies", "exercise", "exercises", "method", "scale", "sonatina", "czerny", "hanon", "wohlfahrt", "kreutzer", "burgmuller")):
        d01.append("solo_etudes_pedagogical")
    if acc_density > 0.25 or int(meta.get("meter_changes", 0)) >= 3:
        d01.append("classical_modern_atonal")
    era = classify_composer_era(title_blob)
    if era and era not in d01:
        d01.append(era)
    if not d01:
        if source_id in ("openscore_lieder", "nifc_chopin_first_editions", "debussy_bnf_and_real_hw_local"):
            d01.append("classical_romantic")
        elif source_id in ("openscore_string_quartets", "malerlab_ossq_omr"):
            d01.append("classical_era")
        elif source_id == "rism_iiif_manuscripts":
            d01.append("classical_baroque")
        else:
            d01.append("classical_era")
    vb["01_genre_tradition"] = d01

    # --- Dim 02: Ensemble & Score Type ---
    d02: List[str] = []
    if is_tier_c:
        if any(w in title_blob for w in ("quartet", "quatuor")):
            d02.append("string_quartet_wind_quintet")
        elif any(w in title_blob for w in ("symphon", "overture", "ouverture", "concerto", "orchestr")):
            d02.append("conductor_orchestral_full")
        elif any(w in title_blob for w in ("lied", "song", "chanson", "gesang", "aria", "voice")):
            d02.append("voice_plus_piano_lieder")
        elif any(w in title_blob for w in ("piano", "klavier", "sonata", "etude", "czerny")):
            d02.append("solo_grand_staff")
        elif any(w in title_blob for w in ("violin", "flute", "cello", "solo", "studies")):
            d02.append("solo_single_staff")
        elif any(w in title_blob for w in ("mass", "salve", "antiphon", "hymn", "choral")):
            d02.append("choral_satb_and_acmp")
        else:
            d02.extend(["solo_grand_staff", "chamber_duet_trio"])
    else:
        if staves >= 8:
            d02.append("conductor_orchestral_full")
        if ("jazz_lead_sheets" in d01 or "band_bigband_marching" in d01) and staves >= 4:
            d02.append("big_band_jazz_combo")
        if "organ" in title_blob and staves == 3:
            d02.append("organ_3staff")
        if source_id == "openscore_lieder" or (staves == 3 and any(w in title_blob for w in ("voice", "soprano", "alto", "tenor", "baritone", "vocal", "lied", "song", "piano"))):
            d02.append("voice_plus_piano_lieder")
        if source_id in ("openscore_string_quartets", "malerlab_ossq_omr") or (staves in (4, 5) and not ("choral_sacred_early" in d01)):
            d02.append("string_quartet_wind_quintet")
        if ("choral_sacred_early" in d01 or any(w in title_blob for w in ("soprano", "choir", "chorus", "satb", "vocal"))) and staves >= 2:
            d02.append("choral_satb_and_acmp")
        if int(meta.get("multibar_rest_count", 0)) >= 2 or (staves == 1 and any(w in title_blob for w in ("part", "violin i", "violin 1", "clarinet", "trumpet", "horn", "flute", "oboe", "bassoon", "cello", "viola", "trombone"))):
            d02.append("extracted_instrumental_part")
        if staves in (2, 3) and n_parts in (2, 3):
            d02.append("chamber_duet_trio")
        if staves == 2 and (n_parts == 1 or any(w in title_blob for w in ("piano", "harp", "harpsichord", "accordion", "keyboard", "klavier"))):
            d02.append("solo_grand_staff")
        if staves == 1:
            d02.append("solo_single_staff")
        if not d02:
            d02.append("solo_grand_staff" if staves == 2 else ("conductor_orchestral_full" if staves >= 6 else "chamber_duet_trio"))
    vb["02_ensemble_score_type"] = d02

    # --- Dim 03: Clefs & Transposition ---
    d03: List[str] = []
    if not is_tier_c:
        if "clef-p-2" in clefs:
            d03.append("percussion_drum_kit")
        if "clef-T-2" in clefs:
            d03.append("guitar_bass_tab")
        if any(c in clefs for c in ("C-1", "C-2", "C-5", "G-1", "F-3")):
            d03.append("historical_c_clefs")
        if any(c in clefs for c in ("G-2-8vb", "G-2-8va", "F-4-8vb")):
            d03.append("octave_clefs")
        if int(meta.get("distinct_keys_in_system", 1)) >= 2:
            d03.append("transposing_multi_key")
        if "C-3" in clefs:
            d03.append("alto_c3")
        if "C-4" in clefs:
            d03.append("tenor_c4")
        if int(meta.get("clef_changes", 0)) >= 1:
            d03.append("mid_measure_clef_changes")
        if clefs == {"F-4"}:
            d03.append("bass_only")
        if "G-2" in clefs and "F-4" in clefs:
            d03.append("treble_and_bass")
        if clefs == {"G-2"} or not d03:
            d03.append("treble_only")
    else:
        d03.append("treble_and_bass")
    vb["03_clefs_transposition"] = d03

    # --- Dim 04: Meter & Rhythm Complexity ---
    d04: List[str] = []
    if not is_tier_c:
        odd_nums = {"5", "7", "10", "11", "13", "15", "17"}
        for ts in time_sigs:
            parts = ts.split("/")
            if len(parts) == 2:
                num, den = parts[0], parts[1]
                if num in odd_nums:
                    d04.append("asymmetric_odd_meters")
                if ts in ("6/8", "9/8", "12/8", "6/4", "9/4", "12/16"):
                    d04.append("compound_meters")
                if ts in ("2/2", "3/2", "4/2", "6/2", "2/1", "3/1"):
                    d04.append("alla_breve_half_meters")
        if time_syms:
            d04.append("common_cut_symbols")
        if int(meta.get("meter_changes", 0)) >= 1:
            d04.append("polymeter_meter_changes")
        if meta.get("has_anacrusis"):
            d04.append("anacrusis_split_repeats")
        if meta.get("has_32nd_or_faster") or meta.get("has_double_dot") or int(meta.get("grace_count", 0)) >= 2:
            d04.append("micro_rhythms_grace")
        if int(meta.get("tuplet_count", 0)) >= 2 or meta.get("non_triplet_tuplet"):
            d04.append("complex_tuplets")
        if meta.get("has_cross_staff_beam") or "jazz_lead_sheets" in d01 or "ragtime" in title_blob:
            d04.append("swing_syncopation_crossbeam")
        if not d04 or any(ts in ("4/4", "3/4", "2/4") for ts in time_sigs):
            d04.append("simple_standard")
    else:
        d04.append("simple_standard")
    vb["04_meter_rhythm_complexity"] = list(dict.fromkeys(d04))

    # --- Dim 05: Keys, Accidentals & Tonality ---
    d05: List[str] = []
    if not is_tier_c:
        if meta.get("has_microtonal"):
            d05.append("microtonal_accidentals")
        if meta.get("has_double_accidental"):
            d05.append("double_accidentals")
        if int(meta.get("key_changes", 0)) >= 1:
            d05.append("mid_system_key_changes")
        if acc_density > 0.22:
            d05.append("atonal_heavy_courtesy")
        if max_abs_fifth >= 4:
            d05.append("key_extreme_4_to_7")
        elif 1 <= max_abs_fifth <= 3:
            d05.append("key_common_1_to_3")
        else:
            d05.append("key_open_0")
    else:
        d05.append("key_common_1_to_3")
    vb["05_keys_accidentals_tonality"] = d05

    # --- Dim 06: Polyphony & Voice Density ---
    d06: List[str] = []
    if not is_tier_c:
        mv = int(meta.get("max_voices_per_staff", 1))
        mc = int(meta.get("max_chord_size", 1))
        rest_frac = rest_cnt / float(max(1, note_cnt + rest_cnt))
        notes_per_meas = note_cnt / float(max(1, int(meta.get("measures", 1))))
        if rest_frac >= 0.40 or int(meta.get("multibar_rest_count", 0)) >= 2:
            d06.append("sparse_rest_heavy")
        if notes_per_meas >= 24 or source_id == "nifc_chopin_first_editions":
            d06.append("dense_romantic_virtuoso")
        if mv >= 3:
            d06.append("counterpoint_3_4voice_staff")
        if mv == 2:
            d06.append("counterpoint_2voice_staff")
        if mc >= 2 and mv == 1:
            d06.append("homophonic_chords")
        if mv == 1 and mc == 1:
            d06.append("pure_monophony")
        if not d06:
            d06.append("homophonic_chords")
    else:
        d06.append("homophonic_chords")
    vb["06_polyphony_voice_density"] = d06

    # --- Dim 07: Articulations, Expressions & Text ---
    d07: List[str] = []
    if not is_tier_c:
        if meta.get("has_harmony") or meta.get("has_slash") or source_id == "openewld_jazz":
            d07.append("jazz_chord_symbols_slash")
        if meta.get("has_lyrics") and int(meta.get("lyric_syllables", 0)) >= 20:
            d07.append("heavy_lyrics_multiverse")
        if meta.get("has_fingering_etc"):
            d07.append("fingerings_bowings_pedal_rehearsal")
        if int(meta.get("artic_count", 0)) >= 8 or int(meta.get("ornament_count", 0)) >= 2:
            d07.append("articulations_ornaments")
        if int(meta.get("dyn_count", 0)) >= 4 and int(meta.get("hairpin_count", 0)) >= 2:
            d07.append("dynamics_hairpins")
        if int(meta.get("slur_count", 0)) >= 6:
            d07.append("slurs_ties_cross_system")
        if not d07:
            d07.append("clean_minimal_markings")
    else:
        d07.append("slurs_ties_cross_system")
    vb["07_articulations_expressions_text"] = d07

    # --- Dim 08: Page Structure & Continuity ---
    d08: List[str] = []
    if meta.get("has_volta_repeat") or int(meta.get("repeat_barlines", 0)) >= 2:
        d08.append("repeats_voltas_navigation")
    if total_pages <= 1:
        d08.append("single_page_complete")
    elif page_num == 1:
        d08.append("multipage_page1_title_indented")
    elif page_num >= total_pages:
        d08.append("multipage_final_stub")
    else:
        d08.append("multipage_interior")
    vb["08_page_structure_continuity"] = d08

    # --- Tier C inherent visual dimensions (09, 10, 11, 12, 14) ---
    if is_tier_c:
        is_hw = source_id in ("rism_iiif_manuscripts", "debussy_bnf_and_real_hw_local")
        vb["09_engraving_renderer_font"] = ["real_hw_and_hw_sim"] if is_hw else ["real_19c_plate_engravings"]
        vb["11_visual_domain_capture"] = ["historical_library_scans"]
        vb["12_handwritten_archival"] = ["real_historical_autograph"] if is_hw else ["printed_engraved"]

        w = int(meta.get("width", 1200))
        h = int(meta.get("height", 1600))
        orient = meta.get("orientation", "landscape" if w > h else "portrait")
        long_edge = max(w, h)
        if orient == "landscape":
            vb["10_page_geometry_resolution"] = ["landscape_quartet_orchestral"]
        elif long_edge <= 1150:
            vb["10_page_geometry_resolution"] = ["low_res_mobile_web"]
        else:
            vb["10_page_geometry_resolution"] = ["portrait_a4_letter_hires", "portrait_b4_folio_large"]

        staff_rows = float(meta.get("staff_rows", 80.0))
        if staff_rows >= 150:
            vb["14_layout_density_spacing"] = ["ultra_dense_cramped_miniature", "dense_multi_system"]
        elif staff_rows <= 40:
            vb["14_layout_density_spacing"] = ["sparse_beginner_large_print", "empty_prerule_and_text_heavy"]
        else:
            vb["14_layout_density_spacing"] = ["standard_engraving_density", "dense_multi_system"]

    return vb, sup_tier, v2_tags


# =============================================================================
# Streaming PDMX Cache Builder
# =============================================================================

def build_or_load_pdmx_cache(
    blocklist: EvalBlocklist,
    rejected_fh: Any,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pdmx_cache = CACHE_DIR / "pdmx_meta.jsonl"
    pdmx_csv = CACHE_DIR / "PDMX.csv"
    pdmx_dir = Path(r"D:\tmp\corpus10x\pdmx")

    # Ensure PDMX.csv is extracted
    if not pdmx_csv.exists():
        print("[PDMX] Extracting PDMX/PDMX.csv from PDMX.tar.gz...", flush=True)
        with tarfile.open(pdmx_dir / "PDMX.tar.gz", "r|gz") as tf:
            for m in tf:
                if m.name == "PDMX/PDMX.csv":
                    f = tf.extractfile(m)
                    with open(pdmx_csv, "wb") as out:
                        while True:
                            buf = f.read(1024 * 1024)
                            if not buf:
                                break
                            out.write(buf)
                    break

    # Load subset sets (no_license_conflict & deduplicated)
    nlc_set: Set[str] = set()
    dedup_set: Set[str] = set()
    with tarfile.open(pdmx_dir / "subset_paths.tar.gz", "r:gz") as tf:
        for m in tf.getmembers():
            if m.name.endswith("no_license_conflict.txt"):
                for line in tf.extractfile(m):
                    nlc_set.add(os.path.basename(line.decode().strip()).replace(".json", ""))
            elif m.name.endswith("deduplicated.txt"):
                for line in tf.extractfile(m):
                    dedup_set.add(os.path.basename(line.decode().strip()).replace(".json", ""))

    # Load CSV metadata for allowed qids
    csv_meta: Dict[str, Dict[str, str]] = {}
    with open(pdmx_csv, "r", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            qid = os.path.basename(row["path"]).replace(".json", "")
            if qid not in nlc_set or qid not in dedup_set:
                continue
            lic = row.get("license", "")
            if not is_permissive_license(lic):
                continue
            csv_meta[qid] = {
                "license": lic,
                "genres": row.get("genres", ""),
                "tags": row.get("tags", ""),
                "title": row.get("title", "") or row.get("song_name", ""),
                "composer": row.get("composer_name", "") or row.get("artist_name", ""),
                "rating": row.get("rating", "0"),
                "n_tracks": row.get("n_tracks", "1"),
            }
    print(f"[PDMX] Eligible deduplicated+no_license_conflict rows: {len(csv_meta)}", flush=True)

    # Check if cache already exists and is complete
    if pdmx_cache.exists() and pdmx_cache.stat().st_size > 10_000_000:
        print(f"[PDMX] Loading cached XML metadata from {pdmx_cache}...", flush=True)
        records: List[Dict[str, Any]] = []
        with open(pdmx_cache, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                records.append(rec)
                if limit and len(records) >= limit:
                    break
        print(f"[PDMX] Loaded {len(records)} cached PDMX records.", flush=True)
        return records

    # Stream mxl.tar.gz ONCE with bounded ThreadPoolExecutor(max_workers=4)
    print(f"[PDMX] Streaming {pdmx_dir / 'mxl.tar.gz'} (max_workers={MAX_WORKERS})...", flush=True)
    t0 = time.time()
    tmp_cache = pdmx_cache.with_suffix(".jsonl.tmp")

    def _process_member(item: Tuple[str, str, bytes, Dict[str, str]]) -> Optional[Dict[str, Any]]:
        qid, member_name, raw_bytes, cmeta = item
        sha = hashlib.sha256(raw_bytes).hexdigest()
        xml_bytes = extract_xml_from_mxl(raw_bytes)
        if not xml_bytes:
            return None
        parsed = parse_musicxml_bytes(xml_bytes)
        if not parsed:
            return None
        # Merge CSV metadata where XML is blank
        if not parsed["title"] and cmeta["title"] not in ("", "NA"):
            parsed["title"] = cmeta["title"][:200]
        if not parsed["composer"] and cmeta["composer"] not in ("", "NA"):
            parsed["composer"] = cmeta["composer"][:120]
        parsed["genre_hint"] = f"{cmeta['genres']} {cmeta['tags']}".replace("NA", "").strip()
        parsed["license"] = "CC0-1.0" if cmeta["license"] == "cc-zero" else "Public Domain"
        parsed["sha256"] = sha
        parsed["qid"] = qid
        parsed["tar_member"] = member_name
        parsed["rating"] = float(cmeta["rating"] or 0.0)
        return parsed

    records = []
    scanned = 0
    matched = 0
    with open(tmp_cache, "w", encoding="utf-8") as out_f:
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            in_flight: collections.deque = collections.deque()
            with tarfile.open(pdmx_dir / "mxl.tar.gz", "r|gz") as tf:
                for m in tf:
                    if not m.isfile() or not m.name.endswith(".mxl"):
                        continue
                    scanned += 1
                    qid = os.path.basename(m.name).replace(".mxl", "")
                    cmeta = csv_meta.get(qid)
                    if cmeta is None:
                        continue
                    fobj = tf.extractfile(m)
                    if fobj is None:
                        continue
                    raw_bytes = fobj.read()
                    matched += 1
                    fut = ex.submit(_process_member, (qid, m.name, raw_bytes, cmeta))
                    in_flight.append(fut)
                    # Keep queue bounded so RAM never spikes
                    if len(in_flight) >= 256:
                        res = in_flight.popleft().result()
                        if res is not None:
                            out_f.write(json.dumps(res, ensure_ascii=False) + "\n")
                            records.append(res)
                    if matched % 10000 == 0:
                        dt = time.time() - t0
                        print(f"[PDMX] scanned={scanned} matched={matched} parsed={len(records)} ({dt:.1f}s)", flush=True)
                    if limit and len(records) >= limit:
                        break
            while in_flight:
                res = in_flight.popleft().result()
                if res is not None:
                    out_f.write(json.dumps(res, ensure_ascii=False) + "\n")
                    records.append(res)

    if not limit:
        os.replace(tmp_cache, pdmx_cache)
    else:
        tmp_cache.unlink(missing_ok=True)
    print(f"[PDMX] Finished streaming mxl.tar.gz: {len(records)} parsed in {time.time()-t0:.1f}s", flush=True)
    return records


# =============================================================================
# Curated Local & Cloned Source Loaders
# =============================================================================

def load_curated_symbolic_sources(
    blocklist: EvalBlocklist,
    seen_sha: Set[str],
    rejected_fh: Any,
) -> List[Dict[str, Any]]:
    """Ingest all local & cloned permissive symbolic pools."""
    out: List[Dict[str, Any]] = []

    def _add_mxl_dir(source_id: str, root: Path, license_str: str, exts: Tuple[str, ...], dedup_stems: Optional[Set[str]] = None) -> int:
        if not root.exists():
            return 0
        files: List[Path] = []
        for dp, _, fns in os.walk(root):
            if ".git" in dp.split(os.sep):
                continue
            for fn in sorted(fns):
                fl = fn.lower()
                if any(fl.endswith(e) for e in exts):
                    files.append(Path(dp) / fn)
        # Prefer _cleaned.musicxml if present in ossq-omr
        if source_id == "malerlab_ossq_omr":
            cleaned = [f for f in files if f.stem.endswith("_cleaned")]
            if cleaned:
                files = cleaned

        def _parse_one(p: Path) -> Optional[Dict[str, Any]]:
            try:
                raw = p.read_bytes()
                sha = hashlib.sha256(raw).hexdigest()
                if p.suffix.lower() == ".mxl":
                    xb = extract_xml_from_mxl(raw)
                    if not xb:
                        return None
                    parsed = parse_musicxml_bytes(xb)
                elif p.suffix.lower() in (".musicxml", ".xml"):
                    parsed = parse_musicxml_bytes(raw)
                elif p.suffix.lower() == ".mscx":
                    parsed = parse_mscx_file(p)
                else:
                    return None
                if not parsed:
                    return None
                parsed["sha256"] = sha
                parsed["path"] = str(p)
                parsed["stem"] = p.stem
                parsed["rel_hint"] = str(p.relative_to(root))
                return parsed
            except Exception:
                return None

        added = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            for res in ex.map(_parse_one, files):
                if res is None:
                    continue
                stem_clean = res["stem"].replace("_cleaned", "").lower()
                if dedup_stems is not None and stem_clean in dedup_stems:
                    continue
                blob = f"{res.get('title','')} {res.get('composer','')} {res.get('rel_hint','')}"
                blk = blocklist.check(blob, res["stem"], res["sha256"])
                if blk:
                    rejected_fh.write(json.dumps({"source": source_id, "path": res["path"], "reason": f"eval_blocklist:{blk}"}) + "\n")
                    continue
                if res["sha256"] in seen_sha:
                    rejected_fh.write(json.dumps({"source": source_id, "path": res["path"], "reason": "duplicate_sha256"}) + "\n")
                    continue
                seen_sha.add(res["sha256"])
                if dedup_stems is not None:
                    dedup_stems.add(stem_clean)
                res["source_id"] = source_id
                res["license"] = license_str
                res["format"] = Path(res["path"]).suffix.lower().lstrip(".")
                out.append(res)
                added += 1
        print(f"[CURATED] {source_id}: admitted {added} unique scores (from {len(files)} files)", flush=True)
        return added

    # 1. OpenScore Lieder (CC0-1.0)
    _add_mxl_dir("openscore_lieder", Path(r"D:\tmp\openscore-lieder\scores"), "CC0-1.0", (".mxl",))

    # 2. OpenScore String Quartets + MALerLab OSSQ-OMR (CC0-1.0, deduplicated by sq stem)
    sq_stems: Set[str] = set()
    _add_mxl_dir("malerlab_ossq_omr", Path(r"D:\tmp\corpus100k\raw\ossq-omr\scores"), "CC0-1.0", (".musicxml",), dedup_stems=sq_stems)
    _add_mxl_dir("openscore_string_quartets", Path(r"D:\tmp\openscore-sq-repo\scores"), "CC0-1.0", (".mxl",), dedup_stems=sq_stems)

    # 3. OpenEWLD Jazz Lead Sheets (Public Domain)
    _add_mxl_dir("openewld_jazz", Path(r"D:\tmp\corpus10x\openewld\scores"), "Public Domain", (".mxl",))

    # 4. DCML Bach Chorales (CC0-1.0)
    _add_mxl_dir("dcml_bach_chorales", Path(r"D:\tmp\corpus100k\raw\dcml-bach-chorales\MS3"), "CC0-1.0", (".mscx",))

    # 5. NIFC Chopin First Editions (CC-BY-4.0) & NIFC Polish Scores (CC-BY-4.0)
    for sid, root in [
        ("nifc_chopin_first_editions", Path(r"D:\tmp\nifc-chopin-repo")),
        ("nifc_polish_scores", Path(r"D:\tmp\nifc-polish-scores-repo")),
    ]:
        if not root.exists():
            continue
        krn_files = sorted(root.rglob("*.krn"))
        added = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            for p, res in zip(krn_files, ex.map(parse_krn_file, krn_files)):
                if res is None:
                    continue
                blob = f"{res.get('title','')} {res.get('composer','')} {p.stem}"
                blk = blocklist.check(blob, p.stem, res["sha256"])
                if blk:
                    rejected_fh.write(json.dumps({"source": sid, "path": str(p), "reason": f"eval_blocklist:{blk}"}) + "\n")
                    continue
                if res["sha256"] in seen_sha:
                    rejected_fh.write(json.dumps({"source": sid, "path": str(p), "reason": "duplicate_sha256"}) + "\n")
                    continue
                seen_sha.add(res["sha256"])
                res["source_id"] = sid
                res["license"] = "CC-BY-4.0"
                res["path"] = str(p)
                res["stem"] = p.stem
                res["format"] = "krn"
                out.append(res)
                added += 1
        print(f"[CURATED] {sid}: admitted {added} unique .krn scores (from {len(krn_files)} files)", flush=True)

    return out


def load_tier_c_archival_sources(
    blocklist: EvalBlocklist,
    seen_sha: Set[str],
    rejected_fh: Any,
) -> List[Dict[str, Any]]:
    """Load all local Tier C verified permissive archival scan pages."""
    out: List[Dict[str, Any]] = []

    def _emit_img(source_id: str, rec_id: str, img_path: str, title: str, creator: str, lic: str,
                  sha: str, w: int, h: int, orient: str, staff_rows: float = 80.0) -> None:
        if not is_permissive_license(lic):
            rejected_fh.write(json.dumps({"source": source_id, "path": img_path, "reason": f"license_rejected:{lic}"}) + "\n")
            return
        stem = Path(img_path).stem
        blk = blocklist.check(f"{title} {creator} {rec_id} {stem}", stem, sha)
        if blk:
            rejected_fh.write(json.dumps({"source": source_id, "path": img_path, "reason": f"eval_blocklist:{blk}"}) + "\n")
            return
        if sha and sha in seen_sha:
            rejected_fh.write(json.dumps({"source": source_id, "path": img_path, "reason": "duplicate_sha256"}) + "\n")
            return
        if sha:
            seen_sha.add(sha)
        out.append({
            "source_id": source_id,
            "id": rec_id,
            "path": img_path,
            "stem": stem,
            "title": title[:200],
            "composer": creator[:120],
            "license": lic,
            "sha256": sha,
            "width": w,
            "height": h,
            "orientation": orient,
            "staff_rows": staff_rows,
            "format": Path(img_path).suffix.lower().lstrip("."),
            "is_tier_c": True,
            "est_pages": 1,
        })

    # 1. IMSLP / Wikimedia / IA Pre-1928 (D:\tmp\unlabeled-10k-real\manifest.jsonl)
    m1 = Path(r"D:\tmp\unlabeled-10k-real\manifest.jsonl")
    if m1.exists():
        c0 = len(out)
        with open(m1, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                w, h = int(r.get("width", 1200)), int(r.get("height", 1600))
                orient = "landscape" if w > h else "portrait"
                sr = float((r.get("filter_stats") or {}).get("staff_rows", 80.0))
                _emit_img(
                    "imslp_pd_plates_and_autographs",
                    r["id"], r["image_path"], r.get("title", ""), "", r.get("license", "Public Domain"),
                    r.get("sha256", ""), w, h, orient, sr,
                )
        print(f"[TIER-C] imslp_pd_plates_and_autographs: {len(out)-c0} pages", flush=True)

    # 2. RISM HW (D:\tmp\unlabeled-10k-real\rism_hw\manifest.jsonl)
    m2 = Path(r"D:\tmp\unlabeled-10k-real\rism_hw\manifest.jsonl")
    if m2.exists():
        c0 = len(out)
        rism_root = m2.parent
        with open(m2, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                ip = str(rism_root / r["image_path"])
                orient = r.get("orientation", "landscape")
                w, h = (1800, 1350) if orient == "landscape" else (1350, 1800)
                sha = hashlib.sha256(r["id"].encode()).hexdigest()  # fast deterministic key; verify file exists
                if not os.path.exists(ip):
                    continue
                _emit_img(
                    "rism_iiif_manuscripts",
                    r["id"], ip, r.get("id", ""), r.get("library", ""), r.get("license", "PDM 1.0"),
                    sha, w, h, orient, 75.0,
                )
        print(f"[TIER-C] rism_iiif_manuscripts: {len(out)-c0} pages", flush=True)

    # 3. Intl HW (D:\tmp\unlabeled-10k-real\intl_hw\manifest.jsonl)
    m3 = Path(r"D:\tmp\unlabeled-10k-real\intl_hw\manifest.jsonl")
    if m3.exists():
        c0 = len(out)
        with open(m3, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                w, h = int(r.get("width", 1350)), int(r.get("height", 1800))
                orient = r.get("orientation", "landscape" if w > h else "portrait")
                _emit_img(
                    "debussy_bnf_and_real_hw_local",
                    r["id"], r["image_path"], r.get("title", ""), r.get("creator", ""),
                    r.get("license", "PDM 1.0"), r.get("sha256", ""), w, h, orient, 75.0,
                )
        print(f"[TIER-C] intl_hw -> debussy_bnf_and_real_hw_local: {len(out)-c0} pages", flush=True)

    # 4. NIFC HW (D:\tmp\unlabeled-10k-real\nifc_hw\images)
    nifc_hw_dir = Path(r"D:\tmp\unlabeled-10k-real\nifc_hw\images")
    if nifc_hw_dir.exists():
        c0 = len(out)
        for p in sorted(nifc_hw_dir.glob("*.jpg")):
            sha = hashlib.sha256(str(p.name).encode()).hexdigest()
            _emit_img(
                "debussy_bnf_and_real_hw_local",
                f"nifc_hw_{p.stem}", str(p), p.stem, "NIFC", "CC0-1.0",
                sha, 1400, 1800, "portrait", 80.0,
            )
        print(f"[TIER-C] nifc_hw -> debussy_bnf_and_real_hw_local: {len(out)-c0} pages", flush=True)

    # 5. Debussy BnF train_pages (D:\tmp\debussy-omr\train_pages\manifest.json)
    if DEBUSSY_MANIFEST.exists():
        c0 = len(out)
        dm = json.loads(DEBUSSY_MANIFEST.read_text(encoding="utf-8"))
        for p in dm.get("pages", []):
            fn = p.get("filename") or p.get("file") or ""
            ip = str(DEBUSSY_MANIFEST.parent / fn) if fn else ""
            if not ip or not os.path.exists(ip):
                continue
            w, h = int(p.get("width", 1400)), int(p.get("height", 1800))
            orient = "landscape" if w > h else "portrait"
            sha = hashlib.sha256(ip.encode()).hexdigest()
            _emit_img(
                "debussy_bnf_and_real_hw_local",
                f"debussy_{Path(ip).stem}", ip, p.get("document_id", Path(ip).stem),
                "Claude Debussy", "Public Domain", sha, w, h, orient, 85.0,
            )
        print(f"[TIER-C] debussy_bnf -> debussy_bnf_and_real_hw_local: {len(out)-c0} pages", flush=True)

    return out


# =============================================================================
# Dynamic Visual / Renderer Assignment (Dims 09, 10, 11, 12, 14)
# =============================================================================

RENDERER_META = {
    "musescore_leland": ("musescore", "Leland"),
    "musescore_bravura": ("musescore", "Bravura"),
    "musescore_emmentaler_gonville": ("musescore", "Emmentaler"),
    "musescore_petaluma_jazz": ("musescore", "Petaluma"),
    "verovio_leipzig_sebastian_gootville": ("verovio", "Leipzig"),
    "dorico_native_doremi": ("musescore", "Bravura"),
    "lilypond_native_mutopia": ("musescore", "Emmentaler"),
    "osmd_vexflow_abcjs": ("osmd", "Bravura"),
    "real_19c_plate_engravings": ("native_scan", "plate"),
    "real_hw_and_hw_sim": ("hw-score-sim", "handwritten"),
}

GEOM_META = {
    "portrait_a4_letter_hires": ("a4", "portrait", 2000),
    "portrait_b4_folio_large": ("b4", "portrait", 2400),
    "portrait_octavo_pocket": ("a5", "portrait", 1600),
    "landscape_quartet_orchestral": ("a4_land", "landscape", 2200),
    "tight_cropped_margins": ("a4", "portrait", 1800),
    "low_res_mobile_web": ("a4", "portrait", 960),
}

CAPTURE_META = {
    "born_digital_clean": ("clean_digital", "none"),
    "phone_photo_mild_med": ("photo_degrade", "medium"),
    "phone_photo_hard": ("photo_degrade", "hard"),
    "historical_library_scans": ("photo_degrade", "archival"),
    "performer_markup_wear": ("photo_degrade", "wear_markup"),
}


def pick_best_deficit_bucket(
    candidates: List[str],
    dim_counts: Dict[str, int],
    dim_targets: Dict[str, int],
    unit_key: str,
    dim_name: str,
) -> str:
    best_b = candidates[0]
    best_score = -1e9
    for b in candidates:
        t = max(1, dim_targets.get(b, 1))
        c = dim_counts.get(b, 0)
        deficit = (t - c) / float(t)
        # Tiny deterministic tie-break
        tb = det_float(SEED, f"tie_{dim_name}_{b}", unit_key) * 1e-4
        sc = deficit + tb
        if sc > best_score:
            best_score = sc
            best_b = b
    return best_b


def assign_visual_dims(
    unit: Dict[str, Any],
    counts: Dict[str, Dict[str, int]],
    targets: Dict[str, Dict[str, int]],
) -> Dict[str, str]:
    """Assign Dims 09, 10, 11, 12, 14 for symbolic Tier A/B units using live deficits."""
    uk = unit["unit_id"]
    staves = int(unit["meta"].get("staves_per_system", 1))
    fmt = unit["meta"].get("format", "mxl")
    d01 = unit["assigned"]["01_genre_tradition"]
    d02 = unit["assigned"]["02_ensemble_score_type"]

    # Dim 12: Handwriting Modality (choose first so Dim 09 & 11 stay consistent!)
    c12 = ["printed_engraved"]
    if d01 == "jazz_lead_sheets" or unit["source_id"] == "openewld_jazz":
        c12.append("modern_handwritten_jazz")
    if staves <= 3:
        c12.append("hw_score_sim_standard")
    if staves >= 4:
        c12.append("hw_score_sim_dense_quartet")
    # Allow modern_handwritten_jazz for any small-ensemble piece if needed
    if staves <= 2 and "modern_handwritten_jazz" not in c12:
        c12.append("modern_handwritten_jazz")
    b12 = pick_best_deficit_bucket(c12, counts["12_handwritten_archival"], targets["12_handwritten_archival"], uk, "12")

    # Dim 09: Engraving / Renderer / Font
    if b12 in ("hw_score_sim_standard", "hw_score_sim_dense_quartet"):
        b09 = "real_hw_and_hw_sim"
    elif b12 == "modern_handwritten_jazz":
        b09 = "musescore_petaluma_jazz"
    else:
        if fmt == "krn":
            c09 = ["verovio_leipzig_sebastian_gootville", "musescore_leland", "musescore_bravura"]
        else:
            c09 = [
                "musescore_leland",
                "musescore_bravura",
                "musescore_emmentaler_gonville",
                "musescore_petaluma_jazz",
                "verovio_leipzig_sebastian_gootville",
                "dorico_native_doremi",
                "lilypond_native_mutopia",
                "osmd_vexflow_abcjs",
            ]
        b09 = pick_best_deficit_bucket(c09, counts["09_engraving_renderer_font"], targets["09_engraving_renderer_font"], uk, "09")

    # Dim 10: Page Geometry & Resolution
    if b12 == "hw_score_sim_dense_quartet" or d02 in ("conductor_orchestral_full", "string_quartet_wind_quintet"):
        c10 = ["landscape_quartet_orchestral", "portrait_b4_folio_large", "portrait_a4_letter_hires"]
    elif d02 == "choral_satb_and_acmp":
        c10 = ["portrait_octavo_pocket", "portrait_a4_letter_hires", "tight_cropped_margins"]
    else:
        c10 = [
            "portrait_a4_letter_hires",
            "portrait_b4_folio_large",
            "portrait_octavo_pocket",
            "landscape_quartet_orchestral",
            "tight_cropped_margins",
            "low_res_mobile_web",
        ]
    b10 = pick_best_deficit_bucket(c10, counts["10_page_geometry_resolution"], targets["10_page_geometry_resolution"], uk, "10")

    # Dim 11: Visual Domain / Capture
    if b12 == "modern_handwritten_jazz":
        c11 = ["phone_photo_mild_med", "phone_photo_hard", "performer_markup_wear"]
    elif b12 in ("hw_score_sim_standard", "hw_score_sim_dense_quartet"):
        c11 = ["historical_library_scans", "phone_photo_mild_med", "phone_photo_hard"]
    else:
        c11 = [
            "born_digital_clean",
            "phone_photo_mild_med",
            "phone_photo_hard",
            "historical_library_scans",
            "performer_markup_wear",
        ]
    b11 = pick_best_deficit_bucket(c11, counts["11_visual_domain_capture"], targets["11_visual_domain_capture"], uk, "11")

    # Dim 14: Layout Density & Spacing
    if staves >= 8:
        c14 = ["ultra_dense_cramped_miniature", "dense_multi_system"]
    elif staves == 1 and d01 in ("solo_etudes_pedagogical", "jazz_lead_sheets", "folk_traditional"):
        c14 = ["sparse_beginner_large_print", "standard_engraving_density", "dense_multi_system", "empty_prerule_and_text_heavy"]
    else:
        c14 = [
            "sparse_beginner_large_print",
            "standard_engraving_density",
            "dense_multi_system",
            "ultra_dense_cramped_miniature",
            "empty_prerule_and_text_heavy",
        ]
    b14 = pick_best_deficit_bucket(c14, counts["14_layout_density_spacing"], targets["14_layout_density_spacing"], uk, "14")

    return {
        "09_engraving_renderer_font": b09,
        "10_page_geometry_resolution": b10,
        "11_visual_domain_capture": b11,
        "12_handwritten_archival": b12,
        "14_layout_density_spacing": b14,
    }


# =============================================================================
# Main Quota-Filling Pipeline
# =============================================================================

def expand_score_to_page_units(
    score: Dict[str, Any],
    is_curated: bool,
    is_rare: bool,
) -> List[Dict[str, Any]]:
    """Expand a multi-page score into 1..N distinct (score_id, page_num) units."""
    is_tier_c = bool(score.get("is_tier_c", False))
    source_id = score["source_id"]
    base_id = score.get("id") or score.get("qid") or score.get("stem") or score["sha256"][:16]
    est_pages = max(1, int(score.get("est_pages", 1)))

    if is_tier_c or est_pages == 1:
        pages = [1]
        total_pages = 1
    else:
        # Curated or rare multi-page scores can contribute up to 4-6 distinct pages
        # (page 1, interior pages, final stub) per quota_blueprint unit_definition!
        if is_curated and source_id in ("openscore_string_quartets", "malerlab_ossq_omr", "nifc_chopin_first_editions", "openscore_lieder"):
            max_p = min(est_pages, 5)
        elif is_rare:
            max_p = min(est_pages, 4)
        else:
            max_p = min(est_pages, 2)

        if max_p == 2:
            pages = [1, est_pages]
        elif max_p >= 3:
            # Always include page 1, final page, and evenly spaced interior pages
            interior_n = max_p - 2
            pages = [1]
            for i in range(1, interior_n + 1):
                p = max(2, min(est_pages - 1, int(round(1 + i * (est_pages - 1) / float(interior_n + 1)))))
                if p not in pages:
                    pages.append(p)
            if est_pages not in pages:
                pages.append(est_pages)
        else:
            pages = [1]
        total_pages = est_pages

    units = []
    for pnum in pages:
        uid = f"{source_id}:{base_id}:p{pnum}"
        vb, sup_tier, v2_tags = classify_inherent_dims(score, source_id, pnum, total_pages, is_tier_c=is_tier_c)
        units.append({
            "unit_id": uid,
            "score_id": f"{source_id}:{base_id}",
            "source_id": source_id,
            "page_num": pnum,
            "total_pages": total_pages,
            "is_curated": is_curated,
            "is_tier_c": is_tier_c,
            "sup_tier": sup_tier,
            "v2_tags": v2_tags,
            "valid_buckets": vb,
            "meta": score,
        })
    return units


def is_rare_score(score: Dict[str, Any]) -> bool:
    clefs = set(score.get("clefs", []))
    if any(c in clefs for c in ("C-3", "C-4", "C-1", "C-2", "C-5", "G-1", "F-3", "G-2-8vb", "G-2-8va", "F-4-8vb", "clef-p-2", "clef-T-2")):
        return True
    if score.get("has_microtonal") or score.get("has_double_accidental") or score.get("has_harmony"):
        return True
    if int(score.get("staves_per_system", 1)) >= 4 or int(score.get("distinct_keys_in_system", 1)) >= 2:
        return True
    if int(score.get("multibar_rest_count", 0)) >= 2 or int(score.get("meter_changes", 0)) >= 1:
        return True
    for ts in score.get("time_sigs", []):
        if ts.split("/")[0] in ("5", "7", "10", "11", "13", "15", "17"):
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2: 100k Quota-Filling Corpus Puller")
    parser.add_argument("--target", type=int, default=100000, help="Target total score-page units (default 100000)")
    parser.add_argument("--pdmx-limit", type=int, default=None, help="Optional limit on PDMX records for quick smoke test")
    args = parser.parse_args()

    t_start = time.time()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    blueprint = json.loads(BLUEPRINT_PATH.read_text(encoding="utf-8"))
    total_target = args.target
    scale = total_target / float(blueprint["total_target"])

    # Build target counts and rarity weights per (dim, bucket)
    targets: Dict[str, Dict[str, int]] = {}
    rarity_w: Dict[str, Dict[str, float]] = {}
    dominant_buckets: Dict[str, Set[str]] = {}
    counts: Dict[str, Dict[str, int]] = {}

    for dim_name, dim_obj in blueprint["dimensions"].items():
        targets[dim_name] = {}
        rarity_w[dim_name] = {}
        dominant_buckets[dim_name] = set()
        counts[dim_name] = {}
        for b in dim_obj["buckets"]:
            bid = b["id"]
            tcnt = max(1, int(round(b["target_count"] * scale)))
            targets[dim_name][bid] = tcnt
            counts[dim_name][bid] = 0
            pct = max(0.5, float(b["target_pct"]))
            rarity_w[dim_name][bid] = 1.0 + 3.0 / math.sqrt(pct)
            if pct >= 15.0:
                dominant_buckets[dim_name].add(bid)

    blocklist = EvalBlocklist()
    seen_sha: Set[str] = set()
    rejected_fh = open(REJECTED_OUT, "w", encoding="utf-8")

    # Step A: Ingest all curated local + cloned pools
    print("=== STEP A: Ingesting curated permissive symbolic pools ===", flush=True)
    curated_scores = load_curated_symbolic_sources(blocklist, seen_sha, rejected_fh)

    print("=== STEP A2: Ingesting Tier C real archival pools ===", flush=True)
    tier_c_scores = load_tier_c_archival_sources(blocklist, seen_sha, rejected_fh)

    # Step B: Stream & classify PDMX deduplicated + no_license_conflict pool
    print("=== STEP B: Loading / Streaming PDMX permissive deduplicated pool ===", flush=True)
    pdmx_raw = build_or_load_pdmx_cache(blocklist, rejected_fh, limit=args.pdmx_limit)
    pdmx_scores: List[Dict[str, Any]] = []
    pdmx_blocked = 0
    pdmx_dup = 0
    for rec in pdmx_raw:
        blob = f"{rec.get('title','')} {rec.get('composer','')} {rec.get('qid','')}"
        blk = blocklist.check(blob, rec.get("qid", ""), rec.get("sha256", ""))
        if blk:
            pdmx_blocked += 1
            rejected_fh.write(json.dumps({"source": "pdmx_dedup_clean", "qid": rec.get("qid"), "reason": f"eval_blocklist:{blk}"}) + "\n")
            continue
        if rec["sha256"] in seen_sha:
            pdmx_dup += 1
            rejected_fh.write(json.dumps({"source": "pdmx_dedup_clean", "qid": rec.get("qid"), "reason": "duplicate_sha256"}) + "\n")
            continue
        seen_sha.add(rec["sha256"])
        rec["source_id"] = "pdmx_dedup_clean"
        rec["format"] = "mxl"
        rec["path"] = f"D:\\tmp\\corpus10x\\pdmx\\mxl.tar.gz::{rec['tar_member']}"
        pdmx_scores.append(rec)
    print(f"[PDMX] Admissible unique scores: {len(pdmx_scores)} (eval_blocked={pdmx_blocked}, dup_sha={pdmx_dup})", flush=True)

    # Expand into score-page candidate units
    print("=== STEP C: Expanding to score-page units & running deficit-scored quota selection ===", flush=True)
    candidates: List[Dict[str, Any]] = []
    for s in tier_c_scores:
        candidates.extend(expand_score_to_page_units(s, is_curated=True, is_rare=True))
    for s in curated_scores:
        candidates.extend(expand_score_to_page_units(s, is_curated=True, is_rare=is_rare_score(s)))
    for s in pdmx_scores:
        candidates.extend(expand_score_to_page_units(s, is_curated=False, is_rare=is_rare_score(s)))
    print(f"[POOL] Total candidate score-page units: {len(candidates)}", flush=True)

    # Inherent dimensions used for greedy selection
    INHERENT_DIMS = [
        "01_genre_tradition",
        "02_ensemble_score_type",
        "03_clefs_transposition",
        "04_meter_rhythm_complexity",
        "05_keys_accidentals_tonality",
        "06_polyphony_voice_density",
        "07_articulations_expressions_text",
        "08_page_structure_continuity",
        "13_supervision_tier",
    ]

    # Pre-sort candidates deterministically by static rarity score + curated priority
    # Then run fast multi-pass deficit-aware admission!
    def static_priority(u: Dict[str, Any]) -> float:
        sc = 0.0
        if u["is_tier_c"]:
            sc += 50.0  # Real archival Tier C scans are scarce & high-value
        elif u["is_curated"]:
            sc += 18.0  # Curated repos prioritized over generic PDMX
        if u["sup_tier"] == "tier_b_v2_tagged_gt":
            sc += 6.0
        for d in INHERENT_DIMS:
            best_rw = max(rarity_w[d].get(b, 1.0) for b in u["valid_buckets"][d])
            sc += best_rw
        sc += det_float(SEED, "order", u["unit_id"]) * 0.5
        return sc

    candidates.sort(key=static_priority, reverse=True)

    # Multi-pass greedy admission with progressive dominant-bucket cap:
    # Pass 0: strict 1.05x cap on dominant buckets (fills all rare strata first!)
    # Pass 1: 1.25x cap
    # Pass 2: 1.60x cap
    # Pass 3: unbounded backfill to reach exact total_target
    selected: List[Dict[str, Any]] = []
    selected_ids: Set[str] = set()
    source_counts: collections.Counter = collections.Counter()

    # For fast dynamic scoring, recompute batch-wise in passes
    cap_schedule = [1.05, 1.20, 1.45, 2.00, 100.0]

    for pass_idx, cap_mult in enumerate(cap_schedule):
        if len(selected) >= total_target:
            break
        # Re-score remaining candidates against current deficits before this pass
        remaining = [u for u in candidates if u["unit_id"] not in selected_ids]
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for u in remaining:
            s_val = 0.0
            if u["is_tier_c"]:
                s_val += 40.0
            elif u["is_curated"]:
                s_val += 8.0
            for d in INHERENT_DIMS:
                best_d = 0.0
                for b in u["valid_buckets"][d]:
                    t = targets[d][b]
                    c = counts[d][b]
                    if c < t:
                        val = ((t - c) / float(t)) * rarity_w[d][b]
                        if val > best_d:
                            best_d = val
                s_val += best_d
            # Slight tie-break
            s_val += det_float(SEED, f"pass{pass_idx}", u["unit_id"]) * 0.05
            scored.append((s_val, u))
        scored.sort(key=lambda x: x[0], reverse=True)

        added_this_pass = 0
        for _, u in scored:
            if len(selected) >= total_target:
                break

            # Choose best bucket per inherent dim and check dominant cap
            chosen: Dict[str, str] = {}
            capped = False
            for d in INHERENT_DIMS:
                b = pick_best_deficit_bucket(u["valid_buckets"][d], counts[d], targets[d], u["unit_id"], d)
                chosen[d] = b
                # Enforce cap on dominant buckets and on supervision tier
                if (b in dominant_buckets[d] or d == "13_supervision_tier") and counts[d][b] >= int(targets[d][b] * cap_mult):
                    # Always admit Tier C real archival scans on pass 0
                    if not (u["is_tier_c"] and d != "13_supervision_tier"):
                        capped = True
                        break
            if capped:
                continue

            u["assigned"] = chosen
            # Assign visual dims (09, 10, 11, 12, 14)
            if u["is_tier_c"]:
                for d in ("09_engraving_renderer_font", "10_page_geometry_resolution",
                          "11_visual_domain_capture", "12_handwritten_archival", "14_layout_density_spacing"):
                    b = pick_best_deficit_bucket(u["valid_buckets"][d], counts[d], targets[d], u["unit_id"], d)
                    chosen[d] = b
            else:
                v_assigned = assign_visual_dims(u, counts, targets)
                chosen.update(v_assigned)

            # Commit counts
            for d, b in chosen.items():
                counts[d][b] += 1
            selected.append(u)
            selected_ids.add(u["unit_id"])
            source_counts[u["source_id"]] += 1
            added_this_pass += 1

        print(f"[SELECT] Pass {pass_idx} (cap={cap_mult:.2f}x): +{added_this_pass} -> total={len(selected)}/{total_target}", flush=True)

    rejected_fh.close()

    # Step D: Emit corpus100k_manifest.jsonl atomically (tmp + rename)
    print(f"=== STEP D: Writing {MANIFEST_OUT} ({len(selected)} rows) ===", flush=True)
    tmp_manifest = MANIFEST_OUT.with_suffix(".jsonl.tmp")
    eval_audit_hits = 0
    license_audit_fails = 0
    seen_unit_ids: Set[str] = set()

    with open(tmp_manifest, "w", encoding="utf-8") as mf:
        for u in selected:
            assert u["unit_id"] not in seen_unit_ids, f"Duplicate unit_id {u['unit_id']}"
            seen_unit_ids.add(u["unit_id"])
            m = u["meta"]
            assigned = u["assigned"]

            # Post-verification audit on every emitted row
            lic = m.get("license", "")
            if not is_permissive_license(lic):
                license_audit_fails += 1
            blob = f"{m.get('title','')} {m.get('composer','')} {u['unit_id']} {m.get('stem','')}"
            if blocklist.check(blob, m.get("stem", ""), m.get("sha256", "")):
                eval_audit_hits += 1

            r_eng, r_font = RENDERER_META.get(assigned["09_engraving_renderer_font"], ("musescore", "Leland"))
            g_paper, g_orient, g_long = GEOM_META.get(assigned["10_page_geometry_resolution"], ("a4", "portrait", 2000))
            c_mode, c_sev = CAPTURE_META.get(assigned["11_visual_domain_capture"], ("clean_digital", "none"))

            row = {
                "unit_id": u["unit_id"],
                "score_id": u["score_id"],
                "source_id": u["source_id"],
                "path": m["path"],
                "format": m.get("format", "mxl"),
                "sha256": m.get("sha256", ""),
                "license": lic,
                "title": m.get("title", ""),
                "composer": m.get("composer", ""),
                "page_num": u["page_num"],
                "total_pages": u["total_pages"],
                "supervision_tier": assigned["13_supervision_tier"],
                "v2_tags": u["v2_tags"],
                "dimensions": assigned,
                "render_plan": {
                    "renderer": r_eng,
                    "font": r_font,
                    "paper": g_paper,
                    "orientation": g_orient,
                    "target_long_edge_px": g_long,
                    "capture_mode": c_mode,
                    "capture_severity": c_sev,
                    "hw_modality": assigned["12_handwritten_archival"],
                    "seed": splitmix64_state(SEED, "render", u["unit_id"]),
                },
                "musical_summary": {
                    "staves_per_system": m.get("staves_per_system", 1),
                    "clefs": m.get("clefs", []),
                    "time_sigs": m.get("time_sigs", []),
                    "fifths": m.get("fifths", []),
                    "measures": m.get("measures", 0),
                    "note_count": m.get("note_count", 0),
                },
            }
            mf.write(json.dumps(row, ensure_ascii=False) + "\n")

    os.replace(tmp_manifest, MANIFEST_OUT)

    # Step E: Emit quota_report.json
    dim_report: Dict[str, Any] = {}
    for dim_name, dim_obj in blueprint["dimensions"].items():
        buckets_rep = []
        for b in dim_obj["buckets"]:
            bid = b["id"]
            tcnt = targets[dim_name][bid]
            acnt = counts[dim_name][bid]
            tpct = round(100.0 * tcnt / max(1, total_target), 2)
            apct = round(100.0 * acnt / max(1, len(selected)), 2)
            fill_ratio = round(acnt / float(max(1, tcnt)), 4)
            buckets_rep.append({
                "id": bid,
                "label": b["label"],
                "target_count": tcnt,
                "actual_count": acnt,
                "target_pct": tpct,
                "actual_pct": apct,
                "fill_ratio": fill_ratio,
            })
        dim_report[dim_name] = buckets_rep

    report = {
        "schema_version": "1.0.0",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_sec": round(time.time() - t_start, 2),
        "seed": SEED,
        "total_target": total_target,
        "total_selected": len(selected),
        "unique_scores_selected": len({u["score_id"] for u in selected}),
        "total_candidates_considered": len(candidates),
        "eval_blocklist_hits_in_output": eval_audit_hits,
        "license_audit_failures": license_audit_fails,
        "source_counts": dict(source_counts.most_common()),
        "supervision_tiers": counts["13_supervision_tier"],
        "dimensions": dim_report,
    }

    tmp_report = REPORT_OUT.with_suffix(".json.tmp")
    tmp_report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp_report, REPORT_OUT)

    print(f"=== DONE in {report['elapsed_sec']}s ===", flush=True)
    print(f"Selected: {len(selected)} units ({report['unique_scores_selected']} unique scores)", flush=True)
    print(f"Eval blocklist hits in output: {eval_audit_hits} | License failures: {license_audit_fails}", flush=True)
    print(f"Tiers: {counts['13_supervision_tier']}", flush=True)
    print(f"Sources: {dict(source_counts.most_common())}", flush=True)


if __name__ == "__main__":
    main()
