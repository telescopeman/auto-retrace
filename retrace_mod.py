#!/usr/bin/env python3
"""
retrace.py - turn a "raw import" FamiTracker text export of NES music
(speed 1, one row per frame, vibrato / envelopes / arps expanded into rows)
into a compact, readable "retrace".

What it does
  1. finds the loop in the raw data and keeps intro + exactly one loop
  2. finds the song's real row grid (speed + phase) and puts every note on it
     (notes that start between grid lines get a Gxx delay)
  3. cuts the raw data into notes and turns everything that happens *inside*
     a note (volume envelope, arpeggio, vibrato/pitch bends, duty changes)
     into reusable macros, grouped into a small set of instruments.
     Chord arpeggios become ONE arpeggio-scheme macro + a 0xy effect.
  4. chops the result into patterns and shares identical patterns
  5. plays the result back with a model of FamiTracker and reports how close
     it is to the raw data
  6. checks the output file against the ranges FamiTracker enforces on import

Chips: every chip FamiTracker supports.  2A03, MMC5 (which uses 2A03 instruments), VRC6 and
Sunsoft 5B get new macro instruments made for them.  The wavetable / FM chips (N163, FDS,
VRC7) get a rough retrace: notes, timing and volume, played with the original module's own
instruments copied across.  Modules with chips other than the 2A03 / VRC6 are written as
FamiTracker text (File > Import Text), since the .ftm writer only knows those two.

DPCM: the game's samples are copied across with one "DPCM samples" instrument whose key map
plays each sample at the pitch the game used.  Every hit, stop and output-level write (Zxx)
on the DPCM channel is placed on the row grid, with Gxx when it falls between rows.

Usage
    python retrace.py raw.txt                         # writes retrace.ftm by default
    python retrace.py raw.txt --format txt -o retrace.txt
    python retrace.py raw.txt --subsong-range 2 5
    python retrace.py raw.txt --format txt --max-instruments 48
"""
import os
import struct
import re
import argparse
import collections
import sys
import subprocess
import tempfile
import shutil

import numpy as np

NOTE_NAMES = ['C-', 'C#', 'D-', 'D#', 'E-', 'F-', 'F#', 'G-', 'G#', 'A-', 'A#', 'B-']
NOTE_INDEX = {n: i for i, n in enumerate(NOTE_NAMES)}
SQ1, SQ2, TRI, NOI, DPCM, V6A, V6B, SAW = range(8)
ALL_NAMES = ['sq1', 'sq2', 'tri', 'noi', 'dpcm', 'vrc6a', 'vrc6b', 'saw']
CHIP_OF = {SQ1: '2A03', SQ2: '2A03', TRI: '2A03', NOI: '2A03', V6A: 'VRC6', V6B: 'VRC6', SAW: 'VRC6'}
KIND = {SQ1: 'pulse', SQ2: 'pulse', TRI: 'tri', NOI: 'noise', DPCM: 'dpcm',
        V6A: 'pulse', V6B: 'pulse', SAW: 'saw'}
CH_NAMES, MUSIC, NCH = ALL_NAMES[:5], (SQ1, SQ2, TRI, NOI), 5
# expansion chips in FamiTracker's channel order: (bit, chip, channel kinds, short names)
EXPANSION_LAYOUT = [(1, 'VRC6', ['pulse', 'pulse', 'saw'], ['vrc6a', 'vrc6b', 'saw']),
                    (8, 'MMC5', ['pulse', 'pulse'], ['mmc5a', 'mmc5b']),
                    (16, 'N163', None, None),
                    (4, 'FDS', ['wave'], ['fds']),
                    (2, 'VRC7', ['fm'] * 6, ['vrc7_%d' % i for i in range(1, 7)]),
                    (32, 'S5B', ['s5b'] * 3, ['s5b_a', 's5b_b', 's5b_c'])]
# instruments: MMC5 plays 2A03 instruments; these chips get macro instruments made for them;
# the others (wavetable / FM) keep the original module's instruments ("pass-through")
INST_GROUP = {'2A03': '2A03', 'MMC5': '2A03', 'VRC6': 'VRC6', 'S5B': 'S5B'}
MACRO_CHIPS = ('2A03', 'VRC6', 'S5B')
SUPPORTED_EXPANSIONS = {0: 'none'}


def configure(expansion, nch, n163=None):
    """set up the channel list for the chip set named in the file (any FamiTracker chips)"""
    global CH_NAMES, MUSIC, NCH, CHIP_OF, KIND
    chans = [('2A03', 'pulse', 'sq1'), ('2A03', 'pulse', 'sq2'), ('2A03', 'tri', 'tri'),
             ('2A03', 'noise', 'noi'), ('2A03', 'dpcm', 'dpcm')]
    fixed = sum(len(k) for bit, _, k, _ in EXPANSION_LAYOUT if k and expansion & bit)
    for bit, chip, kinds, names in EXPANSION_LAYOUT:
        if not expansion & bit:
            continue
        if kinds is None:                                  # N163: 1-8 channels
            n = n163 if n163 else max(1, nch - 5 - fixed)
            kinds, names = ['wave'] * n, ['n163_%d' % i for i in range(1, n + 1)]
        chans += [(chip, k, nm) for k, nm in zip(kinds, names)]
    if len(chans) != nch:
        # unknown layout: keep going, treating the extra channels as generic pass-through ones
        chans = chans[:nch] + [('EXT', 'other', 'ext%d' % i) for i in range(len(chans), nch)]
    NCH = nch
    CH_NAMES = [c[2] for c in chans]
    CHIP_OF = {i: c[0] for i, c in enumerate(chans) if c[1] != 'dpcm'}
    KIND = {i: c[1] for i, c in enumerate(chans)}
    MUSIC = tuple(i for i, c in enumerate(chans) if c[1] != 'dpcm')


def inst_group(ch):
    """which instrument table a channel's notes use ('2A03', 'VRC6', 'S5B') or None when the
    channel keeps the original module's instruments"""
    return INST_GROUP.get(CHIP_OF.get(ch))


M_VOL, M_ARP, M_PIT, M_HIP, M_DUTY = range(5)       # FamiTracker macro types
ARP_ABS, ARP_FIXED, ARP_REL, ARP_SCHEME = range(4)  # arpeggio macro modes
BIG = 1e7

# cost weights used when deciding whether an instrument is "close enough"
W_ARP, W_VOL, W_PIT, W_DUTY = 6.0, 1.0, 0.5, 8.0
W_CENT = W_ARP / 100.0      # pitch error is judged in cents, as the ear hears it


# ----------------------------------------------------------------------------
# parsing
# ----------------------------------------------------------------------------
class Module:
    pass


def parse_module(text, subsong=0):
    """Parse one track from a FamiTracker text export.

    FamiTracker text files store every track in its own TRACK/ORDER/PATTERN
    section, while INFO, PARAMS, macros, and instruments are module-wide.
    """
    lines = text.replace('\r\n', '\n').split('\n')
    m = Module()
    m.info = [l for l in lines if l.startswith(('TITLE', 'AUTHOR', 'COPYRIGHT'))]
    m.params = [l for l in lines if l.startswith(('MACHINE', 'FRAMERATE', 'EXPANSION',
                                                  'VIBRATO', 'SPLIT', 'PLAYBACKRATE'))]
    m.params += [l for l in lines if l.startswith(('N163CHANNELS', 'NAMCO'))]
    m.expansion, m.n163 = 0, None
    for l in m.params:
        if l.startswith('EXPANSION'):
            m.expansion = int(l.split()[1])
        elif l.startswith(('N163CHANNELS', 'NAMCO')):
            m.n163 = int(l.split()[1])
    # the original instruments (needed for wavetable / FM chips, which are passed through)
    m.raw_inst = [l for l in lines if l.startswith(('INST', 'MACRO', 'N163WAVE', 'FDSWAVE',
                                                     'FDSMOD', 'FDSMACRO', 'KEYDPCM'))]
    m.dpcm_samples = text_dpcm_samples(lines)

    starts = [i for i, line in enumerate(lines) if line.startswith('TRACK')]
    if not starts:
        raise SystemExit('no TRACK / ORDER data found - is this a FamiTracker text export?')
    m.num_subsongs = len(starts)
    tracks = []
    for ti, begin in enumerate(starts):
        section = lines[begin:starts[ti + 1] if ti + 1 < len(starts) else len(lines)]
        p = section[0].split(None, 4)
        track = dict(rows=int(p[1]), speed=int(p[2]), tempo=int(p[3]))
        name = p[4].strip().strip('"') if len(p) > 4 else 'Track %d' % (ti + 1)
        orders, patterns, cur = [], {}, None
        for line in section:
            if line.startswith('ORDER'):
                orders.append([int(x, 16) for x in line.split(' : ')[1].split()])
            elif line.startswith('PATTERN'):
                cur = int(line.split()[1], 16)
                patterns[cur] = []
            elif line.startswith('ROW') and cur is not None:
                patterns[cur].append([c.split() for c in line.split(' : ')[1:]])
        if not orders or not patterns:
            raise SystemExit('track %d has no ORDER / PATTERN data' % ti)
        tracks.append((track, orders, patterns, name))

    if not 0 <= subsong < len(tracks):
        raise ValueError('Subsong %d not found (file has %d subsongs)' % (subsong, len(tracks)))
    m.track, m.orders, m.patterns, _ = tracks[subsong]
    m.track_names = [t[3] for t in tracks]
    m.nch = len(next(iter(m.patterns.values()))[0]) if m.patterns else 5
    # Dn-FamiTracker text always lists all 8 N163 channels; only the first N163CHANNELS are real
    pos, full = n163_text_layout(m.expansion)
    if m.n163 and m.n163 < 8 and m.nch == full:
        drop = set(range(pos + m.n163, pos + 8))
        m.orders = [[v for c, v in enumerate(o) if c not in drop] for o in m.orders]
        m.patterns = {k: [[cell for c, cell in enumerate(row) if c not in drop] for row in rows]
                      for k, rows in m.patterns.items()}
        m.nch -= len(drop)
    return m


def n163_text_layout(expansion):
    """(index of the first N163 channel, channel count of a text export with all 8 N163
    channels) - FamiTracker's channel order is 2A03, VRC6, MMC5, N163, FDS, VRC7, 5B"""
    pos = 5 + (3 if expansion & 1 else 0) + (2 if expansion & 8 else 0)
    full = pos + (8 if expansion & 16 else 0) + (1 if expansion & 4 else 0) + \
        (6 if expansion & 2 else 0) + (3 if expansion & 32 else 0)
    return pos, full


def pad_n163_text(text, expansion, n163):
    """add the empty N163 channels Dn-FamiTracker's text import expects (it always reads 8)"""
    if not expansion & 16 or not n163 or n163 >= 8:
        return text
    pos, _ = n163_text_layout(expansion)
    at, extra = pos + n163, 8 - n163
    nl = '\r\n' if '\r\n' in text else '\n'
    out = []
    for line in text.split(nl):
        if line.startswith('COLUMNS :'):
            v = line.split(':', 1)[1].split()
            line = 'COLUMNS : ' + ' '.join(v[:at] + ['1'] * extra + v[at:])
        elif line.startswith('ORDER '):
            head, v = line.split(' : ', 1)
            v = v.split()
            line = head + ' : ' + ' '.join(v[:at] + ['00'] * extra + v[at:])
        elif line.startswith('ROW '):
            parts = line.split(' : ')
            cells = parts[1:]
            line = ' : '.join([parts[0]] + cells[:at] + ['... .. . ...'] * extra + cells[at:])
        out.append(line)
    return nl.join(out)


EF_CHARS = {
    1: 'F', 2: 'B', 3: 'D', 4: 'C', 5: 'E', 6: '3',
    8: 'H', 9: 'I', 10: '0', 11: '4', 12: '7', 13: 'P',
    14: 'G', 15: 'Z', 16: '1', 17: '2', 18: 'V', 19: 'Y',
    20: 'Q', 21: 'R', 22: 'A', 23: 'S', 24: 'X', 25: 'M',
    29: 'W'
}


# Every effect FamiTracker / 0CC / Dn-FamiTracker can store, by its internal number (the
# letter depends on the number only; H/I/J/Z/E/W mean different things on different chips).
EFFECT_LETTERS = ['', 'F', 'B', 'D', 'C', 'E', '3', '', 'H', 'I', '0', '4', '7', 'P', 'G', 'Z',
                  '1', '2', 'V', 'Y', 'Q', 'R', 'A', 'S', 'X', 'M', 'H', 'I', 'J', 'W', 'H', 'I',
                  'J', 'W', 'H', 'I', 'L', 'O', 'T', 'Z', 'E', 'Z', '=', 'K', 'N']
# Modules older than 0x450 and Dn-FamiTracker modules store the effects after the Sunsoft ones
# in a different order (Dn-FamiTracker's EFF_CONVERSION_050): file number -> internal number
_EFF_050 = {33: 36, 34: 37, 35: 38, 36: 39, 37: 40, 38: 41, 39: 33, 40: 34, 41: 35}


def effect_letter(num, file_version, dn_module):
    if file_version < 0x450 or dn_module:
        num = _EFF_050.get(num, num)
    return EFFECT_LETTERS[num] if 0 <= num < len(EFFECT_LETTERS) and EFFECT_LETTERS[num] else '?'


class _Reader:
    def __init__(self, data):
        self.d, self.i = data, 0

    def u8(self):
        v = self.d[self.i]
        self.i += 1
        return v

    def s8(self):
        v = struct.unpack_from('<b', self.d, self.i)[0]
        self.i += 1
        return v

    def i32(self):
        v = struct.unpack_from('<i', self.d, self.i)[0]
        self.i += 4
        return v

    def raw(self, n):
        if self.i + n > len(self.d):
            raise ValueError('block is truncated')
        v = self.d[self.i:self.i + n]
        self.i += n
        return v

    def left(self):
        return len(self.d) - self.i


def ftm_instruments_as_text(blocks):
    """Decode the INSTRUMENTS and SEQUENCES* blocks of a binary module into the lines a
    FamiTracker text export would contain (MACRO*, INST*, N163WAVE, FDSWAVE, FDSMOD,
    FDSMACRO).  Covers every instrument type: 2A03, VRC6, VRC7, FDS, N163 and 5B."""
    out = []

    def seq_block(name, prefix):
        """FamiTracker's sequence blocks: 2A03 and VRC6 store release point and setting
        inline only in block version 4 and in a trailing table from version 6; N163 and 5B
        always inline.  2A03 blocks older than version 3 use an obsolete layout (skipped)."""
        if name not in blocks:
            return
        ver, data = blocks[name]
        split_kind = prefix in ('MACRO', 'MACROVRC6')
        if split_kind and ver < 3:
            return
        inline = (ver == 4) if split_kind else True
        r = _Reader(data)
        count = r.i32()
        found = []
        for _ in range(count):
            idx, typ = r.i32(), r.i32()
            n = r.u8()
            loop = r.i32()
            rel, setting = -1, 0
            if inline:
                rel, setting = r.i32(), r.i32()
            vals = [r.s8() for _ in range(n)]
            found.append([idx, typ, loop, rel, setting, vals])
        if split_kind and ver >= 6:
            for f in found:
                f[3], f[4] = r.i32(), r.i32()
        for idx, typ, loop, rel, setting, vals in found:
            out.append('%s %7d %3d %3d %3d %3d : %s' % (prefix, typ, idx, loop, rel, setting,
                                                       ' '.join(map(str, vals))))

    seq_block('SEQUENCES', 'MACRO')
    seq_block('SEQUENCES_VRC6', 'MACROVRC6')
    seq_block('SEQUENCES_N163', 'MACRON163')
    seq_block('SEQUENCES_S5B', 'MACROS5B')

    if 'INSTRUMENTS' not in blocks:
        return out
    ver, data = blocks['INSTRUMENTS']
    r = _Reader(data)
    count = r.i32()
    for _ in range(count):
        idx, typ = r.i32(), r.u8()
        lines = []

        def seq_refs():
            n = r.i32()
            refs = []
            for _ in range(n):
                en, ix = r.u8(), r.u8()
                refs.append(ix if en else -1)
            return (refs + [-1] * 5)[:5]

        if typ in (1, 2, 6):                                   # 2A03 / VRC6 / 5B
            refs = seq_refs()
            if typ == 1:                                       # DPCM key map -> KEYDPCM lines
                def key_line(octave, note, smp, pitch, delta=-1):
                    if 1 <= smp <= 64:
                        lines.append('KEYDPCM %3d %3d %3d   %3d %3d %3d %5d %3d' % (
                            idx, octave, note, smp - 1, pitch & 0x0F, 1 if pitch & 0x80 else 0, 0,
                            delta if 0 <= delta <= 127 else -1))
                if ver >= 7:                                   # 0.5+: only the assigned keys
                    for _ in range(r.i32()):
                        midi = r.u8()                          # octave * 12 + key (C = 0)
                        smp, pitch, delta = r.u8(), r.u8(), r.s8()
                        key_line(midi // 12, midi % 12, smp, pitch, delta)
                else:
                    octaves = 6 if ver == 1 else 8
                    for octave in range(octaves):
                        for note in range(12):
                            smp, pitch = r.u8(), r.u8()
                            delta = r.s8() if ver > 5 else -1
                            key_line(octave, note, smp, pitch, delta)
            kind = {1: 'INST2A03', 2: 'INSTVRC6', 6: 'INSTS5B'}[typ]
            head = '%s %3d %5d %3d %3d %3d %3d' % ((kind, idx) + tuple(refs))
        elif typ == 5:                                         # N163
            refs = seq_refs()
            size, pos = r.i32(), r.i32()
            if ver >= 8:
                r.i32()                                        # automatic wave position flag
            nwaves = r.i32()
            if not (0 < size <= 240 and 0 <= nwaves <= 64):
                raise ValueError('implausible N163 wave data')
            waves = [list(r.raw(size)) for _ in range(nwaves)]
            head = 'INSTN163 %3d %5d %3d %3d %3d %3d %3d %3d %3d' % ((idx,) + tuple(refs) + (size, pos, nwaves))
            for w, vals in enumerate(waves):
                lines.append('N163WAVE %3d %3d : %s' % (idx, w, ' '.join(map(str, vals))))
        elif typ == 3:                                         # VRC7
            patch = r.i32()
            regs = list(r.raw(8))
            head = 'INSTVRC7 %3d %5d %s' % (idx, patch, ' '.join('%02X' % x for x in regs))
        elif typ == 4:                                         # FDS
            wave = list(r.raw(64))
            mod = list(r.raw(32))
            rate, depth, delay = r.i32(), r.i32(), r.i32()
            head = 'INSTFDS %3d %5d %3d %3d %3d' % (idx, 1, rate, depth, delay)
            lines.append('FDSWAVE %3d : %s' % (idx, ' '.join(map(str, wave))))
            lines.append('FDSMOD %3d : %s' % (idx, ' '.join(map(str, mod))))
            for t in range(3):                                 # volume, arpeggio, pitch
                n = r.u8()
                loop, rel, setting = r.i32(), r.i32(), r.i32()
                vals = [r.s8() for _ in range(n)]
                if n:
                    lines.append('FDSMACRO %3d %3d %3d %3d %3d : %s' % (idx, t, loop, rel, setting,
                                                                       ' '.join(map(str, vals))))
        else:
            raise ValueError('unknown instrument type %d' % typ)
        nlen = r.i32()
        if not 0 <= nlen <= 128:
            raise ValueError('implausible instrument name length')
        name = r.raw(nlen).decode('latin1', errors='replace').replace('"', "'")
        out.append('%s "%s"' % (head, name))
        out.extend(lines)
    return out


FTM_MAGICS = {b'FamiTracker Module': 18, b'Dn-FamiTracker Module': 21}   # .ftm / .0cc and .dnm


def ftm_dpcm_samples(blocks):
    """DPCM SAMPLES block -> {index: (name, bytes)}"""
    if 'DPCM SAMPLES' not in blocks:
        return {}
    _, data = blocks['DPCM SAMPLES']
    r = _Reader(data)
    out = {}
    if r.left() < 1:
        return out
    for _ in range(r.u8()):
        idx = r.u8()
        name = r.raw(r.i32()).decode('latin1', errors='replace')
        size = r.i32()
        out[idx] = (name, bytes(r.raw(size)))
    return out


def text_dpcm_samples(lines):
    """DPCMDEF / DPCM lines of a FamiTracker text export -> {index: (name, bytes)}"""
    out, cur = {}, None
    for l in lines:
        if l.startswith('DPCMDEF'):
            p = l.split(None, 3)
            cur = int(p[1])
            name = p[3].strip().strip('"') if len(p) > 3 else 'sample'
            out[cur] = (name, bytearray(), int(p[2]))
        elif l.startswith('DPCM ') and cur is not None:
            out[cur][1].extend(int(x, 16) for x in l.split(':', 1)[1].split())
    return {k: (n, bytes(d[:size]) + bytes(max(0, size - len(d)))) for k, (n, d, size) in out.items()}


def dpcm_text_lines(samples):
    """{index: (name, bytes)} -> DPCMDEF / DPCM lines"""
    out = []
    for idx in sorted(samples):
        name, data = samples[idx]
        out.append('DPCMDEF %3d %5d "%s"' % (idx, len(data), name.replace('"', "'")))
        for i in range(0, len(data), 32):
            out.append('DPCM : ' + ' '.join('%02X' % b for b in data[i:i + 32]))
    return out


def parse_ftm(data, subsong=0):
    magic = next((n for k, n in FTM_MAGICS.items() if data[:n] == k), None)
    if magic is None or len(data) < magic + 4:
        raise ValueError('Not a valid FamiTracker Module (.ftm / .dnm)')
    ver = struct.unpack('<I', data[magic:magic + 4])[0]
    idx = magic + 4
    blocks = {}
    while idx + 24 <= len(data):
        bid, bver, bsize = struct.unpack('<16sII', data[idx:idx + 24])
        idx += 24
        name = bid.split(b'\x00')[0].decode('ascii', errors='ignore')
        blocks[name] = (bver, data[idx:idx + bsize])
        idx += bsize

    m = Module()

    # PARAMS - field layout follows FamiTracker's own loader (ReadBlock_Parameters) for
    # every block version, from 0.2 files (v1) to Dn-FamiTracker 0.5 (v9)
    pver, pdata = blocks.get('PARAMS', (0, b''))
    r = _Reader(pdata)
    speed_v1 = None
    if pver == 1:
        speed_v1, chip = r.i32(), 0
    else:
        chip = r.u8()
    channels = r.i32()
    machine = r.i32()
    playback = None
    if pver >= 7:
        playback = (r.i32(), r.i32())
    else:
        r.i32()                                   # engine speed
    vib = r.i32() if pver > 2 else 0
    if pver >= 9:
        r.i32()                                   # sweep reset
    if 3 < pver <= 6:
        r.i32(); r.i32()                          # row highlights
    if channels == 5:
        chip = 0                                  # some files store 0xFF here
    _namco_channels = None
    if pver >= 5 and chip & 16 and r.left() >= 4:
        _namco_channels = r.i32()
    split = r.i32() if pver >= 6 and r.left() >= 4 else 21     # FamiTracker's OLD_SPEED_SPLIT_POINT
    m.params = [
        f'MACHINE         {machine}',
        'FRAMERATE       0',
        f'EXPANSION       {chip}',
        f'VIBRATO         {vib}',
        f'SPLIT           {split}',
    ]
    if playback:
        m.params.append(f'PLAYBACKRATE    {playback[0]} {playback[1]}')
    m.speed_v1 = speed_v1
    m.expansion = chip
    m.nch = channels
    m.n163 = None
    if chip & 16 and _namco_channels:
        m.n163 = max(1, min(8, int(_namco_channels)))
        if m.n163:
            m.params.append(f'N163CHANNELS    {m.n163}')
    # the original instruments and macros, as text lines (needed to pass wavetable / FM
    # chips through, and harmless otherwise)
    try:
        m.raw_inst = ftm_instruments_as_text(blocks)
        m.dpcm_samples = ftm_dpcm_samples(blocks)
    except (struct.error, IndexError, ValueError) as exc:
        print('warning: could not read the instruments of this .ftm (%s); expansion-chip '
              'channels will have no instrument - export it as text if that matters' % exc,
              file=sys.stderr)
        m.raw_inst = []
        m.dpcm_samples = {}

    # INFO
    iver, idata = blocks.get('INFO', (0, b''))
    title = idata[:32].split(b'\x00')[0].decode('latin1', errors='ignore')
    author = idata[32:64].split(b'\x00')[0].decode('latin1', errors='ignore')
    copyr = idata[64:96].split(b'\x00')[0].decode('latin1', errors='ignore')
    m.info = [
        f'TITLE           "{title}"',
        f'AUTHOR          "{author}"',
        f'COPYRIGHT       "{copyr}"',
    ]

    # HEADER
    hver, hdata = blocks.get('HEADER', (0, b''))
    num_tracks = hdata[0] + 1
    h_idx = 1
    track_names = []
    for _ in range(num_tracks):
        end = hdata.find(b'\x00', h_idx)
        track_names.append(hdata[h_idx:end].decode('latin1', errors='ignore'))
        h_idx = end + 1
    chan_types = []
    eff_cols = {}
    for c in range(channels):
        chan_types.append(hdata[h_idx])
        h_idx += 1
        for t in range(num_tracks):
            eff_cols[(t, c)] = hdata[h_idx]
            h_idx += 1

    # FRAMES
    fver, fdata = blocks.get('FRAMES', (0, b''))
    f_idx = 0
    m.track = None
    m.orders = []
    m.num_subsongs = num_tracks
    m.track_names = track_names
    for t in range(num_tracks):
        frame_count, trk_speed, trk_tempo, pat_len = struct.unpack_from('<IIII', fdata, f_idx)
        f_idx += 16
        orders_for_t = []
        for f in range(frame_count):
            row = [fdata[f_idx + c] for c in range(channels)]
            f_idx += channels
            orders_for_t.append(row)
        if t == subsong:
            m.track = dict(rows=pat_len, speed=trk_speed, tempo=trk_tempo)
            m.orders = orders_for_t

    if m.track is None:
        raise ValueError(f'Subsong {subsong} not found (file has {num_tracks} subsongs)')

    pat_len = m.track['rows']
    m.patterns = {}
    max_pat = max(max(od) for od in m.orders) if m.orders else 0
    for p in range(max_pat + 1):
        m.patterns[p] = []
        for r in range(pat_len):
            row_cells = []
            for c in range(channels):
                cols = eff_cols.get((subsong, c), 0) + 1
                row_cells.append(['...', '..', '.'] + ['...'] * cols)
            m.patterns[p].append(row_cells)

    # PATTERNS
    pver, pdata = blocks.get('PATTERNS', (0, b''))
    p_idx = 0
    while p_idx < len(pdata):
        if p_idx + 16 > len(pdata):
            raise ValueError('truncated PATTERNS record header')
        trk, ch, pat, items = struct.unpack_from('<IIII', pdata, p_idx)
        p_idx += 16
        if ch >= channels:
            raise ValueError('PATTERNS channel index %d is out of range' % ch)
        cols = eff_cols.get((trk, ch), 0) + 1
        # PATTERNS v6 stores a one-byte row index and four effect slots.
        # In v6 an empty effect is only one zero byte; its parameter byte is
        # present only when the effect code is nonzero. Older blocks store
        # fixed code/value pairs and use a 32-bit row index.
        fixed_v6 = pver >= 6
        stored_effects = 4 if fixed_v6 else cols
        if trk == subsong and pat not in m.patterns:
            m.patterns[pat] = []
            for r in range(pat_len):
                row_cells = []
                for c in range(channels):
                    row_cells.append(['...', '..', '.'] + ['...'] * (eff_cols.get((subsong, c), 0) + 1))
                m.patterns[pat].append(row_cells)
        for _ in range(items):
            if fixed_v6:
                if p_idx + 5 > len(pdata):
                    raise ValueError('truncated PATTERNS v6 row header')
                r, note, octv, inst, vol = struct.unpack_from('<BBBBB', pdata, p_idx)
                p_idx += 5
            else:
                if p_idx + 8 > len(pdata):
                    raise ValueError('truncated PATTERNS row header')
                r, note, octv, inst, vol = struct.unpack_from('<IBBBB', pdata, p_idx)
                p_idx += 8
            eff_tokens = []
            for effect_idx in range(stored_effects):
                if p_idx >= len(pdata):
                    raise ValueError('truncated PATTERNS effect data')
                num = pdata[p_idx]
                p_idx += 1
                if fixed_v6:
                    if num:
                        if p_idx >= len(pdata):
                            raise ValueError('truncated PATTERNS effect parameter')
                        param = pdata[p_idx]
                        p_idx += 1
                    else:
                        param = 0
                else:
                    if p_idx >= len(pdata):
                        raise ValueError('truncated PATTERNS effect parameter')
                    param = pdata[p_idx]
                    p_idx += 1
                if effect_idx < cols:
                    if num == 0:
                        eff_tokens.append('...')
                    else:
                        eff_tokens.append(f'{effect_letter(num, ver, magic == 21)}{param:02X}')
            if trk == subsong:
                if r >= pat_len:
                    continue                    # rows beyond the pattern length are kept but never played
                if note == 0:
                    note_str = '...'
                elif note == 13:
                    note_str = '==='
                elif note == 14:
                    note_str = '---'
                elif ch == 3:  # Noise
                    val = (octv * 12 + note - 1) - 48
                    note_str = f'{val & 15:X}-#'
                else:
                    note_str = f'{NOTE_NAMES[(note - 1) % 12]}{octv}'
                inst_str = '..' if inst >= 64 else f'{inst:02X}'
                vol_str = '.' if vol >= 16 else f'{vol:X}'
                m.patterns[pat][r][ch] = [note_str, inst_str, vol_str] + eff_tokens

    return m


def parse_furnace_text(text, subsong=0):
    lines = text.replace('\r\n', '\n').split('\n')
    m = Module()
    title = ""
    author = ""
    copyright_ = ""
    expansion = 0
    channels = 5
    for l in lines:
        if l.startswith('- name:'):
            title = l.split(':', 1)[1].strip()
        elif l.startswith('- author:'):
            author = l.split(':', 1)[1].strip()
        elif 'VRC6' in l:
            expansion = 1
            channels = 8

    m.expansion = expansion
    m.nch = channels
    m.num_subsongs = 1
    m.track_names = [title or 'Furnace export']
    m.info = [
        f'TITLE           "{title}"',
        f'AUTHOR          "{author}"',
        f'COPYRIGHT       "{copyright_}"',
    ]
    m.params = [
        'MACHINE         0',
        'FRAMERATE       0',
        f'EXPANSION       {expansion}',
        'VIBRATO         1',
        'SPLIT           21',
    ]

    pat_len = 256
    speed = 1
    tempo = 150
    m.orders = []
    m.patterns = {}
    in_orders = False
    fence_count = 0
    cur_order = None

    for line in lines:
        l = line.strip()
        if l.startswith('- speeds:'):
            speed = int(l.split(':', 1)[1].strip().split()[0])
        elif l.startswith('- virtual tempo:'):
            tempo = int(l.split(':', 1)[1].strip().split('/')[0])
        elif l.startswith('- pattern length:'):
            pat_len = int(l.split(':', 1)[1].strip())
        elif l == 'orders:':
            in_orders = True
            fence_count = 0
            continue
        elif in_orders:
            if l.startswith('```'):
                fence_count += 1
                if fence_count == 2:
                    in_orders = False
            elif '|' in l:
                pats = [int(x, 16) for x in l.split('|')[1].split()]
                m.orders.append(pats)
        elif l.startswith('----- ORDER'):
            cur_order = int(l.split()[2], 16)
        elif cur_order is not None and '|' in l and not l.startswith('```'):
            parts = l.split('|')
            row_idx = int(parts[0].strip(), 16)
            chan_cells = parts[1:]
            for ch, cell_str in enumerate(chan_cells):
                tokens = cell_str.strip().split()
                if not tokens:
                    continue
                raw_note = tokens[0] if len(tokens) > 0 else '...'
                raw_inst = tokens[1] if len(tokens) > 1 else '..'
                raw_vol = tokens[2] if len(tokens) > 2 else '.'
                raw_effs = tokens[3:] if len(tokens) > 3 else []

                if raw_note == '...':
                    note_tok = '...'
                elif raw_note == 'OFF':
                    note_tok = '---'
                elif raw_note == '===':
                    note_tok = '==='
                elif ch == 3:
                    if len(raw_note) >= 3 and raw_note[:2] in NOTE_INDEX:
                        octv = int(raw_note[2])
                        val = (octv - 4) * 12 + NOTE_INDEX[raw_note[:2]]
                        note_tok = f'{val & 15:X}-#'
                    else:
                        note_tok = '...'
                else:
                    note_tok = raw_note

                inst_tok = raw_inst if raw_inst != '..' else '..'
                vol_tok = raw_vol[-1] if raw_vol not in ('.', '..') else '.'

                eff_toks = []
                for eff in raw_effs:
                    if eff == '....' or len(eff) < 4:
                        continue
                    code = eff[:2].upper()
                    param = eff[2:]
                    if code == 'E5': eff_toks.append(f'P{param}')
                    elif code == '12': eff_toks.append(f'V{param}')
                    elif code == '09': eff_toks.append(f'F{param}')
                    elif code == '0A': eff_toks.append(f'B{param}')
                    elif code == '0B': eff_toks.append(f'D{param}')
                    elif code == '00': eff_toks.append(f'0{param}')
                    elif code == '0D': eff_toks.append(f'G{param}')
                    elif code == '0E': eff_toks.append(f'S{param}')

                if not eff_toks:
                    eff_toks = ['...']

                if cur_order < len(m.orders) and ch < len(m.orders[cur_order]):
                    pat_id = m.orders[cur_order][ch]
                    if pat_id not in m.patterns:
                        m.patterns[pat_id] = [[['...', '..', '.', '...'] for _ in range(channels)] for _ in range(pat_len)]
                    m.patterns[pat_id][row_idx][ch] = [note_tok, inst_tok, vol_tok] + eff_toks

    m.track = dict(rows=pat_len, speed=speed, tempo=tempo)
    return m


def load_input(input_source, subsong=0):
    if isinstance(input_source, Module):
        return input_source
    if isinstance(input_source, (bytes, bytearray)):
        if input_source[:18] == b'FamiTracker Module' or input_source[:21] == b'Dn-FamiTracker Module':
            return parse_ftm(input_source, subsong=subsong)
        text = input_source.decode('utf-8', errors='replace')
        if text.startswith('# Furnace Text Export'):
            return parse_furnace_text(text, subsong=subsong)
        return parse_module(text, subsong=subsong)
    if isinstance(input_source, str):
        if '\n' not in input_source and os.path.exists(input_source):
            if input_source.lower().endswith(('.ftm', '.dnm', '.0cc')):
                with open(input_source, 'rb') as f:
                    data = f.read()
                try:
                    return parse_ftm(data, subsong=subsong)
                except (SystemExit, ValueError):
                    raise
                except Exception as exc:
                    raise SystemExit('could not read %s directly (%s). Export it as text from '
                                     'FamiTracker (File > Export Text, or put it through render.bat) '
                                     'and run this on the .txt instead.' % (input_source, exc))
            with open(input_source, 'r', encoding='utf-8', errors='replace', newline='') as f:
                text = f.read()
            if text.startswith('# Furnace Text Export'):
                return parse_furnace_text(text, subsong=subsong)
            return parse_module(text, subsong=subsong)
        if input_source.startswith('# Furnace Text Export'):
            return parse_furnace_text(input_source, subsong=subsong)
        return parse_module(input_source, subsong=subsong)
    raise TypeError(f'Unsupported input type: {type(input_source)}')


def decode_note(tok, ch):
    if tok == '...':
        return None
    if tok in ('---', '==='):
        return 'cut'
    if ch == NOI:
        return int(tok[0], 16)
    return int(tok[2]) * 12 + NOTE_INDEX[tok[:2]]


def note_text(n, ch):
    if ch == NOI:
        return '%X-#' % (n & 15)
    return '%s%d' % (NOTE_NAMES[n % 12], n // 12)


def flatten_frames(m):
    """rows in playing order, one per tick, honouring the flow effects the newer nsfimport
    writes: Dxx ends a pattern early, Bxx jumps back to the loop (the song stops there; the
    loop start is remembered in m.loop_tick), C00 halts.  Older raw imports have none of
    these and come out unchanged."""
    rows_per = m.track['rows']
    frames, start_of = [], {}
    m.loop_tick, m.halted = None, False
    oi, r, guard = 0, 0, 0
    while oi < len(m.orders) and guard < 1000000:
        guard += 1
        if r == 0:
            start_of.setdefault(oi, len(frames))
        od = m.orders[oi]
        row = [m.patterns[od[c]][r][c] for c in range(m.nch)]
        frames.append(row)
        jump = skip = halt = None
        for cell in row:
            for e in cell[3:]:
                if len(e) == 3 and e != '...':
                    if e[0] == 'B':
                        jump = int(e[1:], 16)
                    elif e[0] == 'D':
                        skip = int(e[1:], 16)
                    elif e[0] == 'C':
                        halt = True
        if halt:
            m.halted = True
            break
        if jump is not None:
            if jump in start_of:
                m.loop_tick = start_of[jump]
                break
            oi, r = jump, 0
            continue
        if skip is not None:
            oi, r = oi + 1, min(skip, rows_per - 1)
            continue
        r += 1
        if r >= rows_per:
            oi, r = oi + 1, 0
    return frames


class ChanState:
    pass


CONT_CENTS = 50     # a note change smaller than this (in cents) without a volume re-attack
                    # is the same note continuing (the importer re-labels wide bends)


def period_of(n, ch=SQ1):
    """period register value FamiTracker / NSFImport use for note number n on channel ch.
    FamiTracker's note 0 (C-0) is 32.7032 Hz, so its A-4 is 880 Hz.  Pulses, triangle and
    noise share one table (the triangle simply sounds an octave lower); the VRC6 sawtooth
    has its own (CPU / 14 instead of CPU / 16)."""
    freq = 32.7032 * 2.0 ** (int(n) / 12.0)
    div = 14.0 if KIND.get(ch) == 'saw' else 16.0
    return int(1789773.0 / (div * freq) - 0.5)


def estimate_k(frames):
    """How many period-register units one fine-pitch step (Pxx in the raw import) is worth.
    NSFImport writes P = (table period of the nearest note) - (actual period register), so
    it is exactly 1 (checked against FamiTracker's own NSF output)."""
    return 1.0


def build_state(frames, ch, k=None):
    N = len(frames)
    s = ChanState()
    s.n = N
    s.note = np.zeros(N, int)
    s.onset = np.zeros(N, bool)
    s.cut = np.zeros(N, bool)
    s.snd = np.zeros(N, bool)
    s.fine = np.zeros(N, int)
    s.duty = np.zeros(N, int)
    colev = np.zeros(N, bool)
    vol_arr = np.zeros(N, int)
    cur, sounding, v, fi, du = 0, False, 15, 0, 0
    s.inst = np.full(N, -1, int)
    raw_i = -1
    for f in range(N):
        c = frames[f][ch]
        if len(c) > 1 and c[1] != '..':
            raw_i = int(c[1], 16)
        s.inst[f] = raw_i
        n = decode_note(c[0], ch)
        if n == 'cut':
            sounding = False
            s.cut[f] = True
        elif n is not None:
            cur, sounding = n, True
            colev[f] = True
        if c[2] != '.' and ch != TRI:
            v = int(c[2], 16)
        for e in c[3:]:
            if e[0] == 'P':
                fi = int(e[1:], 16) - 0x80
            elif e[0] == 'V':
                du = int(e[1:], 16)
        s.note[f], s.snd[f], vol_arr[f], s.fine[f], s.duty[f] = cur, sounding, v, fi, du
    s.eff = np.where(s.snd, 15 if ch == TRI else vol_arr, 0)
    s.onset = colev.copy()
    if ch != TRI:
        # a note retriggered on the same pitch has no note cell, just a volume jump back up
        e = s.eff
        lvl = int(e[0])
        for f in range(1, N):
            if s.onset[f]:
                lvl = int(e[f])
                continue
            if not s.snd[f] or e[f] < e[f - 1] + 2:
                continue
            if e[f - 1] == 0 and f >= 2 and e[f - 2] > 0 and e[f] <= e[f - 2] + 1:
                continue                        # one-tick dropout that comes back to the same level
            # a real retrigger: the envelope had decayed, and the volume goes back up to where the
            # note started (an accent spike on an undecayed note is not a new note)
            if e[f - 1] <= lvl - 2 and e[f] >= lvl - 1:
                s.onset[f] = True
                lvl = int(e[f])
    s.reclassified = 0
    if k is not None and ch != NOI:
        tri = ch
        for f in np.flatnonzero(colev):
            if f == 0 or not s.snd[f - 1] or s.eff[f - 1] <= 0 or s.cut[f]:
                continue
            pe_prev = period_of(s.note[f - 1], tri) - k * s.fine[f - 1]
            pe = period_of(s.note[f], tri) - k * s.fine[f]
            if pe_prev > 0 and pe > 0 and abs(1200 * np.log2(pe_prev / pe)) <= CONT_CENTS \
                    and s.eff[f] <= s.eff[f - 1]:
                s.onset[f] = False            # same note, just re-labelled
                s.reclassified += 1
        # stable note + pitch offset measured against it (continuous across relabels)
        raw_note = s.note.copy()
        stable = np.zeros(N, int)
        cur_s = 0
        for f in range(N):
            if s.onset[f]:
                cur_s = raw_note[f]
            stable[f] = cur_s
        fine_c = s.fine.astype(float).copy()
        for f in range(N):
            if stable[f] != raw_note[f]:
                fine_c[f] += (period_of(stable[f], tri) - period_of(raw_note[f], tri)) / k
        s.note = stable
        s.fine = np.rint(fine_c).astype(int)
    return s


def trim_state(s, T):
    for k in ('note', 'onset', 'cut', 'snd', 'fine', 'duty', 'eff', 'inst'):
        setattr(s, k, getattr(s, k)[:T])
    s.n = T


# ----------------------------------------------------------------------------
# loop detection
# ----------------------------------------------------------------------------
def detect_loop(states, min_len=256, min_score=0.8, tol=1):
    """Returns ((start, length) | None, (score, length)).  Matching is done on note onsets
    and tolerates +-tol ticks of timing jitter between passes."""
    N = states[SQ1].n
    melodic = [c for c in MUSIC if c != NOI]
    arr = np.full((len(melodic), N), -1, int)
    for i, c in enumerate(melodic):
        st = states[c]
        arr[i, st.onset] = st.note[st.onset]
    exact = []
    for L in range(min_len, N - min_len):
        a, b = arr[:, :N - L], arr[:, L:]
        mask = a >= 0
        cnt = mask.sum()
        if cnt >= 20:
            exact.append((((a == b) & mask).sum() / cnt, L))
    exact.sort(reverse=True)

    def tolerant(L):
        M = N - L - tol
        a = arr[:, :M]
        mask = a >= 0
        hit = np.zeros(a.shape, bool)
        for d in range(-tol, tol + 1):
            hit |= arr[:, L + d:L + d + M] == a
        return (hit & mask).sum() / max(1, mask.sum()), mask & ~hit
    best = (0.0, None, None)
    for _, L in exact[:25]:
        sc, bad = tolerant(L)
        if sc > best[0] + 1e-9:
            best = (sc, L, bad)
    if best[1] is None or best[0] < min_score:
        return None, (best[0], best[1])
    L, bad = best[1], best[2]
    pos = np.flatnonzero(bad.any(0))
    half = (N - L) // 2
    early = pos[pos < half]
    start = int(early[-1]) + 1 if len(early) else 0
    return (start, L), (best[0], L)


# ----------------------------------------------------------------------------
# row grid
# ----------------------------------------------------------------------------
def structural_onsets(st, merge_gap):
    ons = np.flatnonzero(st.onset)
    aud = np.concatenate([[0], np.cumsum(st.eff > 0)])
    out, last = [], None
    for i, f in enumerate(ons):
        nxt = ons[i + 1] if i + 1 < len(ons) else st.n
        if aud[nxt] - aud[f] == 0:
            continue
        if last is not None and f - last < merge_gap:
            last = f
            continue
        out.append(int(f))
        last = f
    return out


def int_phase(states, s):
    ph = collections.Counter()
    for c in MUSIC:
        o = structural_onsets(states[c], 3)
        for a, b in zip(o, o[1:]):
            if b - a >= s:
                ph[b % s] += 1
    return ph.most_common(1)[0][0] if ph else 0


def row_starts(S, T, K):
    """frames at which FamiTracker starts rows k=0..K-1 for speed S, tempo T (60 Hz)"""
    D = (T * 24) // S
    return -(-(np.arange(K) * 3600) // D)


def choose_grid(states, forced=None):
    """Find the row grid.  Songs on a whole speed (tempo 150) use that speed.  Songs whose
    rows are a fractional number of ticks long (e.g. 5.5) get a speed+tempo pair."""
    N = states[SQ1].n
    ons = np.array([f for c in MUSIC for f in structural_onsets(states[c], 3)])
    if forced:
        return dict(kind='int', s=forced, phase=int_phase(states, forced), msg='speed %d (forced)' % forced)
    cands = [(S, 150) for S in (3, 4, 5, 6, 7, 8, 9, 10, 12)]
    for S in (3, 4, 6, 8):
        for T in range(32, 256):
            if T != 150 and 3 <= 150 * S / T <= 12:
                cands.append((S, T))
    results = []
    for S, T in cands:
        D = (T * 24) // S
        r = 3600 / D
        R = row_starts(S, T, int((N + 20) * D / 3600) + 4)
        mask = np.zeros(N + 20, bool)
        mask[R[R < N + 20]] = True
        best = (0.0, 0)
        for sh in range(int(np.ceil(r))):
            A = mask[ons + sh].mean()
            if A > best[0]:
                best = (A, sh)
        results.append((best[0], r, S, T, best[1]))
    top = max(x[0] for x in results)
    ok = [x for x in results if x[0] >= top - 0.10 and x[0] - 1 / x[1] >= 0.25] or [max(results)]
    A, r, S, T, sh = min(ok, key=lambda x: (round(x[1], 3), x[3] != 150, -x[0]))
    if T == 150:
        return dict(kind='int', s=S, phase=int_phase(states, S),
                    msg='speed %d (%.0f%% of note starts on the row grid)' % (S, A * 100))
    return dict(kind='frac', S=S, T=T, shift=sh, r=r,
                msg='rows are %.3f ticks long -> speed %d, tempo %d (%.0f%% of note starts on the row grid)'
                    % (r, S, T, A * 100))


class Grid:
    """whole-speed grid: maps a raw tick to (row, delay-within-row)"""
    tempo = 150

    def __init__(self, s, phase, preroll=True):
        self.s = s
        if phase and preroll:
            self.shift, self.first = 0, s + phase    # row 0 is longer (F effect)
        else:
            self.shift, self.first = (s - phase) % s, s

    def locate(self, f):
        t = f + self.shift
        if t < self.first:
            return 0, t
        return 1 + (t - self.first) // self.s, (t - self.first) % self.s

    def row_start(self, row):
        return -self.shift if row == 0 else self.first + (row - 1) * self.s - self.shift


class GridF:
    """fractional grid (FamiTracker speed + tempo): rows start at ceil(k*r) ticks"""

    def __init__(self, S, T, shift, nframes):
        self.s, self.tempo, self.shift, self.first = S, T, shift, S
        D = (T * 24) // S
        self.r = 3600 / D
        self.R = row_starts(S, T, int((nframes + shift + 16) * D / 3600) + 4)

    def locate(self, f):
        t = f + self.shift
        k = int(np.searchsorted(self.R, t, 'right')) - 1
        return k, t - int(self.R[k])

    def row_start(self, row):
        return int(self.R[row]) - self.shift


# ----------------------------------------------------------------------------
# notes (segments)
# ----------------------------------------------------------------------------
class Segment:
    pass


def find_period(seg, max_t=24):
    """smallest T such that the first cycle of a note run repeats (an arpeggio)"""
    n = len(seg)
    for T in range(2, min(max_t, n // 2) + 1):
        if len(set(seg[:T].tolist())) >= 2 and np.array_equal(seg[T:2 * T], seg[:T]):
            return T
    return None


def split_arpeggios(st, a, b, aligned_idx):
    """a merged run [a,b) that is a repeating arpeggio is cut wherever the chord changes
    (at the row boundary just before the first deviating frame)"""
    cuts, start = [a], a
    while True:
        seg = st.note[start:b]
        T = find_period(seg)
        if T is None:
            break
        mism = np.flatnonzero(seg[T:] != seg[:-T])
        if len(mism) == 0:
            break
        f = start + T + int(mism[0])
        k = int(np.searchsorted(aligned_idx, f, 'right')) - 1
        c = int(aligned_idx[k]) if k >= 0 else f
        if c <= start:
            c = f
        cuts.append(c)
        start = c
    return cuts


def fast_runs(st, ons, pa, pb, min_notes=6):
    """steady up/down runs of note changes 1-2 ticks apart (sweeps, scale runs).
    Returns (start ticks, ticks that must not be cut)."""
    inside = [int(f) for f in ons if pa <= f < pb]
    starts, protected = [], set()
    i = 0
    while i < len(inside) - 1:
        if inside[i + 1] - inside[i] > 2:
            i += 1
            continue
        sign, j, changes = 0, i, 0
        while j + 1 < len(inside) and inside[j + 1] - inside[j] <= 2:
            dn = int(st.note[inside[j + 1]]) - int(st.note[inside[j]])
            sd = (dn > 0) - (dn < 0)
            if sd and sign and sd != sign:
                break
            if sd:
                sign = sd
                changes += 1
            j += 1
        if changes >= min_notes - 1:
            # the run's flat lead-in (repeats of the first note) belongs to the run too
            if inside[i] > pa:
                starts.append(inside[i])
            protected.update(range(inside[i] + 1, inside[j] + 1))
            i = j
        else:
            i = max(j, i + 1)
    return starts, protected


def split_transient(st, a, b, max_pre=9, min_span=6, min_hold=4):
    """a short steep sweep (drum hit) followed by a held note is two notes: the triangle has no
    volume, so the second note has no onset of its own.  Returns the start of the held note
    or None."""
    n = st.note[a:b].astype(int)
    for p in range(3, min(max_pre, len(n) - 1) + 1):
        pre, post = n[:p], n[p:p + min_hold]
        if len(post) < min(min_hold, len(n) - p) or np.any(post != post[0]):
            continue
        d = np.diff(pre)
        mono = np.all(d <= 0) or np.all(d >= 0)
        if mono and np.count_nonzero(d) >= 2 and post[0] != pre[-1] and \
                (abs(int(pre[-1]) - int(pre[0])) >= min_span or abs(int(post[0]) - int(pre[-1])) >= 4):
            return a + p
    return None


def find_segments(st, ch, merge_gap, aligned):
    N = st.n
    ons = np.flatnonzero(st.onset)
    aud = np.concatenate([[0], np.cumsum(st.eff > 0)])
    cutpos = np.flatnonzero(st.cut)
    starts, last = [], None
    for i, f in enumerate(ons):
        nxt = ons[i + 1] if i + 1 < len(ons) else N
        if aud[nxt] - aud[f] == 0:
            continue
        reattack = f > 0 and st.eff[f] >= st.eff[f - 1] + 2      # volume jumped back up: a new note
        if last is not None and f - last < merge_gap and not reattack and \
                not np.any((cutpos > last) & (cutpos <= f)):
            last = f
            continue
        starts.append(int(f))
        last = f
    aligned_idx = np.flatnonzero(aligned)
    if ch != NOI:
        refined = []
        for i, a in enumerate(starts):
            b = starts[i + 1] if i + 1 < len(starts) else N
            pieces = split_arpeggios(st, a, b, aligned_idx) if b - a >= 8 else [a]
            for j, pa in enumerate(pieces):
                pb = pieces[j + 1] if j + 1 < len(pieces) else b
                refined.append(pa)
                # long chain of onsets that is not a repeating arpeggio = several real
                # notes (one per row) that were glued together: restart at each grid onset
                if pb - pa >= 6 * merge_gap and find_period(st.note[pa:pb]) is None:
                    run_starts, protected = fast_runs(st, ons, pa, pb)
                    refined.extend(int(f) for f in ons if pa < f < pb and aligned[f]
                                   and f not in protected)
                    refined.extend(run_starts)          # a fast run starts a note of its own
        starts = sorted(set(refined))
    if ch == TRI:
        extra = []
        for i, a in enumerate(starts):
            b = starts[i + 1] if i + 1 < len(starts) else N
            p = split_transient(st, a, b)
            if p is not None:
                extra.append(p)
        starts = sorted(set(starts) | set(extra))
    segs = []
    for i, a in enumerate(starts):
        b = starts[i + 1] if i + 1 < len(starts) else N
        sg = Segment()
        sg.ch, sg.start, sg.end, sg.L = ch, a, b, b - a
        sg.base = int(st.note[a])
        nn = st.note[a:b]
        lag = False
        if ch != NOI and len(nn) >= 4 and nn[0] != nn[1] and nn[1] == nn[2] == nn[3] and st.eff[a + 1] > 0:
            # the note really is the one that follows tick 0.  Tick 0 is either register lag
            # (it still has exactly the previous tick's pitch: dropped) or an attack at its own
            # pitch, like a guitar pluck (kept, as a 1-tick arpeggio step)
            sg.base = int(nn[1])
            lag = a > 0 and period_of(st.note[a - 1], ch) - st.fine[a - 1] == period_of(nn[0], ch) - st.fine[a]
        eff = st.eff[a:b]
        aud_i = np.flatnonzero(eff > 0)
        last_aud = int(aud_i[-1]) if len(aud_i) else 0
        arp = (st.note[a:b] - sg.base).astype(int)
        if lag:
            arp[0] = 0
        fine = st.fine[a:b].astype(int).copy()
        duty = st.duty[a:b].astype(int).copy()
        if last_aud + 1 < sg.L:          # after the sound dies nothing matters: hold
            arp[last_aud + 1:] = arp[last_aud]
            fine[last_aud + 1:] = fine[last_aud]
            duty[last_aud + 1:] = duty[last_aud]
        sg.arp, sg.duty = arp, duty
        sg.fine, sg.detune = simplify_pitch(fine, eff > 0, cents_per_unit(sg.base, ch))
        if ch == NOI:
            sg.detune = 0
        sg.vol = None if ch == TRI else eff.astype(int).copy()
        sg.cut_at = None
        if ch == TRI:
            sil = np.flatnonzero(~st.snd[a:b])
            if len(sil):
                sg.cut_at = a + int(sil[0])
        segs.append(sg)
    return segs


PITCH_TOL = 9      # cents: pitch wobble smaller than this (peak to peak) is ignored


def cents_per_unit(note, ch):
    """how many cents one period-register unit is worth around this note (more at high notes)"""
    if ch == NOI:
        return W_PIT / W_CENT
    return 1731.2 / max(1, period_of(int(note), ch))


def simplify_pitch(fine, audible, cpu=4.0):
    """drop tiny wobble (peak to peak <= PITCH_TOL cents) and keep real bends exactly.
    Returns (curve, detune): a steady offset is returned as `detune`, which becomes a Pxx
    effect on the note instead of a pitch macro (so detuned notes can share instruments)."""
    vals = fine[audible] if audible.any() else fine
    if (vals.max() - vals.min()) * cpu <= PITCH_TOL:
        return np.zeros_like(fine), int(np.round(np.median(vals)))
    return fine, 0


# ----------------------------------------------------------------------------
# macros
# ----------------------------------------------------------------------------
def compress(vals, hold):
    """Shortest (macro values, loop index) that expands back to vals.
    hold='last': value stays at the last entry; 'zero': deltas, nothing more after."""
    vals = [int(x) for x in vals]
    L = len(vals)
    v = list(vals)
    if hold == 'last':
        while len(v) > 1 and v[-1] == v[-2]:
            v.pop()
    else:
        while v and v[-1] == 0:
            v.pop()
    best, bestlen = (v, -1), len(v)
    for T in range(1, min(L // 2, 64) + 1):
        k = L - T
        while k > 0 and vals[k - 1] == vals[k - 1 + T]:
            k -= 1
        if L - k >= 2 * T and k + T < bestlen:
            best, bestlen = (vals[:k + T], k), k + T
    vs, lp = best
    vs = vs[:250]
    if lp >= len(vs):
        lp = -1
    return tuple(vs), lp


def expand(vals, loop, L, hold):
    n = len(vals)
    if n == 0:
        return np.zeros(L, int)
    arr = np.asarray(vals)
    idx = np.arange(L)
    if loop >= 0 and n - loop > 0:
        return arr[np.where(idx < n, idx, loop + (idx - n) % (n - loop))]
    if hold == 'last':
        return arr[np.minimum(idx, n - 1)]
    out = np.zeros(L, int)
    out[:min(n, L)] = arr[:min(n, L)]
    return out


def scheme_info(arp):
    """If the arp curve is a 3-note chord pattern return (shift, x, y, symbols)."""
    arp = np.asarray(arp)
    if arp.min() == arp.max():
        return None
    shift = int(arp.min()) if arp.min() < 0 else 0
    a = arp - shift
    res = sorted(set(int(r) for r in (a % 12)) - {0})
    if len(res) != 2:
        return None
    x, y = res
    sym = []
    for v in a:
        o, r = divmod(int(v), 12)
        sym.append(12 * o if r == 0 else (0x40 + 12 * o if r == x else -128 + 12 * o))
    if max(abs(z) for z in sym) > 127 or any(0x40 + 12 * (int(v) // 12) > 127 for v in a):
        return None
    return shift, x, y, sym


def scheme_decode(sym, x, y):
    out = np.zeros(len(sym), int)
    for i, v in enumerate(sym):
        out[i] = x + (v - 64) if v >= 64 else (y + (v + 128) if v < 0 else v)
    return out


def cluster_curves(curves, weights, tol, kmax, hold, cumulative=False):
    """Replace many similar per-note curves with a small library.
    curves: list of 1-D int arrays (distinct), weights: occurrences.
    Returns list of canonical curves (same length as the input curves)."""
    n = len(curves)
    if n == 0:
        return []
    cands, cidx = [], {}
    for cv in curves:
        arr = np.diff(np.concatenate([[0], cv])) if cumulative else cv
        k = compress(arr, hold)
        if k not in cidx:
            cidx[k] = len(cands)
            cands.append(k)
    Lmax = max(len(c) for c in curves)
    E = np.zeros((len(cands), Lmax))
    for i, (vals, lp) in enumerate(cands):
        e = expand(vals, lp, Lmax, hold)
        E[i] = np.cumsum(e) if cumulative else e
    M = np.zeros((n, len(cands)))
    for u, cv in enumerate(curves):
        M[u] = np.abs(E[:, :len(cv)] - cv[None, :]).sum(1)
    W = np.asarray(weights, float)
    lim = np.array([tol * len(c) for c in curves])
    EX = np.maximum(0.0, M - lim[:, None])
    cur = np.full(n, 3 * BIG)
    chosen = []
    while len(chosen) < kmax:
        gain = (W[:, None] * np.maximum(0, cur[:, None] - EX)).sum(0)
        if chosen:
            gain[chosen] = -1
        c = int(np.argmax(gain))
        if gain[c] <= max(1e-9, FACILITY_COST * W.sum() / 3000.0):
            break
        chosen.append(c)
        cur = np.minimum(cur, EX[:, c])
        if cur.max() <= 0:
            break
    assign = np.argmin(M[:, chosen], axis=1)
    return [E[chosen[assign[u]]][:len(curves[u])].astype(int) for u in range(n)]


def quantize_segments(segs, tol, vol_max=24, pit_max=16):
    """snap volume envelopes and pitch curves to a small shared library"""
    for kind, hold, cum, kmax, tl in (('vol', 'last', False, vol_max, 2.0 * tol),
                                      ('fine', 'zero', True, pit_max, tol)):
        uniq, idx, cnt = [], {}, []
        for c in MUSIC:
            for sg in segs[c]:
                cv = getattr(sg, kind)
                if cv is None:
                    continue
                key = (c == NOI and kind == 'vol', tuple(cv))
                if key not in idx:
                    idx[key] = len(uniq)
                    uniq.append(cv)
                    cnt.append(0)
                cnt[idx[key]] += 1
        canon = cluster_curves(uniq, cnt, tl, kmax, hold, cum)
        for c in MUSIC:
            for sg in segs[c]:
                cv = getattr(sg, kind)
                if cv is not None:
                    setattr(sg, kind, canon[idx[(c == NOI and kind == 'vol', tuple(cv))]])


class Req:
    pass


def make_requirement(sg):
    """what a note needs from its instrument; returns (Req, base-note shift)"""
    r = Req()
    r.ch, r.L = sg.ch, sg.L
    r.cpu = cents_per_unit(sg.base, sg.ch)
    shift, r.scheme, r.arp = 0, None, sg.arp
    sch = scheme_info(sg.arp) if sg.ch != NOI else None
    if sch is not None:
        shift, x, y, sym = sch
        r.arp, r.scheme = sg.arp - shift, (x, y, sym)
    r.vol, r.fine, r.colvol = sg.vol, sg.fine, None
    # duty (noise: mode) per tick; only a duty that CHANGES inside the note needs a macro,
    # a steady one is set with Vxx on the note's row
    r.duty = None if sg.ch == TRI or getattr(sg, 'duty', None) is None else np.asarray(sg.duty, int)
    r.duty_var = r.duty is not None and len(set(r.duty.tolist())) > 1
    if sg.vol is not None and len(set(sg.vol.tolist())) == 1 and 1 <= int(sg.vol[0]) <= 15:
        r.colvol, r.vol = int(sg.vol[0]), None      # one steady level: the volume column does it
    r.base = int(sg.base + shift)
    r.head = None
    if r.scheme is None and sg.ch != NOI:
        nz = np.flatnonzero(r.arp != 0)
        if len(nz) and nz[0] == 0 and nz.max() < 2 and len(r.arp) > nz.max() + 1:
            # an attack of 1-2 ticks at its own pitch (a pluck, a click): FamiTracker plays this
            # with a FIXED arpeggio, which returns to the real note when it ends - one
            # instrument then fits every note, whatever its pitch
            r.head = tuple(int(r.base + a) for a in r.arp[:nz.max() + 1])
    r.cand = candidate_from(r)
    r.key = (sg.ch == TRI, r.cand, None if r.scheme is None else r.scheme[:2], r.colvol, r.head,
             int(round(2 * np.log2(r.cpu))),
             None if r.vol is None else r.vol.tobytes(),
             r.duty.tobytes() if r.duty_var else None)
    return r, shift


def reload_mask(arp, L):
    """ticks on which FamiTracker's driver reloads the period from the note table, throwing
    away what the pitch macro had added so far: every tick an arpeggio macro step runs, plus
    one more tick after a non-absolute arpeggio ends ('restore')."""
    R = np.zeros(L, bool)
    if arp is None:
        return R
    vals, loop = arp[1], arp[2]
    n = len(vals)
    if loop >= 0:
        R[:] = True
    else:
        R[:min(n, L)] = True
        if arp[0] != 'A' and n < L:
            R[n] = True
    return R


def pitch_effect(pit_steps, R):
    """fine pitch heard on each tick (positive = higher) for pitch-macro steps and reload ticks"""
    out = np.zeros(len(pit_steps), int)
    acc = 0
    for t, v in enumerate(pit_steps):
        if R[t]:
            acc = 0
        acc += int(v)
        out[t] = -acc
    return out


def candidate_from(r):
    """instrument spec (arp, vol, pit, duty) that reproduces r exactly"""
    arp = None
    if r.scheme is not None:
        arp = ('S',) + compress(r.scheme[2], 'last')
    elif np.any(r.arp != 0):
        arp = ('A',) + compress(r.arp, 'last')
    vol = None if r.vol is None else compress(r.vol, 'last')
    # FamiTracker pitch macros are period-based: a POSITIVE value LOWERS the pitch,
    # while Pxx above 80 (the sign of the raw "fine" values) raises it
    R = reload_mask(arp, r.L)
    f = np.asarray(r.fine, int)
    prev = np.concatenate([[0], f[:-1]])
    d = -(f - np.where(R, 0, prev))
    pit = compress(d, 'zero')
    if len(pit[0]) == 0:
        pit = None
    duty = compress(r.duty, 'last') if getattr(r, 'duty_var', False) else None
    return (arp, vol, pit, duty)


def cost_matrix(reqs, cands):
    U, C = len(reqs), len(cands)
    Lmax = max(r.L for r in reqs)
    A = np.zeros((C, Lmax)); Asym = [None] * C; isS = np.zeros(C, bool); isF = np.zeros(C, bool)
    hasA = np.array([c[0] is not None for c in cands])
    V = np.zeros((C, Lmax)); hasV = np.zeros(C, bool)
    P = np.zeros((C, Lmax)); D = np.zeros((C, Lmax)); hasD = np.zeros(C, bool)
    hasP = np.array([c[2] is not None for c in cands])
    for i, (arp, vol, pit, duty) in enumerate(cands):
        if arp is not None:
            if arp[0] == 'A':
                A[i] = expand(arp[1], arp[2], Lmax, 'last')
            elif arp[0] == 'F':
                isF[i] = True
            else:
                isS[i] = True
                Asym[i] = expand(arp[1], arp[2], Lmax, 'last')
        if vol is not None:
            hasV[i] = True
            V[i] = expand(vol[0], vol[1], Lmax, 'last')
        if pit is not None:
            P[i] = pitch_effect(expand(pit[0], pit[1], Lmax, 'zero'),
                                reload_mask(None if arp is None else (arp[0], arp[1], arp[2]), Lmax))
        if duty is not None:
            hasD[i] = True
            D[i] = expand(duty[0], duty[1], Lmax, 'last')
    sch_idx = np.flatnonzero(isS)
    fix_idx = np.flatnonzero(isF)
    M = np.zeros((U, C))
    vkeys = {}
    vid = np.full(C, -1)
    for i, c in enumerate(cands):
        if c[1] is not None:
            vid[i] = vkeys.setdefault(c[1], len(vkeys))
    nv = max(1, len(vkeys))
    VU = np.zeros((nv, Lmax))
    for key, j in vkeys.items():
        VU[j] = expand(key[0], key[1], Lmax, 'last')
    colbest = []
    for u, r in enumerate(reqs):
        L = r.L
        arpc = np.abs(A[:, :L] - r.arp[None, :]).sum(1)
        if len(sch_idx):
            if r.scheme is None:
                arpc[sch_idx] = BIG
            else:
                x, y, _ = r.scheme
                for i in sch_idx:
                    arpc[i] = np.abs(scheme_decode(Asym[i][:L], x, y) - r.arp).sum()
        for i in (fix_idx if r.ch != NOI else []):   # fixed notes, then back to the real note
            head = np.asarray(cands[i][0][1]) - r.base
            eff = np.zeros(L)
            eff[:min(L, len(head))] = head[:L]
            arpc[i] = np.abs(eff - r.arp).sum()
        if r.ch == NOI and len(fix_idx):
            arpc[fix_idx] = BIG                # noise pitches are not notes: no fixed arpeggios
        none_arp = np.abs(r.arp).sum()
        wrong = (arpc > 0) & (np.arange(C) != -1) & hasA
        arpc = np.where(wrong, np.maximum(arpc, none_arp) * 1.5 + 1, arpc)
        tot = W_ARP * arpc
        if r.vol is not None:
            # the volume column scales the envelope: out = macro * column / 15 (never 0 unless
            # the macro is 0); pick the best column level for every envelope
            errv = np.full(nv, 1e18); cbest = np.full(nv, 15)
            for cl in range(15, 0, -1):
                eff = np.floor(VU[:, :L] * cl / 15.0)
                eff = np.where((VU[:, :L] > 0) & (eff == 0), 1, eff)
                e = (np.abs(eff - r.vol[None, :]) + 3.0 * ((eff > 0) != (r.vol[None, :] > 0))).sum(1)
                better = e < errv
                errv = np.where(better, e, errv); cbest = np.where(better, cl, cbest)
            colbest.append((u, cbest))
            vc = np.where(vid >= 0, errv[np.maximum(vid, 0)], BIG)
            tot = tot + W_VOL * vc
        else:
            tot = tot + np.where(hasV, BIG, 0)
        pc = np.abs(P[:, :L] - r.fine[None, :]).sum(1)
        none_pit = np.abs(r.fine).sum()
        pc = np.where(hasP & (pc > 0), np.maximum(pc, none_pit) * 1.5 + 1, pc)
        tot = tot + W_CENT * r.cpu * pc
        if getattr(r, 'duty', None) is not None:
            # with a duty macro the note follows it; without one it keeps the duty it starts
            # with (set by Vxx), so every later change is a miss
            dmac = (D[:, :L] != r.duty[None, :]).sum(1)
            dnone = int((r.duty != r.duty[0]).sum())
            tot = tot + W_DUTY * np.where(hasD, dmac, dnone)
        M[u] = tot
    return M, (vid, dict(colbest))


ERR_CAP = 1e9          # one note can't be worth more than this much error (one-offs get no instrument)
FACILITY_COST = 0.0     # per 3000 notes: an instrument must remove at least this much (weighted) error to be worth adding


SEL_CONVEX = 6.0       # error per tick at which a note counts double (convex: big errors dominate)
SEL_REPEAT = 1.0       # how much repeats count: 1 = linearly, 0.5 = square root
WRONG_PER_TICK = 12.0  # reported as "sounds plainly wrong" above this (~2 semitones off every tick)
LAST_SELECTION = {}
ONE_OFF_COUNT = 2      # a note shape heard this few times counts as a one-off event
ONE_OFF_GATE = 2.0     # one-offs try a pattern effect once they are this far off per tick
SLIDES = True          # play badly-fitting one-off sweeps with Qxy/Rxy note slides
BAD_PER_TICK = float('inf')    # off: extra instruments for plainly-wrong one-offs (see choose_instruments)


SLIDE_FIRST_TICK = True   # the driver applies a note slide on the note's first tick already
VOLUME_FOLLOW = True      # badly-fitting volumes follow the game with volume-column changes per row


def slide_speed(x, y):
    """period units per tick of Qxy / Rxy in FamiTracker's driver: ((x << 4 | y) >> 3) | 1"""
    return (((x << 4) | y) >> 3) | 1


def slide_periods(p0, pt, speed, L):
    """period register on each tick of a note slide from p0 toward pt (stops there)"""
    out = np.empty(L)
    p = p0
    for t in range(L):
        if t > 0 or SLIDE_FIRST_TICK:
            if pt > p:
                p = min(pt, p + speed)
            elif pt < p:
                p = max(pt, p - speed)
        out[t] = p
    return out


def _vol_fit(r, vspec):
    """(weighted volume error, volume column) for requirement r played with envelope vspec"""
    if r.vol is None:
        return (0.0, r.colvol if r.colvol is not None else 15) if vspec is None else (BIG, 15)
    if vspec is None:
        return BIG, 15
    V = expand(vspec[0], vspec[1], r.L, 'last').astype(float)
    best = (BIG, 15)
    for cl in range(15, 0, -1):
        eff = np.floor(V * cl / 15.0)
        eff = np.where((V > 0) & (eff == 0), 1, eff)
        e = float((np.abs(eff - r.vol) + 3.0 * ((eff > 0) != (r.vol > 0))).sum())
        if e < best[0]:
            best = (e, cl)
    return W_VOL * best[0], best[1]


def fit_slide(sg, r, plain):
    """try to play a note the instruments fit badly as instrument + Qxy/Rxy note slide.
    plain: [(instrument number, candidate)] without arpeggio or pitch macros (an arpeggio
    macro reloads the period every tick and would cancel the slide).
    Returns (cost, effect, instrument, volume column) or None."""
    if r.scheme is not None or sg.ch == NOI or not plain:
        return None
    base = int(sg.base + sg.shift)
    p0 = period_of(base, sg.ch)
    tgt = np.array([period_of(base + int(a), sg.ch) for a in r.arp], float) - np.asarray(r.fine, float)
    if np.all(tgt == p0):
        return None
    vol = min((_vol_fit(r, cand[1]) + (inst,) for inst, cand in plain), key=lambda v: v[0])
    if vol[0] >= BIG:
        return None
    best = None
    for name, sign in (('Q', 1), ('R', -1)):
        for y in range(1, 16):
            n2 = base + sign * y
            if n2 < 0 or n2 > 95:
                continue
            pt = period_of(n2, sg.ch)
            for x in range(16):
                ps = slide_periods(p0, pt, slide_speed(x, y), r.L)
                cost = W_ARP * float(np.abs(12 * np.log2(np.maximum(ps, 1) / np.maximum(tgt, 1))).sum())
                if best is None or cost < best[0]:
                    best = (cost, '%s%X%X' % (name, x, y))
    if best is None:
        return None
    return best[0] + vol[0], best[1], vol[2], vol[1]


def choose_instruments(reqs, weights, cands, max_inst, tolerance, hard_max=None):
    """greedy facility location: add instruments until every note is within
    `tolerance` (average weighted error per frame) or max_inst is reached"""
    M, (vid, colbest) = cost_matrix(reqs, cands)
    M = np.where(M >= BIG, M, np.minimum(M, ERR_CAP))
    L = np.array([r.L for r in reqs], float)
    # what counts is how wrong a note SOUNDS: a note that is badly off (a flattened riser, a
    # wrong drum) is far worse than one that is slightly off, so the error per tick is
    # weighted convexly; and repeats count less than linearly, so a one-off that sounds
    # wrong still gets attention next to a slightly-off note that repeats 200 times
    e = M / L[:, None]
    P = np.where(M >= BIG, M, L[:, None] * (e + e * e / SEL_CONVEX))
    W = np.asarray(weights, float) ** SEL_REPEAT
    tol = np.array([tolerance * r.L for r in reqs])
    EX = np.maximum(0.0, P - tol[:, None])
    cur = np.full(len(reqs), 3 * BIG)
    chosen = []
    while len(chosen) < max_inst:
        gain = (W[:, None] * np.maximum(0, cur[:, None] - EX)).sum(0)
        if chosen:
            gain[chosen] = -1
        c = int(np.argmax(gain))
        if gain[c] <= max(1e-9, FACILITY_COST):
            break
        chosen.append(c)
        cur = np.minimum(cur, EX[:, c])
        if cur.max() <= 0:
            break
    # notes that would still sound plainly wrong (a one-off drum fill, a riser) get an
    # instrument of their own while FamiTracker's hard limit leaves room, worst first
    hard_max = max_inst if hard_max is None else hard_max
    while len(chosen) < hard_max:
        best = M[:, chosen].min(1) / L
        worst = int(np.argmax(np.where(best > BAD_PER_TICK, best * W, -1)))
        if best[worst] <= BAD_PER_TICK:
            break
        c = int(np.argmin(M[worst]))
        if c in chosen:
            break
        chosen.append(c)
    assign = np.argmin(M[:, chosen], axis=1)
    errs = M[np.arange(len(reqs)), np.array(chosen)[assign]]
    wrong = errs / L > WRONG_PER_TICK
    LAST_SELECTION.update(wrong_notes=int(np.asarray(weights)[wrong].sum()), wrong_shapes=int(wrong.sum()))
    def column_for(u, ci):
        """volume-column level that scales candidate ci's envelope onto requirement u"""
        return int(colbest[u][vid[ci]]) if u in colbest and vid[ci] >= 0 else 15
    return chosen, assign, errs, column_for


# ----------------------------------------------------------------------------
# rows / patterns
# ----------------------------------------------------------------------------
class Cell:
    def __init__(self):
        self.note, self.inst, self.eff, self.vol, self.seg = None, None, [], None, None

    def text(self, ncols):
        e = list(self.eff[:ncols]) + ['...'] * (ncols - len(self.eff[:ncols]))
        return ' '.join([self.note or '...',
                         '%02X' % self.inst if self.inst is not None else '..',
                         '%X' % self.vol if self.vol is not None else '.'] + e)


STATS = collections.Counter()
MAX_DELAY = 9


def dpcm_events(frames, T):
    """what the game did on the DPCM channel, tick by tick: sample hits (instrument, key,
    start offset), stops (=== / ---) and writes to the DPCM output level (Zxx, repeats of the
    same level dropped unless a sample played in between)"""
    ev, inst, last_z, played, sounding = [], None, None, False, False
    for t in range(min(T, len(frames))):
        row = frames[t]
        if len(row) <= DPCM:
            continue
        cell = row[DPCM]
        note = cell[0]
        if cell[1] not in ('..', '.'):
            inst = int(cell[1], 16)
        effs = [e for e in cell[3:] if e not in ('...', '.')]
        y = next((int(e[1:], 16) for e in effs if e[0] == 'Y'), None)
        z = next((int(e[1:], 16) for e in effs if e[0] == 'Z'), None)
        if z is not None and (z != last_z or played):
            ev.append((t, 'Z', z))
            last_z, played = z, False
        if note in ('===', '---'):
            if sounding:                              # a stop with nothing playing does nothing
                ev.append((t, 'stop', note))
            sounding = False
        elif note not in ('...', '', None):
            ev.append((t, 'hit', (inst if inst is not None else 0, note, y)))
            played = sounding = True
    return ev


def raw_dpcm_keys(raw_inst):
    """KEYDPCM lines -> {(instrument, octave, note): (sample, pitch, loop, loop offset, delta)}"""
    keys = {}
    for l in raw_inst:
        if l.startswith('KEYDPCM'):
            p = list(map(int, l.split()[1:9]))
            keys[(p[0], p[1], p[2])] = tuple(p[3:8])
    return keys


def dpcm_key_of(note):
    """'C#0' -> (octave, note index)"""
    return int(note[2]), NOTE_INDEX[note[:2]]


def place_dpcm(rows, ctx, grid):
    """put the game's DPCM hits, stops and level writes on the row grid (Gxx for hits between
    rows); returns how many hits landed on their exact tick, late, or were dropped"""
    res = collections.Counter()
    keymap, inst = getattr(ctx, 'dpcm_map', {}), getattr(ctx, 'dpcm_inst', None)
    for t, kind, val in getattr(ctx, 'dpcm', []):
        r, d = grid.locate(t)
        if r >= len(rows):
            continue
        if kind == 'Z':
            cell = rows[r][DPCM]
            cell.eff = [e for e in cell.eff if e[0] != 'Z'] + ['Z%02X' % val]
            continue
        if kind == 'hit':
            out = keymap.get((val[0], val[1]))
            if out is None:
                res['unmapped'] += 1
                continue
        cell = rows[r][DPCM]
        if cell.note is not None:                    # two events in one row: the later one waits
            if r + 1 < len(rows) and rows[r + 1][DPCM].note is None:
                r, d = r + 1, 0
                cell = rows[r][DPCM]
                res['late'] += kind == 'hit'
            else:
                res['dropped'] += kind == 'hit'
                continue
        else:
            res['exact'] += kind == 'hit'
        if kind == 'hit':
            cell.note, cell.inst = out, inst
            if val[2]:
                cell.eff.append('Y%02X' % val[2])
        else:
            cell.note = val
        if d:
            cell.eff = ['G%02X' % d] + [e for e in cell.eff if e[0] != 'G']
    return res


def build_rows(segs, grid, nframes, states, duty_insts=frozenset(), env_of=None):
    maxrow = grid.locate(nframes - 1)[0] + 1
    rows = [[Cell() for _ in range(NCH)] for _ in range(maxrow + 1)]
    placed = {}                      # (channel, row) -> (delay,) of the note placed there
    for ch in MUSIC:
        arp_xy = (0, 0)        # a 0xy effect stays active on the channel until replaced
        col_vol = 15           # channel volume (volume column) stays until replaced
        fine_p = 0             # a Pxx fine pitch effect stays until replaced
        for sg in segs[ch]:
            r, d = grid.locate(sg.start)
            if rows[r][ch].note is not None:          # two notes landed in one row
                occ = rows[r][ch]
                dA = placed.get((ch, r), (0,))[0]
                # option 1: play the earlier note at the end of the previous row (early by dA+1)
                # option 2: push this note to the next row (late by what is left of this row)
                prev_len = grid.row_start(r) - grid.row_start(r - 1) if r > 0 else 0
                this_len = grid.row_start(r + 1) - grid.row_start(r)
                early, late = dA + 1, this_len - d
                if r > 0 and rows[r - 1][ch].note is None and prev_len >= 2 and early < late \
                        and not any(e[0] == 'G' for e in occ.eff if int(e[1:], 16) > 0):
                    occ.eff = [e for e in occ.eff if e[0] != 'G'] + ['G%02X' % (prev_len - 1)]
                    rows[r - 1][ch] = occ
                    rows[r][ch] = Cell()
                    placed[(ch, r - 1)] = (prev_len - 1,)
                    placed.pop((ch, r), None)
                    if occ.seg is not None:
                        occ.seg.row, occ.seg.delay = r - 1, prev_len - 1
                    STATS['moved earlier'] += 1
                elif r + 1 < len(rows) and rows[r + 1][ch].note is None:
                    r, d = r + 1, 0
                    STATS['pushed'] += 1
                else:
                    STATS['dropped'] += 1
                    continue
            cell = rows[r][ch]
            placed[(ch, r)] = (d,)
            cell.seg, sg.row, sg.delay = sg, r, d
            cell.note = note_text(sg.base + sg.shift, ch)
            cell.inst = sg.inst
            if ch != TRI:
                want_v = sg.colvol
                if want_v != col_vol:
                    cell.vol = want_v
                    col_vol = want_v
            if d:
                cell.eff.append('G%02X' % d)
            if getattr(sg, 'slide', None):
                cell.eff.append(sg.slide)
            want_p = max(-0x7F, min(0x7F, getattr(sg, 'detune', 0)))
            if want_p != fine_p:
                cell.eff.append('P%02X' % (0x80 + want_p))   # P80 = no detune
                fine_p = want_p
            want = tuple(sg.req.scheme[:2]) if sg.uses_scheme else (0, 0)
            if want != arp_xy:
                cell.eff.append('0%X%X' % want)    # '000' switches the old chord off
                arp_xy = want
        if ch != TRI:
            # FamiTracker: Vxx sets the channel's default duty, every new note starts at it, and
            # a duty macro (when the instrument has one) takes over while it runs.  So: notes
            # with a duty macro need nothing; other notes get Vxx where their duty differs
            # from the default, at the note's own row and wherever it changes inside the note
            du = states[ch].duty
            default = 0
            for sg in segs[ch]:
                if not hasattr(sg, 'row') or rows[sg.row][ch].seg is not sg:
                    continue
                if getattr(sg, 'inst', None) in duty_insts:
                    continue
                end = min(nframes, sg.start + sg.L)
                last_aud = sg.start + (int(np.flatnonzero(sg.vol > 0)[-1]) if sg.vol is not None
                                       and np.any(sg.vol > 0) else sg.L - 1)
                for f in range(sg.start, min(end, last_aud + 1)):
                    if du[f] == default:
                        continue
                    r = sg.row if f == sg.start else grid.locate(f)[0]
                    if r >= len(rows) or (r != sg.row and rows[r][ch].note is not None):
                        continue
                    rows[r][ch].eff = [e for e in rows[r][ch].eff if e[0] != 'V'] + ['V%02X' % du[f]]
                    default = int(du[f])
        if ch == TRI:
            for sg in segs[ch]:
                if sg.cut_at is None:
                    continue
                rc, dc = grid.locate(sg.cut_at)
                if rc >= len(rows) or not hasattr(sg, 'row'):
                    continue
                here = rows[rc][ch]
                if here.note is None and not any(e[0] == 'S' for e in here.eff):
                    if dc == 0:
                        here.note = '---'
                    else:
                        here.eff.append('S%02X' % dc)
                    continue
                if here.note is not None and here.seg is not None and here.seg is not sg:
                    if here.seg.delay == 0 or dc >= here.seg.delay:
                        continue                   # the next note starts first: no cut needed
                # the cut row is busy (FamiTracker reads a delayed row, its Sxx included, at the
                # delay tick), so the note cuts itself: Sxx on its own cell, counted from its start
                own = rows[sg.row][ch]
                length = sg.cut_at - (grid.row_start(sg.row) + sg.delay)
                if 1 <= length <= 0x7F and not any(e[0] == 'S' for e in own.eff):
                    own.eff.append('S%02X' % length)
        if inst_group(ch) is None:
            passthrough_envelopes(rows, segs[ch], grid, ch)
        elif env_of is not None and KIND.get(ch) in ('pulse', 'noise'):
            STATS['volume column follows'] += column_follow(rows, segs[ch], grid, ch, states[ch], env_of)
    return rows


def column_follow(rows, segs, grid, ch, st, env_of, min_gain=3):
    """a note whose volume no instrument fits well (a slow swell or fade, a one-off level):
    pick the best volume-column level for each of its rows, the way people write these by
    hand, and keep it only when it brings the volume clearly closer to the game.
    FamiTracker: heard volume = floor(macro * column / 15), at least 1 when both are nonzero."""
    def heard(env, c):
        v = env * c // 15
        return np.where((env > 0) & (c > 0) & (v == 0), 1, v)

    def err(out, want):
        d = np.abs(out - want)
        # a level within 1 sounds right; beyond that every tick counts, and more so the further
        return int(4 * (d > 1).sum() + np.maximum(d - 1, 0).sum() + 3 * ((out > 0) != (want > 0)).sum())

    used = 0
    order = [sg for sg in segs if hasattr(sg, 'row') and sg.row < len(rows) and rows[sg.row][ch].seg is sg]
    for i, sg in enumerate(order):
        if sg.colvol is None or sg.vol is None:
            continue
        p0 = grid.row_start(sg.row) + sg.delay                  # tick the note really starts
        end = sg.start + sg.L
        if i + 1 < len(order):
            nx = order[i + 1]
            end = min(end, grid.row_start(nx.row) + nx.delay)
        if end - p0 < 4:
            continue
        env_spec = env_of(getattr(sg, 'inst', None))
        ticks = np.arange(p0, end)
        if env_spec is None:
            env = np.full(len(ticks), 15)
        else:
            env = np.asarray(expand(env_spec[0], env_spec[1], len(ticks), 'last'), int)
        want = np.asarray(st.eff[p0:end], int) if end <= len(st.eff) else None
        if want is None or len(want) != len(ticks):
            continue
        base = err(heard(env, sg.colvol), want)
        if base < min_gain:
            continue
        # rows the note covers, and the best level for each
        k, plan, total, prev = sg.row, [], 0, None
        while k < len(rows):
            a = max(p0, grid.row_start(k))
            b = min(end, grid.row_start(k + 1))
            if a >= end:
                break
            if k != sg.row and rows[k][ch].note is not None:
                break
            seg_t = slice(a - p0, b - p0)
            scores = [(err(heard(env[seg_t], c), want[seg_t]), c != prev, c) for c in range(16)]
            e, _, c = min(scores)
            plan.append((k, c))
            total += e
            prev = c
            k += 1
        if not plan or base - total < min_gain:
            continue
        cur = None
        for k, c in plan:
            if c != cur:
                rows[k][ch].vol = c
                cur = c
        used += 1
        # the next note must set its own level: the column now holds this note's last one
        if i + 1 < len(order) and order[i + 1].colvol is not None:
            rows[order[i + 1].row][ch].vol = order[i + 1].colvol
    return used


def passthrough_envelopes(rows, segs, grid, ch):
    """wavetable / FM channels keep their original instruments, which know nothing of the
    game's envelopes: follow the volume in the volume column, row by row, and stop the
    note (=== cut / Sxx) where the game silences it"""
    for sg in segs:
        if not hasattr(sg, 'row') or rows[sg.row][ch].seg is not sg or sg.vol is None:
            continue
        rows[sg.row][ch].vol = int(sg.colvol)       # always explicit: later rows change it
        cur = int(sg.colvol)
        end = sg.start + sg.L
        zeros = np.flatnonzero(sg.vol <= 0)
        cut = sg.start + int(zeros[0]) if len(zeros) else None
        k = sg.row + 1
        while k < len(rows):
            f = grid.row_start(k)
            if f >= end or (cut is not None and f >= cut) or rows[k][ch].note is not None:
                break
            v = int(sg.vol[f - sg.start])
            if v != cur and v > 0:
                rows[k][ch].vol = v
                cur = v
            k += 1
        if cut is None or cut >= end:
            continue
        rc, dc = grid.locate(cut)
        if rc >= len(rows):
            continue
        here = rows[rc][ch]
        if rc == sg.row:                               # silenced inside its own first row
            length = cut - (grid.row_start(sg.row) + sg.delay)
            if 1 <= length <= 0x7F and not any(e[0] == 'S' for e in here.eff):
                here.eff.append('S%02X' % length)
        elif here.note is None and not any(e[0] == 'S' for e in here.eff):
            if dc == 0:
                here.note = '---'
            else:
                here.eff.append('S%02X' % dc)


def add_global_effect(rows, r, text):
    """F / D / B effects live in the unused DPCM channel so the music channels stay clean"""
    for ch in (DPCM, NOI, SQ1, SQ2, TRI):
        if len(rows[r][ch].eff) < 4:
            rows[r][ch].eff.append(text)
            return


def pattern_chunks(rows, pr):
    out = []
    for i in range(0, len(rows), pr):
        ch = rows[i:i + pr]
        while len(ch) < pr:
            ch.append([Cell() for _ in range(NCH)])
        out.append(ch)
    return out


def layout_chunks(rows, pr, preroll, loop_row):
    """Cut rows into patterns.  The intro (rows before loop_row) and the loop each start on
    a fresh pattern, so the song can jump back to the first loop pattern.
    Returns (chunks, order index where the loop starts)."""
    blank = lambda: [Cell() for _ in range(NCH)]
    chunks, pos = [], 0
    if preroll:
        chunks.append([rows[0]] + [blank() for _ in range(pr - 1)])
        pos = 1
    sections = [(pos, len(rows))]
    if loop_row > pos:
        sections = [(pos, loop_row), (loop_row, len(rows))]
    loop_order = 0
    for a, b in sections:
        if a == loop_row and loop_row > pos:
            loop_order = len(chunks)
        for i in range(a, b, pr):
            ch = rows[i:min(i + pr, b)]
            while len(ch) < pr:
                ch.append(blank())
            chunks.append(ch)
    return chunks, loop_order


def pick_pattern_rows(rows, columns, preroll, loop_row, candidates=(16, 24, 32, 48, 64)):
    """fewest stored rows wins, but 32 (the usual length) is kept unless another
    size is at least 25% more compact"""
    scores = {}
    for pr in candidates:
        chunks, _ = layout_chunks(rows, pr, preroll, loop_row)
        total = sum(len({tuple(r[c].text(columns[c]) for r in ch) for ch in chunks}) for c in range(NCH))
        scores[pr] = total * pr
    best = min(scores, key=scores.get)
    return 32 if 32 in scores and scores[32] <= 1.33 * scores[best] else best


# ----------------------------------------------------------------------------
# writer
# ----------------------------------------------------------------------------
def sanitize_params(params):
    """FamiTracker's importer range-checks every field; NSFImport can write values outside it
    (e.g. 'PLAYBACKRATE 16639 16639' - the first number is a mode 0..2)"""
    out = []
    for l in params:
        p = l.split()
        if p[0] == 'PLAYBACKRATE' and len(p) >= 3 and not (p[1].isdigit() and int(p[1]) <= 2):
            l = 'PLAYBACKRATE    0 %s' % p[2]
        out.append(l)
    return out


def write_module(m, macros, instruments, chunks, columns, speed, tempo, pr, comment):
    out = []
    A = out.append
    A('# Dn-FamiTracker text export 0.5.3.64')
    A('# Module version 0450')
    A('')
    A('# INFO block')
    out.extend(m.info)
    A('')
    A('# COMMENTS block')
    A('COMMENT "%s"' % comment)
    A('')
    A('# PARAMS block')
    out.extend(sanitize_params(m.params))
    A('')
    A('# SEQUENCES block')
    raw_entries = [e for e in instruments if e[0] == 'RAW']
    if raw_entries:
        for l in raw_entries[0][2]:
            if l.startswith('MACRON163'):
                A(l)
    for (chip, t, idx), (vals, loop, setting) in sorted(macros.items()):
        A('%s %7d %3d %3d %3d %3d : %s' % ('MACRO' if chip == '2A03' else 'MACRO' + chip,
                                          t, idx, loop, -1, setting, ' '.join(map(str, vals))))
    A('')
    A('# DPCM SAMPLES block')
    for entry in instruments:
        if entry[0] == 'DPCM':
            out.extend(dpcm_text_lines(entry[2]))
    A('')
    A('# DETUNETABLES block')
    A('')
    A('# GROOVES block')
    A('')
    A('# Tracks using default groove:')
    A('')
    A('# INSTRUMENTS block')
    for idx, entry in enumerate(instruments):
        if entry[0] == 'DPCM':                     # the drum/sample instrument
            A('INST2A03 %3d %5d %3d %3d %3d %3d "DPCM samples"' % (idx, -1, -1, -1, -1, -1))
            for o, n, smp, pitch, loop, loff, delta in entry[1]:
                A('KEYDPCM %3d %3d %3d   %3d %3d %3d %5d %3d' % (idx, o, n, smp, pitch, loop, loff, delta))
            continue
        if entry[0] == 'RAW':                      # an original instrument, renumbered
            _, raw, raw_lines = entry
            for l in raw_lines:
                p = l.split()
                if len(p) > 1 and p[0].startswith(('INST', 'N163WAVE', 'FDSWAVE', 'FDSMOD', 'FDSMACRO')) \
                        and p[1].isdigit() and int(p[1]) == raw:
                    A(re.sub(r'^(\S+\s+)\d+', lambda mm: mm.group(1) + str(idx), l, count=1))
            continue
        chip, vol, arp, pit, hip, duty, name = entry
        A('%s %3d %5d %3d %3d %3d %3d "%s"' % ('INST' + chip, idx, vol, arp, pit, hip, duty, name))
    A('')
    A('# track HEADER block')
    A('TRACK %3d %3d %3d "New song"' % (pr, speed, tempo))
    A('COLUMNS : ' + ' '.join(map(str, columns)))
    A('')
    A('# track FRAMES block')
    pmaps, plists, orders = [{} for _ in range(NCH)], [[] for _ in range(NCH)], []
    for chunk in chunks:
        row = []
        for c in range(NCH):
            key = tuple(r[c].text(columns[c]) for r in chunk)
            if key not in pmaps[c]:
                pmaps[c][key] = len(plists[c])
                plists[c].append(key)
            row.append(pmaps[c][key])
        orders.append(row)
    npat = max(len(p) for p in plists)
    if npat > 128:
        print('WARNING: %d patterns on one channel (FamiTracker limit 128)' % npat, file=sys.stderr)
    for i, row in enumerate(orders):
        A('ORDER %02X : %s' % (i, ' '.join('%02X' % x for x in row)))
    A('')
    A('# track PATTERNS block')
    for p in range(npat):
        A('PATTERN %02X' % p)
        for r in range(pr):
            cells = [plists[c][p][r] if p < len(plists[c])
                     else ' '.join(['...', '..', '.'] + ['...'] * columns[c]) for c in range(NCH)]
            A('ROW %02X : %s' % (r, ' : '.join(cells)))
    A('')
    return '\r\n'.join(out), orders, npat


# ----------------------------------------------------------------------------
# playback model (to check the result against the raw data)
# ----------------------------------------------------------------------------
def parse_macros_and_insts(text):
    """macros[(chip, type, index)] and insts[index] = (chip, [vol, arp, pit, hipit, duty refs])"""
    macros, insts = {}, {}
    for l in text.replace('\r\n', '\n').split('\n'):
        if l.startswith('MACRO'):
            head, vals = l.split(':')
            h = head.split()
            chip = h[0][5:] or '2A03'
            t, idx, loop, rel, setting = (int(x) for x in h[1:6])
            macros[(chip, t, idx)] = ([int(v) for v in vals.split()], loop, setting)
        elif l.startswith('INST') and l[4:5] != 'R':
            p = l.split('"')[0].split()
            try:
                refs = [int(x) for x in p[2:7]]
            except ValueError:
                refs = []
            if p[0] not in ('INST2A03', 'INSTVRC6', 'INSTS5B', 'INSTN163') or len(refs) < 5:
                refs = [-1] * 5               # FDS / VRC7 instruments: no sequence macros to model
            insts[int(p[1])] = (p[0][4:], refs)
    return macros, insts


class _Ch:
    pass


def _scaled_vol(macro_v, col):
    v = macro_v * col // 15
    return 1 if (macro_v > 0 and col > 0 and v == 0) else v


def _apply_cell(o, ev, f, c):
    """read one pattern cell at tick f the way FamiTracker's driver does: volume column and
    effects first, then the note (which clears a note cut set by an earlier row)"""
    cell = ev['cell']
    ev['done'] = True
    if cell[2] != '.':
        o.vol = int(cell[2], 16)
    cut_now = False
    pending_slide = None
    for e in cell[3:]:
        if e == '...':
            continue
        if e[0] == 'V':
            o.duty = o.cur_duty = int(e[1:], 16)     # Vxx sets the default and the current duty
        elif e[0] == '0':
            o.xy = (int(e[1], 16), int(e[2], 16))
        elif e[0] == 'S':
            o.cut_at = f + int(e[1:], 16)
            cut_now = True
        elif e[0] == 'P':
            o.finep = int(e[1:], 16) - 0x80
        elif e[0] in 'QR':
            pending_slide = (1 if e[0] == 'Q' else -1, int(e[1], 16), int(e[2], 16))
    note = None if cell[0] == '...' else cell[0]
    if note is None:
        return
    if note in ('---', '==='):
        o.playing = False
        return
    o.playing, o.base = True, decode_note(note, c)
    if cell[1] != '..':
        o.inst = int(cell[1], 16)
    o.pos, o.cum, o.t0 = [0] * 5, 0, f
    o.cur_duty = o.duty                # a new note starts at the channel's default duty (Vxx)
    if not cut_now:
        o.cut_at = None
    o.slide, o.soff = None, 0          # a new note ends a note slide
    if pending_slide is not None:
        sign, x, y = pending_slide
        p0 = period_of(o.base, c)
        o.slide = (period_of(o.base + sign * y, c) - p0, slide_speed(x, y))


def simulate(text, nframes):
    """Play the module back tick by tick: FamiTracker's tempo accumulator, F/D/B effects,
    note delays/cuts, macros, and a 0xy effect that stays on until replaced."""
    m = parse_module(text)
    macros, insts = parse_macros_and_insts(text)
    speed, tempo = m.track['speed'], m.track['tempo']
    D = (tempo * 24) // speed
    out = {c: dict(snd=np.zeros(nframes, bool), note=np.zeros(nframes, int),
                   vol=np.zeros(nframes, int), fine=np.zeros(nframes, int),
                   duty=np.zeros(nframes, int)) for c in MUSIC}
    chs = {}
    for c in MUSIC:
        o = _Ch()
        o.playing, o.inst, o.base, o.pos, o.cum, o.t0 = False, None, 0, [0] * 5, 0, 0
        o.duty, o.xy, o.cut_at, o.vol, o.finep = 0, (0, 0), None, 15, 0
        o.cur_duty = 0
        o.slide, o.soff = None, 0
        chs[c] = o

    def macro_of(inst, t):
        if inst is None:
            return None
        chip, refs = insts.get(inst, ('2A03', [-1] * 5))
        return macros.get((chip, t, refs[t])) if refs[t] >= 0 else None

    f, oi, r, accum, row_f, events = 0, 0, 0, 0, 0, {}
    while f < nframes:
        if accum <= 0:
            if oi >= len(m.orders):
                break
            order = m.orders[oi]
            skip, jump = False, None
            for c in range(NCH):
                for e in m.patterns[order[c]][r][c][3:]:
                    if e[0] == 'F':
                        v = int(e[1:], 16)
                        if v < 32:
                            speed = max(1, v)
                        else:
                            tempo = v
                        D = (tempo * 24) // speed
                    elif e[0] == 'D':
                        skip = True
                    elif e[0] == 'B':
                        jump = int(e[1:], 16)
            row_f = f
            # a delayed note from the previous row that never got its turn plays now
            # (FamiTracker's driver reads it at the start of the next row)
            for c in MUSIC:
                ev = events.get(c)
                if ev is not None and not ev['done']:
                    _apply_cell(chs[c], ev, f, c)
            events = {}
            for c in MUSIC:
                cell = m.patterns[order[c]][r][c]
                ev = dict(cell=cell, delay=0, done=False)
                for e in cell[3:]:
                    if e[0] == 'G':
                        ev['delay'] = int(e[1:], 16)
                # the driver reads a delayed row - note, volume column and effects - at the
                # delay tick, not at the start of the row
                if ev['delay'] == 0:
                    _apply_cell(chs[c], ev, f, c)
                events[c] = ev
            if jump is not None:
                oi, r = jump, 0
            elif skip or r + 1 >= m.track['rows']:
                oi, r = oi + 1, 0
            else:
                r += 1
            accum += 3600
        k = f - row_f
        for c in MUSIC:
            o, ev, d = chs[c], events.get(c), out[c]
            if ev is not None and not ev['done'] and k == ev['delay']:
                _apply_cell(o, ev, f, c)
            if o.cut_at is not None and f >= o.cut_at:
                o.playing, o.cut_at = False, None
            if o.playing:
                d['snd'][f] = True
                mv = macro_of(o.inst, M_VOL)
                d['vol'][f] = _scaled_vol(mv[0][min(o.pos[M_VOL], len(mv[0]) - 1)], o.vol) if mv else o.vol
                note, ma = o.base, macro_of(o.inst, M_ARP)
                if o.xy != (0, 0) and not (ma and ma[2] == ARP_SCHEME):
                    note += (0, o.xy[0], o.xy[1])[(f - o.t0) % 3]   # a 0xy effect left on
                arp_reload = bool(ma) and (o.pos[M_ARP] < len(ma[0]) or
                                           (ma[2] != ARP_ABS and o.pos[M_ARP] == len(ma[0])))
                if ma and ma[2] != ARP_ABS and ma[1] < 0 and o.pos[M_ARP] >= len(ma[0]):
                    ma = None                    # the driver restores the real note
                if ma:
                    v = ma[0][min(o.pos[M_ARP], len(ma[0]) - 1)]
                    if ma[2] == ARP_SCHEME:
                        note += int(scheme_decode([v], *o.xy)[0])
                    elif ma[2] == ARP_FIXED:
                        note = v
                    else:
                        note += v
                d['note'][f] = note
                mp = macro_of(o.inst, M_PIT)
                if arp_reload:
                    o.cum = 0            # the arpeggio reloaded the period: the bend starts over
                if mp and o.pos[M_PIT] < len(mp[0]):
                    o.cum += mp[0][o.pos[M_PIT]]
                if o.slide is not None and (f > o.t0 or SLIDE_FIRST_TICK):
                    goal, sp = o.slide
                    o.soff = min(goal, o.soff + sp) if goal > o.soff else max(goal, o.soff - sp)
                d['fine'][f] = -o.cum + o.finep - o.soff  # positive macro value = lower pitch
                md = macro_of(o.inst, M_DUTY)
                if md and o.pos[M_DUTY] < len(md[0]):
                    o.cur_duty = md[0][o.pos[M_DUTY]]   # a running duty macro sets it every tick;
                d['duty'][f] = o.cur_duty               # after it ends the last value stays
                for t in (M_VOL, M_ARP, M_PIT, M_DUTY):
                    mm = macro_of(o.inst, t)
                    if mm:
                        o.pos[t] += 1
                        if o.pos[t] >= len(mm[0]) and mm[1] >= 0:
                            o.pos[t] = mm[1]
            else:
                d['duty'][f], d['note'][f] = o.cur_duty, o.base
        f += 1
        accum -= D
    return out


def compare(states, sim, shift, T):
    rep = {}
    for c in MUSIC:
        st, sm = states[c], sim[c]
        t = np.arange(T) + shift
        n = int((t < len(sm['snd'])).sum())
        t = t[:n]
        raw_aud = st.eff[:n] > 0
        sim_aud = sm['snd'][t] & ((sm['vol'][t] > 0) if c != TRI else True)
        both = raw_aud & sim_aud
        pitch = st.note[:n] == sm['note'][t]
        vol = np.abs(st.eff[:n] - sm['vol'][t]) <= 1 if c != TRI else np.ones(n, bool)
        fine = np.abs(st.fine[:n] - sm['fine'][t]) <= 2
        duty = st.duty[:n] == sm['duty'][t]
        A = max(1, int(raw_aud.sum()))
        good = raw_aud & both & pitch & vol & fine & duty
        rep[c] = dict(audible=int(raw_aud.sum()),
                      good=good.sum() / A * 100,
                      onoff=(raw_aud == sim_aud).mean() * 100,
                      pitch=(pitch & both).sum() / A * 100,
                      vol=(vol & both).sum() / A * 100,
                      fine=(fine & both).sum() / A * 100,
                      duty=(duty & both).sum() / A * 100)
    return rep


# ----------------------------------------------------------------------------
# main conversion
# ----------------------------------------------------------------------------
def inst_name(spec, n):
    arp, vol, pit, duty = spec
    tags = []
    if arp is not None:
        tags.append('chord' if arp[0] == 'S' else 'arp')
    if pit is not None:
        tags.append('vib' if pit[1] >= 0 else 'bend')
    if duty is not None:
        tags.append('duty')
    return ('%s %s' % ('tri' if vol is None else 'env', '+'.join(tags) if tags else 'plain'))[:40]


class _Ctx:
    pass


def prepare(text_or_mod, speed=None, tolerance=0.1, keep_all=False, verbose=True, subsong=0):
    """stage 1: load one song, find its loop and row grid, and cut it into notes"""
    log = print if verbose else (lambda *a, **k: None)
    m = load_input(text_or_mod, subsong=subsong)
    configure(m.expansion, m.nch, getattr(m, "n163", None))
    frames = flatten_frames(m)
    N = len(frames)
    k_unit = estimate_k(frames)
    states = {c: build_state(frames, c, k_unit) for c in MUSIC}
    log('one fine-pitch step ~ %.2f period units; %d note-column changes treated as the same '
        'note bending (not new notes)' % (k_unit, sum(st.reclassified for st in states.values())))

    # ---- loop: keep the intro plus exactly one pass of the loop
    T, loop_start = N, 0
    halts = False
    if not keep_all and getattr(m, 'loop_tick', None) is not None:
        loop_start = m.loop_tick
        log('loop marked in the module (Bxx): starts at tick %d, %d ticks long' % (loop_start, N - loop_start))
    elif not keep_all and getattr(m, 'halted', False):
        halts = True
        log('the module ends with C00 (no loop): %d ticks' % N)
    elif not keep_all:
        loop, best = detect_loop(states)
        if loop is None:
            log('no clear loop found (best match %.0f%%) - keeping all %d ticks' % (best[0] * 100, N))
        else:
            ls, ll = loop
            if ls < 6:
                ls = 0          # only song-setup noise in the first ticks
            T, loop_start = min(N, ls + ll), ls
            log('loop: %d ticks long (match %.1f%%), starts at tick %d -> keeping the first %d of %d ticks'
                % (ll, best[0] * 100, ls, T, N))
        for st in states.values():
            trim_state(st, T)

    # ---- row grid
    info = choose_grid(states, speed)
    log('grid: ' + info['msg'])
    if info['kind'] == 'int':
        s, phase = info['s'], info['phase']
        grid = Grid(s, phase, True)
        merge_gap = s
    else:
        s = info['S']
        grid = GridF(info['S'], info['T'], info['shift'], T)
        merge_gap = max(3, int(grid.r))
    aligned = np.array([grid.locate(f)[1] == 0 for f in range(T)])
    segs = {c: find_segments(states[c], c, merge_gap, aligned) for c in MUSIC}
    quantize_segments(segs, tolerance)

    if info['kind'] == 'int':
        def conflicts(g):
            return any(len({g.locate(sg.start)[0] for sg in segs[c]}) != len(segs[c]) for c in MUSIC)
        if conflicts(grid):
            grid = Grid(s, phase, False)
            log('two notes would share the long first row -> delaying song by %d tick(s) instead' % grid.shift)
    ctx = _Ctx()
    ctx.m, ctx.states, ctx.T, ctx.N, ctx.loop_start, ctx.halts = m, states, T, N, loop_start, halts
    ctx.dpcm = dpcm_events(frames, T)
    ctx.info, ctx.s, ctx.grid, ctx.segs, ctx.log = info, s, grid, segs, log
    return ctx


def build_dpcm_instrument(ctxs, log=print):
    """map every (instrument, key) the songs hit on the DPCM channel to a key of one shared
    instrument; keys that play the same sample the same way share a key.  Sets ctx.dpcm_map
    and returns ('DPCM', [(octave, note, sample, pitch, loop, loop offset, delta)], samples)"""
    m0 = ctxs[0].m
    raw_keys = raw_dpcm_keys(getattr(m0, 'raw_inst', []))
    samples = getattr(m0, 'dpcm_samples', {}) or {}
    by_sig, used, keys = {}, set(), []
    for x in ctxs:
        x.dpcm_map = {}
        for t, kind, val in getattr(x, 'dpcm', []):
            if kind != 'hit' or (val[0], val[1]) in x.dpcm_map:
                continue
            try:
                octave, note = dpcm_key_of(val[1])
            except (KeyError, ValueError, IndexError):
                continue
            sig = raw_keys.get((val[0], octave, note))
            if sig is None or sig[0] not in samples:
                continue                                   # silent in FamiTracker as well
            if sig not in by_sig:
                slot = (octave, note)
                if slot in used:                           # original key taken: next free key
                    slot = next((o, n) for o in range(8) for n in range(12) if (o, n) not in used)
                used.add(slot)
                by_sig[sig] = slot
                keys.append(slot + tuple(sig))
            o, n = by_sig[sig]
            x.dpcm_map[(val[0], val[1])] = '%s%d' % (NOTE_NAMES[n], o)
    if not keys:
        return None
    used_samples = {k[2]: samples[k[2]] for k in keys}
    log('DPCM: %d sample(s) on %d key(s) of one DPCM instrument' % (len(used_samples), len(keys)))
    return ('DPCM', sorted(keys), used_samples)


def choose_shared_instruments(ctxs, max_inst, log=print):
    """stage 2: one set of instruments for all the given songs (tracks of one module share
    FamiTracker's instrument list); each sound chip has its own instruments and macros"""
    available_chips = [c for c in MACRO_CHIPS if any(inst_group(ch) == c for ch in MUSIC)]
    chip_notes = {c: sum(len(x.segs[ch]) for x in ctxs for ch in MUSIC if inst_group(ch) == c)
                  for c in available_chips}
    # wavetable / FM chips (N163, FDS, VRC7, unknown): their notes keep the original module's
    # instruments, copied across - a rough retrace (notes, timing, volume), not new macros
    passthrough = {}
    for x in ctxs:
        for ch in MUSIC:
            if inst_group(ch) is None:
                for sg in x.segs[ch]:
                    seg_insts = x.states[ch].inst[sg.start:sg.start + sg.L]
                    seg_insts = seg_insts[seg_insts >= 0]
                    sg.raw_inst = int(np.bincount(seg_insts).argmax()) if len(seg_insts) else 0
                    passthrough.setdefault(sg.raw_inst, None)
    chips = [c for c in available_chips if chip_notes[c] > 0]
    if not chips and not passthrough:
        raise ValueError('no audible notes found in this subsong')
    total_notes = sum(chip_notes[c] for c in chips)
    instruments, macros = [], {}
    # DPCM: one shared instrument whose key map plays every sample the songs use
    dpcm_entry = build_dpcm_instrument(ctxs, log)
    # Allocate exactly max_inst choices across active chips. At least one choice per
    # active chip avoids an empty candidate set for quiet expansion-chip parts.
    max_inst = min(max_inst, 64 - len(chips) - len(passthrough) - (1 if dpcm_entry else 0))
    if chips and max_inst < len(chips):
        raise ValueError('--max-instruments must be at least %d for the active sound chips' % len(chips))
    budgets = {c: 1 for c in chips}
    remaining = max_inst - len(chips)
    if remaining and chips:
        quotas = {c: remaining * chip_notes[c] / total_notes for c in chips}
        for c in chips:
            budgets[c] += int(quotas[c])
        left = max_inst - sum(budgets.values())
        for c in sorted(chips, key=lambda x: quotas[x] - int(quotas[x]), reverse=True)[:left]:
            budgets[c] += 1
    for chip in chips:
        chans = [ch for ch in MUSIC if inst_group(ch) == chip]
        budget = budgets[chip]
        reqs, key_idx, counts = [], {}, []
        for c, sg in ((c, sg) for x in ctxs for c in chans for sg in x.segs[c]):
            if True:
                r, shift = make_requirement(sg)
                if r.key not in key_idx:
                    key_idx[r.key] = len(reqs)
                    reqs.append(r)
                    counts.append(0)
                u = key_idx[r.key]
                if r.L > reqs[u].L:
                    reqs[u] = r
                counts[u] += 1
                sg.req, sg.shift, sg.u = r, shift, u
        cands, cand_idx = [], {}
        for r in reqs:
            ck = r.cand
            if ck not in cand_idx:
                cand_idx[ck] = len(cands)
                cands.append(ck)
            if r.head is not None:
                ck = (('F', r.head, -1), r.cand[1], r.cand[2], None)
                if ck not in cand_idx:
                    cand_idx[ck] = len(cands)
                    cands.append(ck)
            if r.vol is not None and 0 < r.vol.max() < 15:
                # the same envelope scaled up to full level: the volume column brings it back down
                shape = np.rint(r.vol * 15.0 / r.vol.max()).astype(int)
                ck = (r.cand[0], compress(shape, 'last'), r.cand[2], None)
                if ck not in cand_idx:
                    cand_idx[ck] = len(cands)
                    cands.append(ck)
        # cross candidates: every arpeggio paired with the most common envelope / pitch settings,
        # including combinations that never occur together in the music
        wv, wp = collections.Counter(), collections.Counter()
        for r, w in zip(reqs, counts):
            wv[(r.ch == TRI, r.cand[1])] += w
            wp[(r.ch == TRI, r.cand[2])] += w
        top_v = {t: [k[1] for k, _ in wv.most_common() if k[0] == t][:12] for t in (False, True)}
        top_p = {t: [None] + [k[1] for k, _ in wp.most_common() if k[0] == t and k[1] is not None][:6]
                 for t in (False, True)}
        arps = {(r.ch == TRI, r.cand[0]) for r in reqs}
        for tri, arp in arps:
            for vol in top_v[tri]:
                for pit in top_p[tri]:
                    ck = (arp, vol, pit, None)
                    if ck not in cand_idx:
                        cand_idx[ck] = len(cands)
                        cands.append(ck)
        log('%s: %d notes, %d distinct note shapes, %d candidate instruments'
            % (chip, chip_notes[chip], len(reqs), len(cands)))
        slack = max(0, (64 - len(chips)) - len(instruments) - sum(budgets[c] for c in chips[chips.index(chip):]))
        chosen, assign, errs, column_for = choose_instruments(reqs, counts, cands, budget, 0.2,
                                                              hard_max=budget + slack)
        if len(chosen) > budget:
            log('%s: %d extra instrument(s) for notes that would otherwise sound plainly wrong'
                % (chip, len(chosen) - budget))
        approx = int((np.asarray(counts) * (errs > 1e-9)).sum())
        log('%s: %d instruments; %d of %d notes use a "close enough" instrument; %d sound plainly wrong'
            % (chip, len(chosen), approx, chip_notes[chip], LAST_SELECTION.get('wrong_notes', 0)))

        macro_idx = {}

        def macro_ref(t, spec, setting):
            if spec is None:
                return -1
            vals, loop_i = spec
            k = (t, vals, loop_i, setting)
            if k not in macro_idx:
                n = sum(1 for kk in macro_idx if kk[0] == t)
                macro_idx[k] = n
                macros[(chip, t, n)] = (list(vals), loop_i, setting)
            return macro_idx[k]

        base = len(instruments)
        instruments.append((chip, -1, -1, -1, -1, -1, 'blank'))
        for ci in chosen:
            arp, vol, pit, duty = cands[ci]
            a = -1 if arp is None else macro_ref(M_ARP, (arp[1], arp[2]),
                                                 {'S': ARP_SCHEME, 'F': ARP_FIXED}.get(arp[0], ARP_ABS))
            instruments.append((chip, macro_ref(M_VOL, vol, 0), a, macro_ref(M_PIT, pit, 0), -1,
                                macro_ref(M_DUTY, duty, 0), inst_name(cands[ci], 0)))
        n_slides = 0
        for c, sg in ((c, sg) for x in ctxs for c in chans for sg in x.segs[c]):
            if True:
                ci = chosen[int(assign[sg.u])]
                arp = cands[ci][0]
                sg.inst = base + 1 + chosen.index(ci)
                sg.colvol = sg.req.colvol if sg.req.colvol is not None else \
                    (column_for(sg.u, ci) if sg.ch != TRI else 15)
                sg.uses_scheme = arp is not None and arp[0] == 'S' and sg.req.scheme is not None
                sg.slide = None
                # a note no instrument fits (a one-off sweep, a tom fill): try a pattern effect
                rare = counts[sg.u] <= ONE_OFF_COUNT
                if SLIDES and errs[sg.u] / sg.req.L > (ONE_OFF_GATE if rare else WRONG_PER_TICK):
                    plain = [(base + 1 + j, cands[cj]) for j, cj in enumerate(chosen)
                             if cands[cj][0] is None and cands[cj][2] is None]
                    fit = fit_slide(sg, sg.req, plain)
                    if fit is not None and fit[0] < 0.6 * errs[sg.u]:
                        sg.slide, sg.inst, sg.colvol = fit[1], fit[2], fit[3]
                        sg.uses_scheme = False
                        sg.detune = 0
                        n_slides += 1
        if n_slides:
            log('%s: %d badly-fitting note(s) played with a Qxy/Rxy note slide instead' % (chip, n_slides))
    if passthrough:
        raw_lines = getattr(ctxs[0].m, 'raw_inst', [])
        for raw in sorted(passthrough):
            passthrough[raw] = len(instruments)
            instruments.append(('RAW', raw, raw_lines))
        log('%d original instrument(s) kept for wavetable / FM channels (rough retrace)' % len(passthrough))
        for x in ctxs:
            for ch in MUSIC:
                if inst_group(ch) is None:
                    for sg in x.segs[ch]:
                        sg.inst, sg.shift, sg.uses_scheme, sg.slide = passthrough[sg.raw_inst], 0, False, None
                        aud = sg.vol[sg.vol > 0] if sg.vol is not None and len(sg.vol) else []
                        sg.colvol = int(aud[0]) if len(aud) else 15     # the volume it starts at
                        sg.req = Req()
                        sg.req.scheme, sg.req.colvol = None, sg.colvol
    if dpcm_entry:
        for x in ctxs:
            x.dpcm_inst = len(instruments)
        instruments.append(dpcm_entry)
    for chip in chips:
        for t, nm in ((M_VOL, 'volume'), (M_ARP, 'arpeggio'), (M_PIT, 'pitch'), (M_DUTY, 'duty')):
            if sum(1 for k in macros if k[0] == chip and k[1] == t) > 128:
                print('WARNING: more than 128 %s macros (FamiTracker limit)' % nm, file=sys.stderr)
    if len(instruments) > 64:
        print('WARNING: %d instruments (FamiTracker limit 64) - lower --max-instruments' % len(instruments),
              file=sys.stderr)
    return instruments, macros


def finish(ctx, instruments, macros, pattern_rows=None):
    """stage 3: put the notes on rows and patterns, write the module, check it by playing it"""
    m, states, T, N, loop_start, halts = ctx.m, ctx.states, ctx.T, ctx.N, ctx.loop_start, ctx.halts
    info, s, grid, segs, log = ctx.info, ctx.s, ctx.grid, ctx.segs, ctx.log
    configure(m.expansion, m.nch, getattr(m, "n163", None))
    STATS.clear()
    duty_insts = {i for i, e in enumerate(instruments)
                  if e[0] not in ('RAW', 'DPCM') and len(e) == 7 and e[5] >= 0}
    def env_of(inst):
        """(values, loop) of an instrument's volume macro, None for a flat 15"""
        if inst is None or not 0 <= inst < len(instruments):
            return None
        e = instruments[inst]
        if e[0] in ('RAW', 'DPCM') or len(e) != 7 or e[1] < 0:
            return None
        vals, loop, _setting = macros[(e[0], M_VOL, e[1])]
        return (vals, loop)
    rows = build_rows(segs, grid, T, states, duty_insts, env_of if VOLUME_FOLLOW else None)
    if STATS['volume column follows']:
        log('%d note(s) follow the game\'s volume with the volume column, row by row'
            % STATS['volume column follows'])
    dres = place_dpcm(rows, ctx, grid)
    n_hits = sum(1 for e in getattr(ctx, 'dpcm', []) if e[1] == 'hit')
    if n_hits:
        log('DPCM: %d sample hits - %d on their exact tick, %d one row late (row already used), '
            '%d dropped, %d with no sample assigned' % (n_hits, dres['exact'], dres['late'],
                                                        dres['dropped'], dres['unmapped']))
    n_rows = grid.locate(T - 1)[0] + 1
    rows = rows[:n_rows]
    if STATS['pushed'] or STATS['dropped']:
        log('%d notes shared a row with another note: %d moved to the next row, %d dropped'
            % (STATS['pushed'] + STATS['dropped'], STATS['pushed'], STATS['dropped']))
    preroll = grid.shift == 0 and grid.first > s and info['kind'] == 'int'
    loop_row = 0
    if loop_start > 0:
        r0, d0 = grid.locate(loop_start)
        loop_row = r0 if d0 == 0 else r0 + 1
        if loop_row >= len(rows) - 1:
            loop_row = 0
    columns = [max(1, min(4, max(len(r[c].eff) for r in rows))) for c in range(NCH)]
    pr = pattern_rows or pick_pattern_rows(rows, columns, preroll, loop_row)
    if preroll:
        add_global_effect(rows, 0, 'F%02X' % grid.first)
        add_global_effect(rows, 0, 'D00')
        add_global_effect(rows, 1, 'F%02X' % s)
    if loop_row:
        add_global_effect(rows, loop_row - 1, 'D00')          # intro ends early
        _, loop_order = layout_chunks(rows, pr, preroll, loop_row)
        add_global_effect(rows, len(rows) - 1, 'B%02X' % loop_order)   # jump back to the loop
        log('loop starts at row %d (order %02X); the last row jumps back to it' % (loop_row, loop_order))
    if halts:
        add_global_effect(rows, len(rows) - 1, 'C00')          # the song ends here, as in the game
    columns = [max(1, min(4, max(len(r[c].eff) for r in rows))) for c in range(NCH)]
    chunks, _ = layout_chunks(rows, pr, preroll, loop_row)
    out_text, orders, npat = write_module(m, macros, instruments, chunks, columns, s, grid.tempo,
                                          pr, 'auto retrace')
    log('%d raw rows -> %d rows in %d orders of %d rows; %d patterns, %d instruments, %d macros'
        % (N, len(rows), len(orders), pr, npat, len(instruments), len(macros)))
    sim = simulate(out_text, T + grid.shift + 8)
    rep = compare(states, sim, grid.shift, T)
    rep['dpcm_hits'], rep['dpcm_exact'] = n_hits, dres['exact']
    return out_text, rep


def convert(text_or_mod, speed=None, pattern_rows=None, max_inst=63, tolerance=0.1,
            keep_all=False, verbose=True, subsong=0):
    ctx = prepare(text_or_mod, speed, tolerance, keep_all, verbose, subsong)
    instruments, macros = choose_shared_instruments([ctx], max_inst, ctx.log)
    return finish(ctx, instruments, macros, pattern_rows)


def combine_retraces(module_texts, track_names):
    """Combine independently retraced songs as tracks in one FamiTracker text module.

    Instruments and macros are module-wide in FamiTracker, whereas orders and patterns
    belong to individual tracks. This merges identical definitions and remaps references
    in every track so pattern data continues to point at the correct shared instruments.
    """
    if not module_texts:
        raise ValueError('no subsongs were converted')
    if len(module_texts) != len(track_names):
        raise ValueError('internal error: each retrace must have a track name')

    parsed = []
    macro_defs = {}              # (chip, type, global index) -> (loop, relative, setting, values)
    macro_signatures = {}        # (chip, type, signature) -> global index
    next_macro = collections.Counter()
    instrument_defs = []         # (chip, macro references, name)
    instrument_signatures = {}   # (chip, macro references) -> global index

    for track_i, text in enumerate(module_texts):
        lines = text.replace('\r\n', '\n').split('\n')
        try:
            seq_i = lines.index('# SEQUENCES block')
            dpcm_i = lines.index('# DPCM SAMPLES block')
            inst_i = lines.index('# INSTRUMENTS block')
            track_i0 = lines.index('# track HEADER block')
        except ValueError as exc:
            raise ValueError('could not parse generated retrace module: %s' % exc)

        local_macro_map = {}
        for line in lines[seq_i + 1:dpcm_i]:
            if not line.startswith(('MACRO ', 'MACROVRC6 ')):
                continue
            try:
                head, values_text = line.split(':', 1)
                fields = head.split()
                macro_kind = fields[0]
                chip = macro_kind[5:] or '2A03'
                macro_type, local_idx, loop, relative, setting = map(int, fields[1:6])
                values = tuple(map(int, values_text.split()))
            except (ValueError, IndexError) as exc:
                raise ValueError('could not parse macro line %r: %s' % (line, exc))
            signature = (loop, relative, setting, values)
            signature_key = (chip, macro_type, signature)
            if signature_key not in macro_signatures:
                global_idx = next_macro[(chip, macro_type)]
                if global_idx > 127:
                    raise ValueError('the merged module needs more than 128 %s macros of type %d; '
                                     'reduce --max-instruments' % (chip, macro_type))
                macro_signatures[signature_key] = global_idx
                macro_defs[(chip, macro_type, global_idx)] = (loop, relative, setting, values)
                next_macro[(chip, macro_type)] += 1
            local_macro_map[(chip, macro_type, local_idx)] = macro_signatures[signature_key]

        local_instrument_map = {}
        local_keys = collections.defaultdict(list)       # DPCM key map lines per local instrument
        for line in lines[inst_i + 1:track_i0]:
            if line.startswith('KEYDPCM '):
                f = line.split()
                local_keys[int(f[1])].append(tuple(f[2:]))
        for line in lines[inst_i + 1:track_i0]:
            if not line.startswith(('INST2A03 ', 'INSTVRC6 ')):
                continue
            try:
                left, _, tail = line.partition('"')
                name = tail.rsplit('"', 1)[0] if '"' in tail else 'instrument'
                fields = left.split()
                chip = fields[0][4:]
                local_idx = int(fields[1])
                refs = tuple(map(int, fields[2:7]))
                if len(refs) != 5:
                    raise ValueError('expected five macro references')
                remapped_refs = tuple(
                    -1 if ref < 0 else local_macro_map[(chip, macro_type, ref)]
                    for macro_type, ref in enumerate(refs)
                )
            except (ValueError, IndexError, KeyError) as exc:
                raise ValueError('could not parse instrument line %r: %s' % (line, exc))
            keys = tuple(sorted(local_keys.get(local_idx, [])))
            signature_key = (chip, remapped_refs, keys)
            if signature_key not in instrument_signatures:
                global_idx = len(instrument_defs)
                if global_idx > 63:
                    raise ValueError('the merged module needs more than FamiTracker\'s 64 instruments; '
                                     'reduce --max-instruments')
                instrument_signatures[signature_key] = global_idx
                instrument_defs.append((chip, remapped_refs, name, keys))
            local_instrument_map[local_idx] = instrument_signatures[signature_key]

        # A track's order/pattern data must stay intact, except for instrument references
        # and the track name. Pattern numbers are local to each TRACK section.
        track_lines = lines[track_i0:]
        replaced_track_name = False
        clean_name = str(track_names[track_i]).replace('"', "'").replace('\r', ' ').replace('\n', ' ')
        for j, line in enumerate(track_lines):
            if line.startswith('TRACK '):
                fields = line.split(None, 4)
                if len(fields) >= 4:
                    track_lines[j] = '%s %s %s %s "%s"' % (fields[0], fields[1], fields[2], fields[3], clean_name)
                    replaced_track_name = True
            elif line.startswith('ROW '):
                parts = line.split(' : ')
                for cell_i in range(1, len(parts)):
                    tokens = parts[cell_i].split()
                    if len(tokens) >= 2 and tokens[1] != '..':
                        try:
                            tokens[1] = '%02X' % local_instrument_map[int(tokens[1], 16)]
                        except (ValueError, KeyError) as exc:
                            raise ValueError('could not remap instrument in %r: %s' % (line, exc))
                        parts[cell_i] = ' '.join(tokens)
                track_lines[j] = ' : '.join(parts)
        if not replaced_track_name:
            raise ValueError('generated retrace did not contain a TRACK header')
        while track_lines and not track_lines[-1]:
            track_lines.pop()
        parsed.append((lines, seq_i, dpcm_i, inst_i, track_lines))

    # Keep the first module's metadata and empty global sections, replacing only the
    # shared sequences/instruments and appending each track's own header/orders/patterns.
    first, seq_i, dpcm_i, inst_i, _ = parsed[0]
    out = first[:seq_i + 1]
    for (chip, macro_type, idx), (loop, relative, setting, values) in sorted(macro_defs.items()):
        label = 'MACRO' if chip == '2A03' else 'MACRO' + chip
        out.append('%s %7d %3d %3d %3d %3d : %s' %
                   (label, macro_type, idx, loop, relative, setting, ' '.join(map(str, values))))
    out.append('')
    out.extend(first[dpcm_i:inst_i + 1])
    for idx, (chip, refs, name, keys) in enumerate(instrument_defs):
        out.append('%s %3d %5d %3d %3d %3d %3d "%s"' %
                   ('INST' + chip, idx, *refs, name.replace('"', "'")))
        for k in keys:
            out.append('KEYDPCM %3d %s' % (idx, ' '.join(k)))
    out.append('')
    for _, _, _, _, track_lines in parsed:
        out.extend(track_lines)
        out.append('')
    return '\r\n'.join(out)


def validate_text(text):
    """final cleanup check: does the output obey the ranges FamiTracker's importer enforces?"""
    errs = []
    lines = text.split('\r\n')
    def err(i, msg): errs.append('line %d: %s' % (i + 1, msg))
    sect = None; ninst = 0; instchip = {}; macros = {}; ncol = None; real_of = {}; rows = None; patids = set(); maxpat = -1
    for i, l in enumerate(lines):
        if not l: continue
        if l.startswith('#'): sect = l; continue
        p = l.split()
        k = p[0]
        if k in ('MACHINE', 'FRAMERATE', 'EXPANSION', 'VIBRATO'):
            if not p[1].lstrip('-').isdigit(): err(i, 'bad int')
        elif k == 'SPLIT':
            if not 0 <= int(p[1]) <= 255: err(i, 'SPLIT range')
        elif k == 'PLAYBACKRATE':
            if not (p[1].isdigit() and 0 <= int(p[1]) <= 2): err(i, 'PLAYBACKRATE mode must be 0..2, got %s' % p[1])
        elif k.startswith('MACRO'):
            chipk = k[5:] or '2A03'
            t, idx, loop, rel, setting = map(int, p[1:6])
            vals = list(map(int, l.split(':')[1].split()))
            lim = {0: (0, 15), 1: (-128, 127), 2: (-128, 127), 3: (-128, 127), 4: (0, 255)}[t]
            if not 0 <= t <= 4: err(i, 'macro type')
            if not 0 <= idx <= 127: err(i, 'macro index')
            if len(vals) > 252: err(i, 'macro too long (%d)' % len(vals))
            if loop >= len(vals): err(i, 'loop point past end')
            if any(not lim[0] <= v <= lim[1] for v in vals): err(i, 'macro value out of range %s' % (lim,))
            macros[(chipk, t, idx)] = len(vals)
        elif k.startswith('INST'):
            n = int(p[1]); ninst = max(ninst, n + 1); instchip[n] = k[4:]
            if not 0 <= n <= 63: err(i, 'instrument index %d > 63' % n)
            for t, ref in (enumerate(map(int, p[2:7])) if k in ('INST2A03', 'INSTVRC6', 'INSTS5B', 'INSTN163') else ()):
                if ref >= 0 and (k[4:], t, ref) not in macros: err(i, 'instrument %d refers to missing macro type %d #%d' % (n, t, ref))
        elif k == 'TRACK':
            rows, sp, tp = int(p[1]), int(p[2]), int(p[3])
            if not 1 <= rows <= 256: err(i, 'pattern length')
            if not 1 <= sp <= 255 or not 32 <= tp <= 255: err(i, 'speed/tempo %d/%d' % (sp, tp))
        elif k == 'COLUMNS':
            ncol = list(map(int, p[2:]))
            if len(ncol) < 5 or any(not 1 <= c <= 4 for c in ncol): err(i, 'columns')
            # text channel -> real channel (Dn-FamiTracker text pads N163 to 8 channels)
            ex = int(next((x.split()[1] for x in lines if x.startswith('EXPANSION')), 0))
            n1 = int(next((x.split()[1] for x in lines if x.startswith('N163CHANNELS')), 8))
            pos, full = n163_text_layout(ex)
            real_of, rc = {}, 0
            for tc in range(len(ncol)):
                if ex & 16 and n1 < 8 and len(ncol) == full and pos + n1 <= tc < pos + 8:
                    real_of[tc] = None
                else:
                    real_of[tc], rc = rc, rc + 1
        elif k == 'ORDER':
            ids = [int(x, 16) for x in l.split(' : ')[1].split()]
            maxpat = max(maxpat, max(ids))
            if max(ids) > 127: err(i, 'pattern id > 7F')
        elif k == 'PATTERN':
            patids.add(int(p[1], 16)); nrow = 0
        elif k == 'ROW':
            cells = l.split(' : ')[1:]
            if len(cells) != len(ncol): err(i, 'need %d channels' % len(ncol)); continue
            for c, cell in enumerate(cells):
                t = cell.split()
                if len(t) != 3 + ncol[c]: err(i, 'channel %d has %d fields, expected %d' % (c, len(t), 3 + ncol[c])); continue
                note, inst, vol = t[:3]
                if note not in ('...', '---', '==='):
                    if c == 3:
                        if not re.fullmatch(r'[0-9A-F]-#', note): err(i, 'bad noise note %s' % note)
                    elif not re.fullmatch(r'[A-G][-#][0-7]|\^-[0-9]', note): err(i, 'bad note %s' % note)
                if inst != '..' and not (len(inst) == 2 and int(inst, 16) < max(ninst, 1)): err(i, 'bad/undefined instrument %s' % inst)
                elif inst != '..' and real_of.get(c, c) is None:
                    err(i, 'instrument on an unused N163 channel %d' % c)
                elif inst != '..' and CHIP_OF.get(real_of.get(c, c), '2A03') != 'EXT' and \
                        instchip.get(int(inst, 16)) != INST_GROUP.get(CHIP_OF.get(real_of.get(c, c), '2A03'),
                                                                      CHIP_OF.get(real_of.get(c, c))):
                    err(i, 'channel %d uses instrument %s of the wrong chip type' % (c, inst))
                for e in t[3:]:
                    if e != '...' and not re.fullmatch(r'[0-9A-Z][0-9A-F]{2}', e): err(i, 'bad effect %s' % e)
    if maxpat >= 0 and maxpat not in patids: err(len(lines), 'ORDER refers to missing pattern %X' % maxpat)
    return errs



# ----------------------------------------------------------------------------
# Native FamiTracker module writer (.ftm)
# ----------------------------------------------------------------------------
def _ftm_block(name, version, payload):
    """Wrap a legacy FamiTracker block (header + payload)."""
    encoded_name = name.encode('ascii')
    if len(encoded_name) > 16:
        raise ValueError('FTM block name is too long: %s' % name)
    return struct.pack('<16sII', encoded_name, version, len(payload)) + payload


def _ftm_string(text, max_bytes=None):
    """Encode strings the way the legacy FamiTracker format stores them."""
    data = str(text).encode('latin1', errors='replace').replace(b'\x00', b' ')
    if max_bytes is not None:
        data = data[:max_bytes]
    return data


def _ftm_fixed_string(text, size=32):
    data = _ftm_string(text, size)
    return data + b'\x00' * (size - len(data))


def _ftm_parse_text(text):
    """Read the relevant global and per-track fields from our generated FT text."""
    lines = text.replace('\r\n', '\n').replace('\r', '\n').split('\n')

    def quoted_field(key, default=''):
        prefix = key + ' '
        for line in lines:
            if line.startswith(prefix):
                parts = line.split('"', 2)
                if len(parts) >= 3:
                    return parts[1]
        return default

    def integer_field(key, default):
        for line in lines:
            if line.startswith(key + ' '):
                fields = line.split()
                if len(fields) > 1:
                    try:
                        return int(fields[1], 10)
                    except ValueError:
                        pass
        return default

    title = quoted_field('TITLE')
    author = quoted_field('AUTHOR')
    copyright_ = quoted_field('COPYRIGHT')
    expansion = integer_field('EXPANSION', 0)
    machine = integer_field('MACHINE', 0)
    vibrato = integer_field('VIBRATO', 1)
    split_point = integer_field('SPLIT', 21)
    playback_type, playback_rate = 0, 0
    for line in lines:
        if line.startswith('PLAYBACKRATE '):
            fields = line.split()
            try:
                playback_type = int(fields[1], 10)
                playback_rate = int(fields[2], 10) if len(fields) > 2 else 0
            except (ValueError, IndexError):
                pass
            break

    macros = []
    for line in lines:
        if not line.startswith(('MACRO ', 'MACROVRC6 ')):
            continue
        try:
            head, values_text = line.split(':', 1)
            fields = head.split()
            kind = fields[0]
            chip = kind[5:] or '2A03'
            macro_type, index, loop, release, setting = map(int, fields[1:6])
            values = [int(value, 10) for value in values_text.split()]
        except (ValueError, IndexError) as exc:
            raise ValueError('cannot encode macro line %r: %s' % (line, exc))
        if chip not in ('2A03', 'VRC6'):
            raise ValueError('unsupported instrument chip in macro: %s' % chip)
        if not 0 <= macro_type <= 4 or not 0 <= index <= 127 or len(values) > 255:
            raise ValueError('macro is outside FamiTracker limits: %r' % line)
        macros.append((chip, macro_type, index, loop, release, setting, values))

    instruments = []
    for line in lines:
        if not line.startswith(('INST2A03 ', 'INSTVRC6 ')):
            continue
        try:
            left, sep, tail = line.partition('"')
            name = tail.rsplit('"', 1)[0] if sep and '"' in tail else 'Instrument'
            fields = left.split()
            kind = fields[0]
            chip = kind[4:]
            index = int(fields[1], 10)
            refs = [int(value, 10) for value in fields[2:7]]
        except (ValueError, IndexError) as exc:
            raise ValueError('cannot encode instrument line %r: %s' % (line, exc))
        if chip not in ('2A03', 'VRC6') or len(refs) != 5:
            raise ValueError('unsupported or malformed instrument line: %r' % line)
        if not 0 <= index <= 63:
            raise ValueError('instrument index is outside FamiTracker limits: %d' % index)
        instruments.append((index, chip, refs, name))
    instruments.sort(key=lambda inst: inst[0])
    dpcm_keys = collections.defaultdict(list)          # instrument -> [(midi, sample, pitch byte, delta)]
    for line in lines:
        if line.startswith('KEYDPCM '):
            f = list(map(int, line.split()[1:9]))
            inst_i, octave, note, smp, pitch, loop, _loff, delta = f
            dpcm_keys[inst_i].append((octave * 12 + note, smp, (pitch & 0x0F) | (0x80 if loop else 0), delta))
    dpcm_samples = text_dpcm_samples(lines)
    if len(instruments) > 64:
        raise ValueError('FamiTracker supports at most 64 instruments in one module')

    track_starts = [i for i, line in enumerate(lines) if line.startswith('TRACK ')]
    if not track_starts:
        raise ValueError('no TRACK section found in generated text')
    if len(track_starts) > 64:
        raise ValueError('FamiTracker supports at most 64 tracks in one module')
    tracks = []
    for ti, start in enumerate(track_starts):
        section = lines[start:track_starts[ti + 1] if ti + 1 < len(track_starts) else len(lines)]
        try:
            match = re.match(r'^TRACK\s+(\d+)\s+(\d+)\s+(\d+)\s+"(.*)"$', section[0])
            if not match:
                raise ValueError('malformed TRACK header')
            rows, speed, tempo = map(int, match.group(1, 2, 3))
            name = match.group(4)
        except (ValueError, IndexError) as exc:
            raise ValueError('cannot encode track header %r: %s' % (section[0], exc))

        columns = None
        orders = []
        patterns = {}
        current_pattern = None
        for line in section[1:]:
            if line.startswith('COLUMNS'):
                try:
                    columns = list(map(int, line.split(':', 1)[1].split()))
                except ValueError as exc:
                    raise ValueError('malformed COLUMNS line %r: %s' % (line, exc))
            elif line.startswith('ORDER '):
                try:
                    orders.append([int(value, 16) for value in line.split(':', 1)[1].split()])
                except ValueError as exc:
                    raise ValueError('malformed ORDER line %r: %s' % (line, exc))
            elif line.startswith('PATTERN '):
                try:
                    current_pattern = int(line.split()[1], 16)
                except (ValueError, IndexError) as exc:
                    raise ValueError('malformed PATTERN line %r: %s' % (line, exc))
                patterns[current_pattern] = {}
            elif line.startswith('ROW '):
                if current_pattern is None:
                    raise ValueError('ROW appears before PATTERN: %r' % line)
                try:
                    pieces = line.split(' : ')
                    row_index = int(pieces[0].split()[1], 16)
                    cells = [piece.split() for piece in pieces[1:]]
                except (ValueError, IndexError) as exc:
                    raise ValueError('malformed ROW line %r: %s' % (line, exc))
                patterns[current_pattern][row_index] = cells

        if columns is None or not orders or not patterns:
            raise ValueError('track %d is missing COLUMNS, ORDER, or PATTERN data' % ti)
        if len(columns) not in (5, 8) or any(not 1 <= count <= 4 for count in columns):
            raise ValueError('track %d has unsupported channel/effect-column layout' % ti)
        if any(len(order) != len(columns) for order in orders):
            raise ValueError('track %d has an ORDER with the wrong number of channels' % ti)
        for pattern_id, row_map in patterns.items():
            for row_id, cells in row_map.items():
                if len(cells) != len(columns):
                    raise ValueError('track %d pattern %02X row %02X has the wrong channel count' %
                                     (ti, pattern_id, row_id))
                for channel, cell in enumerate(cells):
                    if len(cell) != 3 + columns[channel]:
                        raise ValueError('track %d pattern %02X row %02X channel %d has %d fields; expected %d' %
                                         (ti, pattern_id, row_id, channel, len(cell), 3 + columns[channel]))
        tracks.append(dict(name=name, rows=rows, speed=speed, tempo=tempo,
                           columns=columns, orders=orders, patterns=patterns))

    channel_count = len(tracks[0]['columns'])
    if any(len(track['columns']) != channel_count for track in tracks):
        raise ValueError('all tracks in one FTM module must use the same channel layout')
    if channel_count != (8 if expansion == 1 else 5):
        raise ValueError('module expansion/channel count mismatch (%d expansion, %d channels)' %
                         (expansion, channel_count))
    return dict(title=title, author=author, copyright=copyright_, expansion=expansion,
                machine=machine, vibrato=vibrato, split_point=split_point,
                playback_type=playback_type, playback_rate=playback_rate,
                macros=macros, instruments=instruments, tracks=tracks,
                channel_count=channel_count, dpcm_keys=dict(dpcm_keys), dpcm_samples=dpcm_samples)


def _ftm_note(note_text, channel):
    """Convert a FamiTracker text note into (note byte, octave byte)."""
    if note_text == '...':
        return 0, 0
    if note_text == '===':
        return 13, 0
    if note_text == '---':
        return 14, 0
    if channel == NOI and re.fullmatch(r'[0-9A-F]-#', note_text):
        noise_value = int(note_text[0], 16)
        semitone = 48 + noise_value
        return semitone % 12 + 1, semitone // 12
    match = re.fullmatch(r'([A-G][-#])([0-7])', note_text)
    if not match:
        raise ValueError('cannot encode note %r in channel %d' % (note_text, channel))
    return NOTE_INDEX[match.group(1)] + 1, int(match.group(2))


def write_ftm(text):
    """Convert our FamiTracker text module into a native 0x0450 .ftm file."""
    data = _ftm_parse_text(text)
    tracks = data['tracks']
    nch = data['channel_count']

    # PARAMS v9: expansion/channels/system/playback, vibrato, sweep reset,
    # optional N163 channel count (only if expansion bit 16 is set), split point,
    # then the two global tuning bytes. retrace supports only 2A03 and VRC6.
    params = struct.pack('<B7Ibb', data['expansion'], nch, data['machine'],
                         data['playback_type'], data['playback_rate'], data['vibrato'],
                         0, data['split_point'], 0, 0)
    info = (_ftm_fixed_string(data['title']) + _ftm_fixed_string(data['author']) +
            _ftm_fixed_string(data['copyright']))

    # HEADER v4 stores track names, channel identifiers, per-track effect columns,
    # and row highlights. The column count in the text includes the first effect.
    header = bytearray([len(tracks) - 1])
    for track in tracks:
        header.extend(_ftm_string(track['name']))
        header.append(0)
    for channel in range(nch):
        header.append(channel)
        for track in tracks:
            header.append(track['columns'][channel] - 1)
    for _track in tracks:
        # In HEADER v4 the major/minor row highlights are two single bytes.
        header.extend((4, 16))

    # Instruments and sequences share module-global indices across all tracks.
    instrument_payload = bytearray(struct.pack('<I', len(data['instruments'])))
    for index, chip, refs, name in data['instruments']:
        inst_type = 1 if chip == '2A03' else 2
        instrument_payload.extend(struct.pack('<IB', index, inst_type))
        instrument_payload.extend(struct.pack('<I', 5))
        # INSTRUMENTS v8 stores each sequence's presence flag and index as an
        # interleaved pair, not as two separate arrays.
        for ref in refs:
            instrument_payload.extend((1 if ref >= 0 else 0, ref if ref >= 0 else 0))
        if chip == '2A03':
            # INSTRUMENTS v8 DPCM key map: count, then (octave * 12 + key, sample + 1, pitch | loop, delta)
            # (FamiTracker writes MIDI_NOTE(octave, key) + 1, and its MIDI_NOTE subtracts one)
            keys = sorted(data['dpcm_keys'].get(index, []))
            instrument_payload.extend(struct.pack('<I', len(keys)))
            for midi, smp, pitch_byte, delta in keys:
                instrument_payload.extend(struct.pack('<BBBb', midi, smp + 1, pitch_byte,
                                                      delta if 0 <= delta <= 127 else -1))
        encoded_name = _ftm_string(name)
        instrument_payload.extend(struct.pack('<I', len(encoded_name)))
        instrument_payload.extend(encoded_name)

    def encode_sequences(chip):
        selected = sorted((item for item in data['macros'] if item[0] == chip),
                          key=lambda item: (item[2], item[1]))
        payload = bytearray(struct.pack('<I', len(selected)))
        release_settings = []
        for _chip, macro_type, index, loop, release, setting, values in selected:
            if len(values) > 255:
                raise ValueError('macro %d has too many values for FamiTracker' % index)
            if not 0 <= loop < max(len(values), 1) and loop != -1:
                raise ValueError('macro %d has an invalid loop point %d' % (index, loop))
            if release < -1 or release >= len(values):
                raise ValueError('macro %d has an invalid release point %d' % (index, release))
            # SEQUENCES v6 stores all sequence bodies first, then a separate
            # release-point / setting pair for each sequence at the end.
            payload.extend(struct.pack('<IIBI', index, macro_type, len(values), loop & 0xFFFFFFFF))
            payload.extend(bytes(value & 0xFF for value in values))
            release_settings.append((release, setting))
        for release, setting in release_settings:
            payload.extend(struct.pack('<II', release & 0xFFFFFFFF, setting & 0xFFFFFFFF))
        return bytes(payload)

    sequences = encode_sequences('2A03')
    vrc6_sequences = encode_sequences('VRC6')

    # DPCM SAMPLES v1: count, then (index, name length, name, size, data) per sample
    dpcm_block = bytearray([len(data['dpcm_samples'])])
    for sidx in sorted(data['dpcm_samples']):
        sname, sdata = data['dpcm_samples'][sidx]
        encoded = _ftm_string(sname)[:255]
        dpcm_block.extend(struct.pack('<BI', sidx, len(encoded)))
        dpcm_block.extend(encoded)
        dpcm_block.extend(struct.pack('<I', len(sdata)))
        dpcm_block.extend(sdata)
    dpcm_block = bytes(dpcm_block)

    # FRAMES v3 carries one frame list plus defaults per track.
    frames = bytearray()
    for track in tracks:
        if not 1 <= len(track['orders']) <= 256:
            raise ValueError('track %r has an invalid number of frames (%d)' %
                             (track['name'], len(track['orders'])))
        frames.extend(struct.pack('<IIII', len(track['orders']), track['speed'],
                                  track['tempo'], track['rows']))
        frames.extend(bytes(value for order in track['orders'] for value in order))

    effect_codes = {char: code for code, char in EF_CHARS.items()}
    patterns = bytearray()
    for track_index, track in enumerate(tracks):
        for channel in range(nch):
            for pattern_id in sorted(track['patterns']):
                row_map = track['patterns'][pattern_id]
                encoded_rows = []
                effect_columns = track['columns'][channel]
                for row_index in sorted(row_map):
                    if not 0 <= row_index < track['rows']:
                        raise ValueError('row index %d exceeds track pattern length %d' %
                                         (row_index, track['rows']))
                    cell = row_map[row_index][channel]
                    note_byte, octave = _ftm_note(cell[0], channel)
                    instrument = 0x40 if cell[1] == '..' else int(cell[1], 16)
                    volume = 0x10 if cell[2] in ('.', '..') else int(cell[2], 16)
                    effects = []
                    for token in cell[3:3 + effect_columns]:
                        if token == '...':
                            effects.append((0, 0))
                        else:
                            if len(token) != 3 or token[0].upper() not in effect_codes:
                                raise ValueError('unsupported FamiTracker effect token %r' % token)
                            effects.append((effect_codes[token[0].upper()], int(token[1:], 16)))
                    if len(effects) < effect_columns:
                        effects.extend([(0, 0)] * (effect_columns - len(effects)))
                    has_event = (note_byte != 0 or instrument != 0x40 or volume != 0x10 or
                                 any(code != 0 for code, _param in effects))
                    if not has_event:
                        continue
                    # PATTERNS v6 stores a one-byte row index and four effect
                    # slots. An empty slot is encoded as just a zero effect byte;
                    # the parameter byte is present only for nonzero effects.
                    row_data = bytearray(struct.pack('<BBBBB', row_index, note_byte, octave,
                                                     instrument, volume))
                    padded_effects = effects[:4] + [(0, 0)] * max(0, 4 - len(effects))
                    for code, param in padded_effects:
                        row_data.append(code)
                        if code != 0:
                            row_data.append(param)
                    encoded_rows.append(bytes(row_data))
                if encoded_rows:
                    patterns.extend(struct.pack('<IIII', track_index, channel, pattern_id, len(encoded_rows)))
                    for row in encoded_rows:
                        patterns.extend(row)

    # This is a standard FamiTracker (0x0450) legacy-block module. The VRC6 sequence
    # block uses its own block version; it shares the current sequence record layout.
    blocks = [
        _ftm_block('PARAMS', 9, params),
        _ftm_block('INFO', 1, info),
        _ftm_block('HEADER', 4, bytes(header)),
        _ftm_block('INSTRUMENTS', 8, bytes(instrument_payload)),
        _ftm_block('SEQUENCES', 6, sequences),
        _ftm_block('FRAMES', 3, bytes(frames)),
        _ftm_block('PATTERNS', 6, bytes(patterns)),
        _ftm_block('DPCM SAMPLES', 1, dpcm_block),
    ]
    if data['expansion'] == 1:
        blocks.append(_ftm_block('SEQUENCES_VRC6', 1, vrc6_sequences))
    # Legacy FTM files terminate the block stream with the literal END marker.
    return b'FamiTracker Module' + struct.pack('<I', 0x0450) + b''.join(blocks) + b'END'

def convert_all(input_path, subsongs, track_names, settings, max_inst, pattern_rows=None,
                keep_all=False, verbose=True):
    """analyse the chosen tracks, pick one shared set of instruments, and write each track.
    settings: speed (None = auto), tolerance, pitch_tol, instrument_cost.
    Returns (module texts, playback reports, track names)."""
    global PITCH_TOL, FACILITY_COST
    PITCH_TOL, FACILITY_COST = settings['pitch_tol'], settings['instrument_cost']
    say = print if verbose else (lambda *a, **k: None)
    ctxs, names = [], []
    for i in subsongs:
        name = track_names[i] or ('Track %d' % (i + 1))
        say('\n=== Subsong %d/%d: %s ===' % (i + 1, len(track_names), name))
        try:
            ctx = prepare(input_path, settings['speed'], settings['tolerance'], keep_all,
                          verbose=verbose, subsong=i)
        except ValueError as exc:
            say('skipped: %s' % exc)
            continue
        if not any(len(ctx.segs[c]) for c in MUSIC):
            say('skipped: no audible notes')
            continue
        ctxs.append(ctx)
        names.append(name)
    if not ctxs:
        raise SystemExit('nothing to convert')
    say('\n=== Instruments shared by %d track%s ===' % (len(ctxs), '' if len(ctxs) == 1 else 's'))
    instruments, macros = choose_shared_instruments(ctxs, max_inst, say)
    outputs, reports = [], []
    for ctx, name in zip(ctxs, names):
        say('\n=== %s ===' % name)
        out, rep = finish(ctx, instruments, macros, pattern_rows)
        rep['instruments'] = len(instruments)
        rep['speed'] = ctx.s
        outputs.append(out)
        reports.append(rep)
    return outputs, reports, names


def score_reports(reports):
    """one number for how close the retrace sounds to the game: the share of audible frames
    that are fully right (note, volume, pitch, duty), over every channel of every track, plus
    DPCM hits on their exact tick"""
    good = total = 0.0
    for rep in reports:
        for c in MUSIC:
            r = rep.get(c)
            if r and r['audible']:
                good += r['good'] / 100.0 * r['audible']
                total += r['audible']
        if rep.get('dpcm_hits'):
            good += rep['dpcm_exact']
            total += rep['dpcm_hits']
    return 100.0 * good / max(total, 1.0)


def search_settings(input_path, subsongs, track_names, settings, max_inst, pattern_rows=None,
                    keep_all=False):
    """try alternatives for one setting at a time, play each version back with the simulator,
    and keep a change only when it makes the retrace closer to the game.  Ties (within 0.05%)
    go to the simpler module: fewer instruments, then a coarser row grid."""
    import contextlib, io

    def evaluate(st):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            try:
                _, reps, _ = convert_all(input_path, subsongs, track_names, st, max_inst,
                                         pattern_rows, keep_all, verbose=False)
            except (ValueError, SystemExit) as exc:
                return None
        return (score_reports(reps), -max(r['instruments'] for r in reps),
                min(r['speed'] for r in reps))

    def better(a, b):
        if b is None:
            return a is not None
        if a is None:
            return False
        if abs(a[0] - b[0]) > 0.05:
            return a[0] > b[0]
        return a[1:] > b[1:]

    best = dict(settings)
    best_val = evaluate(best)
    if best_val is None:
        return settings
    auto_speed = best_val[2]
    print('\n=== Searching for better settings (each version is played back and scored) ===')
    print('start: %.2f%% of audible frames fully right (speed %d, tolerance %g, pitch tolerance %g)'
          % (best_val[0], auto_speed, best['tolerance'], best['pitch_tol']))
    options = [
        ('speed', sorted({auto_speed, 2 * auto_speed} | ({auto_speed // 2} if auto_speed % 2 == 0 and auto_speed >= 4 else set()))),
        ('tolerance', [0.0, 0.05, 0.1, 0.2, 0.3, 0.45]),
        ('pitch_tol', [4, 6, 9, 13, 18]),
    ]
    tried = 0
    for _round in range(2):
        changed = False
        for key, values in options:
            for v in values:
                cand = dict(best, **{key: v})
                if key == 'speed' and v == auto_speed and best['speed'] is None:
                    continue
                if cand == best:
                    continue
                val = evaluate(cand)
                tried += 1
                if better(val, best_val):
                    print('  %-10s %-6s -> %.2f%% (was %.2f%%)' % (key, v, val[0], best_val[0]))
                    best, best_val, changed = cand, val, True
        if not changed:
            break
    print('tried %d versions; best: %.2f%% fully right (speed %s, tolerance %g, pitch tolerance %g)'
          % (tried, best_val[0], best['speed'] or auto_speed, best['tolerance'], best['pitch_tol']))
    return best


def main(argv=None):
    global PITCH_TOL, FACILITY_COST
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('input')
    ap.add_argument('-o', '--output', default=None,
                    help='output path (default: retrace.ftm, or retrace.txt with --format txt)')
    ap.add_argument('--format', choices=('ftm', 'txt'), default=None,
                    help='output format: native binary FTM by default, or FamiTracker text with --format txt')
    ap.add_argument('--txt', action='store_true',
                    help='shortcut for --format txt')
    ap.add_argument('--famitracker-exe', default=None, metavar='PATH',
                    help='optional FamiTracker executable that supports: <txt> -import <ftm>; '
                         'use it to generate the binary FTM with that application')
    ap.add_argument('--speed', type=int, help='frames per row (default: auto-detect)')
    ap.add_argument('--pattern-rows', type=int, help='rows per pattern (default: auto)')
    ap.add_argument('--max-instruments', type=int, default=None,
                    help='most instruments to create (default: as many as FamiTracker allows, 64 '
                         'including one blank per sound chip). Lower = fewer, more general '
                         'instruments and a less exact result')
    ap.add_argument('--tolerance', type=float, default=0.1,
                    help='how imprecise an instrument may be: average weighted error per frame. '
                         '0 = exact (many instruments), higher = fewer, rougher instruments (default 0.1)')
    ap.add_argument('--pitch-tol', type=float, default=PITCH_TOL,
                    help='pitch wobble this small (peak to peak, in cents) is ignored (default %g)' % PITCH_TOL)
    ap.add_argument('--instrument-cost', type=float, default=FACILITY_COST,
                    help='how reluctant to create a new instrument (default %g). Higher = fewer, '
                         'more general instruments and less exact notes; 0 = as many as needed' % FACILITY_COST)
    ap.add_argument('--keep-all', action='store_true', help='do not trim to one loop')
    ap.add_argument('--search', action='store_true',
                    help='try other settings (row grid, instrument tolerance, pitch tolerance), play '
                         'each version back, and keep whichever is closest to the game (slower)')
    selection = ap.add_mutually_exclusive_group()
    selection.add_argument('--subsong', type=int, default=None,
                           help='convert only this zero-based subsong/track index (default: all subsongs)')
    selection.add_argument('--subsong-range', type=int, nargs=2, metavar=('START', 'END'),
                           help='convert an inclusive range of zero-based subsong indices, e.g. --subsong-range 2 5')
    args = ap.parse_args(argv)
    if args.txt and args.format == 'ftm':
        ap.error('--txt cannot be combined with --format ftm')
    output_format = args.format or ('txt' if args.txt else 'ftm')
    output_path = args.output or ('retrace.txt' if output_format == 'txt' else 'retrace.ftm')
    PITCH_TOL = args.pitch_tol
    FACILITY_COST = args.instrument_cost

    probe = load_input(args.input, subsong=0)
    num_subsongs = max(1, int(getattr(probe, 'num_subsongs', 1)))
    track_names = list(getattr(probe, 'track_names', []))
    if len(track_names) < num_subsongs:
        track_names.extend('Track %d' % (i + 1) for i in range(len(track_names), num_subsongs))
    if args.subsong is not None:
        if not 0 <= args.subsong < num_subsongs:
            raise SystemExit('Subsong %d not found (file has %d subsongs)' % (args.subsong, num_subsongs))
        subsongs = [args.subsong]
    elif args.subsong_range is not None:
        start, end = args.subsong_range
        if start < 0 or end < 0 or start > end or end >= num_subsongs:
            raise SystemExit('Invalid subsong range %d..%d (file has %d subsongs; indices are 0..%d)' %
                             (start, end, num_subsongs, num_subsongs - 1))
        subsongs = list(range(start, end + 1))
    else:
        subsongs = list(range(num_subsongs))

    # The instrument table is shared by all tracks of a module (FamiTracker caps it at 64), so
    # the tracks are analysed first and one set of instruments is chosen for all of them.
    chips_in_module = 2 if getattr(probe, 'expansion', 0) == 1 else 1
    if args.max_instruments is None:
        max_inst = 64 - chips_in_module
    else:
        max_inst = min(args.max_instruments, 64 - chips_in_module)

    settings = dict(speed=args.speed, tolerance=args.tolerance, pitch_tol=args.pitch_tol,
                    instrument_cost=args.instrument_cost)
    if args.search:
        settings = search_settings(args.input, subsongs, track_names, settings, max_inst,
                                   args.pattern_rows, args.keep_all)
    outputs, reports, output_names = convert_all(args.input, subsongs, track_names, settings, max_inst,
                                                 args.pattern_rows, args.keep_all, verbose=True)
    for rep, name in zip(reports, output_names):
        print('\n%s - playback check vs raw (%% of audible frames; "good" = note+volume+pitch+duty all right):'
              % name)
        print('%-5s %8s %7s %7s %7s %7s %7s %7s' %
              ('chan', 'audible', 'good', 'on/off', 'note', 'vol+-1', 'fine', 'duty'))
        for c in MUSIC:
            r = rep[c]
            if r['audible'] == 0:
                continue
            print('%-5s %8d %7.1f %7.1f %7.1f %7.1f %7.1f %7.1f' %
                  (CH_NAMES[c], r['audible'], r['good'], r['onoff'], r['pitch'], r['vol'], r['fine'], r['duty']))

    exotic = getattr(probe, 'expansion', 0) & ~1
    try:
        if len(outputs) == 1:
            out = outputs[0]
        elif exotic:
            raise ValueError('several tracks with N163 / FDS / VRC7 / MMC5 / 5B chips cannot be merged '
                             'into one module yet: convert one at a time with --subsong N')
        else:
            out = combine_retraces(outputs, output_names)
    except ValueError as exc:
        raise SystemExit(str(exc))
    if exotic and output_format == 'ftm':
        # the binary .ftm writer only knows the 2A03 and VRC6; text works for every chip
        output_format = 'txt'
        if output_path.lower().endswith('.ftm'):
            output_path = output_path[:-4] + '.txt'
        print('this module uses expansion chips the .ftm writer does not support: writing FamiTracker '
              'text instead (%s) - open it with File > Import Text' % output_path)
    if output_format == 'txt':
        out = pad_n163_text(out, getattr(probe, 'expansion', 0), getattr(probe, 'n163', None))
    problems = validate_text(out)
    if problems:
        print('\nFORMAT CHECK: %d problem(s) FamiTracker may reject:' % len(problems))
        for p in problems[:20]:
            print('  ', p)
        return 1

    if output_format == 'txt':
        with open(output_path, 'w', encoding='utf-8', newline='') as f:
            f.write(out)
        print('\nwrote %s (%d subsong%s)' %
              (output_path, len(outputs), '' if len(outputs) == 1 else 's'))
        print('text format check: OK (FamiTracker text import should accept this module)')
    else:
        try:
            binary = None
            if args.famitracker_exe:
                # Some forks/custom builds expose text import on the command line.
                # Keep this opt-in because stock FamiTracker builds do not all accept
                # the -import switch.
                exe = os.path.abspath(os.path.expandvars(os.path.expanduser(args.famitracker_exe)))
                if not os.path.isfile(exe):
                    raise ValueError('FamiTracker executable not found: %s' % exe)
                temp_path = None
                try:
                    with tempfile.NamedTemporaryFile('w', encoding='utf-8', newline='',
                                                     suffix='.txt', delete=False) as tmp:
                        tmp.write(out)
                        temp_path = tmp.name
                    proc = subprocess.run([exe, temp_path, '-import', os.path.abspath(output_path)],
                                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                          text=True, timeout=120, check=False)
                    if proc.returncode != 0:
                        detail = (proc.stderr or proc.stdout or '').strip()
                        raise ValueError('FamiTracker -import failed (exit %d)%s' %
                                         (proc.returncode, ': ' + detail if detail else ''))
                    with open(output_path, 'rb') as f:
                        binary = f.read()
                    if len(binary) < 22 or binary[:18] != b'FamiTracker Module':
                        raise ValueError('FamiTracker did not create a valid .ftm file; check that this build supports -import')
                finally:
                    if temp_path and os.path.exists(temp_path):
                        os.remove(temp_path)
            else:
                binary = write_ftm(out)
            # Validate block bounds and every track's decoded frames/patterns.
            for track_index in range(len(outputs)):
                check = parse_ftm(binary, subsong=track_index)
                if int(getattr(check, 'num_subsongs', 0)) != len(outputs):
                    raise ValueError('FTM track-count check failed')
                if len(check.orders) == 0 or check.track is None:
                    raise ValueError('FTM track/frame round-trip check failed at track %d' % track_index)
        except (ValueError, struct.error, IndexError, OSError, subprocess.SubprocessError) as exc:
            print('\nFTM export failed: %s' % exc, file=sys.stderr)
            return 1
        with open(output_path, 'wb') as f:
            f.write(binary)
        print('\nwrote %s (%d subsong%s, %d bytes)' %
              (output_path, len(outputs), '' if len(outputs) == 1 else 's', len(binary)))
        if args.famitracker_exe:
            print('FTM export used the configured FamiTracker executable.')
        else:
            print('FTM structure check: OK (header, tracks, frames, patterns, and block bounds)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
