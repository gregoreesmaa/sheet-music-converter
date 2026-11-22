"""Lossless bekern <-> event-tuple compiler (S18 step-1 gate).

One tuple = one atomic visual event. ``bekern_to_tuples`` parses a token
list; ``tuples_to_bekern`` rebuilds it in canonical emitter order, so the
round-trip is byte-exact on decode-validated data. Anything outside the
175-token census raises ``ValueError`` (fail loud, never invent).
"""

STEP_IDX = {'C': 0, 'D': 1, 'E': 2, 'F': 3, 'G': 4, 'A': 5, 'B': 6}
NO_PITCH = 63

DURS = {'whole', 'half', 'quarter', 'eighth', '16th', '32nd', '64th',
        '128th', 'breve'}
ALTERS = {'a-2', 'a-1', 'a1', 'a2'}
ARTICS = {'stacc', 'accent', 'tenuto', 'fermata'}
DYNS = {'dyn-pp', 'dyn-p', 'dyn-mp', 'dyn-mf', 'dyn-f', 'dyn-ff', 'dyn-sfz'}
HAIRPINS = {'hairpin-cresc-start', 'hairpin-dim-start', 'hairpin-stop'}
BAR_VARIANTS = {'bar-double': 'double', 'bar-light-heavy': 'light-heavy',
                'bar-repeat-left': 'repeat-left',
                'bar-repeat-right': 'repeat-right'}


def diatonic_index(step, octv):
    """C0 -> 0 .. B8 -> 62."""
    return int(octv) * 7 + STEP_IDX[step]


def bekern_to_tuples(tokens):
    evs = []
    i, n = 0, len(tokens)
    pending_voice = None

    def peek():
        return tokens[i] if i < n else None

    def mods():
        # Ordered verbatim mod tokens (rebuild uses these: the emitter has
        # no fixed mod order, e.g. 'slur-stop tie-start' occurs) plus
        # parsed facets (first occurrence wins) for S18 heads.
        m = {'mods': [], 'alter': None, 'dots': 0, 'ties': [],
             'slurs': [], 'articulations': [], 'dyn': None,
             'hairpins': []}
        nonlocal i
        while i < n:
            t = tokens[i]
            if t in ALTERS:
                if m['alter'] is None:
                    m['alter'] = t
            elif t == 'dot':
                m['dots'] += 1
            elif t in ('tie-start', 'tie-stop'):
                m['ties'].append(t[4:])
            elif t in ('slur-start', 'slur-stop'):
                m['slurs'].append(t[5:])
            elif t in ARTICS:
                m['articulations'].append(t)
            elif t in DYNS:
                if m['dyn'] is None:
                    m['dyn'] = t
            elif t in HAIRPINS:
                m['hairpins'].append(t)
            else:
                break
            m['mods'].append(t)
            i += 1
        return m

    while i < n:
        t = tokens[i]
        if t.startswith('voice-'):
            pending_voice = int(t[6:])
            evs.append({'kind': 'VOICE', 'n': pending_voice,
                        'diatonic_pitch': NO_PITCH})
            i += 1
        elif t == 'tuplet-stop':
            evs.append({'kind': 'TUPLET_STOP', 'diatonic_pitch': NO_PITCH})
            i += 1
        elif t.startswith('tuplet-start-'):
            evs.append({'kind': 'TUPLET_START',
                        'ratio': t[len('tuplet-start-'):],
                        'diatonic_pitch': NO_PITCH})
            i += 1
        elif t == 'bar' or t in BAR_VARIANTS:
            evs.append({'kind': 'BAR',
                        'variant': BAR_VARIANTS.get(t),
                        'diatonic_pitch': NO_PITCH})
            i += 1
        elif (t.startswith('clef-') or t.startswith('key')
              or t.startswith('time-')):
            evs.append({'kind': 'HEADER', 'token': t,
                        'diatonic_pitch': NO_PITCH})
            i += 1
        elif t in DYNS:
            evs.append({'kind': 'DYN', 'mark': t,
                        'diatonic_pitch': NO_PITCH})
            i += 1
        elif t in HAIRPINS:
            evs.append({'kind': 'HAIRPIN', 'mark': t,
                        'diatonic_pitch': NO_PITCH})
            i += 1
        elif t in ARTICS:
            evs.append({'kind': 'ARTIC', 'mark': t,
                        'diatonic_pitch': NO_PITCH})
            i += 1
        elif t in ('tie-start', 'tie-stop'):
            evs.append({'kind': 'TIE', 'mark': t[4:],
                        'diatonic_pitch': NO_PITCH})
            i += 1
        elif t in ('slur-start', 'slur-stop'):
            evs.append({'kind': 'SLUR', 'mark': t[5:],
                        'diatonic_pitch': NO_PITCH})
            i += 1
        else:
            chord, grace = False, False
            if t == 'chord':
                chord, i = True, i + 1
                t = peek()
            if t == 'grace':
                grace, i = True, i + 1
                t = peek()
            if t == 'bar-rest':
                voice, pending_voice = pending_voice, None
                evs.append({'kind': 'BAR_REST', 'voice': voice,
                            'diatonic_pitch': NO_PITCH})
                i += 1
            elif t == 'rest':
                i += 1
                dur = tokens[i]
                if dur not in DURS:
                    raise ValueError('bad rest dur %r' % dur)
                i += 1
                m = mods()
                voice, pending_voice = pending_voice, None
                evs.append({'kind': 'REST', 'voice': voice, 'dur': dur,
                            'diatonic_pitch': NO_PITCH, **m})
            elif t in DURS:
                dur, i = t, i + 1
                step, octv = tokens[i], tokens[i + 1]
                if step not in STEP_IDX or not (len(octv) == 1
                                                and octv.isdigit()):
                    raise ValueError('bad pitch %r %r' % (step, octv))
                i += 2
                m = mods()
                if chord:
                    kind, voice = 'CHORD_NOTE', None
                else:
                    kind = 'GRACE_NOTE' if grace else 'NOTE'
                    voice, pending_voice = pending_voice, None
                evs.append({'kind': kind, 'voice': voice, 'dur': dur,
                            'step': step, 'oct': int(octv),
                            'diatonic_pitch': diatonic_index(step, octv),
                            'grace': grace, **m})
            else:
                raise ValueError('unknown token %r' % t)
    return evs


def tuples_to_bekern(evs):
    out = []
    for e in evs:
        k = e['kind']
        if k == 'VOICE':
            out.append('voice-%d' % e['n'])
        elif k == 'TUPLET_STOP':
            out.append('tuplet-stop')
        elif k == 'TUPLET_START':
            out.append('tuplet-start-' + e['ratio'])
        elif k == 'BAR':
            out.append('bar' if e['variant'] is None
                       else 'bar-' + e['variant'])
        elif k == 'HEADER':
            out.append(e['token'])
        elif k == 'DYN':
            out.append(e['mark'])
        elif k == 'HAIRPIN':
            out.append(e['mark'])
        elif k == 'ARTIC':
            out.append(e['mark'])
        elif k == 'TIE':
            out.append('tie-' + e['mark'])
        elif k == 'SLUR':
            out.append('slur-' + e['mark'])
        elif k == 'BAR_REST':
            out.append('bar-rest')
        elif k in ('NOTE', 'CHORD_NOTE', 'GRACE_NOTE', 'REST'):
            if k == 'CHORD_NOTE':
                out.append('chord')
            if k == 'GRACE_NOTE':
                out.append('grace')
            if k == 'REST':
                out.append('rest')
                out.append(e['dur'])
            else:
                out.extend([e['dur'], e['step'], str(e['oct'])])
            out.extend(e['mods'])
        else:
            raise ValueError('unknown event kind %r' % k)
    return out
