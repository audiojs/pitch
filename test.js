import test, { almost, ok, is, throws } from 'tst'
import { execFileSync } from 'node:child_process'
import { yin, mcleod, autocorrelation, amdf, hps, cepstrum, swipe, pyin } from './index.js'
import { track, notes } from '@audio/pitch-pyin'

let fs = 44100

// --- signal generators ---

function sine(freq, n, sampleRate = fs) {
  let d = new Float32Array(n)
  for (let i = 0; i < n; i++) d[i] = Math.sin(2 * Math.PI * freq * i / sampleRate)
  return d
}

// frequency-modulated sine: instantaneous pitch varies ±depth Hz around baseFreq
function vibrato(baseFreq, depth, modFreq, n, sampleRate = fs) {
  let d = new Float32Array(n), phase = 0
  for (let i = 0; i < n; i++) {
    let f = baseFreq + depth * Math.sin(2 * Math.PI * modFreq * i / sampleRate)
    d[i] = Math.sin(phase)
    phase += 2 * Math.PI * f / sampleRate
  }
  return d
}

// two equal-amplitude sines at f1 and f2
function twosines(f1, f2, n, sampleRate = fs) {
  let d = new Float32Array(n)
  for (let i = 0; i < n; i++)
    d[i] = 0.5 * Math.sin(2 * Math.PI * f1 * i / sampleRate)
           + 0.5 * Math.sin(2 * Math.PI * f2 * i / sampleRate)
  return d
}

// Decaying harmonic pluck after a short onset delay. This is harder than a
// stationary sine at low frequencies because the first few periods differ.
function pluck(freq, n, sampleRate, onset = 384) {
  let d = new Float32Array(n)
  for (let i = onset; i < n; i++) {
    let t = (i - onset) / sampleRate
    let attack = 1 - Math.exp(-t * 250)
    for (let h = 1; h <= 10; h++)
      d[i] += attack * Math.exp(-t * (2 + h * 0.8)) *
        Math.sin(2 * Math.PI * h * freq * t + 0.1 * h * h) / Math.pow(h, 0.7)
  }
  return d
}

function silence(n) { return new Float32Array(n) }

// band-limited sawtooth: sum of H harmonics with 1/h amplitude (simulates a pitched instrument)
function saw(freq, n, harmonics = 10, sampleRate = fs) {
  let d = new Float32Array(n)
  for (let i = 0; i < n; i++) {
    let s = 0
    for (let h = 1; h <= harmonics; h++) {
      if (h * freq > sampleRate / 2) break
      s += Math.sin(2 * Math.PI * h * freq * i / sampleRate) / h
    }
    d[i] = s
  }
  return d
}

// chord synthesized as a sum of sawtooths at MIDI pitches
function synthChord(midiNotes, n, harmonics = 6, sampleRate = fs) {
  let d = new Float32Array(n)
  for (let m of midiNotes) {
    let f0 = 440 * Math.pow(2, (m - 69) / 12)
    for (let i = 0; i < n; i++) {
      for (let h = 1; h <= harmonics; h++) {
        if (h * f0 > sampleRate / 2) break
        d[i] += Math.sin(2 * Math.PI * h * f0 * i / sampleRate) / h
      }
    }
  }
  return d
}

// deterministic low-correlation noise: sum of 16 inharmonic sines at irrational ratios
function noise(n, sampleRate = fs) {
  let d = new Float32Array(n)
  let freqs = [317, 641, 1013, 1499, 2003, 2749, 3571, 4201, 5003, 6007, 7109, 8221, 9337, 10613, 11903, 13001]
  for (let i = 0; i < n; i++) {
    let v = 0
    for (let f of freqs) v += Math.sin(2 * Math.PI * f * i / sampleRate)
    d[i] = v / freqs.length
  }
  return d
}

// deterministic white noise via Box–Muller over a seeded LCG
function whiteNoise(n, seed = 42) {
  let d = new Float32Array(n)
  let s = seed
  let rand = () => { s = (s * 1103515245 + 12345) & 0x7fffffff; return s / 0x7fffffff }
  for (let i = 0; i < n; i += 2) {
    let u1 = Math.max(1e-12, rand()), u2 = rand()
    let r = Math.sqrt(-2 * Math.log(u1))
    d[i] = r * Math.cos(2 * Math.PI * u2)
    if (i + 1 < n) d[i + 1] = r * Math.sin(2 * Math.PI * u2)
  }
  return d
}

// =============================================================================
// YIN
// =============================================================================

test('yin — 440 Hz sine', () => {
  let r = yin(sine(440, 2048), { fs })
  ok(r, 'detects pitch')
  almost(r.freq, 440, 1)
  ok(r.clarity > 0.8, 'high clarity')
})

test('yin — 100 Hz sine', () => {
  let r = yin(sine(100, 4096), { fs })
  ok(r, 'detects low pitch')
  almost(r.freq, 100, 1)
})

test('yin — bounded range detects a low E pluck in a short window', () => {
  let sampleRate = 48000
  let r = yin(pluck(80.91, 4096, sampleRate), { fs: sampleRate, minFreq: 60, maxFreq: 520 })
  ok(r, 'detects low E after its onset')
  almost(r.freq, 80.91, 0.5)
})

test('yin — bounded range rejects out-of-range pitches', () => {
  is(yin(sine(40, 4096), { fs, minFreq: 60 }), null)
  is(yin(sine(880, 4096), { fs, maxFreq: 520 }), null)
})

test('yin — 220 Hz sine', () => {
  let r = yin(sine(220, 2048), { fs })
  ok(r, 'detects pitch')
  almost(r.freq, 220, 1)
})

test('yin — 1000 Hz sine', () => {
  let r = yin(sine(1000, 2048), { fs })
  ok(r, 'detects high pitch')
  almost(r.freq, 1000, 2)
})

test('yin — vibrato 440±20 Hz', () => {
  // single window captures one snapshot of the modulated pitch
  let r = yin(vibrato(440, 20, 5, 2048), { fs })
  ok(r, 'detects vibrato pitch')
  ok(r.freq >= 410 && r.freq <= 470, `freq ${r.freq.toFixed(1)} in vibrato range [410, 470]`)
})

test('yin — silence returns null', () => {
  is(yin(silence(2048), { fs }), null)
})

test('yin — noise returns null or low clarity', () => {
  let r = yin(noise(2048), { fs })
  ok(!r || r.clarity < 0.5, 'noise: no confident pitch')
})

test('yin manifest: emits pitch and clarity inside the ranges it declares, as YIN reads each frame', async () => {
  let { pitch: atom } = await import('./packages/pitch-yin/audio.js')
  let { pitch: P, clarity: C } = atom.events.out
  is(P.unit, 'Hz', 'pitch in Hz')
  // a stub ctx collects what the manifest emits, its input fed in blocks of 128
  let run = (x, sampleRate, threshold) => {
    let got = { pitch: [], clarity: [] }
    let proc = atom({ sampleRate, maxBlockSize: 128, maxChannels: 1, emit: (name, v) => got[name].push(v) })
    for (let i = 0; i < x.length; i += 128) proc([[x.subarray(i, i + 128)]], [], { threshold: Float32Array.of(threshold) })
    return got
  }
  // YIN itself on the same 2048-sample frames, kept where its pitch is inside the declared range
  let direct = (x, sampleRate, threshold) => {
    let want = { pitch: [], clarity: [] }
    for (let i = 0; i + 2048 <= x.length; i += 2048) {
      let r = yin(x.subarray(i, i + 2048), { fs: sampleRate, threshold: Math.fround(threshold) })
      if (r && r.freq >= P.min && r.freq <= P.max) want.pitch.push(r.freq), want.clarity.push(r.clarity)
    }
    return want
  }
  let n = 2048 * 4
  let pitched = { 'sine 100': [sine(100, n), fs], 'sine 440': [sine(440, n), fs], 'sine 1000': [sine(1000, n), fs], 'vibrato 440±20': [vibrato(440, 20, 5, n), fs], 'sines 220+440': [twosines(220, 440, n), fs], 'saw 110': [saw(110, n), fs], 'saw 880': [saw(880, n), fs], 'pluck 80.91 at 48 kHz': [pluck(80.91, n, 48000), 48000] }
  let unpitched = { noise: [noise(n), fs], 'white noise': [whiteNoise(n), fs], silence: [silence(n), fs] }
  // YIN reads a 21 kHz sine at 44.1 kHz as 23152 Hz, above Nyquist, and a 15 Hz sine at 8 kHz as 15 Hz: neither is emitted
  let outside = { 'sine 21000': [sine(21000, n), fs], 'sine 15 at 8 kHz': [sine(15, n, 8000), 8000] }
  let { min, max, default: def } = atom.params.threshold
  for (let threshold of [min, def, max]) {
    for (let [name, [x, sr]] of Object.entries({ ...pitched, ...unpitched, ...outside })) {
      let got = run(x, sr, threshold)
      ok(got.pitch.every(v => v >= P.min && v <= P.max) && got.clarity.every(v => v >= C.min && v <= C.max), `${name}, threshold ${threshold}: inside ${P.min}..${P.max} Hz, clarity ${C.min}..${C.max}`)
      is(got, direct(x, sr, threshold), `${name}, threshold ${threshold}: YIN's own reading of each frame`)
    }
    for (let [name, [x, sr]] of Object.entries(pitched)) ok(run(x, sr, threshold).pitch.length > 0, `${name}, threshold ${threshold}: emits a pitch`)
    for (let [name, [x, sr]] of Object.entries(outside)) {
      let r = yin(x.subarray(0, 2048), { fs: sr, threshold: Math.fround(threshold) })
      ok(r && (r.freq < P.min || r.freq > P.max) && !run(x, sr, threshold).pitch.length, `${name}, threshold ${threshold}: YIN reads ${r?.freq.toFixed(0)} Hz, not emitted`)
    }
  }
})

test('yin manifest: a block allocates nothing, a frame YIN reads included', () => {
  // A child Node runs the manifest 2^17 blocks of 8 frames at 44.1 kHz (512 frames analyzed) after 2^15 warm ones, and
  // reads the array-buffer stores and the young generation's used size before and after, with a 64 MB semi-space so no
  // collection falls inside (checked), twice; the flags are @audio/compile's validator's. The reads themselves leave
  // 2 to 5 KB in the young generation, so a processor that does nothing runs beside it and its bytes are taken off.
  // Inputs: noise (no pitch), and a 21 kHz sine that YIN reads as 23152 Hz (the whole analysis, a pitch found, none
  // emitted: an emitted number is a double handed to the host's emit, which boxes it where the engine does not inline
  // that). The line is the validator's, 3/64 of a byte a block (6144 bytes a run), over the smaller of the two runs; a
  // frame's two scratch arrays, made anew, are 2 × 8 KB of stores
  let code = `
    import v8 from 'node:v8'
    import { PerformanceObserver } from 'node:perf_hooks'
    let { pitch } = await import('@audio/pitch-yin/audio')
    let young = () => v8.getHeapSpaceStatistics().find(s => s.space_name === 'new_space').space_used_size
    let n = 1 << 16, B = n / 8, s = 3, out = {}
    let inputs = { noise: Float32Array.from({ length: n }, () => 0.15 * ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 2147483648 - 1)), 'sine 21 kHz': Float32Array.from({ length: n }, (_, i) => 0.5 * Math.sin(2 * Math.PI * 21000 * i / 44100)) }
    for (let [name, x] of [['nothing', inputs.noise], ...Object.entries(inputs)]) {
      let views = Array.from({ length: B }, (_, k) => [[x.subarray(k * 8, k * 8 + 8)]]), O = [], params = { threshold: Float32Array.of(0.1) }, emitted = new Int32Array(1)
      let proc = name === 'nothing' ? function nothing (i, o, p) {} : pitch({ sampleRate: 44100, maxBlockSize: 128, maxChannels: 1, params, emit: function emit () { emitted[0]++ } })
      for (let i = 0; i < 1 << 15; i++) proc(views[i % B], O, params)
      out[name] = []
      for (let r = 0; r < 2; r++) {
        globalThis.gc()
        let gcs = 0, obs = new PerformanceObserver(l => { gcs += l.getEntries().length })
        obs.observe({ entryTypes: ['gc'] })
        let e0 = process.memoryUsage().arrayBuffers, y0 = young()
        for (let i = 0; i < 1 << 17; i++) proc(views[i % B], O, params)
        let y1 = young(), e1 = process.memoryUsage().arrayBuffers
        await new Promise(r => setTimeout(r, 10))
        obs.disconnect()
        out[name].push({ young: y1 - y0, stores: e1 - e0, gcs, emitted: emitted[0] })
      }
    }
    process.stdout.write(JSON.stringify(out))`
  let flags = ['--expose-gc', '--no-concurrent-recompilation', '--no-allocation-site-pretenuring', '--no-concurrent-array-buffer-sweeping', '--min-semi-space-size=64', '--max-semi-space-size=64']
  let { nothing, ...res } = JSON.parse(execFileSync(process.execPath, [...flags, '--input-type=module', '-e', code], { cwd: new URL('.', import.meta.url).pathname, encoding: 'utf8', timeout: 120000 }))
  let floor = Math.min(...nothing.map(r => r.young + r.stores))
  for (let [name, runs] of Object.entries(res)) {
    is([...nothing, ...runs].map(r => r.gcs), [0, 0, 0, 0], `${name}: no collection inside the runs`)
    is(runs[1].emitted, 0, `${name}: nothing emitted`)
    let least = runs.reduce((a, r) => r.young + r.stores < a.young + a.stores ? r : a), bytes = least.young + least.stores - floor
    ok(bytes <= 6144, `${name}: ${bytes} bytes in 2^17 blocks beyond a processor that does nothing (${least.young} young, ${least.stores} of stores, against ${floor}; ≤ 6144)`)
  }
}, { timeout: 120000 })


// =============================================================================
// McLeod
// =============================================================================

test('mcleod — 440 Hz sine', () => {
  let r = mcleod(sine(440, 2048), { fs })
  ok(r, 'detects pitch')
  almost(r.freq, 440, 1)
  ok(r.clarity > 0.8, 'high clarity')
})

test('mcleod — 100 Hz sine', () => {
  let r = mcleod(sine(100, 4096), { fs })
  ok(r, 'detects low pitch')
  almost(r.freq, 100, 1)
})

test('mcleod — 220 Hz sine', () => {
  let r = mcleod(sine(220, 2048), { fs })
  ok(r, 'detects pitch')
  almost(r.freq, 220, 1)
})

test('mcleod — 1000 Hz sine', () => {
  let r = mcleod(sine(1000, 2048), { fs })
  ok(r, 'detects high pitch')
  almost(r.freq, 1000, 2)
})

test('mcleod — vibrato 440±20 Hz', () => {
  let r = mcleod(vibrato(440, 20, 5, 2048), { fs })
  ok(r, 'detects vibrato pitch')
  ok(r.freq >= 410 && r.freq <= 470, `freq ${r.freq.toFixed(1)} in vibrato range`)
})

test('mcleod — silence returns null', () => {
  is(mcleod(silence(2048), { fs }), null)
})

test('mcleod — noise returns null', () => {
  let r = mcleod(noise(2048), { fs })
  ok(!r || r.clarity < 0.9, 'noise: no confident pitch')
})

test('mcleod — prefers lower octave for two sines 220+440', () => {
  // MPM tends to return the first qualifying peak — typically the lower pitch
  let r = mcleod(twosines(220, 440, 2048), { fs })
  ok(r, 'returns a result')
  ok(r.freq >= 200 && r.freq <= 460, `freq ${r.freq.toFixed(1)} is one of the two pitches`)
})


// =============================================================================
// Autocorrelation
// =============================================================================

test('autocorrelation — 440 Hz sine', () => {
  let r = autocorrelation(sine(440, 2048), { fs })
  ok(r, 'detects pitch')
  almost(r.freq, 440, 1)
})

test('autocorrelation — 100 Hz sine', () => {
  let r = autocorrelation(sine(100, 4096), { fs })
  ok(r, 'detects low pitch')
  almost(r.freq, 100, 1)
})

test('autocorrelation — 220 Hz sine', () => {
  let r = autocorrelation(sine(220, 2048), { fs })
  ok(r, 'detects pitch')
  almost(r.freq, 220, 1)
})

test('autocorrelation — silence returns null', () => {
  is(autocorrelation(silence(2048), { fs }), null)
})

test('autocorrelation — noise returns null or low clarity', () => {
  let r = autocorrelation(noise(2048), { fs })
  ok(!r || r.clarity < 0.7, 'noise: no confident pitch')
})


// =============================================================================
// AMDF
// =============================================================================

test('amdf — 440 Hz sine', () => {
  let r = amdf(sine(440, 2048), { fs })
  ok(r, 'detects pitch')
  almost(r.freq, 440, 1)
})

test('amdf — 100 Hz sine', () => {
  let r = amdf(sine(100, 4096), { fs })
  ok(r, 'detects low pitch')
  almost(r.freq, 100, 1)
})

test('amdf — 220 Hz sine', () => {
  let r = amdf(sine(220, 2048), { fs })
  ok(r, 'detects pitch')
  almost(r.freq, 220, 1)
})

test('amdf — silence returns null', () => {
  is(amdf(silence(2048), { fs }), null)
})

test('amdf — noise returns null or low clarity', () => {
  let r = amdf(noise(2048), { fs })
  ok(!r || r.clarity < 0.5, 'noise: no confident pitch')
})


// =============================================================================
// HPS — harmonic signals only
// =============================================================================

test('hps — 440 Hz sawtooth', () => {
  let r = hps(saw(440, 4096), { fs })
  ok(r, 'detects pitch')
  almost(r.freq, 440, 2)
})

test('hps — 220 Hz sawtooth', () => {
  let r = hps(saw(220, 4096), { fs })
  ok(r, 'detects pitch')
  almost(r.freq, 220, 2)
})

test('hps — 110 Hz sawtooth', () => {
  let r = hps(saw(110, 8192), { fs })
  ok(r, 'detects pitch')
  almost(r.freq, 110, 2)
})

test('hps — 880 Hz sawtooth', () => {
  let r = hps(saw(880, 4096), { fs })
  ok(r, 'detects pitch')
  almost(r.freq, 880, 2)
})

test('hps — silence returns null', () => {
  is(hps(silence(4096), { fs }), null)
})

test('hps — rejects white noise', () => {
  let r = hps(whiteNoise(4096), { fs })
  ok(!r || r.clarity < 0.9, `noise: no confident pitch, got ${r ? r.clarity.toFixed(2) : 'null'}`)
})


// =============================================================================
// Cepstrum — harmonic signals only
// =============================================================================

test('cepstrum — 440 Hz sawtooth', () => {
  let r = cepstrum(saw(440, 4096), { fs })
  ok(r, 'detects pitch')
  almost(r.freq, 440, 3)
})

test('cepstrum — 220 Hz sawtooth', () => {
  let r = cepstrum(saw(220, 4096), { fs })
  ok(r, 'detects pitch')
  almost(r.freq, 220, 3)
})

test('cepstrum — 110 Hz sawtooth', () => {
  let r = cepstrum(saw(110, 8192), { fs })
  ok(r, 'detects pitch')
  almost(r.freq, 110, 3)
})

test('cepstrum — silence returns null', () => {
  is(cepstrum(silence(4096), { fs }), null)
})

test('cepstrum — rejects inharmonic noise', () => {
  let r = cepstrum(noise(4096), { fs })
  ok(!r || r.clarity < 0.5, 'noise: no confident pitch')
})


// =============================================================================
// SWIPE′
// =============================================================================

test('swipe — 440 Hz sine', () => {
  let r = swipe(sine(440, 4096), { fs })
  ok(r, 'detects pitch')
  almost(r.freq, 440, 2)
})

test('swipe — 100 Hz sine', () => {
  let r = swipe(sine(100, 8192), { fs })
  ok(r, 'detects low pitch')
  almost(r.freq, 100, 2)
})

test('swipe — 440 Hz sawtooth', () => {
  let r = swipe(saw(440, 4096), { fs })
  ok(r, 'detects pitch')
  almost(r.freq, 440, 2)
})

test('swipe — 220 Hz sawtooth', () => {
  let r = swipe(saw(220, 4096), { fs })
  ok(r, 'detects pitch')
  almost(r.freq, 220, 2)
})

test('swipe — silence returns null', () => {
  is(swipe(silence(4096), { fs }), null)
})

test('swipe — rejects white noise', () => {
  let r = swipe(whiteNoise(4096), { fs })
  ok(!r || r.clarity < 0.9, `noise: no confident pitch, got ${r ? r.clarity.toFixed(2) : 'null'}`)
})


// =============================================================================
// pYIN — probabilistic YIN with candidate posterior
// =============================================================================

test('pyin — 440 Hz sine', () => {
  let r = pyin(sine(440, 2048), { fs })
  ok(r, 'detects pitch')
  almost(r.freq, 440, 1)
  ok(Array.isArray(r.candidates) && r.candidates.length >= 1, 'exposes candidate distribution')
  ok(Math.abs(r.candidates.reduce((a, c) => a + c.prob, 0) - 1) < 1e-6, 'candidate probs sum to 1')
})

test('pyin — 100 Hz sine', () => {
  let r = pyin(sine(100, 4096), { fs })
  ok(r, 'detects low pitch')
  almost(r.freq, 100, 1)
})

test('pyin — silence returns null', () => {
  is(pyin(silence(2048), { fs }), null)
})

test('pyin — noise returns null or low clarity', () => {
  let r = pyin(noise(2048), { fs })
  ok(!r || r.clarity < 0.5, 'noise: no confident pitch')
})



// =============================================================================
// pYIN sequence stages: track (pitch HMM, Viterbi) and notes (Tony's note HMM)
// =============================================================================

// harmonic tone following f(t) Hz, phase integrated per sample: the label is f itself
function tone(f, dur, sampleRate = fs) {
  let n = Math.round(dur * sampleRate), d = new Float32Array(n), ph = 0
  for (let i = 0; i < n; i++) {
    d[i] = 0.4 * Math.sin(ph) + 0.2 * Math.sin(2 * ph) + 0.1 * Math.sin(3 * ph)
    ph += 2 * Math.PI * f(i / sampleRate) / sampleRate
  }
  return d
}

// notes rendered independently and overlap-added, linear ADSR with @audio/synth-voice's defaults
// (attack 0.01 s, decay 0.15 s, sustain 0.6, release 0.25 s)
function played(list, dur) {
  let d = new Float32Array(Math.round(dur * fs))
  for (let { time, midi, duration } of list) {
    let f = hz(midi), s0 = Math.round(time * fs), ph = 0
    for (let i = 0; i < Math.round((duration + 0.25) * fs) && s0 + i < d.length; i++) {
      let t = i / fs
      let env = t < 0.01 ? t / 0.01 : t < 0.16 ? 1 - 0.4 * (t - 0.01) / 0.15 : t < duration ? 0.6 : 0.6 * Math.max(0, 1 - (t - duration) / 0.25)
      d[s0 + i] += env * (0.4 * Math.sin(ph) + 0.2 * Math.sin(2 * ph))
      ph += 2 * Math.PI * f / fs
    }
  }
  return d
}

// sawtooth whose every other period drops to 0.4 during 30 ms bursts every 250 ms: the period
// doubles there, so YIN frame by frame reports the octave below
function shimmer(f, dur) {
  let d = new Float32Array(Math.round(dur * fs))
  for (let i = 0; i < d.length; i++) {
    let t = i / fs, p = Math.floor(f * t)
    d[i] = 0.4 * (2 * (f * t % 1) - 1) * (t % 0.25 < 0.03 && p % 2 ? 0.4 : 1)
  }
  return d
}

let hz = m => 440 * 2 ** ((m - 69) / 12)
let cents = (a, b) => 1200 * Math.log2(a / b)
let midiOf = f => 69 + 12 * Math.log2(f / 440)

test('pyin track: a frame per hop, steady tone within 0.1 cent', () => {
  let x = tone(() => 220, 1), r = track(x, { fs })
  is(r.f0.length, 1 + Math.floor(x.length / 256), '1 + ⌊n / hop⌋ frames, hop 256 at 44.1 kHz (as librosa centred frames)')
  almost(r.times[100], 100 * 256 / fs, 1e-12, 'frame i at i·hop')
  let worst = 0
  for (let i = 0; i < r.f0.length; i++) if (r.times[i] > 0.05 && r.times[i] < 0.95) worst = Math.max(worst, Math.abs(cents(r.f0[i], 220)))
  ok(worst < 0.1, `max error ${worst.toFixed(3)} cents`)
  ok(r.voiced.every(v => v === 1), 'every frame voiced')
})

test('pyin track: follows ±50 cent vibrato at 5.5 Hz within 3 cents', () => {
  let f = t => hz(60) * 2 ** (50 / 1200 * Math.sin(2 * Math.PI * 5.5 * t))
  let r = track(tone(f, 2), { fs }), worst = 0
  // YIN compares its window with the one a period later: the estimate sits half a period after
  // the frame time, averaged over the 23 ms window (≈ 2 cents of this modulation)
  for (let i = 0; i < r.f0.length; i++) if (r.times[i] > 0.05 && r.times[i] < 1.95) worst = Math.max(worst, Math.abs(cents(r.f0[i], f(r.times[i] + 0.5 / r.f0[i]))))
  ok(worst < 3, `max error ${worst.toFixed(2)} cents`)
})

test('pyin track: silence around a tone is unvoiced', () => {
  let x = new Float32Array(fs), t = tone(() => 330, 0.4)
  x.set(t, Math.round(0.3 * fs))
  let r = track(x, { fs })
  for (let i = 0; i < r.f0.length; i++) {
    let s = r.times[i]
    if (s < 0.25 || s > 0.75) ok(!r.voiced[i] && r.f0[i] === 0, `frame at ${s.toFixed(3)} s unvoiced`)
    if (s > 0.35 && s < 0.65) ok(r.voiced[i] && Math.abs(cents(r.f0[i], 330)) < 1, `frame at ${s.toFixed(3)} s voiced`)
  }
  let silent = track(new Float32Array(fs), { fs })
  ok(silent.voiced.every(v => !v), 'all-zero input: nothing voiced')
})

test('pyin track: holds the pitch through period-doubling bursts (octave errors)', () => {
  let x = shimmer(220, 1), r = track(x, { fs }), framewise = 0, tracked = 0
  for (let i = 4; i < r.f0.length - 4; i++) {
    let c = x.subarray(i * 256 - 512, i * 256 + 1536), p = c.length === 2048 && pyin(c, { fs })
    if (p && Math.abs(cents(p.freq, 110)) < 50) framewise++
    if (Math.abs(cents(r.f0[i], 220)) > 50) tracked++
  }
  ok(framewise > 5, `frame by frame, ${framewise} frames pick the octave below`)
  is(tracked, 0, 'the track never leaves 220 Hz')
})

test('pyin track: an empty pitch range is an error', () => {
  throws(() => track(new Float32Array(4096), { fs, minFreq: 500, maxFreq: 100 }), RangeError)
  is(track(new Float32Array(0), { fs }).f0.length, 0, 'no samples, no frames')
})

test('pyin track: streaming in odd blocks equals one call', () => {
  let x = tone(t => 200 * 2 ** t, 1.5), a = track(x, { fs }), write = track({ fs }), f0 = []
  for (let i = 0; i < x.length; i += 777) f0.push(...write(x.subarray(i, i + 777)).f0)
  f0.push(...write().f0)
  is(f0.length, a.f0.length)
  ok(f0.every((v, i) => v === a.f0[i]), 'identical frames')
})

// A stream with a fixed latency: every frame decided within `lag` frames of the last one read, whatever the blocks;
// 24 frames (140 ms) still hold the octave through the bursts, 16 did not (5 frames an octave down)
test('pyin track: a short lag decides every frame within it, octaves held', () => {
  let x = shimmer(220, 1), whole = track(x, { fs, lag: 24 }), write = track({ fs, lag: 24 }), f0 = [], late = 0
  for (let i = 0, k = 0; i < x.length; k++) {
    let m = [17, 1000, 4096, 333][k % 4]
    f0.push(...write(x.subarray(i, i += m)).f0)
    // frames read so far: those whose 2048 samples, from 512 before their time, are all in
    late = Math.max(late, Math.floor((Math.min(i, x.length) + 512 - 2048) / 256) + 1 - f0.length)
  }
  f0.push(...write().f0)
  ok(late <= 24, `at most ${late} frames undecided`)
  ok(f0.length === whole.f0.length && f0.every((v, i) => v === whole.f0[i]), 'odd blocks ≡ one call')
  is(f0.filter((v, i) => i > 3 && i < f0.length - 4 && Math.abs(cents(v, 220)) > 50).length, 0, 'the track never leaves 220 Hz')
  throws(() => track(x, { fs, lag: 0 }), RangeError)
})

test('pyin notes: ±50 cent vibrato at 5.5 Hz stays one note', () => {
  // centred 25 cents above C4, so rounding each frame would flip between C4 and C#4
  let ns = notes(tone(t => hz(60.25) * 2 ** (50 / 1200 * Math.sin(2 * Math.PI * 5.5 * t)), 2), { fs })
  is(ns.length, 1, 'one note')
  is(ns[0].midi, 60)
  ok(Math.abs(midiOf(ns[0].freq) - 60.25) < 0.1, `median pitch at the vibrato centre (${midiOf(ns[0].freq).toFixed(2)})`)
  ok(ns[0].duration > 1.9, `spans the tone (${ns[0].duration.toFixed(2)} s)`)
})

test('pyin notes: a scoop into a note stays inside it', () => {
  let ns = notes(tone(t => hz(60) * 2 ** (t < 0.08 ? -2 / 12 * (1 - t / 0.08) : 0), 1), { fs })
  is(ns.length, 1)
  is(ns[0].midi, 60)
})

test('pyin notes: glides: a fast one joins its two notes, a slow one steps up', () => {
  // A3 → A4 over 0.2 s: the passage is shorter than a note (0.1 s per step), so two notes remain
  let fast = notes(tone(t => 220 * 2 ** Math.min(1, Math.max(0, (t - 0.5) / 0.2)), 1.2), { fs })
  is(fast.map(n => n.midi), [57, 69], 'A3 then A4')
  ok(fast[1].time > 0.5 && fast[1].time < 0.75, `A4 starts inside or right after the glide (${fast[1].time.toFixed(3)} s)`)
  // over 0.5 s it passes semitone regions long enough to be notes: a rising staircase A3 … A4
  let slow = notes(tone(t => 220 * 2 ** Math.min(1, Math.max(0, (t - 0.5) / 0.5)), 1.5), { fs })
  is(slow[0].midi, 57, 'starts on A3')
  is(slow.at(-1).midi, 69, 'ends on A4')
  ok(slow.every((n, i) => !i || n.midi > slow[i - 1].midi), `rising: ${slow.map(n => n.midi)}`)
})

test('pyin notes: a note played twice on one pitch is two notes', () => {
  let ns = notes(played([{ time: 0, midi: 60, duration: 0.5 }, { time: 0.5, midi: 60, duration: 0.5 }], 1.1), { fs })
  is(ns.map(n => n.midi), [60, 60])
  ok(Math.abs(ns[1].time - 0.5) < 0.03, `second note at its attack (${ns[1].time.toFixed(3)} s)`)
})

test('pyin notes: period-doubling bursts give no octave notes', () => {
  let ns = notes(shimmer(220, 2), { fs })
  is(ns.map(n => n.midi), [57])
})

test('pyin notes: silence, short notes, streaming', () => {
  is(notes(new Float32Array(fs), { fs }), [])
  let x = new Float32Array(Math.round(1.2 * fs))
  x.set(tone(() => hz(64), 0.06), Math.round(0.1 * fs))
  x.set(tone(() => hz(67), 0.4), Math.round(0.5 * fs))
  is(notes(x, { fs }).map(n => n.midi), [67], 'a 60 ms note is under the 0.1 s minimum')
  is(notes(x, { fs, minDuration: 0.05 }).map(n => n.midi), [64, 67], 'minDuration 0.05 keeps it')
  let write = notes({ fs }), got = []
  for (let i = 0; i < x.length; i += 1000) got.push(...write(x.subarray(i, i + 1000)))
  got.push(...write())
  is(got, notes(x, { fs }), 'streamed notes equal one call')
})

// librosa 1.0: librosa.pyin(y, fmin=note_to_hz('C2'), fmax=note_to_hz('C7'), sr=44100), defaults
// otherwise (frame 2048, hop 512), on the sequence below; runs of [frames, bin], bin b at
// C2·2^(b/120) Hz, −1 unvoiced. librosa 1.0 zero-extends the frame in its difference function;
// pYIN here keeps YIN's (eq. 6, de Cheveigné & Kawahara 2002), so edges move by a frame and the
// reported pitch (librosa: bin centre) differs by up to a bin and a half.
const LIBROSA = [[16, -1], [1, 192], [1, 191], [32, 190], [1, 189], [1, 192], [1, -1], [32, 230], [1, 233], [1, 254], [32, 260], [1, 261], [2, -1], [35, 310], [23, -1], [2, 281], [42, 280]]

test('pyin track: agrees with librosa.pyin frame by frame', () => {
  // 0.2 s silence, 196/247/294/392 Hz for 0.4 s each, 0.3 s silence, 330 Hz for 0.5 s; phase as numpy's cumsum
  let seg = (f, dur) => Float32Array.from({ length: Math.round(dur * fs) }, (_, i) => {
    let p = 2 * Math.PI * f * (i + 1) / fs
    return f && 0.3 * (Math.sin(p) + 0.5 * Math.sin(2 * p) + 0.33 * Math.sin(3 * p) + 0.25 * Math.sin(4 * p))
  })
  let parts = [seg(0, 0.2), ...[196, 247, 294, 392].map(f => seg(f, 0.4)), seg(0, 0.3), seg(330, 0.5)]
  let x = new Float32Array(parts.reduce((n, p) => n + p.length, 0)), o = 0
  for (let p of parts) { x.set(p, o); o += p.length }
  let C2 = 440 * 2 ** ((36 - 69) / 12), ref = LIBROSA.flatMap(([n, b]) => Array(n).fill(b))
  let r = track(x, { fs, minFreq: C2, maxFreq: 440 * 2 ** ((96 - 69) / 12), frameSize: 2048, hopSize: 512 })
  is(r.f0.length, ref.length, 'same frames')
  let agree = 0, worst = 0
  for (let i = 0; i < ref.length; i++) {
    if (!!r.voiced[i] === ref[i] >= 0) agree++
    if (r.voiced[i] && ref[i] >= 0) worst = Math.max(worst, Math.abs(cents(r.f0[i], C2 * 2 ** (ref[i] / 120))))
  }
  ok(agree >= ref.length - 3, `voicing agrees on ${agree} of ${ref.length} frames`)
  ok(worst <= 15, `pitch within ${worst.toFixed(1)} cents where both are voiced`)
})

// stage 1 stub: one candidate at `f` Hz where the frame's centre half has signal; frames 2N long, centred
function stage1(f, seen = []) {
  return N => {
    let run = frame => {
      let e = 0
      for (let i = N / 2; i < N / 2 + N / 2; i++) e += frame[i] * frame[i]
      run.rms = Math.sqrt(e / (N / 2))
      seen.push({ length: frame.length, at: frame[run.lead] })
      if (!(run.rms > 1e-4)) return 0
      run.freq[0] = f; run.prob[0] = 0.9
      return 1
    }
    return Object.assign(run, { freq: new Float64Array(1), prob: new Float64Array(1), rms: 0, frameSize: 2 * N, lead: N })
  }
}

test('pyin track: `candidates` replaces stage 1, on the same frames', () => {
  let x = tone(() => 220, 1), H = 256, seen = []
  x[100 * H] = 5                                              // frame 100's time: the stub sees it at run.lead
  let r = track(x, { fs, candidates: stage1(330, seen) }), d = track(x, { fs })
  is(r.times, d.times, 'frame times as the default')
  ok(seen.every(s => s.length === 4096), 'frames of the stub\'s frameSize')
  is(seen[100].at, 5, 'frame i centred on sample i·hop at run.lead')
  let voiced = r.f0.filter(v => v > 0)
  ok(voiced.length > 0.9 * r.f0.length && voiced.every(v => v === 330), `voiced at the stub's 330 Hz (${voiced.length} of ${r.f0.length})`)
  is(track(x, { fs, candidates: undefined }).f0, d.f0, 'no candidates: the default')
})

test('pyin notes: `candidates` drives the note model', () => {
  let x = new Float32Array(2 * fs)
  x.set(tone(() => 220, 0.8), Math.round(0.5 * fs))
  let n = notes(x, { fs, candidates: stage1(hz(64)) })
  is(n.length, 1, 'one note')
  is(n[0].midi, 64, 'at the stub\'s pitch')
  ok(Math.abs(n[0].time - 0.5) < 0.05 && Math.abs(n[0].duration - 0.8) < 0.1, `where the tone is (${n[0].time.toFixed(3)} s, ${n[0].duration.toFixed(3)} s)`)
})
