/**
 * YIN pitch detection (de Cheveigné & Kawahara, 2002).
 * Difference fn → cumulative mean normalized difference → threshold → parabolic interpolation.
 *
 * @param {Float32Array | Float64Array} data - audio samples (single window, eg. 2048 samples)
 * @param {{fs?: number, threshold?: number, minFreq?: number, maxFreq?: number}} [params]
 * @returns {{freq: number, clarity: number} | null}
 */
export default function yin(data, params) {
  let minFreq = params?.minFreq, maxFreq = params?.maxFreq, half = data.length >> 1
  let k = Float64Array.of(params?.fs || 44100, params?.threshold ?? 0.15,
    Number.isFinite(minFreq) && minFreq > 0 ? minFreq : NaN, Number.isFinite(maxFreq) && maxFreq > 0 ? maxFreq : NaN, 0, 0)
  return yinInto(data, k, new Float64Array(half), new Float64Array(half)) ? { freq: k[4], clarity: k[5] } : null
}

/**
 * yin() on one frame, allocating nothing, for a caller that runs it frame after frame (the ./audio processor).
 * `k` (Float64Array of 6): fs, threshold, minFreq and maxFreq (NaN where there is none) in; the pitch (Hz) and the
 * clarity out at 4 and 5 where it returns 1 (0: none found). `d`, `cmndf`: scratch, Float64Arrays of half the frame at
 * least. Only arrays and whole numbers cross the call: an engine that does not inline it makes no number object for it.
 */
export function yinInto(data, k, d, cmndf) {
  let fs = k[0], threshold = k[1], minFreq = k[2], maxFreq = k[3]
  let len = data.length
  let half = len >> 1

  // With a known lower bound we only need lags through fs/minFreq. Use the
  // remaining samples as a longer fixed comparison window, which materially
  // improves low-pitch detection near an onset without adding latency.
  let bounded = minFreq > 0
  let limit = bounded ? Math.min(half, Math.ceil(fs / minFreq) + 2) : half
  let window = bounded ? len - (limit - 1) : half
  if (limit < 4 || window < 1) return 0

  // step 1-2: difference function
  for (let tau = 1; tau < limit; tau++) {
    let sum = 0
    for (let i = 0; i < window; i++) {
      let diff = data[i] - data[i + tau]
      sum += diff * diff
    }
    d[tau] = sum
  }

  // step 3: cumulative mean normalized difference
  cmndf[0] = 1
  let running = 0
  for (let tau = 1; tau < limit; tau++) {
    running += d[tau]
    cmndf[tau] = running > 0 ? d[tau] * tau / running : 1
  }

  // step 4: absolute threshold — find first dip below threshold
  let tau = 2
  while (tau < limit - 1) {
    if (cmndf[tau] < threshold) {
      // find local minimum
      while (tau + 1 < limit - 1 && cmndf[tau + 1] < cmndf[tau]) tau++
      break
    }
    tau++
  }
  if (tau >= limit - 1) return 0

  // step 5: parabolic interpolation
  let s0 = cmndf[tau - 1], s1 = cmndf[tau], s2 = cmndf[tau + 1]
  let denom = s0 - 2 * s1 + s2
  let shift = denom !== 0 ? (s0 - s2) / (2 * denom) : 0
  let period = tau + shift
  let freq = fs / period

  if (bounded && freq < minFreq) return 0
  if (maxFreq > 0 && freq > maxFreq) return 0
  k[4] = freq
  k[5] = 1 - s1
  return 1
}
