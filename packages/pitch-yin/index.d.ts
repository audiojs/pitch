/** YIN pitch detection (de Cheveigné & Kawahara, 2002). */
export interface YinOptions {
  /** sample rate (Hz), default 44100 */
  fs?: number
  /** CMND threshold — lower = stricter, fewer detections, default 0.15 */
  threshold?: number
  /** minimum frequency to return (Hz); also improves low-pitch use of the window */
  minFreq?: number
  /** maximum frequency to return (Hz) */
  maxFreq?: number
}

/** Single-frame pitch estimate. Returns null if no periodic structure is found. */
export default function yin(data: Float32Array | Float64Array, options?: YinOptions): { freq: number, clarity: number } | null

/**
 * The same estimate, allocating nothing, for a caller that runs it frame after frame. `k` (6 numbers): fs, threshold,
 * minFreq and maxFreq (NaN where there is none) in; the pitch (Hz) and the clarity out at 4 and 5 where it returns 1
 * (0: none found). `d` and `cmndf`: scratch, each of half the frame's length at least.
 */
export function yinInto(data: Float32Array | Float64Array, k: Float64Array, d: Float64Array, cmndf: Float64Array): 0 | 1
