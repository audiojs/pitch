// atom manifest — YIN as an analyzer: audio in, no audio out, pitch/clarity
// emitted per analysis frame via ctx.emit. A block allocates nothing: the frame, YIN's
// scratch, its settings and results and the frame's fill are typed arrays made here, and
// yinInto takes and gives only those.

import { yinInto } from './yin.js'

const FRAME = 2048
// The range of the pitch it reports. YIN runs here without minFreq/maxFreq: its lags span 2 to 1022 samples of the
// frame, so what it can return moves with the sample rate (fs/1022.5 to fs/1.5 Hz, and lower where the parabola at the
// last lag extrapolates; at 44.1 kHz a 21 kHz sine reads 23153 Hz). A declared range cannot move, so its two numbers
// are those of the contract's example `freq` (contract 1.0 § Parameters: 20 Hz to 20 kHz, log), and a frame whose pitch
// falls outside it emits nothing. Clarity is 1 minus YIN's dip, which is at least 0 and under the threshold: 0..1.
const PITCH = { min: 20, max: 20000, unit: 'Hz', curve: 'log' }

export const pitch = (ctx) => {
	// k: the rate, the threshold, no minFreq or maxFreq, then the pitch and clarity yinInto finds
	const win = new Float32Array(FRAME), d = new Float64Array(FRAME >> 1), cmndf = new Float64Array(FRAME >> 1)
	const k = Float64Array.of(ctx.sampleRate || 44100, 0, NaN, NaN, 0, 0), fill = new Int32Array(1)
	return (inputs, outputs, params) => {
		const x = inputs[0] && inputs[0][0]
		if (!x) return
		let f = fill[0]
		for (let i = 0; i < x.length; i++) {
			win[f++] = x[i]
			if (f === FRAME) {
				k[1] = params.threshold[0]
				if (yinInto(win, k, d, cmndf) && k[4] >= PITCH.min && k[4] <= PITCH.max) {
					ctx.emit('pitch', k[4])
					ctx.emit('clarity', k[5])
				}
				f = 0
			}
		}
		fill[0] = f
	}
}
pitch.channels = { inputs: 1, outputs: [] }
pitch.events = { out: { pitch: PITCH, clarity: { min: 0, max: 1 } } }
pitch.params = {
	threshold: { type: 'number', min: 0.05, max: 0.5, default: 0.1 },
}
