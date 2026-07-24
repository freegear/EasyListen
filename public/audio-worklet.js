class EasyListenerPcmProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.sourceRate = sampleRate;
    this.targetRate = 16000;
    this.pending = [];
    this.emittedSamples = 0;
    this.port.onmessage = (event) => {
      if (event.data?.type !== "flush") return;
      this.emitPending(true);
      this.port.postMessage({
        type: "flushed",
        emittedSamples: this.emittedSamples,
      });
    };
  }

  emitPending(force = false) {
    const ratio = this.sourceRate / this.targetRate;
    const outputLength = force
      ? Math.ceil(this.pending.length / ratio)
      : Math.floor(this.pending.length / ratio);
    if (outputLength === 0 || (!force && outputLength < 512)) {
      return;
    }
    const output = new Int16Array(outputLength);
    for (let index = 0; index < outputLength; index += 1) {
      const sourceIndex = Math.min(
        this.pending.length - 1,
        Math.floor(index * ratio),
      );
      const sample = Math.max(-1, Math.min(1, this.pending[sourceIndex] || 0));
      output[index] = sample < 0 ? sample * 32768 : sample * 32767;
    }
    const consumed = force
      ? this.pending.length
      : Math.floor(outputLength * ratio);
    this.pending = this.pending.slice(consumed);
    this.emittedSamples += output.length;
    this.port.postMessage(output.buffer, [output.buffer]);
  }

  process(inputs) {
    const channels = inputs[0];
    if (!channels || channels.length === 0 || channels[0].length === 0) {
      return true;
    }

    const frameLength = channels[0].length;
    for (let index = 0; index < frameLength; index += 1) {
      let mixed = 0;
      for (const channel of channels) {
        mixed += channel[index] || 0;
      }
      this.pending.push(mixed / channels.length);
    }

    this.emitPending();
    return true;
  }
}

registerProcessor("easylistener-pcm", EasyListenerPcmProcessor);
