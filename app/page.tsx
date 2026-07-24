"use client";

import { useCallback, useEffect, useRef, useState } from "react";

type Segment = {
  text: string;
  rawText?: string;
  start: number;
  end: number;
  confidence?: number | null;
  avgLogprob?: number | null;
  reviewRequired?: boolean;
  reviewReasons?: string[];
  corrections?: Array<{ from: string; to: string }>;
  alternatives?: string[];
};
type RecordingDiagnostics = {
  receivedDuration?: number;
  transcribedDuration?: number;
  speechDuration?: number;
  speechRangeCount?: number;
  windowCount?: number;
  deduplicatedSegments?: number;
  lowConfidenceSegments?: number;
  missingSamples?: number;
  workletFlushed?: boolean;
  model?: string;
  engine?: string;
  vadFallback?: boolean;
};
type Recording = {
  id: string;
  title: string;
  createdAt: string;
  duration: number;
  segments: Segment[];
  waveform?: number[];
  blob?: Blob;
  demo?: boolean;
  hasAudio?: boolean;
  diagnostics?: RecordingDiagnostics;
};

type CaptureStatus =
  | "idle"
  | "requesting_permission"
  | "connecting"
  | "ready"
  | "listening"
  | "recognizing"
  | "stopping"
  | "saving"
  | "completed"
  | "error";

type AudioDiagnostics = {
  sampleRate: number;
  channels: number;
  rms: number;
  peak: number;
  decibels: number;
  contextState: AudioContextState | "unavailable";
  noiseSuppression: boolean;
};

type SystemCapabilities = {
  stt: boolean;
  vad: string;
  sqlite: boolean;
  localAudio: boolean;
  d1: boolean;
  r2: boolean;
  model: string;
  engine?: string;
  recordingCount: number;
  retentionDays: number;
  maxRecordingSeconds: number;
  legalPrompt?: boolean;
  legalCorrections?: number;
};

declare global {
  interface Window {
    webkitAudioContext?: typeof AudioContext;
  }
}

const demoSegments: Segment[] = [
  { text: "현장 지휘소, 통신 상태 확인 바랍니다.", start: 0, end: 2.7 },
  { text: "좌측 채널 정상, 배경 소음 제거 모듈 작동 중입니다.", start: 2.7, end: 6.2 },
  { text: "음성 명령 인식률 96.8%, 전송 준비가 완료되었습니다.", start: 6.2, end: 10.5 },
];

const WAVEFORM_SAMPLE_COUNT = 180;
const WAVEFORM_SAMPLE_INTERVAL_MS = 70;
const WAVEFORM_MIN_LEVEL = 0.06;
const REVIEW_REASON_LABELS: Record<string, string> = {
  low_log_probability: "낮은 인식 확률",
  high_compression_ratio: "반복 문장 의심",
  possible_non_speech: "비음성 가능성",
  character_repetition: "문자 반복",
  repeated_text: "인접 문장 반복",
  dictionary_correction: "용어 사전 교정",
  overlap_conflict: "중첩 구간 불일치",
};
const NO_SIGNAL_THRESHOLD = 0.0001;
const NO_SIGNAL_WARNING_DELAY_MS = 3000;
const EMPTY_WAVEFORM = Array.from(
  { length: WAVEFORM_SAMPLE_COUNT },
  () => WAVEFORM_MIN_LEVEL,
);
const demoWaveform = Array.from({ length: WAVEFORM_SAMPLE_COUNT }, (_, index) => {
  const carrier = Math.abs(Math.sin(index * 0.43));
  const modulation = 0.35 + Math.abs(Math.sin(index * 0.11)) * 0.65;
  return 0.12 + carrier * modulation * 0.72;
});

const demoRecording: Recording = {
  id: "demo",
  title: "전술 통신 음성 샘플",
  createdAt: "2026. 07. 23.  19:42",
  duration: 10.5,
  segments: demoSegments,
  waveform: demoWaveform,
  demo: true,
};

const formatTime = (seconds: number) => {
  const value = Number.isFinite(seconds) ? Math.max(0, seconds) : 0;
  return `${Math.floor(value / 60)
    .toString()
    .padStart(2, "0")}:${Math.floor(value % 60).toString().padStart(2, "0")}`;
};

const analyzeSamples = (samples: Float32Array) => {
  let sumSquares = 0;
  let peak = 0;

  for (const sample of samples) {
    sumSquares += sample * sample;
    peak = Math.max(peak, Math.abs(sample));
  }

  const rms = Math.sqrt(sumSquares / samples.length);
  return {
    rms,
    peak,
    rawLevel: rms * 0.65 + peak * 0.35,
  };
};

const normalizeDecibels = (rawLevel: number) => {
  const decibels = 20 * Math.log10(Math.max(rawLevel, 0.00001));
  const normalized = Math.min(1, Math.max(0, (decibels + 60) / 48));
  return { decibels, normalized };
};

const enhanceWaveformLevel = (
  rawLevel: number,
  previousLevel: number,
  recentPeak: number,
) => {
  const { decibels, normalized } = normalizeDecibels(rawLevel);
  const nextRecentPeak = Math.max(normalized, recentPeak * 0.98);
  const autoGain = Math.min(3, 0.85 / Math.max(nextRecentPeak, 0.1));
  const amplified = Math.min(1, normalized * autoGain);
  const smoothing = amplified > previousLevel ? 0.65 : 0.18;
  const smoothed =
    previousLevel + (amplified - previousLevel) * smoothing;

  return {
    level: Math.max(WAVEFORM_MIN_LEVEL, smoothed),
    recentPeak: nextRecentPeak,
    decibels,
  };
};

function openDatabase(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open("voice-command-center", 1);
    request.onupgradeneeded = () => {
      if (!request.result.objectStoreNames.contains("recordings")) {
        request.result.createObjectStore("recordings", { keyPath: "id" });
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

async function loadRecordings(): Promise<Recording[]> {
  const db = await openDatabase();
  return new Promise((resolve, reject) => {
    const request = db.transaction("recordings").objectStore("recordings").getAll();
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

export default function Home() {
  const [recordings, setRecordings] = useState<Recording[]>([demoRecording]);
  const [selected, setSelected] = useState<Recording>(demoRecording);
  const [recording, setRecording] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [recordingElapsed, setRecordingElapsed] = useState(0);
  const [playbackElapsed, setPlaybackElapsed] = useState(0);
  const [levels, setLevels] = useState({ left: 0.08, right: 0.06 });
  const [audioDiagnostics, setAudioDiagnostics] = useState<AudioDiagnostics>({
    sampleRate: 0,
    channels: 1,
    rms: 0,
    peak: 0,
    decibels: -100,
    contextState: "unavailable",
    noiseSuppression: false,
  });
  const [liveText, setLiveText] = useState("음성인식 버튼을 눌러 한국어 음성 명령을 시작하세요.");
  const [interimText, setInterimText] = useState("");
  const [notice, setNotice] = useState("");
  const [captureStatus, setCaptureStatus] = useState<CaptureStatus>("idle");
  const [systemOnline, setSystemOnline] = useState(false);
  const [modelName, setModelName] = useState("LOCAL STT");
  const [latencyMs, setLatencyMs] = useState<number | null>(null);
  const [realtimeFactor, setRealtimeFactor] = useState<number | null>(null);
  const [helpOpen, setHelpOpen] = useState(false);
  const [capabilities, setCapabilities] = useState<SystemCapabilities | null>(null);
  const [accessInfo, setAccessInfo] = useState({
    origin: "확인 중",
    secure: false,
  });
  const streamRef = useRef<MediaStream | null>(null);
  const audioContext = useRef<AudioContext | null>(null);
  const workletNode = useRef<AudioWorkletNode | null>(null);
  const webSocket = useRef<WebSocket | null>(null);
  const animation = useRef<number | null>(null);
  const startedAt = useRef(0);
  const segments = useRef<Segment[]>([]);
  const waveform = useRef<number[]>(EMPTY_WAVEFORM);
  const lastWaveformSampleAt = useRef(0);
  const previousWaveformLevel = useRef(WAVEFORM_MIN_LEVEL);
  const recentWaveformPeak = useRef(0.1);
  const noSignalStartedAt = useRef(0);
  const signalWarningActive = useRef(false);
  const lastDiagnosticsAt = useRef(0);
  const stopping = useRef(false);
  const recordingActive = useRef(false);
  const intentionalSocketClose = useRef(false);
  const reconnectAttempts = useRef(0);
  const reconnectTimer = useRef<number | null>(null);
  const pcmFrames = useRef<ArrayBuffer[]>([]);
  const workletFlushResolver = useRef<
    ((result: { emittedSamples: number; flushed: boolean }) => void) | null
  >(null);
  const player = useRef<HTMLAudioElement | null>(null);
  const playTimer = useRef<number | null>(null);

  useEffect(() => {
    const accessInfoTimer = window.setTimeout(() => {
      setAccessInfo({
        origin: window.location.origin,
        secure: window.isSecureContext,
      });
    }, 0);
    Promise.all([
      fetch("/stt-api/health").then((response) => response.json()),
      fetch("/stt-api/api/recordings").then((response) => response.json()),
      fetch("/stt-api/api/system/capabilities").then((response) => response.json()),
    ])
      .then(([
        health,
        stored,
        systemCapabilities,
      ]: [
        { whisperReady?: boolean; model?: string },
        Recording[],
        SystemCapabilities,
      ]) => {
        setSystemOnline(Boolean(health.whisperReady));
        setModelName(health.model || "LOCAL STT");
        setCapabilities(systemCapabilities);
        if (stored.length) {
          setRecordings([demoRecording, ...stored]);
          setSelected(stored[0]);
          setLiveText(stored[0].segments.map((segment) => segment.text).join(" "));
        }
      })
      .catch(() => {
        setSystemOnline(false);
        loadRecordings()
          .then((stored) => stored.length && setRecordings([demoRecording, ...stored]))
          .catch(() => undefined);
      });
    const healthTimer = window.setInterval(() => {
      fetch("/stt-api/health")
        .then((response) => {
          if (!response.ok) throw new Error("health check failed");
          return response.json();
        })
        .then((health: { whisperReady?: boolean; model?: string }) => {
          setSystemOnline(Boolean(health.whisperReady));
          setModelName(health.model || "LOCAL STT");
        })
        .catch(() => setSystemOnline(false));
    }, 5000);
    return () => {
      window.clearTimeout(accessInfoTimer);
      window.clearInterval(healthTimer);
      if (reconnectTimer.current) window.clearTimeout(reconnectTimer.current);
      recordingActive.current = false;
      intentionalSocketClose.current = true;
      streamRef.current?.getTracks().forEach((track) => track.stop());
      webSocket.current?.close();
      if (animation.current) cancelAnimationFrame(animation.current);
      if (playTimer.current) cancelAnimationFrame(playTimer.current);
    };
  }, []);

  useEffect(() => {
    if (!helpOpen) return;
    fetch("/stt-api/api/system/capabilities")
      .then((response) => response.json())
      .then((systemCapabilities: SystemCapabilities) => {
        setCapabilities(systemCapabilities);
      })
      .catch(() => undefined);
  }, [helpOpen]);

  const stopPlayback = useCallback(() => {
    player.current?.pause();
    player.current = null;
    if (playTimer.current) cancelAnimationFrame(playTimer.current);
    setPlaying(false);
  }, []);

  const playSelected = useCallback(() => {
    if (playing) {
      stopPlayback();
      return;
    }
    const start = performance.now() - playbackElapsed * 1000;
    setPlaying(true);
    if (selected.blob || selected.hasAudio) {
      const sourceUrl = selected.blob
        ? URL.createObjectURL(selected.blob)
        : `/stt-api/api/recordings/${selected.id}/audio`;
      const audio = new Audio(sourceUrl);
      player.current = audio;
      audio.currentTime = playbackElapsed >= selected.duration ? 0 : playbackElapsed;
      audio.play().catch(() => setNotice("브라우저에서 오디오 재생을 허용해 주세요."));
      audio.onended = () => {
        setPlaybackElapsed(0);
        setPlaying(false);
      };
    }
    const tick = () => {
      const next = (selected.blob || selected.hasAudio) && player.current
        ? player.current.currentTime
        : (performance.now() - start) / 1000;
      if (next >= selected.duration) {
        setPlaybackElapsed(0);
        setPlaying(false);
        return;
      }
      setPlaybackElapsed(next);
      playTimer.current = requestAnimationFrame(tick);
    };
    playTimer.current = requestAnimationFrame(tick);
  }, [playbackElapsed, playing, selected, stopPlayback]);

  const stopRecording = useCallback(async () => {
    if (!recording || stopping.current) return;
    stopping.current = true;
    recordingActive.current = false;
    setCaptureStatus("stopping");
    setInterimText("마지막 음성 프레임을 전송하고 있습니다…");
    let flushResult = { emittedSamples: 0, flushed: false };
    if (workletNode.current) {
      flushResult = await new Promise((resolve) => {
        const timeout = window.setTimeout(() => {
          workletFlushResolver.current = null;
          resolve({ emittedSamples: 0, flushed: false });
        }, 800);
        workletFlushResolver.current = (result) => {
          window.clearTimeout(timeout);
          workletFlushResolver.current = null;
          resolve(result);
        };
        workletNode.current?.port.postMessage({ type: "flush" });
      });
    }
    setInterimText("전체 음성을 고정밀 모델로 최종 인식하고 있습니다…");
    workletNode.current?.disconnect();
    workletNode.current = null;
    streamRef.current?.getTracks().forEach((track) => track.stop());
    if (animation.current) cancelAnimationFrame(animation.current);
    audioContext.current?.close();
    audioContext.current = null;
    signalWarningActive.current = false;
    setLevels({ left: 0.08, right: 0.06 });
    setAudioDiagnostics((current) => ({
      ...current,
      contextState: "unavailable",
    }));
    if (webSocket.current?.readyState === WebSocket.OPEN) {
      setCaptureStatus("saving");
      webSocket.current.send(JSON.stringify({
        type: "stop",
        waveform: waveform.current,
        title: `음성 명령 기록 ${(recordings.length).toString().padStart(2, "0")}`,
        emittedSamples: flushResult.emittedSamples,
        workletFlushed: flushResult.flushed,
      }));
    } else {
      stopping.current = false;
      setRecording(false);
      setCaptureStatus("error");
      setNotice("STT 서버 연결이 종료되어 녹음을 저장할 수 없습니다.");
    }
  }, [recording, recordings.length]);

  const startRecording = async () => {
    setNotice("");
    setCaptureStatus("requesting_permission");
    stopPlayback();
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 2, echoCancellation: true, noiseSuppression: true },
      });
      streamRef.current = stream;
      setCaptureStatus("connecting");
      recordingActive.current = true;
      intentionalSocketClose.current = false;
      reconnectAttempts.current = 0;
      pcmFrames.current = [];
      segments.current = [];
      waveform.current = [...EMPTY_WAVEFORM];
      lastWaveformSampleAt.current = 0;
      previousWaveformLevel.current = WAVEFORM_MIN_LEVEL;
      recentWaveformPeak.current = 0.1;
      noSignalStartedAt.current = 0;
      signalWarningActive.current = false;
      lastDiagnosticsAt.current = 0;
      startedAt.current = performance.now();
      setRecordingElapsed(0);
      setLiveText("");
      setInterimText("로컬 STT 서버에 연결하고 있습니다…");

      const socketProtocol = window.location.protocol === "https:" ? "wss:" : "ws:";
      const socketUrl = `${socketProtocol}//${window.location.host}/stt-api/ws/stt`;
      const sessionId = crypto.randomUUID();
      type SttEvent = {
          type: string;
          text?: string;
          start?: number;
          end?: number;
          latencyMs?: number;
          realtimeFactor?: number;
          model?: string;
          message?: string;
          recording?: Recording;
          segments?: Segment[];
          diagnostics?: RecordingDiagnostics;
          whisperReady?: boolean;
          current?: number;
          total?: number;
      };

      const handleSocketMessage = (socket: WebSocket, message: MessageEvent) => {
        let event: SttEvent;
        try {
          event = JSON.parse(String(message.data)) as SttEvent;
        } catch {
          intentionalSocketClose.current = true;
          recordingActive.current = false;
          setRecording(false);
          setCaptureStatus("error");
          setNotice("STT 서버가 올바르지 않은 응답을 전송했습니다. 서버를 다시 시작해 주세요.");
          socket.close();
          return;
        }
        if (event.type === "ready") {
          setSystemOnline(Boolean(event.whisperReady));
          setModelName(event.model || "LOCAL STT");
          if (!event.whisperReady) {
            intentionalSocketClose.current = true;
            recordingActive.current = false;
            streamRef.current?.getTracks().forEach((track) => track.stop());
            setCaptureStatus("error");
            setNotice("Whisper 모델 서버가 준비되지 않았습니다. Whisper 로그를 확인해 주세요.");
            socket.close();
            return;
          }
          setCaptureStatus("ready");
          setInterimText("듣고 있습니다…");
          window.requestAnimationFrame(() => {
            if (recordingActive.current) setCaptureStatus("listening");
          });
        } else if (event.type === "speech_start") {
          setCaptureStatus("recognizing");
          setInterimText("음성이 감지되었습니다…");
        } else if (event.type === "interim") {
          setInterimText(event.text || "음성을 분석하고 있습니다…");
        } else if (event.type === "final" && event.text) {
          const segment = {
            text: event.text,
            start: event.start || 0,
            end: event.end || 0,
          };
          segments.current.push(segment);
          setLiveText(segments.current.map((part) => part.text).join(" "));
          setInterimText("");
          setCaptureStatus("listening");
        } else if (event.type === "speech_end") {
          setCaptureStatus("recognizing");
          setInterimText("발화를 인식하고 있습니다…");
        } else if (event.type === "finalizing") {
          setCaptureStatus("saving");
          setInterimText(
            `전체 음성 최종 인식 중… ${event.current ?? 0}/${event.total ?? 1}`,
          );
        } else if (event.type === "final_replace" && event.segments) {
          segments.current = event.segments;
          setLiveText(event.segments.map((part) => part.text).join(" "));
          setInterimText("최종 자막을 저장하고 있습니다…");
        } else if (event.type === "metrics") {
          setLatencyMs(event.latencyMs ?? null);
          setRealtimeFactor(event.realtimeFactor ?? null);
          if (event.model) setModelName(event.model);
        } else if (event.type === "saved" && event.recording) {
          const savedRecording = event.recording;
          intentionalSocketClose.current = true;
          recordingActive.current = false;
          setRecordings((current) => [
            savedRecording,
            ...current.filter((item) => item.id !== savedRecording.id),
          ]);
          setSelected(savedRecording);
          setPlaybackElapsed(0);
          setRecordingElapsed(0);
          setLiveText(savedRecording.segments.map((part) => part.text).join(" "));
          setInterimText("");
          setRecording(false);
          setCaptureStatus("completed");
          stopping.current = false;
          socket.close();
        } else if (event.type === "error") {
          intentionalSocketClose.current = true;
          recordingActive.current = false;
          workletNode.current?.disconnect();
          streamRef.current?.getTracks().forEach((track) => track.stop());
          if (animation.current) cancelAnimationFrame(animation.current);
          void audioContext.current?.close();
          audioContext.current = null;
          setRecording(false);
          setNotice(event.message || "음성인식 처리 중 오류가 발생했습니다.");
          setCaptureStatus("error");
          socket.close();
        }
      };

      const scheduleReconnect = () => {
        if (
          intentionalSocketClose.current ||
          !recordingActive.current ||
          stopping.current ||
          reconnectAttempts.current >= 3
        ) {
          if (recordingActive.current && reconnectAttempts.current >= 3) {
            recordingActive.current = false;
            setRecording(false);
            setCaptureStatus("error");
            setNotice("STT 서버 재연결에 실패했습니다. 서버 상태를 확인한 뒤 다시 녹음해 주세요.");
          }
          return;
        }
        reconnectAttempts.current += 1;
        setCaptureStatus("connecting");
        setInterimText(`STT 서버에 재연결하고 있습니다… (${reconnectAttempts.current}/3)`);
        reconnectTimer.current = window.setTimeout(async () => {
          try {
            segments.current = [];
            setLiveText("");
            webSocket.current = await connectSocket(true);
          } catch {
            scheduleReconnect();
          }
        }, 1000);
      };

      const connectSocket = async (replayAudio: boolean) => {
        const socket = new WebSocket(socketUrl);
        await new Promise<void>((resolve, reject) => {
          const timeout = window.setTimeout(() => {
            socket.close();
            reject(new Error("로컬 STT 서버 연결 시간이 초과되었습니다."));
          }, 5000);
          socket.onopen = () => {
            window.clearTimeout(timeout);
            resolve();
          };
          socket.onerror = () => {
            window.clearTimeout(timeout);
            reject(new Error("로컬 STT 서버에 연결할 수 없습니다. run_demo.sh 실행 상태를 확인해 주세요."));
          };
        });
        socket.onmessage = (message) => handleSocketMessage(socket, message);
        socket.onclose = scheduleReconnect;
        socket.send(JSON.stringify({
          type: "start",
          sessionId,
          language: "ko",
          sampleRate: 16000,
          format: "pcm_s16le",
        }));
        if (replayAudio) {
          for (const frame of pcmFrames.current) {
            socket.send(frame);
          }
          setNotice("STT 서버 연결이 복구되어 녹음 데이터를 다시 전송했습니다.");
        }
        reconnectAttempts.current = 0;
        return socket;
      };

      const socket = await connectSocket(false);
      webSocket.current = socket;

      const Context = window.AudioContext || window.webkitAudioContext;
      if (!Context) {
        throw new Error("이 브라우저는 오디오 분석 기능을 지원하지 않습니다.");
      }

      const context = new Context();
      audioContext.current = context;

      if (context.state === "suspended") {
        await context.resume();
      }

      if (context.state !== "running") {
        throw new Error("오디오 분석기를 시작하지 못했습니다. 화면을 클릭한 후 다시 시도해 주세요.");
      }

      const audioTrack = stream.getAudioTracks()[0];
      const trackSettings = audioTrack?.getSettings();
      const detectedChannels = Math.max(
        1,
        Math.min(2, trackSettings?.channelCount ?? 1),
      );
      const source = context.createMediaStreamSource(stream);
      const splitter = context.createChannelSplitter(2);
      const leftAnalyser = context.createAnalyser();
      const rightAnalyser = context.createAnalyser();
      leftAnalyser.fftSize = 256;
      rightAnalyser.fftSize = 256;
      leftAnalyser.smoothingTimeConstant = 0.15;
      rightAnalyser.smoothingTimeConstant = 0.15;
      source.connect(splitter);
      splitter.connect(leftAnalyser, 0);
      splitter.connect(rightAnalyser, detectedChannels > 1 ? 1 : 0);
      const leftWaveformData = new Float32Array(leftAnalyser.fftSize);
      const rightWaveformData = new Float32Array(rightAnalyser.fftSize);

      setAudioDiagnostics({
        sampleRate: context.sampleRate,
        channels: detectedChannels,
        rms: 0,
        peak: 0,
        decibels: -100,
        contextState: context.state,
        noiseSuppression: trackSettings?.noiseSuppression ?? false,
      });
      await context.audioWorklet.addModule("/audio-worklet.js");
      const pcmWorklet = new AudioWorkletNode(context, "easylistener-pcm");
      const silentGain = context.createGain();
      silentGain.gain.value = 0;
      source.connect(pcmWorklet);
      pcmWorklet.connect(silentGain).connect(context.destination);
      pcmWorklet.port.onmessage = (
        event: MessageEvent<ArrayBuffer | {
          type: string;
          emittedSamples?: number;
        }>,
      ) => {
        if (!(event.data instanceof ArrayBuffer)) {
          if (event.data.type === "flushed") {
            workletFlushResolver.current?.({
              emittedSamples: event.data.emittedSamples ?? 0,
              flushed: true,
            });
          }
          return;
        }
        pcmFrames.current.push(event.data.slice(0));
        if (webSocket.current?.readyState === WebSocket.OPEN) {
          webSocket.current.send(event.data);
        }
      };
      workletNode.current = pcmWorklet;
      if (!recordingActive.current) {
        throw new Error("로컬 STT 엔진을 시작하지 못했습니다.");
      }

      const meter = () => {
        const now = performance.now();
        setRecordingElapsed((now - startedAt.current) / 1000);

        if (now - lastWaveformSampleAt.current >= WAVEFORM_SAMPLE_INTERVAL_MS) {
          leftAnalyser.getFloatTimeDomainData(leftWaveformData);
          rightAnalyser.getFloatTimeDomainData(rightWaveformData);

          const leftAnalysis = analyzeSamples(leftWaveformData);
          const rightAnalysis = detectedChannels > 1
            ? analyzeSamples(rightWaveformData)
            : leftAnalysis;
          const dominantAnalysis = leftAnalysis.rawLevel >= rightAnalysis.rawLevel
            ? leftAnalysis
            : rightAnalysis;
          const enhanced = enhanceWaveformLevel(
            dominantAnalysis.rawLevel,
            previousWaveformLevel.current,
            recentWaveformPeak.current,
          );

          previousWaveformLevel.current = enhanced.level;
          recentWaveformPeak.current = enhanced.recentPeak;

          const leftNormalized = normalizeDecibels(leftAnalysis.rawLevel).normalized;
          const rightNormalized = normalizeDecibels(rightAnalysis.rawLevel).normalized;
          setLevels({
            left: Math.max(0.02, leftNormalized),
            right: Math.max(0.02, detectedChannels > 1 ? rightNormalized : leftNormalized),
          });

          const nextWaveform = [...waveform.current.slice(1), enhanced.level];
          waveform.current = nextWaveform;
          lastWaveformSampleAt.current = now;

          if (dominantAnalysis.rawLevel <= NO_SIGNAL_THRESHOLD) {
            if (noSignalStartedAt.current === 0) {
              noSignalStartedAt.current = now;
            } else if (
              now - noSignalStartedAt.current >= NO_SIGNAL_WARNING_DELAY_MS &&
              !signalWarningActive.current
            ) {
              signalWarningActive.current = true;
              setNotice("마이크 입력 신호가 감지되지 않습니다. 입력 장치와 마이크 음량을 확인해 주세요.");
            }
          } else {
            noSignalStartedAt.current = 0;
            if (signalWarningActive.current) {
              signalWarningActive.current = false;
              setNotice("");
            }
          }

          if (now - lastDiagnosticsAt.current >= 350) {
            setAudioDiagnostics({
              sampleRate: context.sampleRate,
              channels: detectedChannels,
              rms: dominantAnalysis.rms,
              peak: dominantAnalysis.peak,
              decibels: enhanced.decibels,
              contextState: context.state,
              noiseSuppression: trackSettings?.noiseSuppression ?? false,
            });
            lastDiagnosticsAt.current = now;
          }
        }

        animation.current = requestAnimationFrame(meter);
      };
      meter();
      stopping.current = false;
      setRecording(true);
    } catch (error: unknown) {
      if (reconnectTimer.current) window.clearTimeout(reconnectTimer.current);
      recordingActive.current = false;
      intentionalSocketClose.current = true;
      webSocket.current?.close();
      streamRef.current?.getTracks().forEach((track) => track.stop());
      if (animation.current) cancelAnimationFrame(animation.current);
      await audioContext.current?.close();
      audioContext.current = null;
      setRecording(false);
      setCaptureStatus("error");
      setLevels({ left: 0.08, right: 0.06 });
      setAudioDiagnostics((current) => ({
        ...current,
        contextState: "unavailable",
      }));
      setNotice(
        error instanceof Error
          ? error.message
          : "마이크 권한이 필요합니다. 주소창의 마이크 권한을 허용해 주세요.",
      );
    }
  };

  const selectRecording = (item: Recording) => {
    if (recording) return;
    stopPlayback();
    setSelected(item);
    setPlaybackElapsed(0);
    setLiveText(item.segments.map((segment) => segment.text).join(" "));
  };

  const safeFilename = (name: string) =>
    name.replace(/[\\/:*?"<>|]/g, "-").replace(/\s+/g, "_");

  const downloadBlob = (blob: Blob, filename: string) => {
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = filename;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  };

  const downloadAudio = () => {
    if (selected.hasAudio) {
      window.location.href = `/stt-api/api/recordings/${selected.id}/audio`;
      return;
    }
    if (!selected.blob) {
      setNotice("기본 데모에는 음성 원본이 없습니다. 직접 녹음한 파일은 다운로드할 수 있습니다.");
      return;
    }
    const mime = selected.blob.type;
    const extension = mime.includes("ogg") ? "ogg" : mime.includes("mp4") ? "m4a" : "webm";
    downloadBlob(selected.blob, `${safeFilename(selected.title)}.${extension}`);
  };

  const downloadTranscript = () => {
    if (selected.hasAudio) {
      window.location.href = `/stt-api/api/recordings/${selected.id}/transcript`;
      return;
    }
    const contents = [
      selected.title,
      `기록 일시: ${selected.createdAt}`,
      `재생 시간: ${formatTime(selected.duration)}`,
      "",
      ...selected.segments.map((segment) => `[${formatTime(segment.start)}] ${segment.text}`),
    ].join("\n");
    downloadBlob(
      new Blob(["\uFEFF", contents], { type: "text/plain;charset=utf-8" }),
      `${safeFilename(selected.title)}_인식텍스트.txt`,
    );
  };

  const clearRecordings = async () => {
    try {
      await fetch("/stt-api/api/recordings", { method: "DELETE" });
      setCapabilities((current) => current
        ? { ...current, recordingCount: 0 }
        : current);
      setRecordings([demoRecording]);
      setSelected(demoRecording);
      setPlaybackElapsed(0);
      setLiveText(demoRecording.segments.map((segment) => segment.text).join(" "));
      setNotice("저장된 데모 기록을 초기화했습니다.");
    } catch {
      setNotice("기록을 초기화하지 못했습니다.");
    }
  };

  const reviewSegment = async (segmentIndex: number, text: string) => {
    if (!selected.hasAudio) {
      setNotice("서버에 저장된 녹음에서만 자막을 수정할 수 있습니다.");
      return;
    }
    const normalizedText = text.trim();
    if (!normalizedText) return;
    try {
      const response = await fetch(
        `/stt-api/api/recordings/${selected.id}/segments/${segmentIndex}`,
        {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: normalizedText }),
        },
      );
      if (!response.ok) throw new Error("review failed");
      const updated = await response.json() as Recording;
      setSelected(updated);
      setRecordings((current) => current.map((item) => (
        item.id === updated.id ? updated : item
      )));
      setLiveText(updated.segments.map((segment) => segment.text).join(" "));
      setNotice("검토한 자막을 저장했습니다.");
    } catch {
      setNotice("검토한 자막을 저장하지 못했습니다.");
    }
  };

  const activeSegment = selected.segments.findIndex(
    (segment) => playbackElapsed >= segment.start && playbackElapsed < segment.end,
  );
  const reviewCount = selected.segments.filter(
    (segment) => segment.reviewRequired,
  ).length;
  const meterBars = Array.from({ length: 18 });
  const displayedWaveform = selected.waveform?.length
    ? selected.waveform
    : demoWaveform;

  return (
    <main className="shell">
      <header className="topbar">
        <div className="brand">
          <div className="brandMark"><span /><span /><span /></div>
          <div><strong>VOICE COMMAND</strong><small>AI RECOGNITION STUDIO</small></div>
        </div>
        <div className={`systemStatus ${systemOnline ? "" : "offline"}`}>
          <i /> {systemOnline ? "SYSTEM ONLINE" : "STT OFFLINE"} <span>{modelName.toUpperCase()}</span>
        </div>
        <button className="profile" aria-label="도움말 열기" onClick={() => setHelpOpen(true)}>?</button>
      </header>

      <section className="hero">
        <div>
          <div className="eyebrow"><span>LIVE</span> KOREAN SPEECH ENGINE · LOW LATENCY</div>
          <h1>VOICE <em>INTELLIGENCE</em><br />COMMAND CENTER</h1>
          <p>실시간 음성 신호를 분석하고, 정확한 한국어 텍스트로 기록합니다.</p>
        </div>
        <div className="signalOrb" aria-hidden="true">
          <div className="orbit orbitOne" /><div className="orbit orbitTwo" />
          <div className="orbCore">AI<small>STT</small></div>
        </div>
      </section>

      <section className="consoleGrid">
        <article className={`panel capturePanel ${recording ? "isRecording" : ""}`}>
          <div className="panelHead">
            <div><span className="step">01</span><div><h2>LIVE CAPTURE</h2><p>실시간 음성 입력</p></div></div>
            <span className="secure">● {systemOnline ? "LOCAL STT READY" : "STT DEGRADED"}</span>
          </div>

          <div className="meters">
            {(["left", "right"] as const).map((channel) => (
              <div className="meterRow" key={channel}>
                <strong>{channel === "left" ? "L" : "R"}</strong>
                <div className="ledTrack">
                  {meterBars.map((_, index) => {
                    const active = index / meterBars.length < levels[channel];
                    return <i key={index} className={active ? "on" : ""} />;
                  })}
                </div>
                <span>{Math.round(-60 + levels[channel] * 60)} dB</span>
              </div>
            ))}
          </div>

          <div className="liveTranscript">
            <div className="liveLabel"><i /> {captureStatus.toUpperCase()}</div>
            <p>{liveText || <span className="muted">음성을 분석하고 있습니다…</span>} <mark>{interimText}</mark></p>
          </div>

          <div className="recordArea">
            <button
              className={`recordButton ${recording ? "active" : ""}`}
              onClick={recording ? stopRecording : startRecording}
              aria-label={recording ? "녹음 중지" : "음성인식 시작"}
              disabled={[
                "requesting_permission",
                "connecting",
                "stopping",
                "saving",
              ].includes(captureStatus)}
            >
              <span className="recordIcon">{recording ? "■" : "●"}</span>
              <span>{recording ? "STOP & SAVE" : "START RECOGNITION"}<small>{recording ? "녹음 종료 및 저장" : "음성인식 시작"}</small></span>
            </button>
            <div className="timer">{formatTime(recordingElapsed)}<small>REC TIME</small></div>
          </div>
          {notice && <div className="notice">{notice}</div>}
        </article>

        <article className={`panel archivePanel ${recording ? "isLocked" : ""}`}>
          <div className="panelHead">
            <div><span className="step orange">02</span><div><h2>RECORD ARCHIVE</h2><p>음성·텍스트 통합 기록</p></div></div>
            <button className="archiveClear" onClick={clearRecordings} disabled={recording}>
              CLEAR · {recordings.length.toString().padStart(2, "0")} FILES
            </button>
          </div>
          <div className="archiveList">
            {recordings.map((item) => (
              <button className={`archiveItem ${selected.id === item.id ? "selected" : ""}`} key={item.id} onClick={() => selectRecording(item)} disabled={recording}>
                <span className="fileIcon">▥</span>
                <span className="fileCopy"><strong>{item.title}</strong><small>{item.createdAt} · {formatTime(item.duration)}</small></span>
                <span className="filePlay">▶</span>
              </button>
            ))}
          </div>
        </article>
      </section>

      <section className={`panel playbackPanel ${recording ? "isLocked" : ""}`}>
        <div className="playbackTop">
          <div className="nowPlaying">
            <span className="step purple">03</span>
            <div><small>{recording ? "ARCHIVE PAUSED" : "NOW ANALYZING"}</small><h2>{selected.title}</h2></div>
          </div>
          <div className="playbackActions">
            <div className="chips">
              <span>KO-KR</span>
              <span>{modelName.toUpperCase()}</span>
              {reviewCount > 0 && <span className="reviewChip">REVIEW {reviewCount}</span>}
              <span
                title={[
                  `AudioContext: ${audioDiagnostics.contextState}`,
                  `Sample rate: ${audioDiagnostics.sampleRate || "-"} Hz`,
                  `Channels: ${audioDiagnostics.channels}`,
                  `RMS: ${audioDiagnostics.rms.toFixed(4)}`,
                  `Peak: ${audioDiagnostics.peak.toFixed(4)}`,
                  `Level: ${audioDiagnostics.decibels.toFixed(1)} dB`,
                ].join("\n")}
              >
                {audioDiagnostics.channels > 1 ? "STEREO" : "MONO"}
              </span>
            </div>
            <button className="downloadButton" onClick={downloadTranscript} aria-label="인식 텍스트 다운로드" disabled={recording}>
              ↓ <span>TEXT</span>
            </button>
            <button
              className="downloadButton audio"
              onClick={downloadAudio}
              aria-label="녹음 음성 파일 다운로드"
              disabled={recording}
              title={selected.blob || selected.hasAudio ? "음성 파일 다운로드" : "직접 녹음한 파일에서 사용할 수 있습니다"}
            >
              ↓ <span>AUDIO</span>
            </button>
          </div>
        </div>

        <div className="transport">
          <button
            className="playMain"
            onClick={playSelected}
            aria-label={playing ? "일시정지" : "재생"}
            disabled={recording}
          >
            {playing ? "Ⅱ" : "▶"}
          </button>
          <div className="timelineWrap">
            <div className="waveform" aria-hidden="true">
              {displayedWaveform.map((amplitude, index) => (
                <i
                  key={index}
                  style={{ height: `${Math.max(4, Math.round(amplitude * 54))}px` }}
                />
              ))}
              <div className="waveProgress" style={{ width: `${Math.min(100, (playbackElapsed / selected.duration) * 100)}%` }} />
            </div>
            <input
              type="range"
              min="0"
              max={selected.duration}
              step="0.05"
              value={playbackElapsed}
              disabled={recording}
              onChange={(event) => {
                const value = Number(event.target.value);
                setPlaybackElapsed(value);
                if (player.current) player.current.currentTime = value;
              }}
              aria-label="재생 위치"
            />
            <div className="timecodes">
              <span>{formatTime(playbackElapsed)}</span>
              <span>{formatTime(selected.duration)}</span>
            </div>
          </div>
        </div>

        <div className="transcriptBox">
          <div className="transcriptHead">
            <span>{recording ? "ARCHIVE TRANSCRIPT" : "LIVE TRANSCRIPT"}</span>
            <span><i /> {recording ? "PLAYBACK PAUSED" : "SYNCED WITH AUDIO"}</span>
          </div>
          <div className="transcriptText">
            {selected.segments.map((segment, index) => (
              <div className="transcriptSegmentWrap" key={`${segment.start}-${index}`}>
                <button
                  className={[
                    "transcriptSegment",
                    activeSegment === index && (playing || playbackElapsed > 0)
                      ? "highlight"
                      : "",
                    segment.reviewRequired ? "needsReview" : "",
                  ].filter(Boolean).join(" ")}
                  title={
                    segment.rawText && segment.rawText !== segment.text
                      ? `원본 인식: ${segment.rawText}`
                      : undefined
                  }
                  onClick={() => {
                    if (recording) return;
                    setPlaybackElapsed(segment.start);
                    if (player.current) player.current.currentTime = segment.start;
                  }}
                  disabled={recording}
                >
                  <time>{formatTime(segment.start)}</time>
                  <span>{segment.text}</span>
                  {segment.reviewRequired && (
                    <small>
                      검토 필요
                      {segment.reviewReasons?.length
                        ? ` · ${segment.reviewReasons
                            .map((reason) => REVIEW_REASON_LABELS[reason] || reason)
                            .join(", ")}`
                        : ""}
                    </small>
                  )}
                  {segment.rawText && segment.rawText !== segment.text && (
                    <small className="rawTranscript">원본: {segment.rawText}</small>
                  )}
                </button>
                {segment.reviewRequired && selected.hasAudio && (
                  <div className="reviewActions">
                    {segment.rawText && segment.rawText !== segment.text && (
                      <button onClick={() => reviewSegment(index, segment.rawText || "")}>
                        원본 적용
                      </button>
                    )}
                    <button onClick={() => {
                      const edited = window.prompt("확정할 자막을 입력하세요.", segment.text);
                      if (edited) void reviewSegment(index, edited);
                    }}>
                      직접 수정
                    </button>
                    <button onClick={() => reviewSegment(index, segment.text)}>
                      교정 확정
                    </button>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      </section>

      <footer>
        <span>VOICE COMMAND CENTER · DEMO BUILD 1.0</span>
        <span><i /> SQLITE + LOCAL WAV STORAGE</span>
        <span>
          LATENCY <b>{latencyMs === null ? "--" : `${latencyMs}ms`}</b>
          {" · "}RTF <b>{realtimeFactor === null ? "--" : realtimeFactor.toFixed(2)}</b>
        </span>
      </footer>
      {helpOpen && (
        <div className="helpBackdrop" role="presentation" onClick={() => setHelpOpen(false)}>
          <section className="helpModal" role="dialog" aria-modal="true" onClick={(event) => event.stopPropagation()}>
            <button className="helpClose" onClick={() => setHelpOpen(false)} aria-label="도움말 닫기">×</button>
            <span className="eyebrow">EASYLISTNER · HELP</span>
            <h2>로컬 음성인식 데모 사용법</h2>
            <ol>
              <li><code>./setup_demo.sh</code>을 한 번 실행해 모델과 백엔드를 준비합니다.</li>
              <li><code>./run_demo.sh</code>로 전체 서버를 실행합니다.</li>
              <li>START RECOGNITION을 누르고 마이크 권한을 허용합니다.</li>
              <li>한국어로 말한 뒤 STOP &amp; SAVE를 누릅니다.</li>
              <li>저장 기록에서 음성 재생과 TEXT·AUDIO 다운로드를 확인합니다.</li>
            </ol>
            <div className="capabilityGrid" aria-label="시스템 진단">
              <span>STT 엔진<strong>{capabilities?.stt ? "READY" : "OFFLINE"}</strong></span>
              <span>VAD<strong>{capabilities?.vad || "UNKNOWN"}</strong></span>
              <span>SQLite<strong>{capabilities?.sqlite ? "READY" : "OFF"}</strong></span>
              <span>로컬 음성<strong>{capabilities?.localAudio ? "READY" : "OFF"}</strong></span>
              <span>D1 바인딩<strong>{capabilities?.d1 ? "ON" : "OFF"}</strong></span>
              <span>R2 바인딩<strong>{capabilities?.r2 ? "ON" : "OFF"}</strong></span>
              <span>저장 기록<strong>{capabilities?.recordingCount ?? 0}개</strong></span>
              <span>보존 정책<strong>{capabilities?.retentionDays === 0 ? "무기한" : `${capabilities?.retentionDays ?? 30}일`}</strong></span>
              <span>법률 프롬프트<strong>{capabilities?.legalPrompt ? "ON" : "OFF"}</strong></span>
              <span>교정 사전<strong>{capabilities?.legalCorrections ?? 0}개</strong></span>
              <span>접속 주소<strong>{accessInfo.origin}</strong></span>
              <span>보안 컨텍스트<strong>{accessInfo.secure ? "HTTPS/LOCAL" : "HTTP 제한"}</strong></span>
            </div>
            <p>원격 접속에서는 HTTPS 터널이 필요하며, 상태가 STT OFFLINE이면 백엔드와 Whisper 로그를 확인하세요.</p>
          </section>
        </div>
      )}
    </main>
  );
}
