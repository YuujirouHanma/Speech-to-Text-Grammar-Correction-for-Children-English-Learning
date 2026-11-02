import React, { useEffect, useRef } from 'react';

interface AudioWaveformProps {
  isRecording: boolean;
  isPlaying?: boolean;
  stream?: MediaStream | null;
}

export const AudioWaveform: React.FC<AudioWaveformProps> = ({
  isRecording,
  isPlaying,
  stream,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let localAnalyser: AnalyserNode | null = null;

    if (isRecording && stream) {
      try {
        const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
        const audioCtx = new AudioContextClass();
        audioContextRef.current = audioCtx;

        const analyser = audioCtx.createAnalyser();
        analyser.fftSize = 64;
        analyser.smoothingTimeConstant = 0.8;
        localAnalyser = analyser;
        analyserRef.current = analyser;

        const source = audioCtx.createMediaStreamSource(stream);
        source.connect(analyser);
        sourceRef.current = source;
      } catch (e) {
        console.warn('Could not initialize audio analyzer node:', e);
      }
    }

    let phase = 0;

    const render = () => {
      const width = canvas.width;
      const height = canvas.height;
      ctx.clearRect(0, 0, width, height);

      const isDark = document.documentElement.classList.contains('dark');
      const primaryColor = 'rgba(239, 68, 68, 0.95)';
      const secondaryColor = 'rgba(185, 28, 28, 0.75)';

      const barCount = 32;
      const barWidth = 3;
      const gap = (width - barCount * barWidth) / (barCount - 1);

      let dataArray = new Uint8Array(barCount);
      if (localAnalyser) {
        localAnalyser.getByteFrequencyData(dataArray);
      }

      phase += 0.08;

      for (let i = 0; i < barCount; i++) {
        let barHeight = 4;

        if (isRecording) {
          if (localAnalyser) {
            const raw = dataArray[i] || 0;
            barHeight = Math.max(4, (raw / 255) * (height * 0.85));
          } else {
            // Synthetic wave if analyser stream is unavailable
            const wave = Math.sin(phase + i * 0.25) * Math.cos(phase * 0.5 + i * 0.1);
            barHeight = Math.max(6, Math.abs(wave) * (height * 0.75));
          }
        } else if (isPlaying) {
          const wave = Math.sin(phase * 1.5 + i * 0.35);
          barHeight = Math.max(5, (Math.sin(wave) * 0.5 + 0.5) * (height * 0.6));
        } else {
          // Idle state - gentle resting height
          barHeight = 3;
        }

        const x = i * (barWidth + gap);
        const y = (height - barHeight) / 2;

        // Gradient for bars
        const grad = ctx.createLinearGradient(0, y, 0, y + barHeight);
        grad.addColorStop(0, primaryColor);
        grad.addColorStop(1, secondaryColor);

        ctx.fillStyle = isRecording || isPlaying ? grad : (isDark ? 'rgba(148, 163, 184, 0.25)' : 'rgba(203, 213, 225, 0.5)');
        ctx.beginPath();
        ctx.roundRect(x, y, barWidth, barHeight, 2);
        ctx.fill();
      }

      animationFrameRef.current = requestAnimationFrame(render);
    };

    render();

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      if (sourceRef.current) {
        sourceRef.current.disconnect();
      }
      if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
        audioContextRef.current.close().catch(() => {});
      }
    };
  }, [isRecording, isPlaying, stream]);

  return (
    <div className="flex flex-col items-center justify-center">
      <canvas
        ref={canvasRef}
        width={220}
        height={36}
        className="block"
      />
    </div>
  );
};
