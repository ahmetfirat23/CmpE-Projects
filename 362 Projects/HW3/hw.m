[audio, fs] = audioread('audio.wav');

lpFilt = designfilt('lowpassfir', 'PassbandFrequency', 480, 'StopbandFrequency', 520, 'SampleRate', fs);
kick_audio = filter(lpFilt, audio);
audiowrite('kick.wav', kick_audio, fs);

bpFilt = designfilt('bandpassfir', 'StopbandFrequency1', 480, 'PassbandFrequency1', 520, 'PassbandFrequency2', 4000, 'StopbandFrequency2', 4050, 'SampleRate', fs);
piano_audio = filter(bpFilt, audio);
audiowrite('piano.wav', piano_audio, fs);

hpFilt = designfilt('highpassfir', 'StopbandFrequency', 3950, 'PassbandFrequency', 4000, 'SampleRate', fs);
cymbal_audio = filter(hpFilt, audio);
audiowrite('cymbal.wav', cymbal_audio, fs);

% frequency responses
figure;
[h, f] = freqz(lpFilt, 'half', fs);
plot(f, 20*log10(abs(h)));
title('Low-Pass Filter Frequency Response (Drum Kick)');
xlabel('Frequency (Hz)');
ylabel('Magnitude (dB)');
grid on;
saveas(gcf, 'low_pass_response.png');
close(gcf);

figure;
[h, f] = freqz(bpFilt, 'half', fs);
plot(f, 20*log10(abs(h)));
title('Band-Pass Filter Frequency Response (Piano)');
xlabel('Frequency (Hz)');
ylabel('Magnitude (dB)');
grid on;
saveas(gcf, 'band_pass_response.png');
close(gcf);

figure;
[h, f] = freqz(hpFilt, 'half', fs);
plot(f, 20*log10(abs(h)));
title('High-Pass Filter Frequency Response (Cymbal)');
xlabel('Frequency (Hz)');
ylabel('Magnitude (dB)');
grid on;
saveas(gcf, 'high_pass_response.png');
close(gcf);

% waveforms
figure;
subplot(4,1,1);
plot(audio);
title('Original Audio Waveform');
xlabel('Sample Number');
ylabel('Amplitude');

subplot(4,1,2);
plot(kick_audio);
title('Drum Kick Waveform');
xlabel('Sample Number');
ylabel('Amplitude');

subplot(4,1,3);
plot(piano_audio);
title('Piano Waveform');
xlabel('Sample Number');
ylabel('Amplitude');

subplot(4,1,4);
plot(cymbal_audio);
title('Cymbal Waveform');
xlabel('Sample Number');
ylabel('Amplitude');

saveas(gcf, 'waveforms.png');
close(gcf);

% spectrograms
figure;
subplot(4,1,1);
spectrogram(audio, 256, [], [], fs, 'yaxis');
title('Original Audio Spectrogram');

subplot(4,1,2);
spectrogram(kick_audio, 256, [], [], fs, 'yaxis');
title('Drum Kick Spectrogram');

subplot(4,1,3);
spectrogram(piano_audio, 256, [], [], fs, 'yaxis');
title('Piano Spectrogram');

subplot(4,1,4);
spectrogram(cymbal_audio, 256, [], [], fs, 'yaxis');
title('Cymbal Spectrogram');

saveas(gcf, 'spectrograms.png');
close(gcf);
