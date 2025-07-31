% Frequency Modulation Example

% Amplitude
A = 0.3;
% Sample Rate (samples per second)
Hz = 48000;
% Signal Length (in seconds)
s = 3;
% Frequency (in Hertz)
f = 1000;
% Sample list
tt = 0:1 / Hz:s;


% An easy way to modulate frequency is by manipulating the time dimension
% Let time dimension quadratically scale (square each element)
tt = tt.^2;
% This can be interpreted as time flowing faster and faster for the wave
% Which means its frequency is increasing as time goes on
% tt = tt .* sin(tt) u can try this

% Generate sine wave with this time dimension
wave = A * sin(2 * pi * f * tt);
% Listen
% sound(wave, Hz)

% You may want to hear sound from reverse.
wave = fliplr(wave);

% Do not forget to plot what you have prepared.
plot(tt, wave)
spectrogram(wave, hamming(length(wave)/20), [], [], Hz, "yaxis");

% Do not forget to play what you have prepared.
sound(wave, Hz)