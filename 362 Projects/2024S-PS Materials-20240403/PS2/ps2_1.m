% Wave summation example

% Number of sine waves in finite computation
n = 15;
% Amplitude
A = 0.5;
% Sample Rate (samples per second) 
sr = 2000;
% Signal length (in seconds)
s = 2;
% Frequency (in Hz)
f = 100;
% Time axis
tt = 0:1 / sr:s;

wave = 0;

for k = 1:n
    wave = wave + cos(k * pi) * sin(2 * pi * f * k * tt) / k;
end

wave = wave * ((-2 * A) / pi);

plot(tt, wave)
axis([0 0.0200 -1 1]);

spectrogram(wave, hamming(length(wave)/5), [], [], sr, "yaxis");
% spectrogram(wave, [], [], [], sr, "yaxis");

% sound(wave, Hz);
