% Amplitude Modulation Example 1

% Amplitude
A = 0.3;
% Sample Rate (samples per second)
Hz = 48000;
% Signal length (in seconds)
s = 5;
% Frequency (in hertz)
f = 700;
% Time axis
tt = 0:1 / Hz:s;
% Regular sine wave
carrier_wave = A * sin(2 * pi * f * tt);


% Shaper function for modulator. You may try another shapes.
envelope = sin(2*pi*3*tt);
% Set zero negative values. This is used to make quiet spaces between waves.
% envelope = arrayfun(@(x) max(x, 0), envelope);

% Element-wise multiplictaion between regular waves and amplitude modulator.
modulate_wave = carrier_wave .* envelope;

% Do not forget to plot what you have prepared.
tiledlayout(3,1)
nexttile
plot(tt, carrier_wave)
title('Carrier')
nexttile
plot(tt, envelope)
title('Envelope')
nexttile
plot(tt, modulate_wave)
title('Modulated')

% Do not forget to play what you have prepared.

% sound(carrier_wave, Hz)
% sound(envelope, Hz)
% sound(modulate_wave, Hz)