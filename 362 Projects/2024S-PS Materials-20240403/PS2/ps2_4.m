% Amplitude Modulation Example 2

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


% Lets encode binary information into our carrier wave
% This is similar to morse code.
% create a row vector consisting of 0's
envelope = zeros(1, length(tt));
% lets put 3 binary 8 bit numbers in it
% numbers are 11101100, 10010011, 00110011 
% let each digit take 0.1 secs
seconds_per_digit = 0.1;
samples_per_digit = length(0 : 1/Hz : seconds_per_digit);
curr_offset = 0;
for k = [1 1 1 0 1 1 0 0 1 0 0 1 0 0 1 1 0 0 1 1 0 0 1 1]
    indices = 1 + curr_offset : samples_per_digit + curr_offset;
    envelope(indices) = k;
    curr_offset = curr_offset + samples_per_digit;
end


% Element-wise multiplictaion between carrier wave and amplitude modulator.
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
sound(modulate_wave, Hz)