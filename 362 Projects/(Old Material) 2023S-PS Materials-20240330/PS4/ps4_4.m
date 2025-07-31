% more examples
% amplitude modulated signal

srate = 44100;
tt = 0:1/srate:2;
freq = 440;
amplt = 0.25;
snd = amplt * 5 *cos(2*pi*12*tt).*sin(2*pi*freq*tt);

sound(snd , srate);





L = length(tt);
Y = fft(snd);
P2 = abs(Y/L);
P1 = P2(1:L/2+1);
P1(2:end-1) = 2*P1(2:end-1);
f = srate*(0:(L/2))/L;
plot(f,P1);
title("Single-Sided Amplitude Spectrum of X(t)")
xlabel("f (Hz)")
ylabel("|P1(f)|")



% spectrogram(snd, 1000,[],[],srate);

