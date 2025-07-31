seconds = 4;
srate = 44000;
tt = 0:1 / srate:seconds;
amplitude = 0.25;
frequency = 440;

wave1 = amplitude*sin(2 * pi * frequency * tt);
wave2 = amplitude*sin(2.55 * pi * frequency * tt);

plot(tt, wave1)

sound(wave1 + wave2, srate)

% soundsc(data, srate)

% [y,Fs] = audioread("my_music.mp3");
% sound(y,Fs);