seconds = 4;
srate = 44000;
sample = 0:1 / srate:seconds;
amplitude = 0.25;
frequency = 440;

data = amplitude*sin(sample * 2 * pi * frequency);

plot(data)

sound(data, srate)

% soundsc(data, srate)

% [y,Fs] = audioread("my_music.mp3");
% sound(y,Fs);