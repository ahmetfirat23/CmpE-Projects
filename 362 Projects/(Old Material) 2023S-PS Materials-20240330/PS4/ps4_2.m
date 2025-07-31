srate = 44100;
tt = 0:1/srate:2;
freq = 440;
amplt = 1;
snd = amplt * sin(2*pi*freq*tt);

sound(snd , srate);

% needs signal processsing toolkit
spectrogram(snd, 1000,[],[],srate);

