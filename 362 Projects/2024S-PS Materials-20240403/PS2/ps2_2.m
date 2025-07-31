% Synthetic Vowel

% sample rate
sr = 44100;
% time dimension
tt = 0:1/sr:2;  
% add the components
wave = exp(1j*2*200*pi*tt).*(771 + 12202j);
wave = wave + exp(1j*2*400*pi*tt).*(-8865 + 28048j);
wave = wave + exp(1j*2*500*pi*tt).*(48001 + 8995j);
wave = wave + exp(1j*2*1600*pi*tt).*(1657 + 13520j);
wave = wave + exp(1j*2*1700*pi*tt).*(4723);


plot(tt, real(wave));
axis([0 0.040 min(real(wave)) max(real(wave))]);
spectrogram(real(wave), [], [], [], sr, 'yaxis');
sound(real(wave), sr);

% sound(real(exp(1j*440*2*pi*tt)), sr);
