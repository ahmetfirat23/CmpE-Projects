[y, Fs] = audioread("fifth.wav");
% pick only 1 channel
y = y(:,1);
% transpose
y = y.';

% specgram(y);



filter_coeffs = [1/3 -1/3 1/3];
y = conv(y, filter_coeffs);


sound(y, Fs);
spectrogram(y, 1000,[],[],Fs, 'yaxis');


figure;
plot(y);

