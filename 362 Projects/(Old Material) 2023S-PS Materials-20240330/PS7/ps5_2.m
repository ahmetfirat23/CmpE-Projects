Fs = 44100;
tt = 0:1/Fs:3;
f = 440;
y = cos(2*pi*f*tt);
y = y + cos(4*pi*f*tt);
y = y + cos(6*pi*f*tt);

total_samples = size(tt, 2);
% sound(y, Fs);

% let's see the frequency spectrum
show_spectrum(y, Fs);

% let's apply a filter
filter_coeffs = [1/3 1/3 1/3];
filter_coeffs = [1/3 -1/3 1/3];
y = conv(y, filter_coeffs);

show_spectrum(y, Fs);