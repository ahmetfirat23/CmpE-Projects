fs = 100;                                % sample frequency (Hz)
t = 0:1/fs:10-1/fs;                      % 10 second span time vector
x = (1.3)*sin(2*pi*5*t+pi/4) ...             % 5 Hz component
  + (2.7)*cos(2*pi*15*t)  ;           % 15 Hz component
  
  %+ 2.5*gallery('normaldata',size(t),4); % Gaussian noise;

subplot(1,4,1)
plot(t,x)
xlabel('time')
ylabel('signal')

y = fft(x);             %discrete fourier transform

n = length(x);          % number of samples
f = (0:n-1)*(fs/n);     % frequency range
subplot(1,4,2)
plot(f,abs(y)/n)
xlabel('Frequency')
ylabel('Amplitude')

y0 = fftshift(y);         % shift y values
f0 = (-n/2:n/2-1)*(fs/n); % 0-centered frequency range

subplot(1,4,3)

plot(f0,abs(y0)/n)
xlabel('Frequency')
ylabel('Amplitude')

tol = 1e-6;
y0(abs(y0) < tol) = 0;

angle0 = angle(y0);    % 0-centered phase

subplot(1,4,4)

plot(f0,180*angle0/pi)
xlabel('Frequency')
ylabel('Phase')