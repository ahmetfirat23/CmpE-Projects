% Wave summation example

% Number of sine waves in finite computation
n = 500;
% Amplitude
A = 0.25;
% Sample Rate (samples per second) 
Hz = 44100;
% Signal length (in seconds)
s = 3;
% Frequency (in Hz)
f = 500;
% Time axis
tt = 0:1 / Hz:s;

% Two examples, select 1 or 2
exm = 1;
% sound(sin(2*pi*f*tt), Hz)
% Create square waves
if exm == 1

    wave = 0;
    
    for k = 1:2:n
        wave = wave + sin(2 * pi * f * k * tt) / k;
    end

    wave = wave * ((4 * A) / pi);
    
    plot(tt, wave)
    
    sound(wave, Hz)

% Create triangular waves
elseif exm == 2

    wave = 0;

    for k = 1:n
        wave = wave + cos(k * pi) * sin(2 * pi * f * k * tt) / k;
    end

    wave = wave * ((-2 * A) / pi);

    plot(tt, wave)
    
    sound(wave, Hz)

end