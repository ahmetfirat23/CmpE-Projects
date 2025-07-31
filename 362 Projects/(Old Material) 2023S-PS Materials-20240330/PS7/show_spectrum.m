function show_spectrum(y, Fs)
    N = size(y, 2);
    x = fftshift(fft(y));
    dF = Fs/N;
    f = -Fs/2:dF:Fs/2-dF;     
    figure;
    plot(f,abs(x)/N);
    xlabel('Frequency (in hertz)');
    title('Magnitude');
end