% these are given values in description
duration = 2;
fundamental_frequency = 100;
amplitude = 0.5;
n_values = [5, 15, 150, 500];

for i = 1:length(n_values)
    n = n_values(i);
    % calculate sample rate according to nyquist theorem
    srate = 2 * fundamental_frequency * n + 1;
    % divide time to samples
    time = 0:1/srate:duration;
    % initialize the approx. square wave
    square_wave = zeros(size(time));
    % formula for the nth order fourier approximation
    for k = 1:2:n
        coeff = (4 * amplitude) / (k * pi);
        square_wave = square_wave + coeff * sin(2 * pi * k * fundamental_frequency * time);
    end

    % plotting and beautifying
    subplot(length(n_values), 1, i);
    plot(time, square_wave, 'LineWidth', 2);
    title(['Square Wave for n = ' num2str(n)]);
    xlabel('Time (s)');
    ylabel('Amplitude');
    % display given ranges
    xlim([0,0.020]) % 20ms
    ylim([-1, 1]);
    grid on;
end
