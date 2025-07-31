% these are given values in description
duration = 2;
fundamental_frequency = 100;
amplitude = 0.5;
n_values = [5, 15, 150, 500];

%initialize error array
overshoot_error = zeros(1,4);

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
    % calculate error using given formula
    overshoot_error(i) = (max(square_wave) - amplitude) / (2 * amplitude);
end

% Plot error and beautify
plot(n_values, overshoot_error, 'o--', 'LineWidth', 1);
title('Overshoot Error vs. n');
xlabel('n');
ylabel('(Max Amplitude - Amplitude) / (2 * Amplitude)');
ylim([0.08, 0.1]);
grid on;
