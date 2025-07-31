duration = 2;
fundamental_frequency = 100;
amplitude = 0.5;



n_values = [5, 15, 150, 500];
for i = 1:length(n_values)
    n = n_values(i);
    srate = 2 * fundamental_frequency * n + 1;
    time = 0:1/srate:duration;
    square_wave = zeros(size(time));
    for k = 1:2:n
        coeff = (4 * amplitude) / (k * pi);
        square_wave = square_wave + coeff * sin(2 * pi * k * fundamental_frequency * time);
    end
    filename = strcat('square_wave_n_', num2str(n), '.wav');
    audiowrite(filename, square_wave, srate);
end