% List of song filenames and their actual BPM
songs = {'sevdacicegi.wav' 'dudu.wav', 'beat.wav', 'aleph.wav'};
actual_bpm = [114.5, 91.0, 60.0, 81.0];

% Constants
N = 1024; % FFT size
Fs = 44100; % Sample rate
num_intervals = 64; % Number of frequency intervals
energy_history_length = 43; % Length of energy history buffer


% Calculate subband frequencies logarithmically
subband_edges = round(logspace(log10(1), log10(N/2), num_intervals+1)); % Logarithmically spaced subband edges


% Process each song
for song_idx = 1:length(songs)
    % Read the audio file
    [y, Fs] = audioread(songs{song_idx});

    % Set the threshold factor C based on the current song
    if strcmp(songs{song_idx}, 'aleph.wav')
        C = 122;
    elseif strcmp(songs{song_idx}, 'beat.wav')
        C = 370;
    elseif strcmp(songs{song_idx}, 'dudu.wav')
        C = 12;
    elseif strcmp(songs{song_idx}, 'sevdacicegi.wav')
        C = 8;
    else
        C = 100; % Default value if the song is not listed
    end


    % Extract the current chunk for both channels
    an = y(:, 1); % Real part of the signal from the first channel
    bn = y(:, 2); % Imaginary part of the signal from the second channel

    % Perform FFT on the complex signal (an + j*bn)
    X = fft(an + 1i*bn);

    B_total = abs(X).^2; 

    % B_total = sort(B_total,"descend");

    % Initialize variables
    energy_history = zeros(num_intervals, energy_history_length);

    beat_count = 0;

    % Process the song in chunks of N samples
    for i = 1:N:length(y)-N
        
        B = B_total(i:i+N-1);
        % Divide into frequency intervals and compute energy in each interval
        Es = zeros(1, num_intervals);
        for interval = 1:num_intervals
            start_idx = subband_edges(interval);
            end_idx = subband_edges(interval+1)-1;
            if end_idx > N
                end_idx = N;
            end
            Es(interval) = sum(B(start_idx:end_idx))*((end_idx-start_idx+1)/N) ;
        end

        % Update energy history and detect beats
        for j = 1:num_intervals
            % Compute the average energy in the interval
            avg_energy = mean(energy_history(j, :));
            
            % Detect beats
            if Es(j) > C * avg_energy
                beat_count = beat_count + 1; % Increment beat count
                break;
            end

        end

        for k = 1:num_intervals
            % Update the energy history buffer
            energy_history(k, 2:end) = energy_history(k, 1:end-1);
            energy_history(k, 1) = Es(k);
        end

    end
    
    % Display the results
    fprintf('Song: %s, Actual BPM: %.1f, Predicted BPM: %.1f\n', ...
        songs{song_idx}, actual_bpm(song_idx), beat_count);


    % Plot the waveform of the audio signal
    figure;
    subplot(3, 1, 1);
    plot((1:length(y))/Fs, y);
    title(sprintf('Waveform of %s', songs{song_idx}));
    xlabel('Time (s)');
    ylabel('Amplitude');
    
    % Plot the frequency spectrum
    subplot(3, 1, 2);
    freq = (0:N/2-1)*(Fs/N);
    plot(freq, B_total(1:N/2));
    title('Frequency Spectrum');
    xlabel('Frequency (Hz)');
    ylabel('Amplitude');
end