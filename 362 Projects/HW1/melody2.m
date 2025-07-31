% Set random durations
durations = 0.2 + (0.5-0.2) .* rand(1,9);
% Set random frequencies
frequencies = 200 + (1000-200) .* rand(1,9);
% Set random amplitudes
amplitudes = 0.2 + (1-0.2) .* rand(1,9)

srate = 44000;
% start with empty melody array
melody = [];
for i = 1:9
    note_sample = 0:1/srate:durations(i)
    note = amplitudes(i)*sin(note_sample * 2 * pi * frequencies(i));
    % concatenate the melody with new note
    melody = [melody note]
end

% sound(melody, srate)

audiowrite("melody2.wav", melody, srate);