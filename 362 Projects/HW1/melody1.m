% Let's create intro of fur elise,
% Intro of fur elise is comprised of these notes: E – D# – E – D# – E – B –
% D – C – A
% Frequencies of notes:
% E = 329.63
% D# = 311.13
% B = 246.94
% D = 293.66
% C = 261.63
% A = 220.00

% Set random durations
durations = 0.2 + (0.5-0.2) .* rand(1,9);
% Frequencies of notes in melody
frequencies = [329.63 311.13 329.63 311.13 329.63 246.94 293.66 261.63 220.00];
srate = 44000;
amplitude = 0.25;
% start with empty melody array
melody = [];
for i = 1:9
    note_sample = 0:1/srate:durations(i);
    note = amplitude*sin(note_sample * 2 * pi * frequencies(i));
    % concatenate the melody with new note
    melody = [melody note];
end

% sound(melody, srate);

audiowrite("melody1.wav", melody, srate);