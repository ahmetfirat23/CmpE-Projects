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

% assume each of the notes will be played for 0.2 seconds
% there are 9 notes in total, which means in total, our wave will be 1.8
% seconds
srate = 44000;
note_sample = 0:1/srate:0.2
amplitude = 0.25;
e_note = amplitude*sin(note_sample * 2 * pi * 329.63);
dsharp_note =  amplitude*sin(note_sample * 2 * pi * 311.13);
b_note = amplitude*sin(note_sample * 2 * pi * 246.94);
d_note = amplitude*sin(note_sample * 2 * pi * 293.66);
c_note = amplitude*sin(note_sample * 2 * pi * 261.63);
a_note = amplitude*sin(note_sample * 2 * pi * 220.00);

% this is just concatenation
melody = [e_note dsharp_note e_note dsharp_note e_note b_note d_note c_note a_note];

plot(melody);

sound(melody, srate)

audiowrite("smp.wav", melody, srate);