function bpm = predict_bpm(song, maxfreq)

% PREDICT_BPM predicts the BPM of a given song.
%
%     BPM = PREDICT_BPM(SONG, MAXFREQ) takes in the name of a .wav file,
%     as a string, and outputs its BPM. MAXFREQ is used to divide the signal
%     for beat-matching.
%
%     Default MAXFREQ is 4096.

  % Set default values if not provided
  if nargin < 2, maxfreq = 4096; end
  
  % Length (in samples) of 2.2 seconds of the song
  sample_size = floor(2.2 * 2 * maxfreq);
  
  % Load the wave file
  [x, fs] = audioread(song);  % Use audioread instead of wavread
  
  % Extract a 2.2-second representative sample from the middle of the song
  start = floor(length(x) / 2 - sample_size / 2);
  stop = floor(length(x) / 2 + sample_size / 2);
  sample = x(start:stop);
  
  % Define 16 subbands
  bandlimits = linspace(0, maxfreq, 17);
  
  % Implement beat detection algorithm
  a = filterbank(sample, bandlimits(1:16), maxfreq); % Adjust bandlimits to have 16 elements
  b = hwindow(a, 0.2, bandlimits(1:16), maxfreq); % Adjust bandlimits to have 16 elements
  c = diffrect(b, 16); % nbands is 16
  
  % Recursively call timecomb to increase accuracy
  d = timecomb(c, 2, 60, 240, bandlimits(1:16), maxfreq); % Adjust bandlimits to have 16 elements
  e = timecomb(c, 0.5, d-2, d+2, bandlimits(1:16), maxfreq); % Adjust bandlimits to have 16 elements
  f = timecomb(c, 0.1, e-0.5, e+0.5, bandlimits(1:16), maxfreq); % Adjust bandlimits to have 16 elements
  g = timecomb(c, 0.01, f-0.1, f+0.1, bandlimits(1:16), maxfreq); % Adjust bandlimits to have 16 elements
  
  % Output the BPM
  bpm = g;
  
  % Print the predicted BPM
  fprintf('Predicted BPM: %.2f\n', bpm);
end
