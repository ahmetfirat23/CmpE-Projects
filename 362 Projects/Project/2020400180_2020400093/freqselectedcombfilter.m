function output = freqselectedcombfilter(song, maxfreq)
% FREQSELECTEDCOMBFILTER predicts the BPM of a given song.
%
%     BPM = FREQSELECTEDCOMBFILTER(SONG, MAXFREQ) takes name of a .wav file, 
%     as a string, and outputs its BPM. MAXFREQ is used to divide the signal 
%     for beat-matching. This function implements the beat detection algorithm
%     described in https://archive.gamedev.net/archive/reference/programming
%     features/beatdetection/page2.html
%
%     Default MAXFREQ is 44100.
    if nargin == 1, maxfreq = 44100; end

    % Set up the parameters
    % Sample size is 5 seconds
    sample_size = 5 * 2 * maxfreq;
    % Number of bands
    bandlimit_count = 16;

    % Read the audio file
    [audio, ~] = audioread(song);

    % Take the middle 5 seconds of the audio
    sample = audio(floor(length(audio) / 2 - sample_size / 2):floor(length(audio) / 2 + sample_size / 2));

    % Create the band limits logarithmically
    bandlimits = logspace(log10(1), log10(maxfreq), bandlimit_count);
    
    % Implements beat detection algorithm for each song
    a = filterbank(sample, bandlimits, maxfreq);
    b = hwindow(a, .2, bandlimits, maxfreq);
    c = diffrect(b, bandlimit_count);
    % Recursively calls timecomb to decrease computational time
    d = timecomb(c, 2, 60, 240, bandlimits, maxfreq);
    e = timecomb(c, .5, d-2, d+2, bandlimits, maxfreq);
    f = timecomb(c, .1, e-.5, e+.5, bandlimits, maxfreq);
    g = timecomb(c, .01, f-.1, f+.1, bandlimits, maxfreq);
    % Output the found BPM
    output = g;


