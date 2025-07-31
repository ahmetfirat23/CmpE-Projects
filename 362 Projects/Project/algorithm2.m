[audio, fs] = audioread('beat.wav');


% Derivation and Combfilter algorithm #1:
start_sample = floor(length(audio) / 2) - 5 * fs;
end_sample = start_sample + 5 * fs - 1;
audio_segment = audio(start_sample:end_sample, :);
sound(audio_segment, fs);
a = audio_segment(:, 1);
b = audio_segment(:, 2);
N = length(a);

complex_signal = real(a) + 1i * imag(b);
transformed = fft(complex_signal);
ta = real(transformed);
tb = imag(transformed);

bpm_range = 60:10:180;
E_bpmc = zeros(1, length(bpm_range));


AmpMax = 32767;
for i = 1:length(bpm_range)
    T_i = 60/bpm_range(i) * fs;

    l = zeros(1, N);
    j = zeros(1, N);
    for k = 1:N
        if mod(k,T_i) == 0
        l(k) = AmpMax;
        j(k) = AmpMax;
        else
        l(k) = 0;
        j(k) = 0;
        end
    end

    tl = fft(l);
    tj = fft(j);

    E_bpmc(i) = sum(abs((ta + 1i * tb) .* (tl + 1i * tj)'));
end

% Find maximum energy
[~, idx] = max(E_bpmc);
bpm = bpm_range(idx);
fprintf('BPM: %d\n', bpm);

% Plot
figure;
plot(bpm_range, E_bpmc);
xlabel('BPM');
ylabel('Energy');
title('Energy vs. BPM');
grid on;


% Frequency selected processing combfilters algorithm #1:
start_sample = floor(length(audio) / 2) - 5 * fs;
end_sample = start_sample + 5 * fs - 1;
audio_segment = audio(start_sample:end_sample, :);
sound(audio_segment, fs);
a = audio_segment(:, 1);
b = audio_segment(:, 2);

N = length(a);
subband = 32;
f_min = 20;
f_max = fs/2;


da = a;
db = b;
for i = 2:N-1
    da(i) = 1/2 * fs * (a(i+1) - a(i-1));
    db(i) = 1/2 * fs * (b(i+1) - b(i-1));
end
complex_signal = real(da) + 1i * imag(db);
transformed = fft(complex_signal);
ta = real(transformed);
tb = imag(transformed);

tas = cell(1, subband);
tbs = cell(1, subband);
subband_edges = logspace(log10(f_min), log10(f_max), subband + 1);
fft_freqs = (0:N-1) * (fs / N);
for i = 1:subband
    subband_indices = (fft_freqs >= subband_edges(i)) & (fft_freqs < subband_edges(i + 1));
    tas{i} = ta(subband_indices);
    tbs{i} = tb(subband_indices);
end

bpm_range = 60:10:180;

E_bpmc_s = zeros(subband, length(bpm_range));
AmpMax = 32767;
for x = 1:subband
    % fprintf('x: %d\n', x);
    ws = length(tas{x});
    % fprintf('ws: %d\n', ws)
    for i = 1:length(bpm_range)
        T_i = 60/bpm_range(i) * fs;
        % fprintf('BPM: %d\n', bpm_range(i));
        l = zeros(1, ws);
        j = zeros(1, ws);
        % fprintf('T_i: %d\n', T_i)
        for k = 1:ws
            if mod(k,T_i) == 0
            l(k) = AmpMax;
            j(k) = AmpMax;
            else
            l(k) = 0;
            j(k) = 0;
            end
        end

        tl = fft(l);
        tj = fft(j);   

        E_bpmc_s(x, i) = sum(abs((tas{x} + 1i * tbs{x}) .* (tl + 1i * tj)'));
        % fprintf('E_bpmc_s: %d\n', E_bpmc_s(x, i));
        
    end
end

E_maxs = zeros(1, subband);
BPM_maxs = zeros(1, subband);
for x = 1:subband
    % Find maximum energy
    [E_max, idx] = max(E_bpmc_s(x,:));

    bpm = bpm_range(idx);
    E_maxs(x) = E_max;
    BPM_maxs(x) = bpm;
end

BPM = 1/sum(E_maxs) * sum(BPM_maxs .* E_maxs);
fprintf('Decided BPM: %d\n', BPM);

figure;
plot(bpm_range, E_bpmc_s);
xlabel('BPM');
ylabel('Energy');
title('Energy vs. BPM');
grid on;


