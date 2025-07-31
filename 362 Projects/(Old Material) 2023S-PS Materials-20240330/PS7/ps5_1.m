% lets see the magnitude response of a filter

my_filter = [1/3 -1/3 1/3];
% my_filter = ones(11)/11;
order = size(my_filter,2);
w_coeffs = 1:order;
w = -pi:1/100:pi;

% w_coeffs.' takes the transpose
% then vector multiplication
y = w_coeffs.'*w;


y = exp(-1i*y);

y = my_filter * y;

y = abs(y);

figure;
plot(w,y);
