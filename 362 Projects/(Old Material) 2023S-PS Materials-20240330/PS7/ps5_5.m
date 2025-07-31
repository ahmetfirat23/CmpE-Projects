img = imread("otter.jpg");

imshow(img);

filter_coeffs = [
    [1/3 -1/3 1/3];
    [-1/3 1/3 -1/3];
    [1/3 -1/3 1/3];
];
num = 15;

% filter_coeffs = ones([num, num])/(num*num);

img = conv2_rgb(img, filter_coeffs);
imshow(img);