img = imread("img1.png");
img = rgb2gray(img);

imshow(img);

filter_coeffs = [
    [1/3 1/3 1/3];
    [1/3 1/3 1/3];
    [1/3 1/3 1/3];
];
num = 15;

filter_coeffs = ones([num, num])/(num*num);

img = conv2(img, filter_coeffs);
imshow(img);