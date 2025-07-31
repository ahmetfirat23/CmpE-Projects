% Image slicing cont'd & resampling

% Load image
im1 = imread("1.png", "png");

% For ease of use, let's use grayscale
im1 = rgb2gray(im1);

% Keep these for later use
height = size(im1, 1);
width = size(im1, 2);

% Resize the image twice the size?
new_im = zeros(width * 2, height * 2, "uint8");
new_im(1:2:height*2, 1:2:width*2) = im1;
new_im(2:2:height*2, 2:2:width*2) = im1;
% imshow(new_im);

% Whats the problem?
% New image has 4 times the volume of first, we only added 2

% We need two more combinations:

new_im(1:2:height*2, 2:2:width*2) = im1;
new_im(2:2:height*2, 1:2:width*2) = im1;

imshow(new_im);

% This called resampling (upsampling to be more specific)




