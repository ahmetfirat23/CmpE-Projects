% Image resampling cont'd

% Load image
im1 = imread("1.png", "png");

% For ease of use, let's use grayscale
im1 = rgb2gray(im1);

% Keep these for later use
height = size(im1, 1);
width = size(im1, 2);

% Downsample the image to half of its size
new_im = im1(1:2:height,1:2:width);

% Create a version of the image with 4 copies
new_im = [new_im new_im ; new_im new_im];
imshow(new_im);





