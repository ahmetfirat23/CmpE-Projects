% Image slicing / cutting

% Load image
im1 = imread("1.png", "png");

% For ease of use, let's use grayscale
im1 = rgb2gray(im1);

% Keep these for later use
height = size(im1, 1);
width = size(im1, 2);

% Duplicate first vertical half into second vertical half

% im1(:,(1:width/2) + width/2) = im1(:, 1:width/2);
% imshow(im1);

% Shuffle columns

% im1 = im1(:, randperm(height));
% imshow(im1);

% Load the second image into first half of first image (sizes should match)
% 
% im2 = rgb2gray(imread("2.png", "png"));
% 
% im1(:,(1:width/2) + width/2) = im2(:, 1:width/2);
% imshow(im1);







