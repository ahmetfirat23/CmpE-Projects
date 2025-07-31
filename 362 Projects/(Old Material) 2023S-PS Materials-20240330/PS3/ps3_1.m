% Image data structure & Image reading

% Generally, images have 3 channels: R G B
% Which means data shape of an RGB image is (Height)x(Width)x3 
% Normally we think in terms of WxH, however, 
% this is matlab and images are
% thought of as matrices, therefore, dimensions are HxW
% Where each entry is of the type uint8 
% (uint8: non-negative numbers from 0 to 255)
im1 = imread("1.png", "png");
im2 = imread("2.png", "png");

% Let's separate the image into channels
im1_r = im1(:,:,1);
im1_g = im1(:,:,2);
im1_b = im1(:,:,3);
im2_r = im2(:,:,1);
im2_g = im2(:,:,2);
im2_b = im2(:,:,3);

% Let's put im2_r into blue channel of the first image

% im1(:,:,3) = im2_r;
% imshow(im1);

% Let's increase the redness of the first image:

% im1(:,:,1) = 2*im1_r; % values bigger than 255 are capped at 255
% imshow(im1);

% Let's blend two images together

% im1(:,:,1) = (im1_r + im2_r) / 2;
% im1(:,:,2) = (im1_g + im2_g) / 2;
% im1(:,:,3) = (im1_b + im2_b) / 2;
% imshow(im1); 

% What is the problem here?
% Problem is that values cap at 255 (similar to overflow)


% Quick and dirty solution
im1(:,:,1) = (im1_r/2 + im2_r/2);
im1(:,:,2) = (im1_g/2 + im2_g/2);
im1(:,:,3) = (im1_b/2 + im2_b/2);
imshow(im1); 









