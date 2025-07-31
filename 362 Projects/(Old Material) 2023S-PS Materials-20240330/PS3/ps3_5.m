% Hide information in least significant bit

% Load the information i want to hide
im4 = rgb2gray(imread("4.png", "png"));
im5 = rgb2gray(imread("5.png", "png"));
im6 = rgb2gray(imread("6.png", "png"));

% I want to hide it in this image
im3 = imread("3.png", "png");


im3_r = im3(:,:,1);
im3_g = im3(:,:,2);
im3_b = im3(:,:,3);

% Logical operations
im4 = uint8(im4 == 0);
im5 = uint8(im5 == 0);
im6 = uint8(im6 == 0);


% Hide the 1's in the LSB of im3
im3(:,:,1) = bitor(bitand(im3_r, 0b11111110), im4);
im3(:,:,2) = bitor(bitand(im3_g, 0b11111110), im5);
im3(:,:,3) = bitor(bitand(im3_b, 0b11111110), im6);

imshow(im3);

imwrite(im3, "3info.png");