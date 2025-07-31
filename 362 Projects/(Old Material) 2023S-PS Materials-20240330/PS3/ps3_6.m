% Retrieve info in the least significant bit

% Load the image
im3 = imread("3info.png", "png");

% get the data in the LSB, multiply it by 255 to make it visible
im3(:,:,1) = bitand(im3(:,:,1), 1) * 255;
im3(:,:,2) = bitand(im3(:,:,2), 1) * 255;
im3(:,:,3) = bitand(im3(:,:,3), 1) * 255;


imshow(im3);