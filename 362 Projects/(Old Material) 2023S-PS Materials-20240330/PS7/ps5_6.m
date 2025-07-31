img = imread("img1.png");
img = imread("otter.jpg");
img = rgb2gray(img);

o1 = dct(double(img), [], 1);
o2 = dct(o1, [], 2);

% imshow(log(abs(o2)),[])
% colormap parula
% colorbar

o2(abs(o2) < 100) = 0;

imshow(log(abs(o2)),[])
colormap parula
colorbar

i1 = idct(o2, [], 2);
img_recons = uint8(idct(i1, [], 1));

imshow(img_recons)