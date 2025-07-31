function im =  conv2_rgb(img, coeffs)
    im_r = conv2(img(:,:,1), coeffs, "same");
    im_g = conv2(img(:,:,2), coeffs, "same");
    im_b = conv2(img(:,:,3), coeffs, "same");
    im = uint8(cat(3, im_r, im_g, im_b));
end