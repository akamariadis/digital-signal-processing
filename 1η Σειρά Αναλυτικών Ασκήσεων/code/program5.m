img = imread('face.jpg'); 
img_double = double(img);
[H, W, C] = size(img_double);
pad_H = mod(8 - mod(H, 8), 8);
pad_W = mod(8 - mod(W, 8), 8);
img_padded = padarray(img_double, [pad_H, pad_W], 'replicate', 'post');
figure('Name', 'Ανακατασκευή με μη επικαλυπτόμενα μπλοκ 8x8', 'Position', [100, 100, 1200, 600]);
m_values = 1:8;
for idx = 1:length(m_values)
    m = m_values(idx);
    mask = zeros(8, 8);
    mask(1:m, 1:m) = 1;
    dct_func = @(block_struct) dct2(block_struct.data);
    mask_func = @(block_struct) block_struct.data .* mask;
    idct_func = @(block_struct) idct2(block_struct.data);
    img_recon = zeros(size(img_padded));
    for c = 1:C
        channel = img_padded(:,:,c);
        dct_coeffs = blockproc(channel, [8 8], dct_func);
        masked_coeffs = blockproc(dct_coeffs, [8 8], mask_func);
        recon_channel = blockproc(masked_coeffs, [8 8], idct_func);
        img_recon(:,:,c) = recon_channel;
    end
    img_final = img_recon(1:H, 1:W, :);
    img_final_uint8 = uint8(img_final);
    mse_val = immse(img_final_uint8, img);
    psnr_val = psnr(img_final_uint8, img);
    subplot(2, 4, idx);
    imshow(img_final_uint8);
    title(sprintf('m=%d | MSE: %.1f | PSNR: %.2fdB', m, mse_val, psnr_val), 'FontSize', 10);
end