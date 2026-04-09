overlaps = [1, 2, 4];
m_test = 4;
mask = zeros(8, 8);
mask(1:m_test, 1:m_test) = 1;
figure('Name', sprintf('Επίδραση επικάλυψης για m=%d', m_test), 'Position', [150, 150, 1000, 400]);
for o_idx = 1:length(overlaps)
    overlap = overlaps(o_idx);
    stride = 8 - overlap;
    img_recon_ov = zeros(size(img_padded));
    weight_map = zeros(size(img_padded));
    for c = 1:C
        channel = img_padded(:,:,c);
        for i = 1:stride:(size(channel, 1) - 7)
            for j = 1:stride:(size(channel, 2) - 7)
                block = channel(i:i+7, j:j+7);
                D = dct2(block);
                D_masked = D .* mask;
                block_inv = idct2(D_masked);
                img_recon_ov(i:i+7, j:j+7, c) = img_recon_ov(i:i+7, j:j+7, c) + block_inv;
                weight_map(i:i+7, j:j+7, c) = weight_map(i:i+7, j:j+7, c) + 1;
            end
        end
    end
    img_recon_ov = img_recon_ov ./ weight_map;
    img_final_ov = img_recon_ov(1:H, 1:W, :);
    img_final_ov_uint8 = uint8(img_final_ov);
    mse_val = immse(img_final_ov_uint8, img);
    psnr_val = psnr(img_final_ov_uint8, img);
    subplot(1, 3, o_idx);
    imshow(img_final_ov_uint8);
    title(sprintf('Overlap=%d | MSE: %.1f | PSNR: %.2fdB', overlap, mse_val, psnr_val));
end