function output = IdealLF(input, D0)
    %利用巴特沃斯低通滤波器对图像进行滤波
    M = 2 * size(input, 1);  %滤波器行数
    N = 2 * size(input, 2);  %滤波器列数
    u = - M/2:(M/2 - 1);
    v = - N/2:(N/2 - 1);
    [U,V] = meshgrid(u,v);
    D = sqrt(U.^2 + V.^2);
    H = zeros(size(D,1),size(D,2));
    H(D <= D0) = 1;
    J = fftshift(fft2(input, size(H,1),size(H,2)));  %转换到频域
    K = J.*H;
    L = ifft2(ifftshift(K));  %傅立叶反变换
    L = L(1:size(input,1), 1:size(input, 2), :);  %改变图像大小
    output = L;
end