function output = GaussianHF(input, n, d0)
    h = size(input, 1);
    w = size(input, 2);
%     n = 1;
%     d0 = 30;
    u = - h/2:(h/2 - 1);
    v = - w/2:(w/2 - 1);
    [U,V] = meshgrid(u, v);
    D = sqrt(U.^2 + V.^2);
    H = 1 - exp(- (D./ d0).^n / 2);
    J = fftshift(fft2(input, size(H,1),size(H,2)));  %ת����Ƶ��
    K = J.*H;
    L = ifft2(ifftshift(K));  %����Ҷ���任
    L = L(1:size(input,1), 1:size(input, 2), :);  %�ı�ͼ���С
    output = L;
end