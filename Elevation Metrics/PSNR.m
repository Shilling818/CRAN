function [outputArg] = PSNR(inputArg1,inputArg2)
    mse = immse(inputArg1, inputArg2);
    peakVal = 255;
    outputArg = 20 * log10(peakVal / sqrt(mse));
end