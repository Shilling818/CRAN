function [outputArg] = l1(inputArg1,inputArg2)
    channel = size(inputArg1,3);
    width = size(inputArg1, 2);
    height = size(inputArg1, 1);
    if channel == 1
        tmp_sum = sum(sum(abs(inputArg1 - inputArg2)));
    elseif channel == 3
        tmp_sum = sum(sum(sum(abs(inputArg1 - inputArg2))));
    end
    outputArg = tmp_sum / channel / width / height;
end

