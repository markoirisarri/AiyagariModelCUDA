function [x,f] = golden_section_search(f, lower, upper, tol)

% This code implements the same golden section search routine
% as in the original CUDA code

max_x = upper;
min_x = lower;

max_x = max_x .* (max_x > min_x) + min_x .* (max_x <= min_x);

d = max_x - min_x;

alpha_1 = (3.0 - sqrt(5.0)) / 2.0;
alpha_2 = 1.0 - alpha_1;

x1 = min_x + alpha_1 .* d;
x2 = min_x + alpha_2 .* d;

f1 = f(x1);
f2 = f(x2);

% Update values based on which (f2, f1) is greater

max_x_value = 1;

while(max_x_value > tol)

    x2_old = x2;
    x1_old = x1;
    f1_old = f1;
    f2_old = f2;

    max_x = max_x .* (f2_old > f1_old) + x2_old .* (f2_old <= f1_old);
    min_x = x1_old .* (f2_old > f1_old) + min_x .* (f2_old <= f1_old);
    x1 = x2_old .* (f2_old > f1_old) + (max_x - alpha_2 .* (max_x - min_x)) .* (f2_old <= f1_old);
    x2 = (min_x + alpha_2 .* (max_x - min_x)) .* (f2_old > f1_old) + x1_old .* (f2_old <= f1_old);

    f2 = f(x2) .* (f2_old > f1_old) + f1_old .* (f2_old <= f1_old);
    f1 = f(x1) .* (f2_old <= f1_old) + f2_old .* (f2_old > f1_old);

    diff = abs(x2-x1);
    max_x_value = max(diff,[],1);

end

x = (x1+x2)/2;
f = (f1+f2)/2;

end
