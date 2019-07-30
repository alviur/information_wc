function [x_lin T] = marginal_uniformization_hardcore(x,porc)

p_aux = porc*abs(max(x)-min(x));
R = sort(x);

C = 1:length(x);
N = max(C);

C = (1-1/N)*C/N;
incr_R = (R(2)-R(1))/2;

R = [min(x)-p_aux R(1:end) max(x)+p_aux];
C = [0 C 1]; 

x_lin = interp1(made_monotonic(R),C,x); 

T.C = C;
T.R = R;

