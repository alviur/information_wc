function A = put_diags(A,v);

% PUT_DIAGS changes the value of the diagonals of a set of squared matrices stacked one below the other.
% The desired diagonals are provided as a single vector (with one diagonal stacked one below the other). 
% 
% USE: B = put_diags(A,v);
%
%    A = matrix of size (d*Nm)*d that contains the Nm stacked matrices.
%    v = column vector with the Nm desired diagonals.
%    B = matrix with the new diagonals (the other elements remain the same)
%

d = size(A,2);
Nm = size(A,1)/d;

A = A';

indexT = repmat([1:d+1:d^2],[Nm 1])+ repmat([0:d^2:numel(A)-1]',[1 d]);
%indexT = indexT(:);
%indexT = sort(indexT);
indexT = flipud(indexT);
indexT = indexT';
indexT = fliplr(indexT);
indexT = indexT(:);

A(indexT)=v;

A = A';
