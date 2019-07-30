function [wf,indf] = focus_on_center(w,ind,pix_remove)

% FOCUS_ON_CENTER removes the outer part of each subband of a wavelet transform
% thus 'focusing' on the center of the spatial domain.
% It leads to a shorter wavelet vector.
%
% [w_focus, ind_focus] = focus_on_center(w,ind,pix_remove)

%

wf = [];
indf = ind;
for i=1:length(ind(:,1))
    dim = ind(i,:);
    indf(i,:) = dim - 2*pix_remove(i);
    B = pyrBand(w, ind, i);
    Bf = B(pix_remove(i)+1:end-pix_remove(i),pix_remove(i)+1:end-pix_remove(i));
    wf = [wf;Bf(:)];
end
end
