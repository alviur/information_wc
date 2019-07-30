function [Hff,indf] = focus_on_center_kernel(H,ind,pix_remove)

% FOCUS_ON_CENTER_KERNEL removes the outer part of each subband of a wavelet-like Kernel
% thus 'focusing' on the center of the spatial domain.
% It leads to a smaller kernel matrix.
%
% [H_focus, ind_focus] = focus_on_center_kernel(H,ind,pix_remove)
%

Hf = [];
for i=1:length(H(1,:))
    p = H(:,i);
    [wf,indf] = focus_on_center(p,ind,pix_remove); 
    Hf = [Hf wf];
end
clear H
Hff = [];
for i=1:length(Hf(:,1))
    p = Hf(i,:);
    [wf,indf] = focus_on_center(p',ind,pix_remove); 
    Hff = [Hff;wf'];    
end
