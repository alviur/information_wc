function [s_H1,s_H2,s_H3,s_H4,s_M1,s_M2,s_M3,s_M4,s_L1,s_L2,s_L3,s_L4,s_low] = scan_wavelet_for_spatial_samples_full(w,ind_focus,tam) 

%
% SCAN_WAVELET_FOR_SPATIAL_SAMPLES_full gets samples for diferent orientations
% and scales corresponding to multiple (but corresponding) spatial
% locations covering the visual field. The differences with the "no_full" version are:
%   - here we dont do subsampling, therefore the samples have all the spatial locations in the considered field
%   - arrangement is different: samples are given in separate matrices that
%     can be combined afterwards as desired.
%
% CAUTION!: hardcoded for 3 scales 4 orientations!
%
% [s_H1,s_H2,s_H3,s_H4,s_M1,s_M2,s_M3,s_M4,s_L1,s_L2,s_L3,s_L4,s_low] = scan_wavelet_for_spatial_samples_full(w,ind,size_spatial_field_in_low_freq);
%
% Examples of posterios data arrangements (all assume stationarity in space -put together samples from different locations-): 
%   
%   samples_space_scale_orient = [s_H1;s_H2;s_H3;s_H4;s_M1;s_M2;s_M3;s_M4;s_L1;s_L2;s_L3;s_L4;s_low];
%
%   samples_orientation_ij = [s_Hi;s_Mi;s_Li;s_Hj;s_Mj;s_Lj;s_low];
%
%      - Extra assumption of stationarity over scale would imply interpolating the
%        different scales to make them of the same spatial size and considering
%        them together (mixing data from different scales).
%
%   samples_scale_ij = [s_i1;s_i2;s_i3;s_i4;s_j1;s_j2;s_j3;s_j4];
%      - Assuming stationarity in orient... [s_i1 s_i2 s_i3 s_i4;s_j1 s_j2 s_j3 s_j4];
%  
%

low_res = pyrBand(w,ind_focus,length(ind_focus(:,1)));

if tam == 2
   low_res = imresize(low_res,[4 4]);
elseif tam == 3
   low_res = low_res;
else
   low_res = low_res(1:2,1:2);
end

s_low = low_res(:)';
% length(s_low)
variable_band = ['s_H1';'s_H2';'s_H3';'s_H4';'s_M1';'s_M2';'s_M3';'s_M4';'s_L1';'s_L2';'s_L3';'s_L4'];

for i = 2:13
    
    B = pyrBand(w,ind_focus,i);
    if i<6
       samples = im2col(B,[4*tam 4*tam],'distinct'); 
    elseif i>=6 & i<10
       samples = im2col(B,[2*tam 2*tam],'distinct'); 
    elseif i>=10
       samples = im2col(B,[tam tam],'distinct');  
    end
    eval([variable_band(i-1,:),'= samples;']);
    
end
