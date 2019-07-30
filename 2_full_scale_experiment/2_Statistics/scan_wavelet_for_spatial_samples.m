function samples = scan_wavelet_for_spatial_samples(w,ind_focus,tam,conservative) 

% SCAN_WAVELET_FOR_SPATIAL_SAMPLES gets samples for diferent orientations
% and scales corresponding to multiple (but corresponding) spatial
% locations covering the visual field.
%
% CAUTION!: hardcoded for 3 scales 4 orientations!
%
% samples = scan_wavelet_for_spatial_samples(w,ind,size_spatial_field,small_big) 
%

increm_list = [1 ceil(tam/2) tam 1 ceil(tam/2) tam 1 ceil(tam/2) tam;
               1 1 1 ceil(tam/2) ceil(tam/2) ceil(tam/2) tam tam tam]-1;
increm_list_cons = [1  tam ceil(tam/2) 1       tam;
                    1  1   ceil(tam/2) tam tam]-1;
                
if conservative == 1
   list = increm_list_cons;
else
   list = increm_list;    
end

for or = 1:4
    % band_L = pyrBand(w,ind_focus,9+or);
    % band_M = pyrBand(w,ind_focus,5+or);
    % band_H = pyrBand(w,ind_focus,1+or);
    
    samples_L_or = [];
    samples_M_or = [];
    samples_H_or = [];
    
    for i=1:ind_focus(10,1)
       for j=1:ind_focus(10,2)
           pos = [];
           posM = [];
           posH = [];
           for ii = 1:length(list(1,:))
               [position,band] = pos_s_pyr([3 or i+list(1,ii) j+list(2,ii)],ind_focus,0);
               pos = [pos;position];
               [positionM,band] = pos_s_pyr([2 or 2*i+2*list(1,ii) 2*j+2*list(2,ii)],ind_focus,0);
               posM = [posM;positionM];
               [positionH,band] = pos_s_pyr([1 or 4*i+4*list(1,ii) 4*j+4*list(2,ii)],ind_focus,0);
               posH = [posH;positionH];
           end
           if isempty(find(pos == 0)) & isempty(find(posM == 0)) & isempty(find(posH == 0))
              samples_L_or = [samples_L_or w(pos)];
              samples_M_or = [samples_M_or w(posM)];
              samples_H_or = [samples_H_or w(posH)];
           end
       end
    end
    % size(samples_L_or),size(samples_M_or),size(samples_H_or)
    samples(1,or).samples = samples_H_or;      
    samples(2,or).samples = samples_M_or;      
    samples(3,or).samples = samples_L_or;      
end