parameters_3D_small;close all
for i=1:2
    % ------------ this is the one
    %% Redundancy_Reduction_small_DN_WC(param,500,['_p3_N500_realiz_',num2str(i)])
    %% Redundancy_Reduction_small_DN_WC(param,1000,['_p3_N1000_realiz_',num2str(i)])
    % Redundancy_Reduction_small_DN_WC(param,2000,['_p3_N2000_realiz_',num2str(i)])
    % Redundancy_Reduction_small_DN_WC(param,4000,['_p3_N4000_realiz_',num2str(i)])
    Redundancy_Reduction_small_DN_WC(param,8000,['_p3_N8000_realiz_',num2str(i)])
    Redundancy_Reduction_small_DN_WC(param,16000,['_p3_N16000_realiz_',num2str(i)])
end

parameters_3D_small;close all
for i=3:4
    %% Redundancy_Reduction_small_DN_WC(param,500,['_p3_N500_realiz_',num2str(i)])
    %% Redundancy_Reduction_small_DN_WC(param,1000,['_p3_N1000_realiz_',num2str(i)])
    % Redundancy_Reduction_small_DN_WC(param,2000,['_p3_N2000_realiz_',num2str(i)])
    % Redundancy_Reduction_small_DN_WC(param,4000,['_p3_N4000_realiz_',num2str(i)])  
    Redundancy_Reduction_small_DN_WC(param,8000,['_p3_N8000_realiz_',num2str(i)])
    Redundancy_Reduction_small_DN_WC(param,16000,['_p3_N16000_realiz_',num2str(i)])
end

parameters_3D_small;close all
for i=5:6
    %% Redundancy_Reduction_small_DN_WC(param,250,['_p3_N250_realiz_',num2str(i)])
    %% Redundancy_Reduction_small_DN_WC(param,500,['_p3_N500_realiz_',num2str(i)])
    %% Redundancy_Reduction_small_DN_WC(param,1000,['_p3_N1000_realiz_',num2str(i)])
    % Redundancy_Reduction_small_DN_WC(param,2000,['_p3_N2000_realiz_',num2str(i)])
    % Redundancy_Reduction_small_DN_WC(param,4000,['_p3_N4000_realiz_',num2str(i)]) 
    Redundancy_Reduction_small_DN_WC(param,8000,['_p3_N8000_realiz_',num2str(i)])
    Redundancy_Reduction_small_DN_WC(param,16000,['_p3_N16000_realiz_',num2str(i)])
end

parameters_3D_small;close all
for i=7:8
    %% Redundancy_Reduction_small_DN_WC(param,250,['_p3_N250_realiz_',num2str(i)])
    %% Redundancy_Reduction_small_DN_WC(param,500,['_p3_N500_realiz_',num2str(i)])
    %% Redundancy_Reduction_small_DN_WC(param,1000,['_p3_N1000_realiz_',num2str(i)])
    % Redundancy_Reduction_small_DN_WC(param,2000,['_p3_N2000_realiz_',num2str(i)])
    % Redundancy_Reduction_small_DN_WC(param,4000,['_p3_N4000_realiz_',num2str(i)])  
    Redundancy_Reduction_small_DN_WC(param,8000,['_p3_N8000_realiz_',num2str(i)])
    Redundancy_Reduction_small_DN_WC(param,16000,['_p3_N16000_realiz_',num2str(i)])
end

%%%%%%%%%%%%%%%%%%

clear all;parameters_3D_small;close all
Redundancy_Reduction_small_DN_WC(param,4e6,['_p3_N4e6_realiz_1'])

clear all;parameters_3D_small;close all
Redundancy_Reduction_small_DN_WC(param,4e6,['_p3_N4e6_realiz_2'])

clear all;parameters_3D_small;close all
Redundancy_Reduction_small_DN_WC(param,4e6,['_p3_N4e6_realiz_3'])

clear all;parameters_3D_small;close all
Redundancy_Reduction_small_DN_WC(param,4e6,['_p3_N4e6_realiz_4'])

clear all;parameters_3D_small;close all
Redundancy_Reduction_small_DN_WC(param,4e6,['_p3_N4e6_realiz_5'])

clear all;parameters_3D_small;close all
Redundancy_Reduction_small_DN_WC(param,4e6,['_p3_N4e6_realiz_6'])

clear all;parameters_3D_small;close all
Redundancy_Reduction_small_DN_WC(param,4e6,['_p3_N4e6_realiz_7'])

clear all;parameters_3D_small;close all
Redundancy_Reduction_small_DN_WC(param,4e6,['_p3_N4e6_realiz_8'])

%%%%%%%%%%%%%%%%%%%%%%%%% FASTER (limits maximum number of samples per L-C region to analyze redundancy reduction)

clear all;parameters_3D_small;close all
Ntot = 5e6;      % Samples extracted from Van Hateren 
Nmax = 10000;    % Maximum number of Samples in eah luminance-contrast bin
integrate = 0;   % If 1 gathers samples and integrates WC, if 0 loads precomputed data and response.
Redundancy_Reduction_small_DN_WC_super_fast(param,5e6,10000,['_N5e6_realiz_1'],integrate)
close all

clear all;parameters_3D_small;close all 
Ntot = 5e6;      % Samples extracted from Van Hateren  
Nmax = 10000;    % Maximum number of Samples in eah luminance-contrast bin 
integrate = 0;   % If 1 gathers samples and integrates WC, if 0 loads precomputed data and response.
Redundancy_Reduction_small_DN_WC_super_fast(param,5e6,10000,['_N5e6_realiz_2'],integrate)
close all

clear all;parameters_3D_small;close all
Ntot = 5e6;      % Samples extracted from Van Hateren 
Nmax = 10000;    % Maximum number of Samples in eah luminance-contrast bin
integrate = 0;   % If 1 gathers samples and integrates WC, if 0 loads precomputed data and response.
Redundancy_Reduction_small_DN_WC_super_fast(param,5e6,10000,['_N5e6_realiz_3'],integrate)

clear all;parameters_3D_small;close all
Ntot = 5e6;      % Samples extracted from Van Hateren 
Nmax = 10000;    % Maximum number of Samples in eah luminance-contrast bin
integrate = 0;   % If 1 gathers samples and integrates WC, if 0 loads precomputed data and response.
Redundancy_Reduction_small_DN_WC_super_fast(param,5e6,10000,['_N5e6_realiz_4'],integrate)

clear all;parameters_3D_small;close all
Ntot = 5e6;      % Samples extracted from Van Hateren 
Nmax = 10000;    % Maximum number of Samples in eah luminance-contrast bin
integrate = 0;   % If 1 gathers samples and integrates WC, if 0 loads precomputed data and response.
Redundancy_Reduction_small_DN_WC_super_fast(param,5e6,10000,['_N5e6_realiz_5'],integrate)

clear all;parameters_3D_small;close all
Ntot = 5e6;      % Samples extracted from Van Hateren 
Nmax = 10000;    % Maximum number of Samples in eah luminance-contrast bin
integrate = 0;   % If 1 gathers samples and integrates WC, if 0 loads precomputed data and response.
Redundancy_Reduction_small_DN_WC_super_fast(param,5e6,10000,['_N5e6_realiz_6'],integrate)

clear all;parameters_3D_small;close all
Ntot = 5e6;      % Samples extracted from Van Hateren 
Nmax = 10000;    % Maximum number of Samples in eah luminance-contrast bin
integrate = 0;   % If 1 gathers samples and integrates WC, if 0 loads precomputed data and response.
Redundancy_Reduction_small_DN_WC_super_fast(param,5e6,10000,['_N5e6_realiz_7'],integrate)

clear all;parameters_3D_small;close all
Ntot = 5e6;      % Samples extracted from Van Hateren 
Nmax = 10000;    % Maximum number of Samples in eah luminance-contrast bin
integrate = 0;   % If 1 gathers samples and integrates WC, if 0 loads precomputed data and response.
Redundancy_Reduction_small_DN_WC_super_fast(param,5e6,10000,['_N5e6_realiz_8'],integrate)

clear all;parameters_3D_small;close all
Ntot = 5e6;      % Samples extracted from Van Hateren 
Nmax = 10000;    % Maximum number of Samples in eah luminance-contrast bin
integrate = 0;   % If 1 gathers samples and integrates WC, if 0 loads precomputed data and response.
Redundancy_Reduction_small_DN_WC_super_fast(param,5e6,10000,['_N5e6_realiz_9'],integrate)

clear all;parameters_3D_small;close all
Ntot = 5e6;      % Samples extracted from Van Hateren 
Nmax = 10000;    % Maximum number of Samples in eah luminance-contrast bin
integrate = 0;   % If 1 gathers samples and integrates WC, if 0 loads precomputed data and response.
Redundancy_Reduction_small_DN_WC_super_fast(param,5e6,10000,['_N5e6_realiz_10'],integrate)

clear all;parameters_3D_small;close all
Ntot = 5e6;      % Samples extracted from Van Hateren 
Nmax = 10000;    % Maximum number of Samples in eah luminance-contrast bin
integrate = 0;   % If 1 gathers samples and integrates WC, if 0 loads precomputed data and response.
Redundancy_Reduction_small_DN_WC_super_fast(param,5e6,10000,['_N5e6_realiz_11'],integrate)

clear all;parameters_3D_small;close all
Ntot = 5e6;      % Samples extracted from Van Hateren 
Nmax = 10000;    % Maximum number of Samples in eah luminance-contrast bin
integrate = 0;   % If 1 gathers samples and integrates WC, if 0 loads precomputed data and response.
Redundancy_Reduction_small_DN_WC_super_fast(param,5e6,10000,['_N5e6_realiz_12'],integrate)

clear all;parameters_3D_small;close all
Ntot = 5e6;      % Samples extracted from Van Hateren 
Nmax = 10000;    % Maximum number of Samples in eah luminance-contrast bin
integrate = 0;   % If 1 gathers samples and integrates WC, if 0 loads precomputed data and response.
Redundancy_Reduction_small_DN_WC_super_fast(param,5e6,10000,['_N5e6_realiz_13'],integrate)

clear all;parameters_3D_small;close all
Ntot = 5e6;      % Samples extracted from Van Hateren 
Nmax = 10000;    % Maximum number of Samples in eah luminance-contrast bin
integrate = 0;   % If 1 gathers samples and integrates WC, if 0 loads precomputed data and response.
Redundancy_Reduction_small_DN_WC_super_fast(param,5e6,10000,['_N5e6_realiz_14'],integrate)

clear all;parameters_3D_small;close all
Ntot = 5e6;      % Samples extracted from Van Hateren 
Nmax = 10000;    % Maximum number of Samples in eah luminance-contrast bin
integrate = 0;   % If 1 gathers samples and integrates WC, if 0 loads precomputed data and response.
Redundancy_Reduction_small_DN_WC_super_fast(param,5e6,10000,['_N5e6_realiz_15'],integrate)

clear all;parameters_3D_small;close all
Ntot = 5e6;      % Samples extracted from Van Hateren 
Nmax = 10000;    % Maximum number of Samples in eah luminance-contrast bin
integrate = 0;   % If 1 gathers samples and integrates WC, if 0 loads precomputed data and response.
Redundancy_Reduction_small_DN_WC_super_fast(param,5e6,10000,['_N5e6_realiz_16'],integrate)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%% FASTER & NICER  (limits maximum number of samples per L-C region to analyze redundancy reduction, focus on the [0 1] square)

clear all;parameters_3D_small;close all
Ntot = 5e6;      % Samples extracted from Van Hateren 
Nmax = 10000;    % Maximum number of Samples in eah luminance-contrast bin
integrate = 1;   % If 1 gathers samples and integrates WC, if 0 loads precomputed data and response.
Redundancy_Reduction_small_DN_WC_super_fast(param,5e6,10000,['_N5e6_01_realiz_1'],integrate)
close all

clear all;parameters_3D_small;close all
Ntot = 5e6;      % Samples extracted from Van Hateren 
Nmax = 10000;    % Maximum number of Samples in eah luminance-contrast bin
integrate = 1;   % If 1 gathers samples and integrates WC, if 0 loads precomputed data and response.
Redundancy_Reduction_small_DN_WC_super_fast(param,5e6,10000,['_N5e6_01_realiz_2'],integrate)
close all

clear all;parameters_3D_small;close all
Ntot = 5e6;      % Samples extracted from Van Hateren 
Nmax = 10000;    % Maximum number of Samples in eah luminance-contrast bin
integrate = 1;   % If 1 gathers samples and integrates WC, if 0 loads precomputed data and response.
Redundancy_Reduction_small_DN_WC_super_fast(param,5e6,10000,['_N5e6_01_realiz_3'],integrate)
close all

clear all;parameters_3D_small;close all
Ntot = 5e6;      % Samples extracted from Van Hateren 
Nmax = 10000;    % Maximum number of Samples in eah luminance-contrast bin
integrate = 1;   % If 1 gathers samples and integrates WC, if 0 loads precomputed data and response.
Redundancy_Reduction_small_DN_WC_super_fast(param,5e6,10000,['_N5e6_01_realiz_4'],integrate)
close all

clear all;parameters_3D_small;close all
Ntot = 5e6;      % Samples extracted from Van Hateren 
Nmax = 10000;    % Maximum number of Samples in eah luminance-contrast bin
integrate = 1;   % If 1 gathers samples and integrates WC, if 0 loads precomputed data and response.
Redundancy_Reduction_small_DN_WC_super_fast(param,5e6,10000,['_N5e6_01_realiz_5'],integrate)
close all

clear all;parameters_3D_small;close all
Ntot = 5e6;      % Samples extracted from Van Hateren 
Nmax = 10000;    % Maximum number of Samples in eah luminance-contrast bin
integrate = 1;   % If 1 gathers samples and integrates WC, if 0 loads precomputed data and response.
Redundancy_Reduction_small_DN_WC_super_fast(param,5e6,10000,['_N5e6_01_realiz_6'],integrate)
close all

clear all;parameters_3D_small;close all
Ntot = 5e6;      % Samples extracted from Van Hateren 
Nmax = 10000;    % Maximum number of Samples in eah luminance-contrast bin
integrate = 1;   % If 1 gathers samples and integrates WC, if 0 loads precomputed data and response.
Redundancy_Reduction_small_DN_WC_super_fast(param,5e6,10000,['_N5e6_01_realiz_7'],integrate)
close all

clear all;parameters_3D_small;close all
Ntot = 5e6;      % Samples extracted from Van Hateren 
Nmax = 10000;    % Maximum number of Samples in eah luminance-contrast bin
integrate = 1;   % If 1 gathers samples and integrates WC, if 0 loads precomputed data and response.
Redundancy_Reduction_small_DN_WC_super_fast(param,5e6,10000,['_N5e6_01_realiz_8'],integrate)
close all

clear all;parameters_3D_small;close all
Ntot = 5e6;      % Samples extracted from Van Hateren 
Nmax = 10000;    % Maximum number of Samples in eah luminance-contrast bin
integrate = 1;   % If 1 gathers samples and integrates WC, if 0 loads precomputed data and response.
Redundancy_Reduction_small_DN_WC_super_fast(param,5e6,10000,['_N5e6_01_realiz_9'],integrate)
close all

clear all;parameters_3D_small;close all
Ntot = 5e6;      % Samples extracted from Van Hateren 
Nmax = 10000;    % Maximum number of Samples in eah luminance-contrast bin
integrate = 1;   % If 1 gathers samples and integrates WC, if 0 loads precomputed data and response.
Redundancy_Reduction_small_DN_WC_super_fast(param,5e6,10000,['_N5e6_01_realiz_10'],integrate)
close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%% FOSTER: FASTER & NICER  (limits maximum number of samples per L-C region to analyze redundancy reduction, focus on the [0 1] square)

% First execute Foster_data_3D (to gather the data)
clear all;parameters_3D_small;close all
Nmax = 10000;    % Maximum number of Samples in eah luminance-contrast bin
integrate = 1;   % If 1 gathers samples and integrates WC, if 0 loads precomputed data and response.
Redundancy_Reduction_small_DN_WC_super_foster(param,10000,['_foster_N7e6_01_realiz_1'],integrate)
close all

clear all;parameters_3D_small;close all
Nmax = 10000;    % Maximum number of Samples in eah luminance-contrast bin
integrate = 1;   % If 1 gathers samples and integrates WC, if 0 loads precomputed data and response.
Redundancy_Reduction_small_DN_WC_super_foster(param,10000,['_foster_N7e6_01_realiz_2'],integrate)
close all

clear all;parameters_3D_small;close all
Nmax = 10000;    % Maximum number of Samples in eah luminance-contrast bin
integrate = 1;   % If 1 gathers samples and integrates WC, if 0 loads precomputed data and response.
Redundancy_Reduction_small_DN_WC_super_foster(param,10000,['_foster_N7e6_01_realiz_3'],integrate)
close all

clear all;parameters_3D_small;close all
Nmax = 10000;    % Maximum number of Samples in eah luminance-contrast bin
integrate = 1;   % If 1 gathers samples and integrates WC, if 0 loads precomputed data and response.
Redundancy_Reduction_small_DN_WC_super_foster(param,10000,['_foster_N7e6_01_realiz_4'],integrate)
close all

clear all;parameters_3D_small;close all
Nmax = 10000;    % Maximum number of Samples in eah luminance-contrast bin
integrate = 1;   % If 1 gathers samples and integrates WC, if 0 loads precomputed data and response.
Redundancy_Reduction_small_DN_WC_super_foster(param,10000,['_foster_N7e6_01_realiz_5'],integrate)
close all

clear all;parameters_3D_small;close all
Nmax = 10000;    % Maximum number of Samples in eah luminance-contrast bin
integrate = 1;   % If 1 gathers samples and integrates WC, if 0 loads precomputed data and response.
Redundancy_Reduction_small_DN_WC_super_foster(param,10000,['_foster_N7e6_01_realiz_6'],integrate)
close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%% FASTER & NICER  (limits maximum number of samples per L-C region to analyze redundancy reduction, focus on the [0 1] square)

clear all;parameters_3D_small;close all
Ntot = 5e6;      % Samples extracted from Van Hateren 
Nmax = 25000;    % Maximum number of Samples in eah luminance-contrast bin
integrate = 1;   % If 1 gathers samples and integrates WC, if 0 loads precomputed data and response.
Redundancy_Reduction_small_DN_WC_super_fast(param,5e6,25000,['_N5e6_25_wide_realiz_1'],integrate)
close all

clear all;parameters_3D_small;close all
Ntot = 5e6;      % Samples extracted from Van Hateren 
Nmax = 25000;    % Maximum number of Samples in eah luminance-contrast bin
integrate = 1;   % If 1 gathers samples and integrates WC, if 0 loads precomputed data and response.
Redundancy_Reduction_small_DN_WC_super_fast(param,5e6,25000,['_N5e6_25_wide_realiz_2'],integrate)
close all

clear all;parameters_3D_small;close all
Ntot = 5e6;      % Samples extracted from Van Hateren 
Nmax = 25000;    % Maximum number of Samples in eah luminance-contrast bin
integrate = 1;   % If 1 gathers samples and integrates WC, if 0 loads precomputed data and response.
Redundancy_Reduction_small_DN_WC_super_fast(param,5e6,25000,['_N5e6_25_wide_realiz_3'],integrate)
close all

clear all;parameters_3D_small;close all
Ntot = 5e6;      % Samples extracted from Van Hateren 
Nmax = 25000;    % Maximum number of Samples in eah luminance-contrast bin
integrate = 1;   % If 1 gathers samples and integrates WC, if 0 loads precomputed data and response.
Redundancy_Reduction_small_DN_WC_super_fast(param,5e6,25000,['_N5e6_25_wide_realiz_4'],integrate)
close all

clear all;parameters_3D_small;close all
Ntot = 5e6;      % Samples extracted from Van Hateren 
Nmax = 25000;    % Maximum number of Samples in eah luminance-contrast bin
integrate = 1;   % If 1 gathers samples and integrates WC, if 0 loads precomputed data and response.
Redundancy_Reduction_small_DN_WC_super_fast(param,5e6,25000,['_N5e6_25_wide_realiz_5'],integrate)
close all

clear all;parameters_3D_small;close all
Ntot = 5e6;      % Samples extracted from Van Hateren 
Nmax = 25000;    % Maximum number of Samples in eah luminance-contrast bin
integrate = 1;   % If 1 gathers samples and integrates WC, if 0 loads precomputed data and response.
Redundancy_Reduction_small_DN_WC_super_fast(param,5e6,25000,['_N5e6_25_wide_realiz_6'],integrate)
close all

clear all;parameters_3D_small;close all
Ntot = 5e6;      % Samples extracted from Van Hateren 
Nmax = 25000;    % Maximum number of Samples in eah luminance-contrast bin
integrate = 1;   % If 1 gathers samples and integrates WC, if 0 loads precomputed data and response.
Redundancy_Reduction_small_DN_WC_super_fast(param,5e6,25000,['_N5e6_25_wide_realiz_7'],integrate)
close all

clear all;parameters_3D_small;close all
Ntot = 5e6;      % Samples extracted from Van Hateren 
Nmax = 25000;    % Maximum number of Samples in eah luminance-contrast bin
integrate = 1;   % If 1 gathers samples and integrates WC, if 0 loads precomputed data and response.
Redundancy_Reduction_small_DN_WC_super_fast(param,5e6,25000,['_N5e6_25_wide_realiz_8'],integrate)
close all

clear all;parameters_3D_small;close all
Ntot = 5e6;      % Samples extracted from Van Hateren 
Nmax = 25000;    % Maximum number of Samples in eah luminance-contrast bin
integrate = 1;   % If 1 gathers samples and integrates WC, if 0 loads precomputed data and response.
Redundancy_Reduction_small_DN_WC_super_fast(param,5e6,25000,['_N5e6_25_wide_realiz_9'],integrate)
close all

clear all;parameters_3D_small;close all
Ntot = 5e6;      % Samples extracted from Van Hateren 
Nmax = 25000;    % Maximum number of Samples in eah luminance-contrast bin
integrate = 1;   % If 1 gathers samples and integrates WC, if 0 loads precomputed data and response.
Redundancy_Reduction_small_DN_WC_super_fast(param,5e6,25000,['_N5e6_25_wide_realiz_10'],integrate)
close all
