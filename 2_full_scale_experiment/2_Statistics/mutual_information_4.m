function [MI]=mutual_information_4(RV1,RV2,Nbins_IM)

%
% MUTUAL_INFORMATION computes mutual information
% measure between two random variables RV1 and RV2 given N realizations of
% such variables.
%
% MI = mutual_information_4(RV1,RV2,Nbins)
%
% * Inputs
%     RV1 = N realizations of the random variable 1 (row vector)
%     RV2 = N realizations of the random variable 2 (row vector)
%     Nbins = number of bins for the joint PDF estimation
%
% * Outputs
%     MI = Mutual information in bits
%

% Preparing data

RV1 = (RV1-min(RV1))/(max(RV1)-min(RV1));
RV2 = (RV2-min(RV2))/(max(RV2)-min(RV2));

% DD=sort(diff(sort(RV1)));
% DD=DD(find(DD>0));
% RV1 = RV1 + 0.2*DD(1)*rand(size(RV1));
%
% DD=sort(diff(sort(RV2)));
% DD=DD(find(DD>0));
% RV2 = RV2 + 0.2*DD(1)*rand(size(RV2));

% (2) Uniformize maginals 
% (this do not change the mutual-information but helps in order to to exploit the full histogram range)

RV1 = marginal_uniformization_hardcore(RV1,0.1);
RV2 = marginal_uniformization_hardcore(RV2,0.1);

% Bidimensional histogram

RV = [RV1' RV2']';
[H,R]=hist3(RV',[round(sqrt(Nbins_IM)) round(sqrt(Nbins_IM))]);

R1=R{1};
R2=R{2};

% Obtaining 2D pdf (Normalizing histogram)

delta1 = R1(3)-R1(2);
delta2 = R2(3)-R2(2); 

% Ontaining 2D entropy (this step can be improved)

pp = H(:);
%pp(pp==0) = 1e-30; % QUITALO!!!!!!!!!

% mle estimator with miller-maddow correction

c = 0.5 * (sum(pp>0)-1)/sum(pp);  % miller maddow correction
pp = pp/sum(pp);               % empirical estimate of the distribution
idx = pp~=0;
h = -sum(pp(idx).*log2(pp(idx))) + c;     % plug-in estimator of the entropy with correction
h = h + log2(delta1*delta2);

% MI = h(RV1) + h(RV2) - h(RV1,RV2)
% after marginal uniformization h(RV1)=h(RV2)=0

MI = -h;
if MI<0,MI=0;end

